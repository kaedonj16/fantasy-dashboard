"""Tests for realized-outcome grading of weekly breakout calls.

The grading math is pure (grade_call / classify_outcome / summarize over
plain dicts, mirroring tests/test_weekly_breakout.py); the orchestrator is
tested with the module's DB loaders/saver monkeypatched to in-memory fakes.
"""
from __future__ import annotations

from datetime import date

import pytest

from data_building.breakout_engine import weekly_grading as wg


def wk(week, snap=None, tgt=None, car=None, ppr=0.0):
    """One player_weekly_metrics-shaped row (shares are 0-100)."""
    return {
        "week": week, "snap_pct": snap, "targets": tgt, "carries": car,
        "ppr_pts": ppr, "snaps": 50,
    }


def make_call(as_of_week=4, *, classification="emerging_breakout",
              scoring_version="weekly-v5", baseline_weeks=(1, 2),
              baseline_source="current_season", baseline_ppg=8.0,
              baseline_snap=40.0, baseline_opp=10.0):
    """A persisted weekly_breakout_scores row with stored-evidence baselines.

    baseline_opp is the RB carries+targets composite the engine stores as
    signals.carry_opportunity_pg; pass None to omit it (non-RB shape).
    """
    signals = {"snap_share": {"baseline": baseline_snap, "recent": 62.0}}
    if baseline_opp is not None:
        signals["carry_opportunity_pg"] = {"baseline": baseline_opp, "recent": 16.0}
    return {
        "player_id": "101", "player_name": "Test Player", "season": 2026,
        "as_of_week": as_of_week, "as_of_date": date(2026, 10, 1),
        "scoring_version": scoring_version, "classification": classification,
        "breakout_score": 55.0, "confidence": 0.8,
        "baseline_source": baseline_source,
        "baseline_weeks": list(baseline_weeks),
        "evidence": {
            "fantasy": {"baseline_ppg": baseline_ppg, "recent_ppg": 14.0},
            "signals": signals,
        },
    }


def _no_db(monkeypatch):
    monkeypatch.setattr(wg, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(wg, "init_weekly_breakout_grades_db", lambda: None)


# ---------------------------------------------------------------------------
# grade_call: hit / miss / partial / ungraded
# ---------------------------------------------------------------------------

def test_hit_when_role_holds_and_production_jumps():
    call = make_call()
    rows = [
        # Baseline-window rows disagree with the stored baseline on purpose:
        # the stored evidence is authoritative and must win.
        wk(1, snap=20, tgt=1, car=2, ppr=3.0),
        wk(2, snap=20, tgt=1, car=2, ppr=3.0),
        # Outcome weeks 5-7: role up (opp 9+11=20/g vs 10, snap 65 vs 40)
        # and production up (18.0 vs 8.0 PPG).
        wk(5, snap=65, tgt=9, car=11, ppr=18.0),
        wk(6, snap=65, tgt=9, car=11, ppr=18.0),
        wk(7, snap=65, tgt=9, car=11, ppr=18.0),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "hit"
    assert grade["outcome_games"] == 3
    assert grade["outcome_weeks"] == [5, 6, 7]
    assert grade["baseline_source"] == "stored"
    assert grade["baseline_ppg"] == 8.0
    assert grade["outcome_ppg"] == 18.0
    assert grade["ppg_delta"] == 10.0
    assert grade["opp_delta"] == 10.0
    assert grade["snap_delta"] == 25.0
    assert grade["detail"]["role_state"] == "held"


def test_miss_when_role_reverts_to_baseline():
    call = make_call()
    rows = [
        wk(5, snap=38, tgt=2, car=7, ppr=9.0),   # opp 9 (delta -1), snap -2
        wk(6, snap=38, tgt=2, car=7, ppr=9.0),
        wk(7, snap=38, tgt=2, car=7, ppr=9.0),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "miss"
    assert grade["detail"]["role_state"] == "reverted"
    assert grade["opp_delta"] == -1.0
    assert grade["snap_delta"] == -2.0


def test_partial_when_role_holds_but_ppg_flat():
    call = make_call(baseline_ppg=12.0)
    rows = [
        wk(5, snap=60, tgt=6, car=9, ppr=12.5),  # opp 15 (+5), snap +20, PPG +0.5
        wk(6, snap=60, tgt=6, car=9, ppr=12.5),
        wk(7, snap=60, tgt=6, car=9, ppr=12.5),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "partial"
    assert grade["detail"]["role_state"] == "held"
    assert grade["ppg_delta"] == 0.5


def test_partial_when_role_partially_retained():
    call = make_call()
    rows = [
        # opp +1.0 (past reverted, short of held), snap +5 (same zone),
        # production way up: still partial, the role did not clearly hold.
        wk(5, snap=45, tgt=4, car=7, ppr=20.0),
        wk(6, snap=45, tgt=4, car=7, ppr=20.0),
        wk(7, snap=45, tgt=4, car=7, ppr=20.0),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "partial"
    assert grade["detail"]["role_state"] == "mixed"


def test_ungraded_when_player_inactive_all_outcome_weeks():
    call = make_call()
    rows = [wk(1, snap=40, tgt=4, car=6, ppr=8.0),
            wk(2, snap=40, tgt=4, car=6, ppr=8.0)]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "ungraded"
    assert grade["outcome_games"] == 0
    assert grade["detail"]["reason"] == "no_outcome_games"


def test_ungraded_when_no_role_baseline_exists():
    call = make_call(baseline_ppg=None, baseline_snap=None, baseline_opp=None,
                     baseline_weeks=(), baseline_source="none")
    rows = [wk(5, snap=70, tgt=8, car=4, ppr=20.0),
            wk(6, snap=70, tgt=8, car=4, ppr=20.0),
            wk(7, snap=70, tgt=8, car=4, ppr=20.0)]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "ungraded"
    assert grade["baseline_source"] == "none"
    assert grade["detail"]["reason"] == "no_role_baseline"


# ---------------------------------------------------------------------------
# baseline resolution
# ---------------------------------------------------------------------------

def test_baseline_falls_back_to_pre_call_weeks_when_not_stored():
    call = make_call(baseline_ppg=None, baseline_snap=None, baseline_opp=None,
                     baseline_weeks=())
    rows = [
        wk(2, snap=30, tgt=3, car=2, ppr=6.0),   # pre-call baseline window
        wk(3, snap=30, tgt=3, car=2, ppr=6.0),
        wk(5, snap=55, tgt=6, car=8, ppr=15.0),  # opp 14 (+9), snap +25, PPG +9
        wk(6, snap=55, tgt=6, car=8, ppr=15.0),
        wk(7, snap=55, tgt=6, car=8, ppr=15.0),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["baseline_source"] == "computed"
    assert grade["baseline_ppg"] == 6.0
    assert grade["baseline_opp_pg"] == 5.0
    assert grade["grade"] == "hit"


def test_prior_season_baseline_never_uses_current_season_weeks():
    # Prior-season calls store a snap baseline but no PPG baseline; the
    # player's current-season games before the call are the *recent* window
    # and must not be promoted into a baseline.
    call = make_call(baseline_ppg=None, baseline_snap=45.0, baseline_opp=9.0,
                     baseline_weeks=(), baseline_source="prior_season")
    rows = [
        wk(2, snap=80, tgt=10, car=10, ppr=25.0),
        wk(3, snap=80, tgt=10, car=10, ppr=25.0),
        wk(5, snap=60, tgt=7, car=9, ppr=20.0),   # opp 16 (+7), snap +15
        wk(6, snap=60, tgt=7, car=9, ppr=20.0),
        wk(7, snap=60, tgt=7, car=9, ppr=20.0),
    ]
    grade = wg.grade_call(call, rows)
    assert grade["baseline_source"] == "stored"
    assert grade["baseline_ppg"] is None
    assert grade["ppg_delta"] is None
    # Role clearly held, but a hit requires a measured production rise.
    assert grade["grade"] == "partial"


# ---------------------------------------------------------------------------
# prior-season PPG fill: baseline PPG resolved from prior-season rows
# ---------------------------------------------------------------------------

def _prior_call():
    """A prior-season-baseline call: stored snap/opp, no stored PPG."""
    return make_call(baseline_ppg=None, baseline_snap=45.0, baseline_opp=9.0,
                     baseline_weeks=(), baseline_source="prior_season")


def _prior_rows(ppg=10.0):
    return [wk(w, snap=50, tgt=5, car=8, ppr=ppg) for w in (1, 2, 3)]


def _held_outcome_rows():
    # opp 16 (+7 vs stored 9), snap 60 (+15 vs stored 45), PPG 20.0.
    return [wk(5, snap=60, tgt=7, car=9, ppr=20.0),
            wk(6, snap=60, tgt=7, car=9, ppr=20.0),
            wk(7, snap=60, tgt=7, car=9, ppr=20.0)]


def test_prior_season_ppg_fill_from_prior_rows_can_hit():
    grade = wg.grade_call(_prior_call(), _held_outcome_rows(), _prior_rows())
    assert grade["grade"] == "hit"
    assert grade["baseline_ppg"] == 10.0     # prior-season mean PPG
    assert grade["ppg_delta"] == 10.0
    # Snap/opp came from the stored evidence, PPG from prior-season rows.
    assert grade["baseline_source"] == "mixed"


def test_prior_season_fill_uses_prior_rows_never_current_season():
    # Pre-call current-season weeks are the recent window: even with prior
    # rows supplied, their 25.0 PPG must not leak into the baseline.
    rows = [wk(2, snap=80, tgt=10, car=10, ppr=25.0),
            wk(3, snap=80, tgt=10, car=10, ppr=25.0)] + _held_outcome_rows()
    grade = wg.grade_call(_prior_call(), rows, _prior_rows())
    assert grade["baseline_ppg"] == 10.0
    assert grade["grade"] == "hit"


def test_prior_season_without_prior_rows_behaves_exactly_as_before():
    for prior_rows in (None, []):
        grade = wg.grade_call(_prior_call(), _held_outcome_rows(), prior_rows)
        assert grade["baseline_ppg"] is None
        assert grade["ppg_delta"] is None
        assert grade["baseline_source"] == "stored"
        # Role held but production rise unmeasured: partial ceiling.
        assert grade["grade"] == "partial"


def test_orchestrator_loads_prior_series_for_calls_that_need_it(monkeypatch):
    _no_db(monkeypatch)
    loaded = []

    def _series(season, through):
        loaded.append(season)
        if season == 2025:
            return {"101": _prior_rows()}
        return {"101": _held_outcome_rows()}

    saved = []
    monkeypatch.setattr(wg, "load_calls", lambda season: [_prior_call()])
    monkeypatch.setattr(wg, "load_existing_grade_keys", lambda season: set())
    monkeypatch.setattr(wg, "load_season_series", _series)
    monkeypatch.setattr(wg, "save_grade_rows",
                        lambda grades: saved.extend(grades) or len(grades))
    summary = wg.grade_weekly_breakouts(2026, through_week=7)
    assert loaded == [2026, 2025]            # current first, then prior
    assert summary["by_grade"] == {"hit": 1}
    assert saved[0]["baseline_ppg"] == 10.0
    assert saved[0]["baseline_source"] == "mixed"


def test_orchestrator_skips_prior_load_when_no_call_needs_it(monkeypatch):
    _no_db(monkeypatch)
    loaded = []

    def _series(season, through):
        loaded.append(season)
        return {"101": [wk(5, snap=65, tgt=9, car=11, ppr=18.0),
                        wk(6, snap=65, tgt=9, car=11, ppr=18.0),
                        wk(7, snap=65, tgt=9, car=11, ppr=18.0)]}

    monkeypatch.setattr(wg, "load_calls", lambda season: [make_call()])
    monkeypatch.setattr(wg, "load_existing_grade_keys", lambda season: set())
    monkeypatch.setattr(wg, "load_season_series", _series)
    monkeypatch.setattr(wg, "save_grade_rows", lambda grades: len(grades))
    summary = wg.grade_weekly_breakouts(2026, through_week=7)
    assert loaded == [2026]                  # no prior-season load paid for
    assert summary["by_grade"] == {"hit": 1}


def test_window_stats_means_and_opportunity_definition():
    stats = wg.window_stats([
        wk(5, snap=50, tgt=4, car=6, ppr=10.0),
        wk(6, snap=70, tgt=6, car=8, ppr=None),
        wk(7, snap=None, tgt=None, car=None, ppr=16.0),
    ])
    assert stats["games"] == 3
    assert stats["opp_pg"] == 12.0          # (10 + 14) / 2 recorded games
    assert stats["snap_pct"] == 60.0        # mean over recorded games only
    assert stats["ppr_ppg"] == 13.0


# ---------------------------------------------------------------------------
# maturity + orchestration
# ---------------------------------------------------------------------------

def test_is_mature_needs_three_subsequent_weeks():
    assert wg.is_mature(4, 7) is True
    assert wg.is_mature(4, 6) is False      # only 2 subsequent weeks
    assert wg.is_mature(6, 7) is False      # 1-week-old call


def test_orchestrator_skips_immature_calls(monkeypatch):
    _no_db(monkeypatch)
    monkeypatch.setattr(wg, "load_calls", lambda season: [make_call(as_of_week=6)])
    monkeypatch.setattr(wg, "load_existing_grade_keys", lambda season: set())
    loaded = {"series": False}

    def _series(season, through):
        loaded["series"] = True
        return {}

    monkeypatch.setattr(wg, "load_season_series", _series)
    summary = wg.grade_weekly_breakouts(2026, through_week=7)
    assert summary["status"] == "completed"
    assert summary["skipped_immature"] == 1
    assert summary["graded"] == 0
    assert loaded["series"] is False       # nothing to grade -> no data load


def test_grading_twice_inserts_exactly_one_grade(monkeypatch):
    _no_db(monkeypatch)
    call = make_call()
    rows = [wk(5, snap=65, tgt=9, car=11, ppr=18.0),
            wk(6, snap=65, tgt=9, car=11, ppr=18.0),
            wk(7, snap=65, tgt=9, car=11, ppr=18.0)]
    store = {"keys": set(), "grades": []}

    monkeypatch.setattr(wg, "load_calls", lambda season: [call])
    monkeypatch.setattr(wg, "load_existing_grade_keys",
                        lambda season: set(store["keys"]))
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through: {"101": rows})

    def _save(grades):
        inserted = 0
        for grade in grades:
            key = wg.call_key(grade)
            if key in store["keys"]:
                continue                    # ON CONFLICT DO NOTHING
            store["keys"].add(key)
            store["grades"].append(grade)
            inserted += 1
        return inserted

    monkeypatch.setattr(wg, "save_grade_rows", _save)

    first = wg.grade_weekly_breakouts(2026, through_week=7)
    assert first["graded"] == 1
    assert first["inserted"] == 1
    assert first["by_grade"] == {"hit": 1}

    second = wg.grade_weekly_breakouts(2026, through_week=7)
    assert second["already_graded"] == 1
    assert second["graded"] == 0
    assert len(store["grades"]) == 1
    assert store["grades"][0]["grade"] == "hit"


# ---------------------------------------------------------------------------
# saver SQL: idempotent insert shape
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Conn:
    """Hands back a row for the first insert, None after (conflict)."""

    def __init__(self):
        self.queries = []
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.queries.append(sql)
        self.calls += 1
        return _Result({"id": 1} if self.calls == 1 else None)


def test_save_grade_rows_uses_on_conflict_do_nothing(monkeypatch):
    _no_db(monkeypatch)
    conn = _Conn()
    monkeypatch.setattr(wg, "get_conn", lambda: conn)
    grade = wg.grade_call(make_call(), [wk(5, snap=65, tgt=9, car=11, ppr=18.0)])
    inserted = wg.save_grade_rows([grade, grade])
    assert inserted == 1                    # second insert conflicted away
    assert len(conn.queries) == 2
    assert "ON CONFLICT (player_id, season, as_of_week, scoring_version)" in conn.queries[0]
    assert "DO NOTHING" in conn.queries[0]


# ---------------------------------------------------------------------------
# summarize: grouping + minimum sample
# ---------------------------------------------------------------------------

def _grade_row(grade, classification="emerging_breakout", version="weekly-v5"):
    return {"grade": grade, "classification": classification,
            "scoring_version": version}


def test_summarize_groups_by_classification_and_version():
    rows = (
        [_grade_row("hit") for _ in range(6)]
        + [_grade_row("partial") for _ in range(3)]
        + [_grade_row("miss") for _ in range(3)]
        + [_grade_row("ungraded") for _ in range(2)]
        + [_grade_row("hit", classification="watchlist") for _ in range(2)]
        + [_grade_row("miss", classification="watchlist") for _ in range(2)]
    )
    summary = wg.summarize_grade_rows(rows, min_sample=10)

    overall = summary["overall"]
    assert overall["calls"] == 18
    assert overall["graded"] == 16
    assert overall["ungraded"] == 2
    assert overall["hit_rate"] == 0.5       # 8 / 16 graded (ungraded excluded)
    assert overall["miss_rate"] == 0.3125

    emerging = summary["by_classification"]["emerging_breakout"]
    assert emerging["graded"] == 12
    assert emerging["hit_rate"] == 0.5
    assert emerging["partial_rate"] == 0.25
    assert emerging["miss_rate"] == 0.25

    # Below the minimum sample: counts stay real, rates are withheld.
    watchlist = summary["by_classification"]["watchlist"]
    assert watchlist["calls"] == 4
    assert watchlist["hit"] == 2
    assert watchlist["miss"] == 2
    assert watchlist["hit_rate"] is None
    assert watchlist["miss_rate"] is None

    assert summary["by_scoring_version"]["weekly-v5"]["graded"] == 16


def test_summarize_empty_rows():
    summary = wg.summarize_grade_rows([])
    assert summary["overall"]["calls"] == 0
    assert summary["overall"]["hit_rate"] is None
    assert summary["by_classification"] == {}
