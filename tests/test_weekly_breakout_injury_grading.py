"""Tests for the "injured" breakout grade (weekly-grading-v3).

Covers the pure injury-window classifier, the grade_call wiring (the
2-games rule), summary exclusion of injured grades from hit-rate
denominators, and the injury-snapshot upsert (latest-wins) plus its
fail-soft behavior. DB loaders are monkeypatched to in-memory fakes,
mirroring tests/test_weekly_breakout_grading.py.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import dashboard_services.api as sleeper_api
from data_building.breakout_engine import weekly_grading as wg


def wk(week, snap=None, tgt=None, car=None, ppr=0.0):
    """One player_weekly_metrics-shaped row (shares are 0-100)."""
    return {
        "week": week, "snap_pct": snap, "targets": tgt, "carries": car,
        "ppr_pts": ppr, "snaps": 50,
    }


def make_call(as_of_week=4, **kwargs):
    """A persisted weekly_breakout_scores row with stored-evidence baselines
    (ppg 8.0, snap 40, opp 10)."""
    params = dict(classification="emerging_breakout",
                  scoring_version="weekly-v5", baseline_weeks=(1, 2),
                  baseline_source="current_season", baseline_ppg=8.0,
                  baseline_snap=40.0, baseline_opp=10.0)
    params.update(kwargs)
    signals = {"snap_share": {"baseline": params["baseline_snap"],
                              "recent": 62.0},
               "carry_opportunity_pg": {"baseline": params["baseline_opp"],
                                        "recent": 16.0}}
    return {
        "player_id": "101", "player_name": "Test Player", "season": 2026,
        "as_of_week": as_of_week, "scoring_version": params["scoring_version"],
        "classification": params["classification"],
        "breakout_score": 55.0, "confidence": 0.8,
        "baseline_source": params["baseline_source"],
        "baseline_weeks": list(params["baseline_weeks"]),
        "evidence": {
            "fantasy": {"baseline_ppg": params["baseline_ppg"],
                        "recent_ppg": 14.0},
            "signals": signals,
        },
    }


def inj_weeks(designations):
    """{week: snapshot row} from {week: (status, injury_status)} pairs."""
    return {w: {"status": s, "injury_status": i}
            for w, (s, i) in designations.items()}


def _no_db(monkeypatch):
    monkeypatch.setattr(wg, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(wg, "init_weekly_breakout_grades_db", lambda: None)
    monkeypatch.setattr(wg, "init_injury_snapshots_db", lambda: None)


# ---------------------------------------------------------------------------
# classify_injury_window: the pure rule
# ---------------------------------------------------------------------------

def test_ir_any_week_with_zero_games_voids():
    out = wg.classify_injury_window(
        0, inj_weeks({5: ("IR", "IR"), 6: ("IR", "IR"), 7: ("IR", "IR")}),
        [5, 6, 7])
    assert out["injured"] is True
    assert out["reason"] == "on_ir_during_window"
    assert out["designation"] == "IR"


def test_pup_voids_like_ir():
    out = wg.classify_injury_window(
        0, inj_weeks({6: ("PUP", "")}), [5, 6, 7])
    assert out["injured"] is True
    assert out["reason"] == "on_ir_during_window"
    assert out["designation"] == "PUP"


def test_out_two_of_three_weeks_voids():
    out = wg.classify_injury_window(
        1, inj_weeks({5: ("Active", "Out"), 6: ("Active", "Out")}), [5, 6, 7])
    assert out["injured"] is True
    assert out["reason"] == "out_during_window"
    assert out["designation"] == "OUT"
    assert out["weeks"] == [5, 6]


def test_doubtful_counts_toward_out_weeks():
    out = wg.classify_injury_window(
        0, inj_weeks({5: ("Active", "Doubtful"), 7: ("Active", "Out")}),
        [5, 6, 7])
    assert out["injured"] is True
    assert out["reason"] == "out_during_window"


def test_questionable_never_voids():
    out = wg.classify_injury_window(
        0, inj_weeks({5: ("Active", "Questionable"),
                      6: ("Active", "Questionable"),
                      7: ("Active", "Questionable")}), [5, 6, 7])
    assert out["injured"] is False


def test_single_out_week_does_not_void():
    out = wg.classify_injury_window(
        1, inj_weeks({5: ("Active", "Out")}), [5, 6, 7])
    assert out["injured"] is False


def test_two_games_played_never_voids_even_on_ir():
    out = wg.classify_injury_window(
        2, inj_weeks({5: ("IR", "IR"), 6: ("IR", "IR"), 7: ("IR", "IR")}),
        [5, 6, 7])
    assert out["injured"] is False


def test_no_snapshots_never_voids():
    assert wg.classify_injury_window(0, {}, [5, 6, 7])["injured"] is False


def test_short_forms_normalize():
    out = wg.classify_injury_window(
        0, inj_weeks({5: ("Active", "O"), 6: ("Active", "D")}), [5, 6, 7])
    assert out["injured"] is True
    assert out["reason"] == "out_during_window"


# ---------------------------------------------------------------------------
# grade_call wiring: the 2-games rule
# ---------------------------------------------------------------------------

def _big_weeks():
    """Two outcome weeks with role held and production up: a hit on merit."""
    return [wk(5, snap=60, tgt=8, car=8, ppr=14.0),
            wk(6, snap=62, tgt=7, car=9, ppr=15.0)]


def test_ir_zero_games_grades_injured():
    grade = wg.grade_call(
        make_call(), [],
        injury_by_week=inj_weeks({5: ("IR", "IR"), 6: ("IR", "IR"),
                                  7: ("IR", "IR")}))
    assert grade["grade"] == wg.GRADE_INJURED
    assert grade["grading_version"] == "weekly-grading-v3"
    assert grade["detail"]["reason"] == "on_ir_during_window"
    assert grade["detail"]["injury"]["designation"] == "IR"
    assert grade["ppg_delta"] is None


def test_ir_with_two_games_grades_on_merit():
    grade = wg.grade_call(
        make_call(), _big_weeks(),
        injury_by_week=inj_weeks({5: ("IR", "IR"), 6: ("IR", "IR"),
                                  7: ("IR", "IR")}))
    assert grade["grade"] == wg.GRADE_HIT


def test_out_all_window_one_game_grades_injured():
    grade = wg.grade_call(
        make_call(), [wk(5, snap=45, tgt=5, car=5, ppr=9.0)],
        injury_by_week=inj_weeks({5: ("Active", "Out"),
                                  6: ("Active", "Out"),
                                  7: ("Active", "Out")}))
    assert grade["grade"] == wg.GRADE_INJURED
    assert grade["detail"]["reason"] == "out_during_window"


def test_questionable_zero_games_stays_ungraded():
    grade = wg.grade_call(
        make_call(), [],
        injury_by_week=inj_weeks({5: ("Active", "Questionable")}))
    assert grade["grade"] == wg.GRADE_UNGRADED
    assert grade["detail"]["reason"] == "no_outcome_games"


def test_out_one_week_two_games_grades_on_merit():
    rows = [wk(5, snap=50, tgt=6, car=6, ppr=9.0),
            wk(6, snap=52, tgt=6, car=6, ppr=9.5)]
    grade = wg.grade_call(
        make_call(), rows,
        injury_by_week=inj_weeks({7: ("Active", "Out")}))
    # role held (opp +2.0) without the production rise: partial, not injured
    assert grade["grade"] == wg.GRADE_PARTIAL


# ---------------------------------------------------------------------------
# summary: injured counted, excluded from rates
# ---------------------------------------------------------------------------

def _summary_row(grade):
    return {"grade": grade, "classification": "emerging_breakout",
            "breakout_score": 55.0, "scoring_version": "weekly-v6"}


def test_summary_excludes_injured_from_rates_but_counts_them():
    rows = ([_summary_row("hit")] * 4 + [_summary_row("partial")] * 4
            + [_summary_row("miss")] * 4 + [_summary_row("ungraded")] * 3
            + [_summary_row("injured")] * 2)
    bucket = wg.summarize_grade_rows(rows, min_sample=10)["overall"]
    assert bucket["calls"] == 17
    assert bucket["graded"] == 12
    assert bucket["injured"] == 2
    assert bucket["ungraded"] == 3
    assert bucket["hit_rate"] == pytest.approx(4 / 12, rel=1e-3)
    assert bucket["partial_rate"] == pytest.approx(4 / 12, rel=1e-3)
    assert bucket["miss_rate"] == pytest.approx(4 / 12, rel=1e-3)


# ---------------------------------------------------------------------------
# snapshot: upsert latest-wins + fail-soft
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self):
        self.sql = None
        self.rows = None

    def executemany(self, sql, rows):
        self.sql = sql
        self.rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def execute(self, *args, **kwargs):
        class _R:
            def fetchall(self):
                return []
        return _R()

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_snapshot_deps(monkeypatch, feed):
    fake = _FakeConn()
    monkeypatch.setattr(wg, "get_conn", lambda: fake)
    monkeypatch.setattr(sleeper_api, "get_nfl_players", lambda: feed)
    # wg.snapshot_injury_statuses imports get_nfl_players lazily from
    # dashboard_services.api, which is the patched attribute above.
    return fake


def test_snapshot_writes_one_row_per_player_with_upsert(monkeypatch):
    _no_db(monkeypatch)
    feed = {
        "101": {"injury_status": "IR", "status": "IR"},
        "202": {"injury_status": "", "status": "Active"},
        "303": {"injury_status": "Questionable", "status": "Active"},
    }
    fake = _patch_snapshot_deps(monkeypatch, feed)
    written = wg.snapshot_injury_statuses(2026, 7)
    assert written == 3
    sql = fake.cursor_obj.sql
    assert "ON CONFLICT (player_id, season, week) DO UPDATE" in sql
    assert "injury_status = EXCLUDED.injury_status" in sql
    assert "captured_at = NOW()" in sql
    assert ("101", 2026, 7, "IR", "IR") in fake.cursor_obj.rows
    assert ("202", 2026, 7, None, "Active") in fake.cursor_obj.rows
    assert ("303", 2026, 7, "Questionable", "Active") in fake.cursor_obj.rows


def test_snapshot_fail_soft_on_feed_error(monkeypatch):
    _no_db(monkeypatch)
    monkeypatch.setattr(wg, "get_conn", lambda: _FakeConn())

    def _boom():
        raise RuntimeError("sleeper down")

    monkeypatch.setattr(sleeper_api, "get_nfl_players", _boom)
    assert wg.snapshot_injury_statuses(2026, 7) == 0


# ---------------------------------------------------------------------------
# orchestrator: injury weeks flow into grade_call; injured is terminal
# ---------------------------------------------------------------------------

def test_orchestrator_grades_ir_call_as_injured(monkeypatch):
    _no_db(monkeypatch)
    monkeypatch.setattr(sleeper_api, "get_nfl_state",
                        lambda: {"week": 7, "season": 2026})
    monkeypatch.setattr(wg, "snapshot_injury_statuses", lambda s, w: 5)
    monkeypatch.setattr(wg, "load_calls", lambda season: [make_call()])
    monkeypatch.setattr(wg, "load_existing_grade_keys", lambda season: set())
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through: {"101": []})
    monkeypatch.setattr(
        wg, "load_injury_weeks",
        lambda season, pids, weeks: {"101": inj_weeks(
            {5: ("IR", "IR"), 6: ("IR", "IR"), 7: ("IR", "IR")})})
    saved = {}

    def _save(grades):
        saved["g"] = list(grades)
        return len(saved["g"])

    monkeypatch.setattr(wg, "save_grade_rows", _save)

    result = wg.grade_weekly_breakouts(2026, through_week=7)

    assert result["by_grade"] == {"injured": 1}
    assert result["injury_snapshot"] == {"week": 7, "rows": 5}
    grade = saved["g"][0]
    assert grade["grade"] == "injured"
    assert grade["grading_version"] == "weekly-grading-v3"


def test_injured_grade_is_terminal_in_frontier(monkeypatch):
    """load_existing_grade_keys keeps every non-ungraded grade, so an
    injured grade is written once and never retried (v2 rule: only
    ungraded is retried)."""
    _no_db(monkeypatch)
    calls = [make_call()]
    monkeypatch.setattr(wg, "load_calls", lambda season: calls)
    # the call was already graded injured under v3
    monkeypatch.setattr(
        wg, "load_existing_grade_keys",
        lambda season: {wg.call_key(calls[0])})
    monkeypatch.setattr(sleeper_api, "get_nfl_state",
                        lambda: {"week": 7, "season": 2026})
    monkeypatch.setattr(wg, "snapshot_injury_statuses", lambda s, w: 0)

    def _fail(*a, **k):
        raise AssertionError("must not load series for graded calls")

    monkeypatch.setattr(wg, "load_season_series", _fail)
    result = wg.grade_weekly_breakouts(2026, through_week=7)
    assert result["already_graded"] == 1
    assert result["graded"] == 0
