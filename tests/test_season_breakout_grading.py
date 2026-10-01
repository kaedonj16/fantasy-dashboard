"""Season-breakout preseason-call grading: snapshot selection, verdict
math, stage gates, idempotency, and summary sample floors.

The module under test (data_building.breakout_engine.season_grading) is
new in this change, so every test here errors on unmodified main (the
import fails). Pure-function tests run on plain dicts; orchestrator tests
monkeypatch the module's DB layer with in-memory fakes, mirroring
tests/test_weekly_breakout_grading.py.
"""
import json

import pytest

from data_building.breakout_engine import season_grading as sg


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def wk(week, ppr, pos="RB"):
    return {"week": week, "ppr_pts": ppr, "position": pos,
            "targets": 5, "carries": 10, "snap_pct": 60.0}


def series(weeks_ppg, pos="RB"):
    """Weekly rows from a list of (week, ppg) pairs."""
    return [wk(week, ppg, pos) for week, ppg in weeks_ppg]


def flat_series(n_games, ppg, pos="RB"):
    return series([(week, ppg) for week in range(1, n_games + 1)], pos)


def make_call(**overrides):
    call = {
        "player_id": "1001",
        "player_name": "Test Back",
        "season": 2026,
        "as_of_date": "2026-08-20",
        "team": "BUF",
        "position": "RB",
        "phase": "preseason",
        "breakout_opportunity_score": 82.5,
        "confidence_score": 71.0,
        "hit_probability": 0.63,
        "projected_role_tag": "RB2",
    }
    call.update(overrides)
    return call


# ---------------------------------------------------------------------------
# final pre-season snapshot selection
# ---------------------------------------------------------------------------

def test_in_season_snapshot_is_not_the_call():
    rows = [
        make_call(as_of_date="2026-08-20", phase="preseason"),
        make_call(as_of_date="2026-10-01", phase="in_season"),
    ]
    calls = sg.select_final_preseason_calls(rows)
    assert len(calls) == 1
    assert calls[0]["as_of_date"] == "2026-08-20"
    assert calls[0]["phase"] == "preseason"


def test_latest_preseason_snapshot_wins():
    rows = [
        make_call(as_of_date="2026-03-05", phase="offseason"),
        make_call(as_of_date="2026-05-01", phase="post_draft"),
        make_call(as_of_date="2026-08-28", phase="preseason"),
    ]
    calls = sg.select_final_preseason_calls(rows)
    assert len(calls) == 1
    assert calls[0]["as_of_date"] == "2026-08-28"


def test_offseason_only_call_still_counts():
    rows = [make_call(as_of_date="2026-03-05", phase="offseason")]
    assert len(sg.select_final_preseason_calls(rows)) == 1


def test_row_without_preseason_phase_is_not_a_call():
    rows = [make_call(phase=None), make_call(player_id="2002", phase="")]
    assert sg.select_final_preseason_calls(rows) == []


def test_selection_is_per_player_and_season():
    rows = [
        make_call(),
        make_call(player_id="2002", as_of_date="2026-08-01"),
        make_call(season=2025, as_of_date="2025-08-19"),
    ]
    assert len(sg.select_final_preseason_calls(rows)) == 3


# ---------------------------------------------------------------------------
# verdict math (hit definition: backtest_multitask.get_breakout_pids)
# prior 10 games @ 8.0 PPG is a valid baseline; bar = 8.0 * 1.15 = 9.2, >= 7.0
# ---------------------------------------------------------------------------

PRIOR = flat_series(10, 8.0)


def test_hit_with_baseline_full_season():
    grade = sg.grade_season_call(
        make_call(), PRIOR, flat_series(17, 10.0), sg.STAGE_FINAL)
    assert grade["grade"] == "hit"
    assert grade["baseline_ppg"] == 8.0
    assert grade["outcome_ppg"] == 10.0
    assert grade["ppg_delta"] == 2.0
    assert grade["outcome_games"] == 17
    assert grade["provisional"] is False


def test_hit_from_scratch_no_prior():
    grade = sg.grade_season_call(
        make_call(), [], flat_series(12, 11.0), sg.STAGE_FINAL)
    assert grade["grade"] == "hit"
    assert grade["detail"]["prior_valid"] is False


def test_hit_bar_boundary_is_inclusive():
    # prior 10.0 PPG -> bar is exactly 11.5; >= clears it.
    grade = sg.grade_season_call(
        make_call(), flat_series(10, 10.0), flat_series(12, 11.5),
        sg.STAGE_FINAL)
    assert grade["grade"] == "hit"


def test_miss_when_no_improvement_over_prior():
    grade = sg.grade_season_call(
        make_call(), flat_series(10, 10.0), flat_series(17, 9.0),
        sg.STAGE_FINAL)
    assert grade["grade"] == "miss"
    assert grade["detail"]["reason"] == "no_improvement_over_prior"


def test_miss_from_scratch_below_floor():
    grade = sg.grade_season_call(
        make_call(), [], flat_series(17, 6.0), sg.STAGE_FINAL)
    assert grade["grade"] == "miss"
    assert grade["detail"]["reason"] == "below_breakout_floor_no_prior"


def test_partial_improved_below_bar():
    # bar is 11.5 (prior 10.0 x 1.15); 11.0 improves but does not clear.
    grade = sg.grade_season_call(
        make_call(), flat_series(10, 10.0), flat_series(17, 11.0),
        sg.STAGE_FINAL)
    assert grade["grade"] == "partial"
    assert grade["detail"]["reason"] == "improved_below_breakout_bar"


def test_partial_bar_cleared_but_season_cut_short():
    # 8 games clears the final stage floor but not the backtest's 10.
    grade = sg.grade_season_call(
        make_call(), PRIOR, flat_series(8, 12.0), sg.STAGE_FINAL)
    assert grade["grade"] == "partial"
    assert grade["detail"]["reason"] == (
        "ppg_bar_cleared_below_backtest_games_floor")


def test_ungraded_with_no_outcome_games():
    grade = sg.grade_season_call(make_call(), PRIOR, [], sg.STAGE_FINAL)
    assert grade["grade"] == "ungraded"
    assert grade["detail"]["reason"] == "no_outcome_games"


def test_games_floor_differs_by_stage():
    rows = flat_series(5, 20.0)
    final = sg.grade_season_call(make_call(), PRIOR, rows, sg.STAGE_FINAL)
    assert final["grade"] == "ungraded"
    assert final["detail"]["reason"] == "insufficient_games"
    early = sg.grade_season_call(make_call(), PRIOR, rows, sg.STAGE_EARLY)
    assert early["grade"] == "hit"
    assert early["provisional"] is True


def test_predicted_fields_recorded_as_made():
    grade = sg.grade_season_call(
        make_call(), PRIOR, flat_series(17, 10.0), sg.STAGE_FINAL)
    assert grade["breakout_score"] == 82.5
    assert grade["confidence"] == 71.0
    assert grade["hit_probability"] == 0.63
    assert grade["phase"] == "preseason"
    assert grade["grading_version"] == "season-grading-v1"


# ---------------------------------------------------------------------------
# stage windows and gates
# ---------------------------------------------------------------------------

def test_stage_eligible_gates():
    assert sg.stage_eligible(sg.STAGE_EARLY, 7) is False
    assert sg.stage_eligible(sg.STAGE_EARLY, 8) is True
    assert sg.stage_eligible(sg.STAGE_FINAL, 16) is False
    assert sg.stage_eligible(sg.STAGE_FINAL, 17) is True
    assert sg.stage_eligible(sg.STAGE_FINAL, None) is False


def test_early_window_excludes_late_weeks():
    outcome = series([(w, 12.0) for w in range(1, 9)]
                     + [(w, 2.0) for w in range(9, 18)])
    early = sg.grade_season_call(make_call(), PRIOR, outcome, sg.STAGE_EARLY)
    assert early["grade"] == "hit"
    assert early["outcome_games"] == 8
    assert early["outcome_ppg"] == 12.0
    assert early["outcome_weeks"] == list(range(1, 9))
    # Same call at final: the collapse drags PPG to 6.71 <= prior 8.0.
    final = sg.grade_season_call(make_call(), PRIOR, outcome, sg.STAGE_FINAL)
    assert final["grade"] == "miss"
    assert final["outcome_games"] == 17


# ---------------------------------------------------------------------------
# position finishes (descriptive detail)
# ---------------------------------------------------------------------------

def test_compute_position_finishes_ranks_by_total_ppr():
    universe = {
        "rb1": flat_series(10, 10.0, "RB"),   # 100 total
        "rb2": flat_series(10, 20.0, "RB"),   # 200 total
        "rb3": flat_series(10, 15.0, "RB"),   # 150 total
    }
    finishes = sg.compute_position_finishes(universe, sg.STAGE_FINAL)
    assert finishes["rb2"]["position_rank"] == 1
    assert finishes["rb3"]["position_rank"] == 2
    assert finishes["rb1"]["position_rank"] == 3
    assert finishes["rb1"]["top_n_finish"] is True


def test_compute_position_finishes_te_cutoff_is_six():
    universe = {
        f"te{i}": flat_series(10, float(80 - 10 * i), "TE")
        for i in range(7)
    }
    finishes = sg.compute_position_finishes(universe, sg.STAGE_FINAL)
    assert finishes["te5"]["position_rank"] == 6
    assert finishes["te5"]["top_n_finish"] is True
    assert finishes["te6"]["position_rank"] == 7
    assert finishes["te6"]["top_n_finish"] is False


def test_compute_position_finishes_early_window_only():
    universe = {
        "rb1": series([(w, 1.0) for w in range(1, 9)]
                      + [(w, 50.0) for w in range(9, 18)], "RB"),
        "rb2": flat_series(17, 5.0, "RB"),
    }
    finishes = sg.compute_position_finishes(universe, sg.STAGE_EARLY)
    # Weeks 9+ do not count at the early stage: rb2 (40) beats rb1 (8).
    assert finishes["rb2"]["position_rank"] == 1
    assert finishes["rb1"]["position_rank"] == 2


# ---------------------------------------------------------------------------
# orchestration (DB layer faked in-memory)
# ---------------------------------------------------------------------------

class _FakeDB:
    def __init__(self, calls, prior_by_player, outcome_by_player):
        self.calls = calls
        self.series = {2025: prior_by_player, 2026: outcome_by_player}
        self.grades = {}
        self.series_loads = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(sg, "init_season_breakout_grades_db", lambda: None)
        monkeypatch.setattr(
            sg, "load_final_preseason_calls", lambda season: list(self.calls))
        monkeypatch.setattr(
            sg, "load_existing_grade_keys", lambda season: set(self.grades))
        monkeypatch.setattr(sg, "load_season_series", self._load_series)
        monkeypatch.setattr(sg, "save_grade_rows", self._save)

    def _load_series(self, season, through_week):
        self.series_loads += 1
        return {
            pid: [r for r in rows if r["week"] <= through_week]
            for pid, rows in self.series.get(season, {}).items()
        }

    def _save(self, grades):
        inserted = 0
        for grade in grades:
            key = sg.call_key(grade, grade["grading_stage"])
            if key not in self.grades:
                self.grades[key] = grade
                inserted += 1
        return inserted


def _standard_db():
    return _FakeDB(
        calls=[make_call()],
        prior_by_player={"1001": flat_series(10, 8.0)},
        outcome_by_player={"1001": flat_series(17, 10.0)},
    )


def test_orchestrator_grades_both_stages_when_season_complete(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    result = sg.grade_season_breakouts(2026, through_week=18)
    assert result["status"] == "completed"
    assert result["inserted"] == 2
    assert result["by_stage"] == {"early": 1, "final": 1}
    assert result["by_grade"] == {"hit": 2}


def test_orchestrator_rerun_is_idempotent(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    sg.grade_season_breakouts(2026, through_week=18)
    again = sg.grade_season_breakouts(2026, through_week=18)
    assert again["already_graded"] == 2
    assert again["graded"] == 0
    assert again["inserted"] == 0
    assert len(db.grades) == 2


def test_orchestrator_early_only_before_season_completes(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    result = sg.grade_season_breakouts(2026, through_week=10)
    assert result["inserted"] == 1
    assert result["by_stage"] == {"early": 1}
    assert result["skipped_stage_gate"] == 1  # the final stage


def test_orchestrator_nothing_grades_before_week_8(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    result = sg.grade_season_breakouts(2026, through_week=5)
    assert result["graded"] == 0
    assert result["skipped_stage_gate"] == 2  # early + final
    assert db.series_loads == 0  # fast skip: no series loaded


def test_orchestrator_skips_without_weekly_metrics(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    monkeypatch.setattr(sg, "default_through_week", lambda season: None)
    result = sg.grade_season_breakouts(2026)
    assert result["status"] == "skipped"
    assert result["reason"] == "no weekly metrics for season"


def test_orchestrator_final_only_when_early_disabled(monkeypatch):
    db = _standard_db()
    db.install(monkeypatch)
    result = sg.grade_season_breakouts(
        2026, through_week=18, stages=(sg.STAGE_FINAL,))
    assert result["by_stage"] == {"final": 1}


# ---------------------------------------------------------------------------
# saver: conflict clause keeps grades immutable
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def fetchone(self):
        return self._outcomes.pop(0) if self._outcomes else None


class _Conn:
    def __init__(self, outcomes):
        self.statements = []
        self._outcomes = outcomes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        return _Result(self._outcomes)


def test_save_grade_rows_conflict_does_nothing(monkeypatch):
    conn = _Conn([{"id": 1}, None])
    monkeypatch.setattr(sg, "get_conn", lambda: conn)
    monkeypatch.setattr(sg, "init_season_breakout_grades_db", lambda: None)
    grades = [
        sg.grade_season_call(make_call(), PRIOR, flat_series(17, 10.0),
                             sg.STAGE_FINAL),
        sg.grade_season_call(make_call(), PRIOR, flat_series(17, 10.0),
                             sg.STAGE_FINAL),
    ]
    assert sg.save_grade_rows(grades) == 1
    sql = conn.statements[0][0]
    assert ("ON CONFLICT (player_id, season, as_of_date, grading_stage) "
            "DO NOTHING") in sql
    params = conn.statements[0][1]
    assert json.loads(params["detail"])["reason"] == "ppg_bar_cleared"


# ---------------------------------------------------------------------------
# summarize: sample floor
# ---------------------------------------------------------------------------

def _summary_row(grade, phase="preseason", score=85.0, season=2026):
    return {"grade": grade, "phase": phase,
            "breakout_score": score, "season": season}


def test_summarize_rates_none_below_sample_floor():
    rows = (
        [_summary_row("hit") for _ in range(6)]
        + [_summary_row("partial") for _ in range(2)]
        + [_summary_row("miss") for _ in range(2)]
        + [_summary_row("ungraded") for _ in range(3)]
        + [_summary_row("hit", phase="offseason", score=65.0)
           for _ in range(3)]
    )
    summary = sg.summarize_grade_rows(rows)
    overall = summary["overall"]
    assert overall["calls"] == 16
    assert overall["graded"] == 13
    assert overall["hit"] == 9
    assert overall["hit_rate"] == pytest.approx(9 / 13, abs=1e-3)
    # Only 3 graded offseason calls: counts real, rates suppressed.
    offseason = summary["by_phase"]["offseason"]
    assert offseason["graded"] == 3
    assert offseason["hit"] == 3
    assert offseason["hit_rate"] is None
    assert offseason["partial_rate"] is None
    # Groupings exist for season and score band too.
    assert summary["by_season"]["2026"]["hit_rate"] == pytest.approx(
        9 / 13, abs=1e-3)
    assert summary["by_score_band"]["80-89"]["graded"] == 10
    assert summary["by_score_band"]["60-69"]["hit_rate"] is None


def test_score_band_edges():
    assert sg.score_band(None) == "unknown"
    assert sg.score_band(59.9) == "<60"
    assert sg.score_band(60) == "60-69"
    assert sg.score_band(85) == "80-89"
    assert sg.score_band(90) == "90+"
