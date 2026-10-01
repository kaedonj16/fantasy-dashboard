"""Tests for rookie-cohort baselines on initial-role (no-baseline) calls.

A call with no baseline at all is measured against the typical rookie
year at its position (see weekly_grading.load_cohort_baselines): the
grader scores it on production alone, and the live forecast bands it
the same way. Pure-math tests use plain dicts; DB loaders are
monkeypatched to in-memory fakes.
"""
from __future__ import annotations

from datetime import date

import pytest

from data_building.breakout_engine import calibration
from data_building.breakout_engine import forecasts
from data_building.breakout_engine import weekly_grading as wg


def wk(week, ppr=0.0):
    return {"week": week, "snap_pct": None, "targets": None, "carries": None,
            "ppr_pts": ppr, "snaps": 50}


def no_baseline_call(as_of_week=1, position="RB", scoring_version="weekly-v6"):
    """A persisted initial-role call: no baseline anywhere in evidence."""
    return {
        "player_id": "101", "player_name": "Rookie Test", "season": 2026,
        "position": position,
        "as_of_week": as_of_week, "as_of_date": date(2026, 9, 8),
        "scoring_version": scoring_version, "classification": "emerging_breakout",
        "breakout_score": 55.0, "confidence": 0.8,
        "baseline_source": "none",
        "baseline_weeks": [],
        "evidence": {
            "fantasy": {"baseline_ppg": None, "recent_ppg": None},
            "signals": {},
        },
    }


def _no_db(monkeypatch):
    monkeypatch.setattr(wg, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(wg, "init_weekly_breakout_grades_db", lambda: None)


# ---------------------------------------------------------------------------
# call_needs_cohort_baseline
# ---------------------------------------------------------------------------

def test_needs_cohort_for_initial_role_call():
    assert wg.call_needs_cohort_baseline(no_baseline_call()) is True


def test_no_cohort_when_a_baseline_exists():
    call = no_baseline_call()
    call["baseline_source"] = "stored"
    call["evidence"]["fantasy"]["baseline_ppg"] = 8.0
    assert wg.call_needs_cohort_baseline(call) is False


def test_no_cohort_when_stored_ppg_exists():
    call = no_baseline_call()
    call["evidence"]["fantasy"]["baseline_ppg"] = 8.0
    assert wg.call_needs_cohort_baseline(call) is False


# ---------------------------------------------------------------------------
# resolve_baseline: cohort fill
# ---------------------------------------------------------------------------

def test_resolve_baseline_cohort_fill():
    baseline, source = wg.resolve_baseline(no_baseline_call(), [], None,
                                           cohort_ppg=7.5)
    assert source == "cohort"
    assert baseline["ppr_ppg"] == 7.5
    assert baseline["snap_pct"] is None
    assert baseline["opp_pg"] is None


def test_resolve_baseline_no_cohort_stays_none():
    baseline, source = wg.resolve_baseline(no_baseline_call(), [], None)
    assert source == "none"
    assert baseline["ppr_ppg"] is None


def test_resolve_baseline_cohort_never_overrides_stored():
    call = no_baseline_call()
    call["baseline_source"] = "stored"
    call["evidence"]["fantasy"]["baseline_ppg"] = 8.0
    baseline, source = wg.resolve_baseline(call, [], None, cohort_ppg=7.5)
    assert source == "stored"
    assert baseline["ppr_ppg"] == 8.0


# ---------------------------------------------------------------------------
# classify_outcome: production-only cohort grading
# ---------------------------------------------------------------------------

def _cohort_baseline(ppg=7.5):
    return {"ppr_ppg": ppg, "snap_pct": None, "opp_pg": None}


def test_cohort_hit_at_plus_two():
    outcome = wg.window_stats([wk(2, ppr=9.5), wk(3, ppr=9.5)])
    verdict = wg.classify_outcome(_cohort_baseline(), outcome, cohort=True)
    assert verdict["grade"] == wg.GRADE_HIT
    assert verdict["role_state"] == "cohort"
    assert verdict["ppg_delta"] == pytest.approx(2.0)


def test_cohort_miss_at_minus_two():
    outcome = wg.window_stats([wk(2, ppr=5.5), wk(3, ppr=5.5)])
    verdict = wg.classify_outcome(_cohort_baseline(), outcome, cohort=True)
    assert verdict["grade"] == wg.GRADE_MISS
    assert verdict["ppg_delta"] == pytest.approx(-2.0)


def test_cohort_partial_between_the_bars():
    outcome = wg.window_stats([wk(2, ppr=8.0), wk(3, ppr=8.0)])
    verdict = wg.classify_outcome(_cohort_baseline(), outcome, cohort=True)
    assert verdict["grade"] == wg.GRADE_PARTIAL


def test_cohort_still_ungraded_with_no_outcome_games():
    verdict = wg.classify_outcome(_cohort_baseline(), {"games": 0},
                                  cohort=True)
    assert verdict["grade"] == wg.GRADE_UNGRADED


def test_non_cohort_path_unchanged_without_baseline():
    # Without the cohort flag, a baselineless call is still ungraded.
    outcome = wg.window_stats([wk(2, ppr=12.0)])
    verdict = wg.classify_outcome(
        {"ppr_ppg": None, "snap_pct": None, "opp_pg": None}, outcome)
    assert verdict["grade"] == wg.GRADE_UNGRADED
    assert verdict["reason"] == "no_role_baseline"


# ---------------------------------------------------------------------------
# grade_call with a cohort baseline
# ---------------------------------------------------------------------------

def test_grade_call_cohort_records_cohort_source():
    call = no_baseline_call(as_of_week=1)
    rows = [wk(2, ppr=12.0), wk(3, ppr=12.0), wk(4, ppr=12.0)]
    grade = wg.grade_call(call, rows, None, cohort_ppg=7.5)
    assert grade["grade"] == "hit"
    assert grade["baseline_source"] == "cohort"
    assert grade["baseline_ppg"] == 7.5
    assert grade["ppg_delta"] == pytest.approx(4.5)
    assert grade["grading_version"] == "weekly-grading-v2"


def test_grade_call_without_cohort_stays_ungraded():
    call = no_baseline_call(as_of_week=1)
    rows = [wk(2, ppr=12.0), wk(3, ppr=12.0), wk(4, ppr=12.0)]
    grade = wg.grade_call(call, rows)
    assert grade["grade"] == "ungraded"
    assert grade["baseline_source"] == "none"


# ---------------------------------------------------------------------------
# load_cohort_baselines SQL mapping
# ---------------------------------------------------------------------------

def _stub_weekly_metrics(monkeypatch):
    import sys
    import types
    stub = types.ModuleType("data_building.weekly_metrics")
    stub.init_weekly_metrics_db = lambda: None
    monkeypatch.setitem(sys.modules, "data_building.weekly_metrics", stub)


def _stub_conn(monkeypatch, rows):
    captured = {}

    class _Cur:
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params
            return self

        def fetchall(self):
            return rows

    class _Conn:
        def __enter__(self):
            return _Cur()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(wg, "get_conn", lambda: _Conn())
    return captured


def test_load_cohort_baselines_maps_positions(monkeypatch):
    _stub_weekly_metrics(monkeypatch)
    captured = _stub_conn(monkeypatch, [
        {"position": "RB", "cohort_ppg": 7.5},
        {"position": "QB", "cohort_ppg": 12.25},
    ])
    out = wg.load_cohort_baselines(2026)
    assert out == {"RB": 7.5, "QB": 12.25}
    # Six-season lookback: 2020 through 2025 for a 2026 season.
    assert captured["params"][0] == 2020
    assert captured["params"][1] == 2025


def test_load_cohort_baselines_empty_on_db_error(monkeypatch):
    _stub_weekly_metrics(monkeypatch)
    monkeypatch.setattr(wg, "get_conn", lambda: 1 / 0)
    assert wg.load_cohort_baselines(2026) == {}


# ---------------------------------------------------------------------------
# weekly_forecast with a cohort baseline
# ---------------------------------------------------------------------------

def test_weekly_forecast_cohort_band_and_basis(monkeypatch):
    monkeypatch.setattr(wg, "default_through_week", lambda season: 2)
    call = no_baseline_call(as_of_week=1)
    forecast = forecasts.weekly_forecast(
        call, [wk(2, ppr=12.0)], 2, None, cohort_ppg=7.5)
    assert forecast is not None
    assert forecast["band"] == "tracking_to_hit"
    assert "typical-rookie RB baseline" in forecast["basis"]
    assert "7.5 PPG" in forecast["basis"]


def test_weekly_forecast_still_pending_without_cohort(monkeypatch):
    monkeypatch.setattr(wg, "default_through_week", lambda season: 2)
    call = no_baseline_call(as_of_week=1)
    forecast = forecasts.weekly_forecast(call, [wk(2, ppr=12.0)], 2)
    assert forecast is not None
    assert forecast["state"] == "no_baseline"
    assert forecast["band"] is None


def test_forecasts_for_calls_loads_cohort_only_when_needed(monkeypatch):
    _stub_weekly_metrics(monkeypatch)
    monkeypatch.setattr(wg, "default_through_week", lambda season: 2)
    monkeypatch.setattr(wg, "load_season_series", lambda season, through: {
        "101": [{"week": 2, "snap_pct": 62.0, "targets": 4, "carries": 16,
                 "ppr_pts": 12.0, "snaps": 62}]})
    loaded = []

    def _cohort(season):
        loaded.append(season)
        return {"RB": 7.5}

    monkeypatch.setattr(wg, "load_cohort_baselines", _cohort)

    call = no_baseline_call(as_of_week=1)
    out = forecasts.weekly_forecasts_for_calls(2026, [call])
    assert loaded == [2026]
    assert out["101"]["band"] == "tracking_to_hit"


def test_forecasts_for_calls_skips_cohort_load_when_unneeded(monkeypatch):
    _stub_weekly_metrics(monkeypatch)
    monkeypatch.setattr(wg, "default_through_week", lambda season: 2)
    monkeypatch.setattr(wg, "load_season_series", lambda season, through: {
        "101": [{"week": 2, "snap_pct": 62.0, "targets": 4, "carries": 16,
                 "ppr_pts": 12.0, "snaps": 62}]})

    def _boom(season):
        raise AssertionError("no call needs a cohort baseline")

    monkeypatch.setattr(wg, "load_cohort_baselines", _boom)
    call = no_baseline_call(as_of_week=1)
    call["baseline_source"] = "stored"
    call["evidence"]["fantasy"]["baseline_ppg"] = 8.0
    call["evidence"]["signals"] = {
        "snap_share": {"baseline": 40.0, "recent": 62.0},
        "carry_opportunity_pg": {"baseline": 10.0, "recent": 16.0},
    }
    out = forecasts.weekly_forecasts_for_calls(2026, [call])
    assert out["101"]["band"] == "tracking_to_hit"


# ---------------------------------------------------------------------------
# orchestrator: cohort flows into grade_call, ungraded never blocks
# ---------------------------------------------------------------------------

def test_orchestrator_passes_cohort_to_grade_call(monkeypatch):
    _no_db(monkeypatch)
    call = no_baseline_call(as_of_week=1)
    monkeypatch.setattr(wg, "load_calls", lambda season: [call])
    monkeypatch.setattr(wg, "load_existing_grade_keys", lambda season: set())
    monkeypatch.setattr(wg, "default_through_week", lambda season: 4)
    monkeypatch.setattr(wg, "load_season_series", lambda season, through: {
        "101": [wk(2, ppr=12.0), wk(3, ppr=12.0), wk(4, ppr=12.0)]})
    monkeypatch.setattr(wg, "load_cohort_baselines", lambda season: {"RB": 7.5})
    saved = {}
    monkeypatch.setattr(wg, "save_grade_rows",
                        lambda grades: saved.setdefault("grades", grades)
                        or len(grades))
    summary = wg.grade_weekly_breakouts(2026)
    assert summary["graded"] == 1
    grade = saved["grades"][0]
    assert grade["grade"] == "hit"
    assert grade["baseline_source"] == "cohort"


def test_existing_grade_keys_excludes_ungraded(monkeypatch):
    _no_db(monkeypatch)
    graded = {"player_id": "1", "season": 2026, "as_of_week": 1,
              "scoring_version": "weekly-v6", "grade": "hit"}
    ungraded = {"player_id": "2", "season": 2026, "as_of_week": 1,
                "scoring_version": "weekly-v6", "grade": "ungraded"}
    all_rows = [graded, ungraded]

    class _Cur:
        def execute(self, sql, params):
            # The SQL itself must do the filtering; the fake mirrors it.
            assert "grade" in sql
            self._rows = [r for r in all_rows if r["grade"] != "ungraded"]
            return self

        def fetchall(self):
            return self._rows

    class _Conn:
        def __enter__(self):
            return _Cur()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(wg, "get_conn", lambda: _Conn())
    keys = wg.load_existing_grade_keys(2026)
    assert keys == {wg.call_key(graded)}


# ---------------------------------------------------------------------------
# calibration: cohort-baselined calls reported as their own group
# ---------------------------------------------------------------------------

def _grade_row(i, *, baseline_source="stored", grade="hit"):
    return {
        "player_id": str(i), "season": 2026,
        "scoring_version": "weekly-v6",
        "breakout_score": 55.0, "confidence": 80.0,
        "grade": grade, "baseline_source": baseline_source,
    }


def test_findings_reports_cohort_group():
    rows = [_grade_row(i) for i in range(195)]
    rows += [_grade_row(1000 + i, baseline_source="cohort",
                        grade="hit" if i % 2 == 0 else "partial")
             for i in range(10)]
    lines = calibration.findings(rows, scoring_version="weekly-v6")
    cohort_lines = [ln for ln in lines if "rookie-cohort baseline" in ln]
    assert len(cohort_lines) == 1
    assert "10 graded calls" in cohort_lines[0]
    assert "5 hits" in cohort_lines[0]


def test_findings_silent_with_no_cohort_rows():
    rows = [_grade_row(i) for i in range(200)]
    lines = calibration.findings(rows, scoring_version="weekly-v6")
    assert not any("rookie-cohort baseline" in ln for ln in lines)
