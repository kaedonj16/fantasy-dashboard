"""Forecasts for open breakout calls + the breakout page track record.

Design rules pinned here:
- A forecast is computed live from partial outcomes; it is NEVER a grade
  and never feeds a hit rate. The track-record payload carries grades
  only; the forecast outlook is a separate, labeled section.
- Zero banked games means no band at all, and a mature call (full 3-week
  window) is never forecast.
- Preseason targets reuse the backtest definition exactly: valid prior
  (>= 6 games, >= 4.0 PPG) -> target = max(prior x 1.15, 7.0); otherwise
  the target is 10.0 PPG.
- Hit rates hide below the grader's 10-graded floor; counts stay real.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from data_building.breakout_engine import forecasts
from data_building.breakout_engine import weekly_grading as wg

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures (stored-evidence calls, same shape as the grading tests)
# ---------------------------------------------------------------------------

def wk(week, snap=60.0, tgt=6.0, car=8.0, ppr=12.0, snaps=60):
    return {
        "week": week, "snap_pct": snap, "targets": tgt, "carries": car,
        "ppr_pts": ppr, "snaps": snaps,
    }


def make_call(as_of_week=4, baseline_ppg=8.0, baseline_snap=40.0,
              baseline_opp=10.0, **overrides):
    call = {
        "player_id": "101", "player_name": "Test Player", "season": 2026,
        "as_of_week": as_of_week, "scoring_version": "weekly-v6",
        "classification": "emerging_breakout", "breakout_score": 55.0,
        "baseline_source": "current_season", "baseline_weeks": [1, 2],
        "evidence": {
            "fantasy": {"baseline_ppg": baseline_ppg, "recent_ppg": 14.0},
            "signals": {
                "snap_share": {"baseline": baseline_snap, "recent": 62.0},
                "carry_opportunity_pg": {"baseline": baseline_opp, "recent": 16.0},
            },
        },
    }
    call.update(overrides)
    return call


HELD_ROW = wk(5, snap=65.0, tgt=9.0, car=11.0, ppr=18.0)      # held + rose
REVERTED_ROW = wk(5, snap=41.0, tgt=4.0, car=6.0, ppr=8.5)    # role reverted
MIXED_ROW = wk(5, snap=45.0, tgt=5.0, car=6.0, ppr=9.0)       # neither


# ---------------------------------------------------------------------------
# weekly forecast bands
# ---------------------------------------------------------------------------

def test_weekly_forecast_tracking_to_hit():
    fc = forecasts.weekly_forecast(make_call(), [HELD_ROW], through_week=5)
    assert fc["state"] == "forecast"
    assert fc["kind"] == "weekly"
    assert fc["band"] == forecasts.BAND_TRACKING_HIT
    assert fc["band_label"] == "Tracking to hit"
    assert fc["weeks_in"] == 1
    assert fc["games"] == 1
    assert "1 of 3 weeks in" in fc["basis"]


def test_weekly_forecast_tracking_to_miss_on_reverted_role():
    fc = forecasts.weekly_forecast(make_call(), [REVERTED_ROW], through_week=5)
    assert fc["band"] == forecasts.BAND_TRACKING_MISS


def test_weekly_forecast_borderline_on_mixed_window():
    fc = forecasts.weekly_forecast(make_call(), [MIXED_ROW], through_week=5)
    assert fc["band"] == forecasts.BAND_BORDERLINE


def test_weekly_forecast_suppressed_with_zero_games():
    # No week has been played since the call.
    fc = forecasts.weekly_forecast(make_call(), [], through_week=4)
    assert fc["state"] == "no_games"
    assert fc["band"] is None
    assert "No games yet" in fc["basis"]
    # A week elapsed but the player has no recorded game in it: same state.
    fc2 = forecasts.weekly_forecast(make_call(), [wk(3)], through_week=5)
    assert fc2["state"] == "no_games"
    assert fc2["band"] is None


def test_weekly_forecast_needs_a_role_baseline():
    call = make_call(evidence={}, baseline_source="prior_season",
                     baseline_weeks=[])
    fc = forecasts.weekly_forecast(call, [HELD_ROW], through_week=5)
    assert fc["state"] == "no_baseline"
    assert fc["band"] is None


def _prior_season_call(**overrides):
    # Stored snap/opp baselines but no stored PPG: the engine's
    # prior-season pseudo-row carries no fantasy points.
    return make_call(baseline_ppg=None, baseline_source="prior_season",
                     baseline_weeks=[], **overrides)


def test_weekly_forecast_prior_season_call_caps_at_borderline_without_prior_rows():
    fc = forecasts.weekly_forecast(_prior_season_call(), [HELD_ROW],
                                   through_week=5)
    # The role clearly held (opp +10, snap +25), but with no PPG baseline
    # the production rise is unmeasured, so a hit is impossible.
    assert fc["state"] == "forecast"
    assert fc["band"] == forecasts.BAND_BORDERLINE
    assert fc["ppg_delta"] is None


def test_weekly_forecast_prior_season_ppg_fill_tracks_to_hit():
    prior_rows = [wk(1, ppr=9.0), wk(2, ppr=11.0)]   # prior mean 10.0 PPG
    fc = forecasts.weekly_forecast(_prior_season_call(), [HELD_ROW],
                                   through_week=5, prior_rows=prior_rows)
    assert fc["state"] == "forecast"
    assert fc["band"] == forecasts.BAND_TRACKING_HIT
    assert fc["ppg_delta"] == pytest.approx(8.0)     # 18.0 vs prior 10.0


def test_weekly_forecasts_for_calls_loads_prior_series_when_needed(monkeypatch):
    loaded = []

    def _series(season, through):
        loaded.append(season)
        if season == 2025:
            return {"101": [wk(1, ppr=9.0), wk(2, ppr=11.0)]}
        return {"101": [HELD_ROW]}

    monkeypatch.setattr(wg, "default_through_week", lambda season: 5)
    monkeypatch.setattr(wg, "load_season_series", _series)
    out = forecasts.weekly_forecasts_for_calls(2026, [_prior_season_call()])
    assert loaded == [2026, 2025]
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_weekly_forecasts_for_calls_skips_prior_load_when_unneeded(monkeypatch):
    loaded = []

    def _series(season, through):
        loaded.append(season)
        return {"101": [HELD_ROW]}

    monkeypatch.setattr(wg, "default_through_week", lambda season: 5)
    monkeypatch.setattr(wg, "load_season_series", _series)
    out = forecasts.weekly_forecasts_for_calls(2026, [make_call()])
    assert loaded == [2026]                  # no prior-season load paid for
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_weekly_forecast_never_for_a_mature_call():
    assert forecasts.weekly_forecast(make_call(), [HELD_ROW], through_week=7) is None


def test_weekly_forecasts_for_calls_skips_mature(monkeypatch):
    monkeypatch.setattr(wg, "default_through_week", lambda season: 5)
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through: {"101": [HELD_ROW]})
    open_call = make_call(as_of_week=4)
    mature_call = make_call(as_of_week=1, player_id="202")
    out = forecasts.weekly_forecasts_for_calls(2026, [open_call, mature_call])
    assert set(out) == {"101"}
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


# ---------------------------------------------------------------------------
# preseason targets + blending (backtest definition, reused exactly)
# ---------------------------------------------------------------------------

def test_season_hit_target_with_valid_prior():
    target, has_prior = forecasts.season_hit_target(8.0, 10)
    assert has_prior is True
    assert target == pytest.approx(9.2)  # 8.0 x 1.15


def test_season_hit_target_floor_applies():
    target, has_prior = forecasts.season_hit_target(4.5, 6)
    assert has_prior is True
    assert target == 7.0  # max(4.5 x 1.15, 7.0)


def test_season_hit_target_without_valid_prior():
    assert forecasts.season_hit_target(12.0, 5) == (10.0, False)   # too few games
    assert forecasts.season_hit_target(3.9, 12) == (10.0, False)   # too few PPG
    assert forecasts.season_hit_target(None, None) == (10.0, False)


def test_blend_forecast_ppg_is_games_weighted():
    blended = forecasts.blend_forecast_ppg(12.0, 4, 9.0, 10)
    assert blended == pytest.approx((12.0 * 4 + 9.0 * 10) / 14, abs=0.01)


def test_blend_forecast_ppg_pace_fallback_without_projection():
    assert forecasts.blend_forecast_ppg(12.0, 4, None, 0) == 12.0


def test_preseason_forecast_bands():
    hit = forecasts.preseason_forecast(
        games=4, current_ppg=12.0, prior_ppg=8.0, prior_games=10,
        ros_ppg=9.0, ros_games=10)
    assert hit["band"] == forecasts.BAND_TRACKING_HIT
    assert hit["ros_available"] is True
    assert hit["target_ppg"] == pytest.approx(9.2)

    borderline = forecasts.preseason_forecast(
        games=4, current_ppg=9.5, prior_ppg=None, prior_games=None)
    assert borderline["band"] == forecasts.BAND_BORDERLINE  # 9.5 vs 10.0 target
    assert borderline["ros_available"] is False
    assert "Pace only" in borderline["basis"]

    miss = forecasts.preseason_forecast(
        games=4, current_ppg=8.0, prior_ppg=None, prior_games=None)
    assert miss["band"] == forecasts.BAND_TRACKING_MISS


def test_preseason_forecast_suppressed_with_zero_games():
    fc = forecasts.preseason_forecast(
        games=0, current_ppg=None, prior_ppg=8.0, prior_games=10)
    assert fc["state"] == "no_games"
    assert fc["band"] is None
    assert fc["forecast_ppg"] is None


# ---------------------------------------------------------------------------
# track record: grades only, floor hides rates, counts stay real
# ---------------------------------------------------------------------------

def _grade_row(pid, classification, grade, ppg_delta, opp_delta=3.0):
    return {
        "player_id": pid, "player_name": f"Player {pid}",
        "classification": classification, "as_of_week": 2,
        "breakout_score": 50.0, "grade": grade, "outcome_games": 3,
        "ppg_delta": ppg_delta, "opp_delta": opp_delta, "snap_delta": 9.0,
    }


def _track_record_rows():
    rows = []
    # emerging_breakout: 12 graded (7 hit / 3 partial / 2 miss) -> rate shown
    for i in range(7):
        rows.append(_grade_row(f"h{i}", "emerging_breakout", "hit", 6.0 - i * 0.1))
    for i in range(3):
        rows.append(_grade_row(f"p{i}", "emerging_breakout", "partial", 1.0))
    rows.append(_grade_row("m0", "emerging_breakout", "miss", -4.0))
    rows.append(_grade_row("m1", "emerging_breakout", "miss", -7.5, opp_delta=-2.0))
    # watchlist: 4 graded -> below the floor, rate hidden
    for i in range(4):
        rows.append(_grade_row(f"w{i}", "watchlist", "hit", 3.0))
    return rows


def test_weekly_track_record_rates_and_floor(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _track_record_rows())
    record = forecasts.weekly_track_record(2026)

    groups = {g["classification"]: g for g in record["groups"]}
    emerging = groups["emerging_breakout"]
    assert emerging["graded"] == 12
    assert emerging["hit_rate"] == pytest.approx(7 / 12, abs=1e-4)
    watch = groups["watchlist"]
    assert watch["graded"] == 4
    assert watch["hit_rate"] is None
    assert record["overall"]["graded"] == 16


def test_weekly_track_record_biggest_hits_and_misses(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _track_record_rows())
    record = forecasts.weekly_track_record(2026)

    hits = record["hits"]
    assert [h["player_id"] for h in hits] == ["h0", "h1", "h2", "h3", "h4"]
    assert hits[0]["ppg_delta"] == pytest.approx(6.0)
    misses = record["misses"]
    assert [m["player_id"] for m in misses] == ["m1", "m0"]
    assert misses[0]["ppg_delta"] == pytest.approx(-7.5)


def test_top_grade_rows_orders_and_converts_decimals():
    rows = [
        _grade_row("a", "emerging_breakout", "miss", Decimal("-2.0"), opp_delta=Decimal("1.0")),
        _grade_row("b", "emerging_breakout", "miss", Decimal("-2.0"), opp_delta=Decimal("-3.0")),
        _grade_row("c", "emerging_breakout", "miss", None, opp_delta=Decimal("-9.0")),
    ]
    misses = forecasts.top_grade_rows(rows, "miss")
    # ppg tie broken by the worse opportunity delta; missing ppg goes last.
    assert [m["player_id"] for m in misses] == ["b", "a", "c"]
    assert isinstance(misses[0]["ppg_delta"], float)


def test_top_grade_rows_excludes_watchlist():
    rows = [
        _grade_row("a", "watchlist", "hit", 6.0),
        _grade_row("b", "monitored", "hit", 5.0),
        _grade_row("c", "emerging_breakout", "hit", 4.0),
        _grade_row("d", "early_watch", "hit", 3.0),
    ]
    hits = forecasts.top_grade_rows(rows, "hit")
    assert [h["player_id"] for h in hits] == ["c", "d"]


# ---------------------------------------------------------------------------
# season track record: pending until the season grades table has rows
# ---------------------------------------------------------------------------

class _SeasonConn:
    def __init__(self, table_exists, rows=None):
        self._table_exists = table_exists
        self._rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        if "information_schema" in query:
            one = {"x": 1} if self._table_exists else None
            return _OneResult(one)
        return _RowsResult(self._rows)


class _OneResult:
    def __init__(self, one):
        self._one = one

    def fetchone(self):
        return self._one


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


def test_season_track_record_pending_without_table(monkeypatch):
    monkeypatch.setattr(forecasts, "get_conn", lambda: _SeasonConn(False))
    record = forecasts.season_track_record(2026)
    assert record["available"] is False
    assert record["groups"] == []


def test_season_track_record_groups_by_phase_with_final_stage(monkeypatch):
    rows = []
    # One player graded twice (early miss, final hit): final wins.
    rows.append({"player_id": "dup", "player_name": "Dup", "phase": "preseason",
                 "grading_stage": "early", "grade": "miss",
                 "as_of_date": date(2026, 8, 20)})
    rows.append({"player_id": "dup", "player_name": "Dup", "phase": "preseason",
                 "grading_stage": "final", "grade": "hit",
                 "as_of_date": date(2026, 8, 20)})
    for i in range(11):
        rows.append({"player_id": f"s{i}", "player_name": f"S{i}",
                     "phase": "preseason", "grading_stage": "final",
                     "grade": "hit", "as_of_date": date(2026, 8, 20)})
    for i in range(3):
        rows.append({"player_id": f"d{i}", "player_name": f"D{i}",
                     "phase": "post_draft", "grading_stage": "final",
                     "grade": "miss", "as_of_date": date(2026, 5, 1)})
    monkeypatch.setattr(forecasts, "get_conn", lambda: _SeasonConn(True, rows))

    record = forecasts.season_track_record(2026)

    assert record["available"] is True
    groups = {g["phase"]: g for g in record["groups"]}
    assert groups["preseason"]["graded"] == 12
    assert groups["preseason"]["hit_rate"] == pytest.approx(1.0)
    assert groups["post_draft"]["graded"] == 3
    assert groups["post_draft"]["hit_rate"] is None


# ---------------------------------------------------------------------------
# service payload: outlook (forecasts) separate from grades; page wiring
# ---------------------------------------------------------------------------

def _patch_track_record(monkeypatch):
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)
    monkeypatch.setattr(forecasts, "weekly_track_record", lambda season: {
        "scoring_version": "weekly-v6", "min_sample": 10,
        "overall": {"calls": 16, "graded": 16, "hit": 11, "partial": 3,
                    "miss": 2, "ungraded": 0, "hit_rate": 0.6875,
                    "partial_rate": 0.1875, "miss_rate": 0.125},
        "groups": [{"classification": "emerging_breakout", "calls": 12,
                    "graded": 12, "hit": 7, "partial": 3, "miss": 2,
                    "ungraded": 0, "hit_rate": 0.5833, "partial_rate": 0.25,
                    "miss_rate": 0.1667}],
        "hits": [{"player_id": "h0", "player_name": "Big Hit",
                  "classification": "emerging_breakout", "call_week": 2,
                  "breakout_score": 55.0, "outcome_games": 3,
                  "ppg_delta": 6.0, "opp_delta": 4.0, "snap_delta": 12.0}],
        "misses": [{"player_id": "m1", "player_name": "Big Miss",
                    "classification": "emerging_breakout", "call_week": 2,
                    "breakout_score": 52.0, "outcome_games": 3,
                    "ppg_delta": -7.5, "opp_delta": -2.0, "snap_delta": -5.0}],
    })
    monkeypatch.setattr(forecasts, "season_track_record", lambda season: {
        "available": False, "groups": [], "overall": None})
    # No reconstructions by default: the pooled outlook's reconstruction
    # read sees this empty set and never touches the store.
    monkeypatch.setattr(forecasts, "load_reconstructed_weeks",
                        lambda season: set())
    board = {"view": "weekly", "candidates": [
        {"player_id": "1", "player_name": "Alpha", "classification": "emerging_breakout",
         "classification_label": "Emerging Breakout", "breakout_score": 55.0,
         "forecast": {"kind": "weekly", "band": "tracking_to_hit", "band_label": "Tracking to hit",
                      "basis": "1 of 3 weeks in, 1 game played"}},
        {"player_id": "2", "player_name": "Beta", "classification": "watchlist",
         "classification_label": "Watchlist", "breakout_score": 30.0,
         "forecast": {"kind": "weekly", "band": "tracking_to_hit", "band_label": "Tracking to hit",
                      "basis": "2 of 3 weeks in, 2 games played"}},
        {"player_id": "3", "player_name": "Gamma", "classification": "watchlist",
         "classification_label": "Watchlist", "breakout_score": 25.0,
         "forecast": {"kind": "weekly", "band": "borderline", "band_label": "Borderline",
                      "basis": "1 of 3 weeks in, 1 game played"}},
        {"player_id": "4", "player_name": "Delta", "classification": "watchlist",
         "classification_label": "Watchlist", "breakout_score": 22.0,
         "forecast": {"kind": "weekly", "band": None, "state": "no_games",
                      "basis": "No games yet in the 3 weeks after the Week 4 call"}},
        {"player_id": "5", "player_name": "Epsilon", "phase": "preseason",
         "breakout_opportunity_score": 70.0,
         "forecast": {"kind": "preseason", "band": "tracking_to_miss",
                      "band_label": "Tracking to miss", "basis": "Forecast 8.0 PPG vs a 10.0 PPG target, 4 games in"}},
    ]}
    monkeypatch.setattr(api, "get_breakout_board_candidates",
                        lambda season, min_score, limit, week=None: board)
    return api


def test_track_record_payload_shape_and_outlook(monkeypatch):
    api = _patch_track_record(monkeypatch)
    payload = api.get_breakout_track_record(2026)

    assert payload["season"] == 2026
    weekly = payload["weekly"]
    assert weekly["groups"][0]["label"] == "Emerging Breakout"
    assert weekly["groups"][0]["hit_rate"] == pytest.approx(0.5833)
    assert "kept the bigger role" in weekly["definition"]
    assert payload["season_engine"]["available"] is False
    assert "15 percent" in payload["season_engine"]["definition"]
    assert payload["pending_text"] == \
        "Still grading, not enough finished calls yet"

    outlook = payload["outlook"]
    assert outlook["weekly"]["counts"] == {
        "tracking_to_hit": 2, "borderline": 1, "tracking_to_miss": 0}
    assert outlook["weekly"]["open_calls"] == 3
    assert outlook["weekly"]["pending_calls"] == 1
    top = outlook["weekly"]["top_tracking_hit"]
    assert [t["player_name"] for t in top] == ["Alpha", "Beta"]
    assert top[0]["group_label"] == "Emerging Breakout"
    assert outlook["preseason"]["counts"]["tracking_to_miss"] == 1

    assert payload["hits"][0]["label"] == "Emerging Breakout"
    assert payload["hits"][0]["call_week"] == 2
    assert payload["misses"][0]["ppg_delta"] == pytest.approx(-7.5)


def test_forecasts_never_leak_into_track_record_grades(monkeypatch):
    api = _patch_track_record(monkeypatch)
    payload = api.get_breakout_track_record(2026)

    import json
    graded_blob = json.dumps({"weekly": payload["weekly"],
                              "season_engine": payload["season_engine"],
                              "hits": payload["hits"],
                              "misses": payload["misses"]})
    assert "tracking_to_hit" not in graded_blob
    assert "borderline" not in graded_blob
    assert "forecast" not in graded_blob.lower()
    # The graded counts come from grades alone: 12 emerging graded calls,
    # untouched by the 5 forecast-bearing open calls on the board.
    assert payload["weekly"]["groups"][0]["graded"] == 12


def test_definitions_have_no_em_dashes_and_pending_text_exact():
    import dashboard_services.breakout_api as api
    assert "—" not in api.WEEKLY_HIT_DEFINITION
    assert "—" not in api.SEASON_HIT_DEFINITION
    assert api.TRACK_RECORD_PENDING_TEXT == \
        "Still grading, not enough finished calls yet"


# ---------------------------------------------------------------------------
# outlook: the preseason board is aggregated even in season (Part A)
# ---------------------------------------------------------------------------

def _patch_two_boards(monkeypatch, *, preseason_raises=False,
                      preseason_board=None):
    """Default board serves weekly calls only; the preseason board is a
    separate load, exactly like production in season."""
    api = _patch_track_record(monkeypatch)
    default_board = {"view": "weekly", "candidates": [
        {"player_id": "1", "player_name": "Alpha", "classification": "emerging_breakout",
         "classification_label": "Emerging Breakout", "breakout_score": 55.0,
         "forecast": {"kind": "weekly", "band": "tracking_to_hit", "band_label": "Tracking to hit",
                      "basis": "1 of 3 weeks in, 1 game played"}},
        {"player_id": "4", "player_name": "Delta", "classification": "watchlist",
         "classification_label": "Watchlist", "breakout_score": 22.0,
         "forecast": {"kind": "weekly", "band": None, "state": "no_games",
                      "basis": "No games yet in the 3 weeks after the Week 4 call"}},
    ]}
    if preseason_board is None:
        preseason_board = {"view": "preseason", "candidates": [
            {"player_id": "5", "player_name": "Epsilon", "phase": "preseason",
             "breakout_opportunity_score": 70.0,
             "forecast": {"kind": "preseason", "band": "tracking_to_miss",
                          "band_label": "Tracking to miss", "basis": "Forecast 8.0 PPG vs a 10.0 PPG target, 4 games in"}},
        ]}

    def _board(season, min_score, limit, week=None):
        if week == "preseason":
            if preseason_raises:
                raise RuntimeError("preseason board unavailable")
            return preseason_board
        return default_board

    monkeypatch.setattr(api, "get_breakout_board_candidates", _board)
    return api


def test_outlook_counts_preseason_board_in_season(monkeypatch):
    api = _patch_two_boards(monkeypatch)
    payload = api.get_breakout_track_record(2026)

    outlook = payload["outlook"]
    assert outlook["weekly"]["counts"] == {
        "tracking_to_hit": 1, "borderline": 0, "tracking_to_miss": 0}
    assert outlook["weekly"]["pending_calls"] == 1
    # The bug: the preseason section read "No open calls right now" while
    # the preseason board carried live forecasts.
    assert outlook["preseason"]["counts"]["tracking_to_miss"] == 1
    assert outlook["preseason"]["open_calls"] == 1


def test_outlook_preseason_load_failure_is_fail_soft(monkeypatch):
    api = _patch_two_boards(monkeypatch, preseason_raises=True)
    payload = api.get_breakout_track_record(2026)

    outlook = payload["outlook"]
    assert outlook["weekly"]["counts"]["tracking_to_hit"] == 1
    assert outlook["preseason"]["open_calls"] == 0
    assert outlook["preseason"]["counts"] == {
        "tracking_to_hit": 0, "borderline": 0, "tracking_to_miss": 0}


def test_outlook_merged_boards_never_double_count(monkeypatch):
    # Out of season the default board IS the preseason board: the same
    # call arrives from both loads and must be counted once.
    shared = {"view": "preseason", "candidates": [
        {"player_id": "5", "player_name": "Epsilon", "phase": "preseason",
         "breakout_opportunity_score": 70.0,
         "forecast": {"kind": "preseason", "band": "tracking_to_miss",
                      "band_label": "Tracking to miss", "basis": "Forecast 8.0 PPG vs a 10.0 PPG target, 4 games in"}},
    ]}
    api = _patch_two_boards(monkeypatch, preseason_board=shared)
    monkeypatch.setattr(api, "get_breakout_board_candidates",
                        lambda season, min_score, limit, week=None: shared)

    payload = api.get_breakout_track_record(2026)

    assert payload["outlook"]["preseason"]["counts"]["tracking_to_miss"] == 1
    assert payload["outlook"]["preseason"]["open_calls"] == 1


def test_outlook_counts_only_the_surfaced_top_15_per_board(monkeypatch):
    # The boards surface only the top BREAKOUT_BOARD_LIMIT candidates, so
    # the sidebar outlook must aggregate that same set. The stub keeps 20
    # forecast-carrying candidates per board and honors the limit exactly
    # like the real loader (score order, then truncate); the track record
    # must ask for the cap and the outlook must reflect 15, not 20.
    api = _patch_track_record(monkeypatch)
    seen_limits = []

    def _candidates(prefix, kind):
        return [
            {"player_id": f"{prefix}{i}", "player_name": f"{prefix} {i}",
             "classification": "watchlist",
             "classification_label": "Watchlist",
             "breakout_score": 100.0 - i,
             "forecast": {"kind": kind, "band": "tracking_to_hit",
                          "band_label": "Tracking to hit",
                          "basis": "stubbed"}}
            for i in range(20)
        ]

    def _board(season, min_score, limit, week=None):
        seen_limits.append(limit)
        if week == "preseason":
            candidates = _candidates("p", "preseason")
            view = "preseason"
        else:
            candidates = _candidates("w", "weekly")
            view = "weekly"
        if limit is not None:
            candidates = candidates[:limit]
        return {"view": view, "candidates": candidates}

    monkeypatch.setattr(api, "get_breakout_board_candidates", _board)

    payload = api.get_breakout_track_record(2026)

    assert api.BREAKOUT_BOARD_LIMIT == 15
    assert seen_limits == [api.BREAKOUT_BOARD_LIMIT,
                           api.BREAKOUT_BOARD_LIMIT]
    outlook = payload["outlook"]
    assert outlook["weekly"]["open_calls"] == 15
    assert outlook["weekly"]["counts"]["tracking_to_hit"] == 15
    assert outlook["preseason"]["open_calls"] == 15
    assert outlook["preseason"]["counts"]["tracking_to_hit"] == 15


# ---------------------------------------------------------------------------
# track record: reconstructed grades pool into the one weekly record
# ---------------------------------------------------------------------------

def _split_rows():
    rows = []
    # Live (week 4): emerging 12 graded (7 hit / 3 partial / 2 miss).
    for i in range(7):
        row = _grade_row(f"lh{i}", "emerging_breakout", "hit", 5.0 - i * 0.1)
        row["as_of_week"] = 4
        rows.append(row)
    for i in range(3):
        row = _grade_row(f"lp{i}", "emerging_breakout", "partial", 1.0)
        row["as_of_week"] = 4
        rows.append(row)
    for i in range(2):
        row = _grade_row(f"lm{i}", "emerging_breakout", "miss", -4.0 - i)
        row["as_of_week"] = 4
        rows.append(row)
    # Reconstructed (weeks 1-2): emerging 12 graded, plus the single
    # biggest hit of the whole set.
    row = _grade_row("rb0", "emerging_breakout", "hit", 9.5)
    row["as_of_week"] = 1
    rows.append(row)
    for i in range(11):
        row = _grade_row(f"rh{i}", "emerging_breakout", "hit", 4.0)
        row["as_of_week"] = 2
        rows.append(row)
    # Reconstructed watchlist: only 3 graded, below the floor.
    for i in range(3):
        row = _grade_row(f"rw{i}", "watchlist", "miss", -3.0)
        row["as_of_week"] = 1
        rows.append(row)
    return rows


def test_weekly_track_record_pools_reconstructed_grades(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _split_rows())
    monkeypatch.setattr(forecasts, "load_reconstructed_weeks",
                        lambda season: {1, 2})
    record = forecasts.weekly_track_record(2026)

    # One pooled record: no separate backtest entry anywhere.
    assert "backtest" not in record
    # Live (12 graded) and reconstructed (15 graded) calls sum into one
    # overall, and the 10-graded floor applies to the pooled groups.
    assert record["overall"]["graded"] == 27
    assert record["overall"]["hit"] == 19
    groups = {g["classification"]: g for g in record["groups"]}
    assert groups["emerging_breakout"]["graded"] == 24
    assert groups["emerging_breakout"]["hit_rate"] == \
        pytest.approx(19 / 24, abs=1e-4)
    assert groups["watchlist"]["graded"] == 3
    assert groups["watchlist"]["hit_rate"] is None

    # Biggest hits span both sets; the reconstructed one is flagged.
    assert record["hits"][0]["player_id"] == "rb0"
    assert record["hits"][0]["reconstructed"] is True
    live_hit = next(h for h in record["hits"] if h["player_id"] == "lh0")
    assert live_hit["reconstructed"] is False


def test_weekly_track_record_without_reconstructions(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _track_record_rows())
    monkeypatch.setattr(forecasts, "load_reconstructed_weeks",
                        lambda season: set())
    record = forecasts.weekly_track_record(2026)

    assert "backtest" not in record
    assert record["overall"]["graded"] == 16
    assert all(h["reconstructed"] is False for h in record["hits"])


def test_track_record_payload_weekly_record_is_combined(monkeypatch):
    api = _patch_track_record(monkeypatch)
    monkeypatch.setattr(forecasts, "weekly_track_record", lambda season: {
        "scoring_version": "weekly-v6", "min_sample": 10,
        "overall": {"calls": 27, "graded": 27, "hit": 19, "partial": 3,
                    "miss": 5, "ungraded": 0, "hit_rate": 19 / 27,
                    "partial_rate": 3 / 27, "miss_rate": 5 / 27},
        "groups": [{"classification": "emerging_breakout",
                    "calls": 24, "graded": 24, "hit": 19, "partial": 3,
                    "miss": 2, "ungraded": 0, "hit_rate": 19 / 24,
                    "partial_rate": 0.125, "miss_rate": 2 / 24}],
        "hits": [], "misses": [],
    })

    payload = api.get_breakout_track_record(2026)

    weekly = payload["weekly"]
    # The combined weekly record carries no backtest section and no
    # provenance week list; the outlook has no backtest entry either.
    assert "backtest" not in weekly
    assert "reconstructed_weeks" not in weekly
    assert weekly["overall"]["graded"] == 27
    assert weekly["groups"][0]["label"] == "Emerging Breakout"
    assert "weekly_backtest" not in payload["outlook"]


def test_track_record_route_registered():
    flask = pytest.importorskip("flask")
    from dashboard_services.breakout_api import register_breakout_routes

    app = flask.Flask(__name__)
    register_breakout_routes(app)
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/api/breakout/track-record" in rules


# ---------------------------------------------------------------------------
# board attach points
# ---------------------------------------------------------------------------

def _weekly_payload():
    return {
        "season": 2026, "as_of_week": 5, "as_of_date": "2026-10-05",
        "count": 1, "candidates": [{
            "player_id": "101", "player_name": "Test Player", "team": "KC",
            "position": "WR", "breakout_score": 55.0, "confidence": 80.0,
            "classification": "emerging_breakout", "as_of_week": 5,
            "evidence": {
                "score_basis": "role_change", "main_board_eligible": True,
                "provisional": False,
                "fantasy": {"baseline_ppg": 8.0, "recent_ppg": 14.0},
                "signals": {},
            },
        }],
        "scoring_version": "weekly-v6", "data_available": True,
        "data_status": "available",
    }


def test_weekly_board_attaches_forecast_for_open_calls(monkeypatch):
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store

    monkeypatch.setattr(weekly_store, "load_weekly_candidates",
                        lambda season, **kw: _weekly_payload())
    monkeypatch.setattr(forecasts, "weekly_forecasts_for_calls",
                        lambda season, calls: {"101": {
                            "kind": "weekly", "band": "tracking_to_hit",
                            "band_label": "Tracking to hit",
                            "basis": "1 of 3 weeks in, 1 game played"}})

    payload = api.get_weekly_breakout_candidates(2026, min_score=0, limit=None)
    candidates = payload["candidates"]

    assert len(candidates) == 1
    assert candidates[0]["forecast"]["band"] == "tracking_to_hit"


def test_weekly_board_attaches_forecast_on_explicit_week_view(monkeypatch):
    # Explicit week views attach forecasts for open calls too: a week 2
    # board whose outcome window is still open is forecast business,
    # not finished history. weekly_forecasts_for_calls skips mature
    # calls itself, so finished weeks get no chips.
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store

    monkeypatch.setattr(weekly_store, "load_weekly_candidates",
                        lambda season, **kw: _weekly_payload())
    monkeypatch.setattr(forecasts, "weekly_forecasts_for_calls",
                        lambda season, calls: {"101": {
                            "kind": "weekly", "band": "tracking_to_hit",
                            "band_label": "Tracking to hit",
                            "basis": "1 of 3 weeks in, 1 game played"}})

    payload = api.get_weekly_breakout_candidates(
        2026, min_score=0, limit=None, as_of_week=2)
    candidates = payload["candidates"]

    assert len(candidates) == 1
    assert candidates[0]["forecast"]["band"] == "tracking_to_hit"


def test_preseason_board_attaches_forecast(monkeypatch):
    import dashboard_services.breakout_api as api

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            return None

        def fetchone(self):
            return {"max": 2026}

        def fetchall(self):
            return [{
                "player_id": "201", "player_name": "Pre Player", "team": "BUF",
                "position": "RB", "breakout_opportunity_score": 72.0,
                "opportunity_opened_score": 30.0, "player_readiness_score": 30.0,
                "hit_probability": 0.42, "phase": "preseason",
                "as_of_date": date(2026, 8, 20), "age": 23.0,
                "cumulative_ppr": 0.0, "peak_ppr": 0.0,
            }]

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return _Cursor()

    monkeypatch.setattr(api, "_weekly_breakout_available", lambda season: False)
    monkeypatch.setattr(api, "_weekly_history_exists", lambda season: False)
    monkeypatch.setattr(api, "opportunity_data_ready", lambda season: True)
    monkeypatch.setattr(api, "get_conn", lambda: _Conn())
    monkeypatch.setattr(forecasts, "preseason_forecasts_for_season",
                        lambda season, player_ids=None: {"201": {
                            "kind": "preseason", "band": "borderline",
                            "band_label": "Borderline",
                            "basis": "Forecast 9.5 PPG vs a 10.0 PPG target, 4 games in"}})

    payload = api.get_breakout_candidates(2026, min_score=40)

    assert payload["candidates"], "expected the preseason candidate"
    assert payload["candidates"][0]["forecast"]["band"] == "borderline"


def test_v5_shaped_row_renders_without_confidence_detail():
    import dashboard_services.breakout_api as api

    row = {
        "player_id": "301", "player_name": "Old Row", "team": "MIA",
        "position": "WR", "breakout_score": 48.0, "confidence": 66.0,
        "classification": "early_watch", "as_of_week": 2,
        "scoring_version": "weekly-v5",
        # v5 evidence predates the v6 confidence_detail block entirely.
        "evidence": {
            "score_basis": "role_change", "fantasy": {}, "signals": {},
            "lifecycle": {}, "opportunity": {},
        },
    }
    card = api._weekly_row_to_candidate(row)
    assert card["player_id"] == "301"
    assert card["confidence"] == 66  # stored value, nothing fabricated
    assert "forecast" not in card


# ---------------------------------------------------------------------------
# pooled outlook: reconstructed weeks' open calls join the weekly outlook
# ---------------------------------------------------------------------------

_HELD = {"snap": 65.0, "tgt": 9.0, "car": 11.0, "ppr": 18.0}
_REVERTED = {"snap": 41.0, "tgt": 4.0, "car": 6.0, "ppr": 8.5}


def _recon_row(pid, week, score):
    return make_call(as_of_week=week, player_id=pid,
                     player_name=f"Player {pid}", breakout_score=score)


def _patch_reconstructed_outlook(monkeypatch, *, through=3, week_rows=None,
                                 series=None):
    """Reconstructed weeks 1-2 with stored runs, layered on the two-board
    track record stubs. ``week_rows`` maps week -> the score rows stored
    under that week's reconstruction run."""
    from data_building.breakout_engine import weekly_store

    api = _patch_two_boards(monkeypatch)
    runs = {1: {"id": 11}, 2: {"id": 12}}
    if week_rows is None:
        week_rows = {
            1: [_recon_row("101", 1, 55.0), _recon_row("102", 1, 50.0)],
            2: [_recon_row("101", 2, 60.0), _recon_row("103", 2, 45.0)],
        }
    rows_by_run = {runs[w]["id"]: rows for w, rows in week_rows.items()}
    if series is None:
        series = {
            "101": [wk(2, **_HELD), wk(3, **_HELD)],
            "102": [wk(2, **_REVERTED), wk(3, **_REVERTED)],
            "103": [],
        }
    monkeypatch.setattr(forecasts, "load_reconstructed_weeks",
                        lambda season: {1, 2})
    monkeypatch.setattr(weekly_store, "get_reconstructed_run",
                        lambda season, week: runs.get(week))
    monkeypatch.setattr(weekly_store, "load_run_score_rows",
                        lambda run_id: rows_by_run.get(run_id, []))
    monkeypatch.setattr(wg, "default_through_week", lambda season: through)
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through_week: series)
    return api


def test_outlook_pools_reconstructed_open_calls(monkeypatch):
    api = _patch_reconstructed_outlook(monkeypatch)
    payload = api.get_breakout_track_record(2026)

    # No separate backtest entry: the reconstructed forecasts are inside
    # the one weekly outlook.
    assert "weekly_backtest" not in payload["outlook"]
    weekly = payload["outlook"]["weekly"]
    # Reconstructed: 101 tracks to hit (its week 2 call), 102 tracks to
    # miss, 103 has no games yet. Live board: Alpha tracks to hit,
    # Delta pending. Pooled, each player counted once:
    assert weekly["counts"] == {
        "tracking_to_hit": 2, "borderline": 0, "tracking_to_miss": 1}
    assert weekly["open_calls"] == 3
    assert weekly["pending_calls"] == 2
    # Both pending calls (Delta on the live board, 103 reconstructed) are
    # bandless only because no outcome game has been played yet.
    assert weekly["pending_no_games"] == 2
    assert weekly["pending_no_baseline"] == 0
    top = weekly["top_tracking_hit"]
    # The named top entries are live-board-only: reconstructed tracking
    # hits still count in the band counts above, but they are never
    # named in the top list.
    assert [t["player_id"] for t in top] == ["1"]
    # The live entry carries no backtest tag.
    assert "reconstructed" not in top[0]


def test_outlook_live_call_beats_reconstructed_call(monkeypatch):
    api = _patch_two_boards(monkeypatch)

    def _view(pid, band, score, name):
        return {"player_id": pid, "player_name": name,
                "classification": "watchlist", "breakout_score": score,
                "call_week": 1,
                "forecast": {"kind": "weekly", "band": band,
                             "band_label": "Stub", "basis": "recon stub"}}

    monkeypatch.setattr(forecasts, "weekly_backtest_forecasts",
                        lambda season, limit=None: ([1], {
                            # Same player as the live board's Alpha,
                            # but the reconstruction says miss.
                            "1": _view("1", "tracking_to_miss", 99.0, "Alpha"),
                            "9": _view("9", "tracking_to_hit", 88.0, "Iota"),
                        }))
    payload = api.get_breakout_track_record(2026)

    weekly = payload["outlook"]["weekly"]
    # The player counts once, with the LIVE forecast: the reconstructed
    # miss never lands. The reconstructed-only tracking hit (9) counts
    # in the band counts but never lands in the named top list, which
    # is live-board-only.
    assert weekly["counts"] == {
        "tracking_to_hit": 2, "borderline": 0, "tracking_to_miss": 0}
    assert weekly["open_calls"] == 2
    assert weekly["pending_calls"] == 1
    top = {t["player_id"]: t for t in weekly["top_tracking_hit"]}
    assert set(top) == {"1"}
    assert "reconstructed" not in top["1"]


def test_outlook_skips_mature_reconstructed_calls(monkeypatch):
    # Through week 6 both the week 1 and week 2 outcome windows are
    # complete: the calls are the grader's business and forecast
    # nothing, so the pooled weekly outlook is the live board alone.
    api = _patch_reconstructed_outlook(monkeypatch, through=6)
    payload = api.get_breakout_track_record(2026)

    weekly = payload["outlook"]["weekly"]
    assert weekly["counts"] == {
        "tracking_to_hit": 1, "borderline": 0, "tracking_to_miss": 0}
    assert weekly["open_calls"] == 1
    assert weekly["pending_calls"] == 1
    assert "weekly_backtest" not in payload["outlook"]


def test_reconstructed_forecasts_cap_each_week_at_the_board_limit(monkeypatch):
    _patch_reconstructed_outlook(
        monkeypatch,
        week_rows={1: [_recon_row(f"c{i}", 1, 100.0 - i) for i in range(20)],
                   2: []},
        series={})
    seen = {"counts": []}

    def _record(season, calls):
        seen["counts"].append(len(calls))
        return {}

    monkeypatch.setattr(forecasts, "weekly_forecasts_for_calls", _record)
    import dashboard_services.breakout_api as api_mod
    weeks, views = forecasts.weekly_backtest_forecasts(
        2026, limit=api_mod.BREAKOUT_BOARD_LIMIT)

    assert seen["counts"][0] == api_mod.BREAKOUT_BOARD_LIMIT == 15
    assert weeks == [1, 2]
    assert views == {}


def test_outlook_without_reconstructions_is_board_only(monkeypatch):
    api = _patch_track_record(monkeypatch)
    payload = api.get_breakout_track_record(2026)

    assert "weekly_backtest" not in payload["outlook"]
    weekly = payload["outlook"]["weekly"]
    assert weekly["open_calls"] == 3
    assert weekly["pending_calls"] == 1
    assert all("reconstructed" not in t
               for t in weekly["top_tracking_hit"])


# ---------------------------------------------------------------------------
# outlook pending split: no games yet vs no baseline to measure against
# ---------------------------------------------------------------------------

def test_forecast_outlook_splits_pending_by_state():
    import dashboard_services.breakout_api as api

    def _cand(pid, kind, band, state=None):
        forecast = {"kind": kind, "band": band, "band_label": None,
                    "basis": "stub"}
        if state is not None:
            forecast["state"] = state
        return {"player_id": pid, "player_name": f"P{pid}",
                "classification": "watchlist",
                "classification_label": "Watchlist", "breakout_score": 30.0,
                "forecast": forecast}

    board = {"candidates": [
        _cand("1", "weekly", "tracking_to_hit", "forecast"),
        _cand("2", "weekly", None, "no_games"),
        _cand("3", "weekly", None, "no_baseline"),
        _cand("4", "weekly", None, "no_baseline"),
        _cand("5", "preseason", None, "no_games"),
    ]}
    outlook = api._forecast_outlook(board)

    weekly = outlook["weekly"]
    assert weekly["open_calls"] == 1
    assert weekly["pending_calls"] == 3
    assert weekly["pending_no_games"] == 1
    assert weekly["pending_no_baseline"] == 2
    # Preseason forecasts have no no-baseline state; the split reports 0.
    preseason = outlook["preseason"]
    assert preseason["pending_calls"] == 1
    assert preseason["pending_no_games"] == 1
    assert preseason["pending_no_baseline"] == 0


def test_forecast_outlook_top_tracking_hit_is_live_only():
    # Reconstructed (backtest) calls count toward the band counts, but
    # the named top entries only ever show live-board calls, even when
    # a reconstructed call would rank first by score.
    import dashboard_services.breakout_api as api

    def _cand(pid, band, score, reconstructed):
        cand = {
            "player_id": pid, "player_name": f"P{pid}",
            "classification": "watchlist", "classification_label": "Watchlist",
            "breakout_score": score,
            "forecast": {"kind": "weekly", "band": band,
                         "band_label": "Stub", "basis": "stub",
                         "state": "forecast"},
        }
        if reconstructed:
            cand["reconstructed"] = True
            cand["call_week"] = 2
        return cand

    board = {"candidates": [
        _cand("9", "tracking_to_hit", 99.0, True),
        _cand("1", "tracking_to_hit", 30.0, False),
    ]}
    weekly = api._forecast_outlook(board)["weekly"]

    assert weekly["counts"]["tracking_to_hit"] == 2
    assert weekly["open_calls"] == 2
    top = weekly["top_tracking_hit"]
    assert [t["player_id"] for t in top] == ["1"]
    assert all("reconstructed" not in t for t in top)


def test_forecast_outlook_preseason_top_label_is_preseason():
    # The top-list label for preseason calls reads "Preseason" (the
    # board's name in the rail), not the raw phase name "Offseason".
    import dashboard_services.breakout_api as api

    board = {"candidates": [{
        "player_id": "7", "player_name": "P7",
        "breakout_opportunity_score": 61.0,
        "forecast": {"kind": "preseason", "phase": "offseason",
                     "band": "tracking_to_hit", "band_label": "Stub",
                     "basis": "stub", "state": "forecast"},
    }]}
    top = api._forecast_outlook(board)["preseason"]["top_tracking_hit"]
    assert len(top) == 1
    assert top[0]["group_label"] == "Preseason"


def test_outlook_reconstructed_no_baseline_counts_separately(monkeypatch):
    # A reconstructed call whose player has outcome games but no baseline
    # at all (initial-role shape) must land in the no-baseline split, not
    # in the no-games count.
    no_base = make_call(as_of_week=1, player_id="104",
                        player_name="Player 104", breakout_score=40.0,
                        evidence={}, baseline_source="none",
                        baseline_weeks=[])
    api = _patch_reconstructed_outlook(
        monkeypatch,
        week_rows={1: [no_base], 2: []},
        series={"104": [wk(2, **_HELD), wk(3, **_HELD)]})
    payload = api.get_breakout_track_record(2026)

    weekly = payload["outlook"]["weekly"]
    # Live board: Alpha tracks to hit, Delta has no games yet.
    # Reconstruction: 104 played both elapsed window weeks, no baseline.
    assert weekly["counts"]["tracking_to_hit"] == 1
    assert weekly["pending_calls"] == 2
    assert weekly["pending_no_games"] == 1
    assert weekly["pending_no_baseline"] == 1


# ---------------------------------------------------------------------------
# rail rendering: the pending lines are split and honestly labeled
# ---------------------------------------------------------------------------

def _extract_bo_outlook_block(src):
    """The _boOutlookBlock function source from the breakout page f-string
    in app.py, with the f-string's doubled braces collapsed back into the
    real JS braces."""
    start = src.index("function _boOutlookBlock(title, block)")
    i = src.index("{{", start)
    depth, j = 0, i
    while True:
        two = src[j:j + 2]
        if two == "{{":
            depth += 1
            j += 2
            continue
        if two == "}}":
            depth -= 1
            j += 2
            if depth == 0:
                break
            continue
        j += 1
    return src[start:j].replace("{{", "{").replace("}}", "}")


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_rail_outlook_block_renders_pending_lines_by_state():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = _extract_bo_outlook_block(src)
    driver = """
var counts = {tracking_to_hit: 0, borderline: 0, tracking_to_miss: 0};
function block(pending, noGames, noBaseline) {
  return {counts: counts, open_calls: 1, pending_calls: pending,
          pending_no_games: noGames, pending_no_baseline: noBaseline,
          top_tracking_hit: []};
}
var out = {
  both: _boOutlookBlock('Weekly calls', block(3, 2, 1)),
  gamesOnly: _boOutlookBlock('Weekly calls', block(2, 2, 0)),
  baselineOnly: _boOutlookBlock('Weekly calls', block(1, 0, 1)),
  none: _boOutlookBlock('Weekly calls', block(0, 0, 0))
};
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    assert "2 more open calls with no games yet" in out["both"]
    assert "1 more open call with no baseline to measure against" in out["both"]
    assert "with no games yet" in out["gamesOnly"]
    assert "no baseline to measure against" not in out["gamesOnly"]
    assert "with no games yet" not in out["baselineOnly"]
    assert "1 more open call with no baseline to measure against" in out["baselineOnly"]
    assert "no band yet" not in out["none"]
    # UI copy carries no em dashes.
    assert "—" not in out["both"]


# ---------------------------------------------------------------------------
# page + stylesheet wiring
# ---------------------------------------------------------------------------

def test_breakout_page_has_sidebar_and_forecast_chip():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Single tabbed track record section (Weekly/Preseason tabs)
    assert "boRailTrackRecord" in src
    assert "/api/breakout/track-record?season=" in src
    assert "loadBreakoutSidebar()" in src
    assert "bo-layout" in src and "bo-main" in src and "bo-rail" in src
    # The chip literally carries the word Forecast on the card.
    assert "Forecast: " in src
    assert "candidate.forecast" in src
    # Combined sections only: no separate backtest outlook subsection,
    # no track-record backtest block, and no provenance note copy. The
    # per-entry Backtest chip and the slim pending lines stay.
    assert "weekly_backtest" not in src
    assert "Weekly backtest" not in src
    assert "never count toward the live rates" not in src
    assert "re-scored later" not in src
    assert "bo-grade-chip-bt" in src
    assert "bo-rail-pending-line" in src


def test_dashboard_css_has_rail_and_chip_styles():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".bo-layout" in css
    assert ".bo-rail" in css
    assert ".bo-rail-section" in css
    assert ".bo-rail-pending-line" in css
    assert ".bo-fc-chip" in css
    assert "border-radius: 999px" not in css.split(".bo-fc-chip")[1].split("}")[0]


# ---------------------------------------------------------------------------
# track record calibration bands (score + confidence, pooled, shared math)
# ---------------------------------------------------------------------------

def _band_grade_row(pid, score, conf, grade, week=4,
                    classification="emerging_breakout"):
    return {
        "player_id": pid, "player_name": f"Player {pid}",
        "classification": classification, "as_of_week": week,
        "breakout_score": score, "confidence": conf, "grade": grade,
        "outcome_games": 3, "ppg_delta": 2.0, "opp_delta": 1.0,
        "snap_delta": 5.0,
    }


def _band_rows():
    rows = []
    # 42-59 score / 70+ confidence: 12 graded emerging calls. Ten are
    # live (week 4); two come from reconstructed week 1 and pool in.
    grades = ["hit"] * 7 + ["partial"] * 3 + ["miss"] * 2
    for i, grade in enumerate(grades):
        rows.append(_band_grade_row(f"a{i}", 50.0, 85.0, grade,
                                    week=1 if i >= 10 else 4))
    # 60+ score / 40-69 confidence: 4 graded, all hits (below the floor).
    for i in range(4):
        rows.append(_band_grade_row(f"b{i}", 75.0, 50.0, "hit"))
    # 18-41 score / Under 40 confidence: 3 graded watchlist misses.
    for i in range(3):
        rows.append(_band_grade_row(f"c{i}", 30.0, 20.0, "miss",
                                    classification="watchlist"))
    # No recorded score: excluded from score bands, still graded overall
    # and inside the 70+ confidence band.
    rows.append(_band_grade_row("d0", None, 85.0, "hit",
                                classification="watchlist"))
    # No recorded confidence: excluded from confidence bands, still in
    # the 60+ score band.
    rows.append(_band_grade_row("e0", 65.0, None, "hit",
                                classification="watchlist"))
    return rows


def test_weekly_track_record_score_and_confidence_bands(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _band_rows())
    monkeypatch.setattr(forecasts, "load_reconstructed_weeks",
                        lambda season: {1, 2})
    record = forecasts.weekly_track_record(2026)

    assert record["overall"]["graded"] == 21
    score = {b["label"]: b for b in record["score_bands"]}
    # Fixed band order; the empty Under 18 band is omitted like a
    # classification with no calls.
    assert [b["label"] for b in record["score_bands"]] == \
        ["18-41", "42-59", "60+"]
    pooled = score["42-59"]
    assert pooled["calls"] == 12          # live 10 + reconstructed 2
    assert pooled["graded"] == 12
    assert pooled["hit"] == 7 and pooled["partial"] == 3
    assert pooled["miss"] == 2
    assert pooled["hit_rate"] == pytest.approx(7 / 12, abs=1e-4)
    assert score["60+"]["graded"] == 5    # includes the confidenceless row
    assert score["60+"]["hit_rate"] is None   # below the 10-graded floor
    assert score["18-41"]["graded"] == 3
    assert score["18-41"]["miss"] == 3
    # The scoreless grade counts overall and by classification, just
    # not in any score band: band graded sums to one less than overall.
    assert sum(b["graded"] for b in record["score_bands"]) == 20
    groups = {g["classification"]: g for g in record["groups"]}
    assert groups["watchlist"]["graded"] == 5

    conf = {b["label"]: b for b in record["confidence_bands"]}
    assert [b["label"] for b in record["confidence_bands"]] == \
        ["Under 40", "40-69", "70+"]
    assert conf["70+"]["graded"] == 13    # includes the scoreless row
    assert conf["70+"]["hit"] == 8
    assert conf["70+"]["hit_rate"] == pytest.approx(8 / 13, abs=1e-4)
    assert conf["40-69"]["graded"] == 4
    assert conf["40-69"]["hit_rate"] is None
    assert conf["Under 40"]["graded"] == 3
    # The confidenceless grade is the one missing from the band sum.
    assert sum(b["graded"] for b in record["confidence_bands"]) == 20


def test_weekly_track_record_no_bands_until_something_grades(monkeypatch):
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: [])
    record = forecasts.weekly_track_record(2026)
    assert record["groups"] == []
    assert record["score_bands"] == []
    assert record["confidence_bands"] == []

    ungraded = [_band_grade_row(f"u{i}", 50.0, 80.0, "ungraded")
                for i in range(3)]
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: ungraded)
    record = forecasts.weekly_track_record(2026)
    assert record["overall"]["graded"] == 0
    assert record["score_bands"] == []
    assert record["confidence_bands"] == []


def test_track_record_payload_carries_bands(monkeypatch):
    api = _patch_track_record(monkeypatch)
    monkeypatch.setattr(forecasts, "weekly_track_record", lambda season: {
        "scoring_version": "weekly-v6", "min_sample": 10,
        "overall": {"calls": 21, "graded": 21, "hit": 12, "partial": 3,
                    "miss": 6, "ungraded": 0, "hit_rate": 12 / 21,
                    "partial_rate": 3 / 21, "miss_rate": 6 / 21},
        "groups": [{"classification": "emerging_breakout", "calls": 16,
                    "graded": 16, "hit": 11, "partial": 3, "miss": 2,
                    "ungraded": 0, "hit_rate": 11 / 16,
                    "partial_rate": 3 / 16, "miss_rate": 2 / 16}],
        "score_bands": [
            {"label": "42-59", "calls": 12, "graded": 12, "hit": 7,
             "partial": 3, "miss": 2, "ungraded": 0,
             "hit_rate": 7 / 12, "partial_rate": 0.25,
             "miss_rate": 2 / 12}],
        "confidence_bands": [
            {"label": "70+", "calls": 13, "graded": 13, "hit": 8,
             "partial": 3, "miss": 2, "ungraded": 0,
             "hit_rate": 8 / 13, "partial_rate": 3 / 13,
             "miss_rate": 2 / 13}],
        "hits": [], "misses": [],
    })

    payload = api.get_breakout_track_record(2026)

    weekly = payload["weekly"]
    assert weekly["score_bands"][0]["label"] == "42-59"
    assert weekly["score_bands"][0]["graded"] == 12
    assert weekly["score_bands"][0]["hit_rate"] == pytest.approx(7 / 12)
    assert weekly["confidence_bands"][0]["label"] == "70+"
    assert weekly["confidence_bands"][0]["hit_rate"] == \
        pytest.approx(8 / 13)


def test_breakout_page_renders_calibration_band_rows():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # The redesigned weekly record renders per-week progress bars (newest
    # first) instead of the old classification/score/confidence band rows.
    assert "by_week" in src
    assert "_boWeekBar" in src
    assert "Show all weeks" in src
