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
HELD_ROW_6 = wk(6, snap=65.0, tgt=9.0, car=11.0, ppr=18.0)    # held + rose
REVERTED_ROW = wk(5, snap=41.0, tgt=4.0, car=6.0, ppr=8.5)    # role reverted
MIXED_ROW = wk(5, snap=45.0, tgt=5.0, car=6.0, ppr=9.0)       # neither


# ---------------------------------------------------------------------------
# weekly forecast bands
# ---------------------------------------------------------------------------

def test_weekly_forecast_tracking_to_hit():
    # Two games at hit pace: a hit needs a 2-game sample (grading v4),
    # so the fixture banks two held weeks.
    fc = forecasts.weekly_forecast(make_call(), [HELD_ROW, HELD_ROW_6],
                                   through_week=6)
    assert fc["state"] == "forecast"
    assert fc["kind"] == "weekly"
    assert fc["band"] == forecasts.BAND_TRACKING_HIT
    assert fc["band_label"] == "Tracking to hit"
    assert fc["weeks_in"] == 2
    assert fc["games"] == 2
    assert "2 of 3 weeks in" in fc["basis"]


def test_weekly_forecast_one_game_hit_pace_is_borderline():
    # The grading-v4 hit floor applied live: one game at hit pace would
    # grade partial (single_game_sample) if the window closed today, so
    # the forecast band is Borderline, not Tracking to hit.
    fc = forecasts.weekly_forecast(make_call(), [HELD_ROW], through_week=5)
    assert fc["state"] == "forecast"
    assert fc["games"] == 1
    assert fc["band"] == forecasts.BAND_BORDERLINE


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
    fc = forecasts.weekly_forecast(_prior_season_call(),
                                   [HELD_ROW, HELD_ROW_6],
                                   through_week=6, prior_rows=prior_rows)
    assert fc["state"] == "forecast"
    assert fc["band"] == forecasts.BAND_TRACKING_HIT
    assert fc["ppg_delta"] == pytest.approx(8.0)     # 18.0 vs prior 10.0


def test_weekly_forecasts_for_calls_loads_prior_series_when_needed(monkeypatch):
    loaded = []

    def _series(season, through):
        loaded.append(season)
        if season == 2025:
            return {"101": [wk(1, ppr=9.0), wk(2, ppr=11.0)]}
        return {"101": [HELD_ROW, HELD_ROW_6]}

    monkeypatch.setattr(wg, "default_through_week", lambda season: 6)
    monkeypatch.setattr(wg, "load_season_series", _series)
    out = forecasts.weekly_forecasts_for_calls(2026, [_prior_season_call()])
    assert loaded == [2026, 2025]
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_weekly_forecasts_for_calls_skips_prior_load_when_unneeded(monkeypatch):
    loaded = []

    def _series(season, through):
        loaded.append(season)
        return {"101": [HELD_ROW, HELD_ROW_6]}

    monkeypatch.setattr(wg, "default_through_week", lambda season: 6)
    monkeypatch.setattr(wg, "load_season_series", _series)
    out = forecasts.weekly_forecasts_for_calls(2026, [make_call()])
    assert loaded == [2026]                  # no prior-season load paid for
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_weekly_forecast_never_for_a_mature_call():
    assert forecasts.weekly_forecast(make_call(), [HELD_ROW], through_week=7) is None


def test_weekly_forecasts_for_calls_skips_mature(monkeypatch):
    monkeypatch.setattr(wg, "default_through_week", lambda season: 6)
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through: {"101": [HELD_ROW, HELD_ROW_6]})
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
    # Overall excludes watchlist/monitored (never surfaced): 12, not 16
    assert record["overall"]["graded"] == 12


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
    monkeypatch.setattr(forecasts, "load_serving_reconstructed_weeks",
                        lambda season: {1, 2})
    record = forecasts.weekly_track_record(2026)

    # One pooled record: no separate backtest entry anywhere.
    assert "backtest" not in record
    # Live (12 graded) and reconstructed (15 graded) calls sum into one
    # overall, and the 10-graded floor applies to the pooled groups.
    assert record["overall"]["graded"] == 24  # excludes watchlist/monitored
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
    monkeypatch.setattr(forecasts, "load_serving_reconstructed_weeks",
                        lambda season: set())
    record = forecasts.weekly_track_record(2026)

    assert "backtest" not in record
    assert record["overall"]["graded"] == 12  # excludes watchlist/monitored
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


def _graded_week_payload():
    # Four surfaced week-2 rows, one per stamp outcome plus a player
    # whose call was voided by injury (never stamped).
    def _row(pid, name):
        return {
            "player_id": pid, "player_name": name, "team": "KC",
            "position": "WR", "breakout_score": 55.0, "confidence": 80.0,
            "classification": "emerging_breakout", "as_of_week": 2,
            "evidence": {
                "score_basis": "role_change", "main_board_eligible": True,
                "provisional": False,
                "fantasy": {"baseline_ppg": 8.0, "recent_ppg": 14.0},
                "signals": {},
            },
        }
    return {
        "season": 2026, "as_of_week": 2, "as_of_date": "2026-09-14",
        "count": 4,
        "candidates": [_row("101", "Hit Player"), _row("102", "Partial Player"),
                       _row("103", "Miss Player"), _row("104", "Hurt Player")],
        "scoring_version": "weekly-v6", "data_available": True,
        "data_status": "available",
    }


def _grade_rows_for_week_two():
    return [
        {"player_id": "101", "as_of_week": 2, "grade": "hit",
         "ppg_delta": Decimal("9.17"), "snap_delta": Decimal("51.7"),
         "opp_delta": None, "outcome_games": 3,
         "detail": {"reason": "role_held_and_production_rose",
                    "role_state": "held"}},
        {"player_id": "102", "as_of_week": 2, "grade": "partial",
         "ppg_delta": 3.1, "snap_delta": -4.0, "opp_delta": 2.6,
         "outcome_games": 3,
         "detail": {"reason": "production_rose_despite_role_reverted",
                    "role_state": "reverted"}},
        {"player_id": "103", "as_of_week": 2, "grade": "miss",
         "ppg_delta": -2.53, "snap_delta": -3.9, "opp_delta": -2.0,
         "outcome_games": 3,
         "detail": {"reason": "role_reverted_to_baseline",
                    "role_state": "reverted"}},
        {"player_id": "104", "as_of_week": 2, "grade": "injured",
         "ppg_delta": None, "snap_delta": None, "opp_delta": None,
         "outcome_games": 1, "detail": {"reason": "out_for_window"}},
        # A week-3 verdict for the hit player must never leak onto the
        # week-2 board.
        {"player_id": "101", "as_of_week": 3, "grade": "miss",
         "ppg_delta": -5.0, "snap_delta": -9.0, "opp_delta": -3.0,
         "outcome_games": 3, "detail": {"reason": "production_collapsed"}},
    ]


def test_weekly_board_attaches_grades_and_receipt_detail(monkeypatch):
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store

    monkeypatch.setattr(weekly_store, "load_weekly_candidates",
                        lambda season, **kw: _graded_week_payload())
    monkeypatch.setattr(forecasts, "weekly_forecasts_for_calls",
                        lambda season, calls: {})
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _grade_rows_for_week_two())

    payload = api.get_weekly_breakout_candidates(
        2026, min_score=0, limit=None, as_of_week=2)
    by_id = {c["player_id"]: c for c in payload["candidates"]}
    assert set(by_id) == {"101", "102", "103", "104"}

    # Hit: grade plus the full receipt evidence, Decimals as floats.
    hit = by_id["101"]
    assert hit["grade"] == "hit"
    detail = hit["grade_detail"]
    assert detail["ppg_delta"] == 9.17
    assert isinstance(detail["ppg_delta"], float)
    assert detail["snap_delta"] == 51.7
    assert detail["opp_delta"] is None
    assert detail["outcome_games"] == 3
    assert detail["reason"] == "role_held_and_production_rose"

    # Partial is stampable now too, with its own evidence.
    partial = by_id["102"]
    assert partial["grade"] == "partial"
    assert partial["grade_detail"]["opp_delta"] == 2.6
    assert partial["grade_detail"]["reason"] == (
        "production_rose_despite_role_reverted")

    assert by_id["103"]["grade"] == "miss"
    assert by_id["103"]["grade_detail"]["ppg_delta"] == -2.53

    # Injured calls are never stamped.
    assert "grade" not in by_id["104"]
    assert "grade_detail" not in by_id["104"]

    # The whole payload stays JSON-serializable (no Decimal leaks).
    json.dumps(payload)


def test_weekly_board_no_grade_attach_on_current_board(monkeypatch):
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import weekly_store

    monkeypatch.setattr(weekly_store, "load_weekly_candidates",
                        lambda season, **kw: _graded_week_payload())
    monkeypatch.setattr(forecasts, "weekly_forecasts_for_calls",
                        lambda season, calls: {})
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: _grade_rows_for_week_two())

    payload = api.get_weekly_breakout_candidates(2026, min_score=0, limit=None)
    for candidate in payload["candidates"]:
        assert "grade" not in candidate
        assert "grade_detail" not in candidate


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
    # Through week 4 with two banked window games per reconstructed
    # player: the week 2 call for 101 needs 2 games to sit at hit pace
    # under the grading-v4 hit floor (1 game would cap at borderline).
    # 102's call sits in week 2 here, not week 1 as in the helper
    # default: at through=4 a week 1 call is mature and forecasts
    # nothing, so the pooled miss has to come from a week 2 call.
    api = _patch_reconstructed_outlook(
        monkeypatch, through=4,
        week_rows={
            1: [_recon_row("101", 1, 55.0)],
            2: [_recon_row("101", 2, 60.0), _recon_row("102", 2, 50.0),
                _recon_row("103", 2, 45.0)],
        },
        series={
            "101": [wk(2, **_HELD), wk(3, **_HELD), wk(4, **_HELD)],
            "102": [wk(3, **_REVERTED), wk(4, **_REVERTED)],
            "103": [],
        })
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
    monkeypatch.setattr(forecasts, "load_serving_reconstructed_weeks",
                        lambda season: {1, 2})
    record = forecasts.weekly_track_record(2026)

    assert record["overall"]["graded"] == 16  # excludes watchlist/monitored
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


# ---------------------------------------------------------------------------
# one surfaced definition: hits/misses floor, grade-row repeat filter,
# the surfaced-by-week helper, and Still Tracking for open live weeks
# ---------------------------------------------------------------------------

def test_top_grade_rows_excludes_non_surfaced_scores():
    rows = [
        # Roman Wilson Week 1: a real graded hit, but his surfaced score
        # was 23.2 and the Week 1 board never showed him.
        {**_grade_row("roman", "early_watch", "hit", 10.1),
         "breakout_score": 23.2},
        # Keenan Allen Week 1: a 0.0 temporary-opportunity miss, likewise
        # never surfaced.
        {**_grade_row("keenan", "temporary_opportunity", "miss", -0.8),
         "breakout_score": 0.0},
        {**_grade_row("golden", "early_watch", "hit", 9.2),
         "breakout_score": 33.5},
        {**_grade_row("realmiss", "emerging_breakout", "miss", -3.0),
         "breakout_score": 41.0},
    ]
    hits = forecasts.top_grade_rows(rows, "hit")
    assert [h["player_id"] for h in hits] == ["golden"]
    misses = forecasts.top_grade_rows(rows, "miss")
    assert [m["player_id"] for m in misses] == ["realmiss"]


class _GradeRowsConn:
    """Fake conn for load_weekly_grade_rows, shaped like the real one:
    execute(query, params) -> result with fetchall() of dict rows."""

    def __init__(self, weeks, score_pids_by_run, grade_rows_by_week):
        self._weeks = weeks
        self._score_pids = score_pids_by_run
        self._grade_rows = grade_rows_by_week

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        if "DISTINCT as_of_week" in query:
            return _RowsResult([{"as_of_week": w} for w in self._weeks])
        if "FROM weekly_breakout_scores" in query:
            run_id = params[0]
            return _RowsResult(
                [{"player_id": pid}
                 for pid in self._score_pids.get(run_id, [])])
        if "SELECT *" in query and wg.GRADES_TABLE in query:
            week = params[1]
            return _RowsResult(self._grade_rows.get(week, []))
        raise AssertionError(f"unexpected query: {query}")


def _patch_grade_row_load(monkeypatch, surfaced):
    """Wire load_weekly_grade_rows to fakes: Week 1 surfaced Golden and
    Coker; Week 2's serving run re-scored Golden plus a new player."""
    from data_building.breakout_engine import weekly_store

    runs = {1: {"id": 11, "scoring_version": "weekly-v6"},
            2: {"id": 22, "scoring_version": "weekly-v6"}}
    grade_rows = {
        1: [{"player_id": "golden", "as_of_week": 1, "grade": "hit",
             "scoring_version": "weekly-v6"}],
        2: [{"player_id": "golden", "as_of_week": 2, "grade": "hit",
             "scoring_version": "weekly-v6"},
            {"player_id": "newguy", "as_of_week": 2, "grade": "miss",
             "scoring_version": "weekly-v6"},
            {"player_id": "ghost", "as_of_week": 2, "grade": "hit",
             "scoring_version": "weekly-v6"}],
    }
    conn = _GradeRowsConn(
        [1, 2], {11: ["golden", "coker"], 22: ["golden", "newguy"]},
        grade_rows)
    monkeypatch.setattr(forecasts, "get_conn", lambda: conn)
    monkeypatch.setattr(wg, "init_weekly_breakout_grades_db", lambda: None)
    monkeypatch.setattr(weekly_store, "get_serving_run",
                        lambda season, week: runs.get(week))
    if isinstance(surfaced, Exception):
        def _boom(season):
            raise surfaced
        monkeypatch.setattr(wg, "surfaced_player_ids_by_week", _boom)
    else:
        monkeypatch.setattr(wg, "surfaced_player_ids_by_week",
                            lambda season: surfaced)


def test_load_weekly_grade_rows_drops_repeats_and_non_surfaced(monkeypatch):
    _patch_grade_row_load(monkeypatch,
                          {1: {"golden", "coker"}, 2: {"newguy"}})
    rows = forecasts.load_weekly_grade_rows(2026)
    # Week 1 Golden stays. Week 2 Golden is a repeat (surfaced Week 1,
    # not Week 2) and drops; the orphan "ghost" grade drops via the
    # serving-run player filter as before.
    assert [(r["player_id"], r["as_of_week"]) for r in rows] == [
        ("golden", 1), ("newguy", 2)]


def test_load_weekly_grade_rows_helper_failure_keeps_old_behavior(monkeypatch):
    _patch_grade_row_load(monkeypatch, RuntimeError("store unavailable"))
    rows = forecasts.load_weekly_grade_rows(2026)
    assert [(r["player_id"], r["as_of_week"]) for r in rows] == [
        ("golden", 1), ("golden", 2), ("newguy", 2)]


def _score_row(pid, classification, score, evidence=None):
    return {"player_id": pid, "classification": classification,
            "breakout_score": score, "evidence": evidence or {}}


def _patch_surfaced_store(monkeypatch):
    """Five completed weeks. Serving runs carry the ids 101..105; run
    999 is a completed NON-serving Week 2 run whose rows must never
    count. Returns the list of run ids the helper actually loaded."""
    from data_building.breakout_engine import weekly_store

    serving = {w: {"id": 100 + w, "scoring_version": "weekly-v6"}
               for w in range(1, 6)}
    rows_by_run = {
        101: [
            _score_row("alpha", "early_watch", Decimal("40.0")),
            _score_row("watchy", "watchlist", 50.0),
            _score_row("tempo", "temporary_opportunity", 45.0,
                       {"main_board_eligible": True}),
            _score_row("initlow", "emerging_breakout", 44.0,
                       {"main_board_eligible": True,
                        "score_basis": "initial_role",
                        "subscores": {"current_role_score": 20.0}}),
            _score_row("inithi", "emerging_breakout", 46.0,
                       json.dumps({"main_board_eligible": True,
                                   "score_basis": "initial_role",
                                   "subscores": {"current_role_score": 40.0}})),
        ],
        102: [
            _score_row("alpha", "early_watch", 45.0),
            _score_row("bravo", "early_watch", 38.0),
            _score_row("watchy", "early_watch", 41.0),
        ],
        103: [
            _score_row("xray", "early_watch", 42.0),
            _score_row("alpha", "early_watch", 44.0),
        ],
        104: [_score_row("alpha", "early_watch", 44.0)],
        105: [_score_row("alpha", "early_watch", 44.0)],
        999: [_score_row("xray", "early_watch", 60.0)],
    }
    loaded_run_ids = []

    def _load(run_id):
        loaded_run_ids.append(run_id)
        return list(rows_by_run.get(run_id, []))

    monkeypatch.setattr(weekly_store, "list_completed_weeks",
                        lambda season: [{"as_of_week": w}
                                        for w in range(1, 6)])
    monkeypatch.setattr(weekly_store, "get_serving_run",
                        lambda season, week: serving.get(week))
    monkeypatch.setattr(weekly_store, "load_run_score_rows", _load)
    return loaded_run_ids


def test_surfaced_player_ids_by_week_canonical(monkeypatch):
    loaded = _patch_surfaced_store(monkeypatch)
    by_week = wg.surfaced_player_ids_by_week(2026)

    # Week 1: the watchlist row and the sub-gate initial-role row are
    # not pool-eligible; the eligible temporary opportunity and the
    # gated initial-role row (evidence as a JSON string) are.
    assert by_week[1] == {"alpha", "tempo", "inithi"}
    # Week 2: alpha is a repeat of Week 1 and is not a new call; watchy
    # surfaces now because its Week 1 watchlist row never counted.
    assert by_week[2] == {"bravo", "watchy"}
    # Week 3: xray surfaces; its 60.0 in the non-serving Week 2 run was
    # never loaded and never counted.
    assert by_week[3] == {"xray"}
    assert 999 not in loaded
    # Alpha stays blocked in Weeks 3 and 4 (the 3 weeks after Week 1)
    # and is a fresh call again in Week 5.
    assert by_week[4] == set()
    assert by_week[5] == {"alpha"}


def test_recently_surfaced_ids_use_canonical_surfaced_sets(monkeypatch):
    _patch_surfaced_store(monkeypatch)
    import dashboard_services.breakout_api as api

    assert api._recently_surfaced_player_ids(2026, 2) == {
        "alpha", "tempo", "inithi"}
    # For the Week 5 board, only Weeks 2-4 block: alpha's Week 1 call
    # has finished its window and does not block.
    assert api._recently_surfaced_player_ids(2026, 5) == {
        "bravo", "watchy", "xray"}


def test_track_record_still_tracking_includes_open_historical_weeks(monkeypatch):
    api = _patch_track_record(monkeypatch)
    from data_building.breakout_engine import weekly_store

    monkeypatch.setattr(weekly_store, "list_completed_weeks",
                        lambda season: [{"as_of_week": 3},
                                        {"as_of_week": 4},
                                        {"as_of_week": 5}])

    def _cand(pid, week, band):
        return {"player_id": pid, "player_name": pid,
                "classification": "emerging_breakout",
                "classification_label": "Emerging Breakout",
                "breakout_score": 50.0,
                "forecast": {"kind": "weekly", "band": band,
                             "band_label": "Tracking to hit",
                             "basis": "stubbed", "call_week": week}}

    def _board(season, min_score, limit, week=None):
        if week == "preseason":
            return {"view": "preseason", "candidates": []}
        if week == 3:
            return {"as_of_week": 3, "candidates": [_cand("w3", 3, "tracking_to_hit")]}
        if week == 4:
            return {"as_of_week": 4, "candidates": [_cand("w4", 4, "borderline")]}
        return {"as_of_week": 5, "candidates": [_cand("w5", 5, "tracking_to_hit")]}

    monkeypatch.setattr(api, "get_breakout_board_candidates", _board)

    payload = api.get_breakout_track_record(2026)

    tracking = {t["week"]: t for t in payload["still_tracking"]}
    assert set(tracking) == {3, 4, 5}
    assert tracking[3]["calls"] == 1
    assert tracking[3]["tracking_to_hit_rate"] == pytest.approx(1.0)
    assert tracking[4]["calls"] == 1
    assert tracking[5]["calls"] == 1


def test_breakout_rail_hero_reads_singular_hit_field():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # The API bucket field is "hit" (weekly_grading._rate_bucket); the
    # hero text once read "hits" and always rendered 0.
    assert "overall.hits" not in src
    assert src.count("var hits = overall.hit || 0;") == 2


# ---------------------------------------------------------------------------
# track record row clicks: jump the board (and the week selector) to the
# row's week, through the selector's own change event
# ---------------------------------------------------------------------------

def _extract_bo_fn(src, signature):
    """A breakout page JS function from the app.py f-string, with the
    f-string's doubled braces collapsed back into real JS braces (same
    extraction as _extract_bo_outlook_block, keyed by signature)."""
    start = src.index(signature)
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
    out = src[start:j].replace("{{", "{").replace("}}", "}")
    # The page f-string escapes quotes inside inline handlers with extra
    # backslashes (\\' in the file, \\\' one nesting level deeper);
    # Python's escape pass collapses each pair when serving, so mirror
    # that (longest first) or node sees a stray backslash and the string
    # literal ends early.
    return out.replace("\\\\\\'", "\\'").replace("\\\\'", "\\'")


_BO_FAKE_DOM = """
var changeEvents = [];
var loadedWeeks = [];
var fakeSel = {
  value: '5',
  options: [{value: 'preseason'}, {value: '1'}, {value: '2'},
            {value: '3'}, {value: '4'}, {value: '5'}],
  _listeners: {},
  addEventListener: function (type, fn) { this._listeners[type] = fn; },
  dispatchEvent: function (ev) {
    changeEvents.push({type: ev.type, bubbles: ev.bubbles});
    if (this._listeners[ev.type]) this._listeners[ev.type]();
    return true;
  }
};
var fakeContainer = {scrolled: false,
                     scrollIntoView: function () { this.scrolled = true; }};
var document = {
  getElementById: function (id) {
    if (id === 'breakoutWeekSelect') return fakeSel;
    if (id === 'breakoutsContainer') return fakeContainer;
    return null;
  }
};
// Stands in for onBreakoutWeekChange: the page's own change listener,
// which loads the board for the selector's current value.
fakeSel.addEventListener('change', function () {
  loadedWeeks.push(fakeSel.value);
});
"""


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_bo_go_week_routes_through_selector_change_event():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = _extract_bo_fn(src, "function boGoWeek(week)")
    driver = _BO_FAKE_DOM + """
boGoWeek(3);
var afterNumeric = {value: fakeSel.value, loaded: loadedWeeks.slice(),
                    events: changeEvents.length,
                    bubbles: changeEvents[0] && changeEvents[0].bubbles,
                    scrolled: fakeContainer.scrolled};
loadedWeeks.length = 0; changeEvents.length = 0;
boGoWeek('preseason');
var afterPreseason = {value: fakeSel.value, loaded: loadedWeeks.slice()};
loadedWeeks.length = 0; changeEvents.length = 0;
boGoWeek(0);
var afterZero = {value: fakeSel.value, loaded: loadedWeeks.slice()};
loadedWeeks.length = 0; changeEvents.length = 0;
boGoWeek(99);
var afterUnknown = {value: fakeSel.value, loaded: loadedWeeks.slice(),
                    events: changeEvents.length};
console.log(JSON.stringify({afterNumeric: afterNumeric,
  afterPreseason: afterPreseason, afterZero: afterZero,
  afterUnknown: afterUnknown}));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    # A numeric week: selector value set, exactly one change dispatched,
    # and the page's change listener (the stand-in above) loads that week.
    assert out["afterNumeric"]["value"] == "3"
    assert out["afterNumeric"]["events"] == 1
    assert out["afterNumeric"]["bubbles"] is True
    assert out["afterNumeric"]["loaded"] == ["3"]
    assert out["afterNumeric"]["scrolled"] is True
    # 'preseason' and week 0 both land on the Preseason option.
    assert out["afterPreseason"] == {"value": "preseason",
                                     "loaded": ["preseason"]}
    assert out["afterZero"] == {"value": "preseason",
                                "loaded": ["preseason"]}
    # A week the selector does not offer changes nothing.
    assert out["afterUnknown"]["value"] == "preseason"
    assert out["afterUnknown"]["loaded"] == []
    assert out["afterUnknown"]["events"] == 0


def test_bo_go_week_never_loads_the_board_directly():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = _extract_bo_fn(src, "function boGoWeek(week)")
    # The old implementation set the value and called loadBreakouts
    # itself, which left the custom dropdown showing the old week. The
    # jump must go through the selector's change event instead.
    assert "loadBreakouts" not in fn_src
    assert "dispatchEvent(new Event('change'" in fn_src
    assert "breakoutWeekSelect" in fn_src


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_bo_row_week_attrs_and_keyboard_jump():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = "\n".join([
        _extract_bo_fn(src, "function boGoWeek(week)"),
        _extract_bo_fn(src, "function _boRowWeekAttrs(week)"),
        _extract_bo_fn(src, "function boWeekRowKey(event, el)"),
        _extract_bo_fn(src, "function _boWeekBar(week, calls, pct, color, label)"),
    ])
    driver = _BO_FAKE_DOM + """
var prevented = 0;
function keyEv(key) {
  return {key: key, preventDefault: function () { prevented += 1; }};
}
var out = {
  attrsNum: _boRowWeekAttrs(4),
  attrsPre: _boRowWeekAttrs('preseason'),
  attrsNull: _boRowWeekAttrs(null),
  attrsEmpty: _boRowWeekAttrs(''),
  bar: _boWeekBar(2, 5, 0.5, null, 'hits')
};
boWeekRowKey(keyEv('Enter'), {dataset: {week: '2'}});
out.afterEnter = fakeSel.value;
boWeekRowKey(keyEv(' '), {dataset: {week: '1'}});
out.afterSpace = fakeSel.value;
boWeekRowKey(keyEv('x'), {dataset: {week: '3'}});
out.afterOtherKey = fakeSel.value;
out.prevented = prevented;
out.loaded = loadedWeeks;
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    assert 'data-week="4"' in out["attrsNum"]
    assert 'role="button"' in out["attrsNum"]
    assert 'tabindex="0"' in out["attrsNum"]
    assert 'onclick="boGoWeek(this.dataset.week)"' in out["attrsNum"]
    assert 'data-week="preseason"' in out["attrsPre"]
    assert "the Preseason board" in out["attrsPre"]
    # Rows without a week stay inert.
    assert out["attrsNull"] == ""
    assert out["attrsEmpty"] == ""
    # The by-week bar renders as a week-jump row for its own week.
    assert out["bar"].startswith('<div class="bo-wk-link" data-week="2"')
    assert "Week 2" in out["bar"]
    # Enter and Space jump; other keys do not.
    assert out["afterEnter"] == "2"
    assert out["afterSpace"] == "1"
    assert out["afterOtherKey"] == "1"
    assert out["prevented"] == 2
    assert out["loaded"] == ["2", "1"]


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_weekly_track_rows_carry_their_week():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = "\n".join([
        "var _boPlayerNames = {};",
        "function boOpenPlayerModal(pid) {}",
        _extract_bo_fn(src, "function _boRowWeekAttrs(week)"),
        _extract_bo_fn(src, "function boWeekRowKey(event, el)"),
        _extract_bo_fn(src, "function _boPlayerLink(name, playerId)"),
        _extract_bo_fn(src, "function _boWeekBar(week, calls, pct, color, label)"),
        _extract_bo_fn(src, "function _boWeeklyTrackHTML(data)"),
    ])
    driver = """
var data = {
  weekly: {overall: {}},
  by_week: [{week: 2, graded: 3, hit_rate: 0.5}],
  hits: [{player_name: 'Hit Player', player_id: '11',
          call_week: 3, ppg_delta: 4.2}],
  misses: [{player_name: 'Miss Player', player_id: '12',
            call_week: null, ppg_delta: -3.1}],
  still_tracking: [{week: 4, calls: 2, tracking_to_hit_rate: 0.5}]
};
console.log(JSON.stringify({html: _boWeeklyTrackHTML(data)}));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    html = json.loads(res.stdout)["html"]

    # By-week bar, hit row and still-tracking bar all carry their week.
    assert 'data-week="2"' in html
    assert 'data-week="3"' in html
    assert 'data-week="4"' in html
    assert html.count('data-week="') == 3
    # Only the hit row (the one rail row with a week) gets the link
    # styling; the weekless miss row stays a plain, inert rail row.
    assert html.count("bo-row-link") == 1
    assert "Week ? call" in html


def test_track_record_week_click_wiring_in_page_source():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Hits and misses rows attach their call week at render time.
    assert src.count("_boRowWeekAttrs(r.call_week)") == 2
    # The by-week / still-tracking bars share the same attributes helper.
    assert "_boRowWeekAttrs(week)" in src
    # The preseason (season engine) rows carry no weekly week and must
    # not navigate: no week-jump wiring inside that renderer.
    start = src.index("function _boPreseasonTrackHTML(data)")
    end = src.index("function _boOutlookBlock(title, block)")
    assert "boGoWeek" not in src[start:end]
    assert "_boRowWeekAttrs" not in src[start:end]


def test_dashboard_css_styles_clickable_rail_rows():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    rule = css.split(".bo-rail-row.bo-row-link {")[1].split("}")[0]
    assert "cursor: pointer" in rule
    hover = css.split(".bo-rail-row.bo-row-link:hover {")[1].split("}")[0]
    assert "background:" in hover


# ---------------------------------------------------------------------------
# grade stamps: animated HIT / PARTIAL / MISS stamps, receipts, definition
# ---------------------------------------------------------------------------

def _bo_stamp_core_fns(src):
    return "\n".join([
        _extract_bo_fn(src, "function _boFmtDelta(v)"),
        _extract_bo_fn(src, "function _boGradeReceiptHtml(candidate, receiptId)"),
        _extract_bo_fn(src, "function _boGradeStampHtml(candidate, idx, receiptId)"),
    ])


def _rc_row(html, label):
    """The receipt row chunk carrying `label` (its pass/fail mark class
    sits in the same chunk, ahead of the label)."""
    for chunk in html.split("bo-rc-row"):
        if label in chunk:
            return chunk
    raise AssertionError(f"no receipt row for {label!r} in {html!r}")


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_grade_stamp_and_receipt_render_on_graded_cards():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = "\n".join([
        _bo_stamp_core_fns(src),
        _extract_bo_fn(src, "function _boWeeklySignal(candidate)"),
        _extract_bo_fn(src, "function _boWeeklySignalText(signal)"),
        _extract_bo_fn(src, "function renderBreakoutCard(candidate, idx)"),
    ])
    driver = """
// Headshot source is stubbed: the real helper's regex escapes do not
// survive source extraction, and the URL is irrelevant to the stamps.
function _boHeadshotSrc(candidate, pid) { return 'hs-' + pid; }
var currentWeek = '2';
function fakeCandidate(over) {
  return Object.assign({
    player_id: '101', player_name: 'Golden Boy', team: 'GB',
    position: 'WR', age: 23, mode: 'weekly', weekly: true,
    breakout_score: 34, classification_label: 'Early Watch',
    confidence: 62, score_basis: 'role_change',
    sample: {recent_games: 3}, as_of_week: 1, grade: 'hit',
    grade_detail: {ppg_delta: 9.2, snap_delta: 51.7, opp_delta: null,
                   outcome_games: 3,
                   reason: 'role_held_and_production_rose'}
  }, over || {});
}
var out = {
  hit: renderBreakoutCard(fakeCandidate(), 0),
  partial: renderBreakoutCard(fakeCandidate({
    player_id: '102', grade: 'partial',
    grade_detail: {ppg_delta: 3.1, snap_delta: -4.0, opp_delta: 2.6,
                   outcome_games: 3}}), 1),
  miss: renderBreakoutCard(fakeCandidate({
    player_id: '103', grade: 'miss',
    grade_detail: {ppg_delta: -2.5, snap_delta: -3.9, opp_delta: -2.0,
                   outcome_games: 3}}), 2),
  deep: renderBreakoutCard(fakeCandidate({player_id: '104'}), 40),
  stampSpan: _boGradeStampHtml({grade: 'hit'}, 3, null),
  stampDelay: _boGradeStampHtml({grade: 'miss'}, 40, 'bo-grc-x')
};
currentWeek = 'latest';
out.latest = renderBreakoutCard(fakeCandidate(), 0);
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    hit = out["hit"]
    assert "bo-stamp bo-stamp-hit" in hit
    assert ">HIT</button>" in hit
    assert 'title="Why this grade?"' in hit
    assert 'id="bo-grc-0-101"' in hit
    # The Hit card flashes its border once; the others do not.
    assert "bo-grade-flash" in hit
    assert "bo-grade-flash" not in out["partial"]
    assert "bo-grade-flash" not in out["miss"]
    # Receipt: actual vs bar per leg, missing touches row hidden.
    assert "Production" in hit and "+9.2 PPG" in hit and "bar +2.0" in hit
    assert "Role (touches)" not in hit
    snap_row = _rc_row(hit, "Role (snap share)")
    assert "bo-rc-pass" in snap_row and "+51.7 pts" in snap_row
    assert "bar +8.0" in snap_row
    assert "Graded over 3 games" in hit

    partial = out["partial"]
    assert "bo-stamp bo-stamp-partial" in partial
    assert ">PARTIAL</button>" in partial
    touch_row = _rc_row(partial, "Role (touches)")
    assert "bo-rc-pass" in touch_row and "+2.6 per game" in touch_row
    snap_row = _rc_row(partial, "Role (snap share)")
    assert "bo-rc-fail" in snap_row and "-4.0 pts" in snap_row

    miss = out["miss"]
    assert "bo-stamp bo-stamp-miss" in miss
    assert ">MISS</button>" in miss
    prod_row = _rc_row(miss, "Production")
    assert "bo-rc-fail" in prod_row and "-2.5 PPG" in prod_row

    # Stagger: 60ms per card, capped at 600ms however deep the list.
    assert "animation-delay:180ms" in out["stampSpan"]
    assert ">HIT</span>" in out["stampSpan"]
    assert "animation-delay:600ms" in out["deep"]
    assert "animation-delay:600ms" in out["stampDelay"]

    # The current board renders exactly as before: no stamp, no receipt.
    assert "bo-stamp" not in out["latest"]
    assert "bo-receipt" not in out["latest"]
    assert "bo-grade-flash" not in out["latest"]


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_grade_receipt_rows_mirror_the_grader():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = "\n".join([
        _extract_bo_fn(src, "function _boFmtDelta(v)"),
        _extract_bo_fn(src, "function _boGradeReceiptHtml(candidate, receiptId)"),
    ])
    driver = """
var out = {
  guardFail: _boGradeReceiptHtml({grade_detail: {
    ppg_delta: 7.5, snap_delta: 20.1, opp_delta: -2.0,
    outcome_games: 3}}, 'r1'),
  snapOnly: _boGradeReceiptHtml({grade_detail: {
    ppg_delta: null, snap_delta: 20.1, opp_delta: null,
    outcome_games: 2}}, 'r2'),
  empty: _boGradeReceiptHtml({grade_detail: {}}, 'r3'),
  noDetail: _boGradeReceiptHtml({}, 'r4'),
  gamesOnly: _boGradeReceiptHtml({grade_detail: {
    outcome_games: 1}}, 'r5')
};
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    # Snap guard: +20.1 pts of snap share still fails the leg when
    # touches fell, exactly as the grader rules it.
    snap_row = _rc_row(out["guardFail"], "Role (snap share)")
    assert "bo-rc-fail" in snap_row and "+20.1 pts" in snap_row
    touch_row = _rc_row(out["guardFail"], "Role (touches)")
    assert "bo-rc-fail" in touch_row and "-2.0 per game" in touch_row

    # Touches unmeasured: the snap leg is judged on snap share alone.
    snap_row = _rc_row(out["snapOnly"], "Role (snap share)")
    assert "bo-rc-pass" in snap_row
    assert "Production" not in out["snapOnly"]
    assert "Role (touches)" not in out["snapOnly"]
    assert "Graded over 2 games" in out["snapOnly"]

    # No evidence at all: no receipt, so the stamp renders plain.
    assert out["empty"] == ""
    assert out["noDetail"] == ""
    # Games alone still earn the footer line.
    assert "Graded over 1 game" in out["gamesOnly"]
    assert "bo-rc-row" not in out["gamesOnly"]


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_grade_receipt_toggle_opens_and_closes():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = _extract_bo_fn(src, "function boToggleGradeReceipt(btn, id)")
    driver = """
var receiptEl = {hidden: true};
var btn = {attrs: {}, setAttribute: function (k, v) { this.attrs[k] = v; }};
var document = {
  getElementById: function (id) {
    return id === 'bo-grc-0-101' ? receiptEl : null;
  }
};
boToggleGradeReceipt(btn, 'bo-grc-0-101');
var opened = {hidden: receiptEl.hidden,
              expanded: btn.attrs['aria-expanded']};
boToggleGradeReceipt(btn, 'bo-grc-0-101');
var closed = {hidden: receiptEl.hidden,
              expanded: btn.attrs['aria-expanded']};
boToggleGradeReceipt(btn, 'bo-grc-missing');
console.log(JSON.stringify({opened: opened, closed: closed}));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["opened"] == {"hidden": False, "expanded": "true"}
    assert out["closed"] == {"hidden": True, "expanded": "false"}


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node not available")
def test_definition_bar_only_on_graded_week_views():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn_src = "\n".join([
        _extract_bo_fn(src, "function _boHitDefinitionBarHtml()"),
        _extract_bo_fn(src, "function _boBoardTopHtml(list)"),
    ])
    driver = """
var currentWeek = '2';
var out = {
  graded: _boBoardTopHtml([{grade: 'partial'}, {}]),
  ungraded: _boBoardTopHtml([{grade: 'injured'}, {}]),
  empty: _boBoardTopHtml([])
};
currentWeek = 'latest';
out.latest = _boBoardTopHtml([{grade: 'hit'}]);
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)

    bar = out["graded"]
    assert "What counts as a Hit:" in bar
    assert "kept the bigger role (snap share +8 pts or touches +2.0 per game)" in bar
    assert "gained at least +2.0 PPG" in bar
    assert "One leg only is a Partial" in bar
    assert "production collapsing by 2.0 PPG or more, is a Miss" in bar
    # Static legend stamps for all three grades.
    assert "bo-stamp bo-stamp-hit bo-stamp-static" in bar
    assert "bo-stamp bo-stamp-partial bo-stamp-static" in bar
    assert "bo-stamp bo-stamp-miss bo-stamp-static" in bar
    # UI copy carries no em dashes.
    assert "—" not in bar

    assert out["ungraded"] == ""
    assert out["empty"] == ""
    assert out["latest"] == ""


def test_grade_stamp_wiring_and_styles():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # The card stamps through the helper; both old flat pills are gone.
    assert "_boGradeStampHtml(candidate, idx || 0" in src
    assert "background:#16a34a;padding:3px 10px" not in src
    assert "background:#dc2626;padding:3px 10px" not in src
    # Both board render paths prepend the definition bar and pass the
    # card index through for the stagger.
    assert src.count("_boBoardTopHtml(filtered)") == 2
    assert "renderBreakoutCard(candidate, i)" in src
    assert "renderBreakoutCard(candidate, start + i)" in src

    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    stamp = css.split(".bo-stamp {")[1].split("}")[0]
    assert "border: 2px solid currentColor" in stamp
    assert "rotate(-8deg)" in stamp
    assert "background: transparent" in stamp
    assert "#16a34a" in css.split(".bo-stamp-hit {")[1].split("}")[0]
    assert "#d97706" in css.split(".bo-stamp-partial {")[1].split("}")[0]
    assert "#b91c1c" in css.split(".bo-stamp-miss {")[1].split("}")[0]
    # Slam keyframes overshoot through the brief's scale stops, the Hit
    # gets its one ring pulse, and Hit cards flash once.
    slam = css.split("@keyframes boStampSlam {")[1].split("@keyframes")[0]
    assert "scale(2.6)" in slam and "scale(0.92)" in slam
    assert "scale(1.07)" in slam
    assert "@keyframes boStampRing" in css
    assert ".bo-grade-flash" in css and "@keyframes boCardFlash" in css
    # Reduced motion degrades the stamps to a plain fade.
    rm_segments = css.split("@media (prefers-reduced-motion: reduce)")
    assert any("bo-stamp-hit" in seg[:600] and "boStampFadeIn" in seg[:600]
               for seg in rm_segments[1:])
