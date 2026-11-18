"""Track record maturity trim: mature weeks' boards are never loaded.

_compute_breakout_track_record merges the current board, the preseason
board, every completed week's board, and reconstructed candidates, then
feeds the merge to _forecast_outlook and _still_tracking_by_week. Both
consumers count only candidates carrying a forecast, and forecasts exist
only for unmatured calls (weekly_grading.is_mature, OUTCOME_WEEKS = 3):
a mature week's board arrives forecast-less and contributes zero.

These tests pin the trim:
- with a known through week, boards are requested only for unmatured
  weeks (plus the current and preseason boards);
- the payload built with mature-week boards present-but-forecastless
  (the old load-everything behavior, via the None-through fallback) is
  deep-equal to the payload built with those boards never loaded;
- when the through week is unknown (None) or its lookup raises, the
  fallback loads every week, exactly the old behavior.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

from data_building.breakout_engine import forecasts, weekly_grading, weekly_store

SEASON = 2026
THROUGH = 9
COMPLETED_WEEKS = list(range(1, 10))  # 1..9, current board is week 9
# is_mature(w, 9) with OUTCOME_WEEKS=3: weeks 1..6 mature, 7..9 open.
MATURE_WEEKS = [w for w in COMPLETED_WEEKS if weekly_grading.is_mature(w, THROUGH)]
OPEN_WEEKS = [w for w in COMPLETED_WEEKS if not weekly_grading.is_mature(w, THROUGH)]


def _weekly_candidate(pid, week, band):
    return {
        "player_id": pid, "player_name": f"Player {pid}",
        "classification": "emerging_breakout",
        "classification_label": "Emerging Breakout",
        "breakout_score": 50.0,
        "forecast": {"kind": "weekly", "band": band,
                     "band_label": band, "basis": "partial window",
                     "call_week": week},
    }


def _board_for(week):
    """Board stub payloads. Mature weeks' candidates carry NO forecast,
    exactly as production serves them (a mature call's forecast is None
    and the board pipeline drops the key's value to nothing usable)."""
    if week is None:  # current board (week 9)
        return {"as_of_week": 9, "candidates": [
            _weekly_candidate("cur9", 9, "tracking_to_hit")]}
    if week == "preseason":
        return {"candidates": [{
            "player_id": "pre1", "player_name": "Preseason One",
            "phase": "preseason", "breakout_opportunity_score": 70.0,
            "forecast": {"kind": "preseason", "band": "tracking_to_miss",
                         "band_label": "Tracking to miss", "basis": "pace"},
        }]}
    if weekly_grading.is_mature(int(week), THROUGH):
        return {"as_of_week": int(week), "candidates": [{
            "player_id": f"old{week}", "player_name": f"Mature {week}",
            "classification": "emerging_breakout",
            "classification_label": "Emerging Breakout",
            "breakout_score": 50.0,
        }]}
    return {"as_of_week": int(week), "candidates": [
        _weekly_candidate(f"w{week}", int(week), "borderline")]}


def _patch(monkeypatch, through):
    """Stub every input of _compute_breakout_track_record; return the
    list recording which board weeks were requested, in order."""
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season=None: SEASON)
    monkeypatch.setattr(forecasts, "weekly_track_record", lambda season: {
        "scoring_version": "weekly-v6", "min_sample": 10,
        "overall": {"calls": 3, "graded": 3, "hit": 2, "partial": 1,
                    "miss": 0, "ungraded": 0, "hit_rate": 0.6667},
        "groups": [], "score_bands": [], "confidence_bands": [],
        "hits": [], "misses": [], "by_week": [],
    })
    monkeypatch.setattr(forecasts, "season_track_record", lambda season: {
        "available": False, "overall": None, "groups": []})
    monkeypatch.setattr(api, "_reconstructed_outlook_candidates",
                        lambda season: [])
    monkeypatch.setattr(
        weekly_store, "list_completed_weeks",
        lambda season: [{"as_of_week": w} for w in COMPLETED_WEEKS])
    if isinstance(through, Exception):
        def _boom(season):
            raise through
        monkeypatch.setattr(weekly_grading, "default_through_week", _boom)
    else:
        monkeypatch.setattr(weekly_grading, "default_through_week",
                            lambda season: through)
    # No mature-ungraded gap rows in this fixture world.
    monkeypatch.setattr(weekly_grading, "surfaced_player_ids_by_week",
                        lambda season: {})

    requested = []

    def _board(season, min_score, limit, week=None):
        requested.append(week)
        return _board_for(week)

    monkeypatch.setattr(api, "get_breakout_board_candidates", _board)
    return api, requested


def _historical(requested):
    return [w for w in requested if isinstance(w, int)]


def test_boards_requested_only_for_unmatured_weeks(monkeypatch):
    api, requested = _patch(monkeypatch, THROUGH)
    api._compute_breakout_track_record(SEASON)

    # Current board and preseason board are always loaded.
    assert None in requested
    assert "preseason" in requested
    # Historical loads: exactly the open weeks other than the current
    # board's own week (9, skipped as before) - never a mature week.
    assert _historical(requested) == [w for w in OPEN_WEEKS if w != 9]
    assert not (set(_historical(requested)) & set(MATURE_WEEKS))


def test_payload_identical_with_and_without_mature_boards(monkeypatch):
    api, requested_trim = _patch(monkeypatch, THROUGH)
    trimmed = api._compute_breakout_track_record(SEASON)

    api, requested_full = _patch(monkeypatch, None)
    full = api._compute_breakout_track_record(SEASON)

    # The fallback run really did load the mature boards (forecast-less),
    # and the trimmed run did not - yet the payloads are deep-equal.
    assert set(MATURE_WEEKS) <= set(_historical(requested_full))
    assert not (set(MATURE_WEEKS) & set(_historical(requested_trim)))
    assert trimmed == full
    # And the shared content is the real pooled outlook, not empty noise.
    assert trimmed["outlook"]["weekly"]["open_calls"] > 0
    assert trimmed["still_tracking"]


def test_none_through_fallback_loads_all_weeks(monkeypatch):
    api, requested = _patch(monkeypatch, None)
    api._compute_breakout_track_record(SEASON)
    assert _historical(requested) == [w for w in COMPLETED_WEEKS if w != 9]


def test_raising_through_lookup_falls_back_to_all_weeks(monkeypatch):
    api, requested = _patch(monkeypatch, RuntimeError("metrics down"))
    api._compute_breakout_track_record(SEASON)
    assert _historical(requested) == [w for w in COMPLETED_WEEKS if w != 9]
