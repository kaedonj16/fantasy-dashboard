"""Read cache for the breakout board and track record payloads.

The board pipeline and the track record aggregate take seconds to build,
but the data behind them only moves when a weekly run completes, grades
are written, a new metrics week lands, or the opportunity scores are
recalculated. These tests pin the cache contract:

- a second identical call under an unchanged data watermark serves the
  cached payload without recomputing;
- a watermark change invalidates and recomputes;
- distinct season / week / min_score / limit keys never collide, while
  equivalent week spellings ("5" vs 5) share one entry;
- when the watermark cannot be read, the cache is bypassed entirely and
  every call computes fresh (the pre-cache behavior);
- the TTL backstop expires an entry even under an unchanged watermark.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_caches():
    import dashboard_services.breakout_api as api

    api._clear_breakout_payload_caches()
    yield
    api._clear_breakout_payload_caches()


class _Watermark:
    """Controllable stand-in for the data watermark query."""

    def __init__(self, value=("run-1", "grades-1", 5, "opp-1")):
        self.value = value

    def __call__(self, season):
        return self.value


def _patch_board(monkeypatch, watermark):
    """Resolve every season to itself and count pipeline computations."""
    import dashboard_services.breakout_api as api

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: season)
    monkeypatch.setattr(api, "_breakout_data_watermark", watermark)
    calls = []

    def candidates(season, min_score=0.0, limit=None, week=None):
        calls.append((season, min_score, limit, week))
        return {
            "season": season,
            "candidates": [{"player_id": "1", "breakout_score": 55.0}],
            "count": 1,
            "data_available": True,
        }

    monkeypatch.setattr(api, "get_breakout_candidates", candidates)
    return api, calls


def test_board_second_call_serves_cache_without_recompute(monkeypatch):
    api, calls = _patch_board(monkeypatch, _Watermark())

    first = api.get_breakout_board_candidates(2026)
    second = api.get_breakout_board_candidates(2026)

    assert len(calls) == 1
    assert first == second
    # Served payloads are copies: mutating one cannot poison the cache.
    assert first is not second
    first["candidates"][0]["breakout_score"] = 0.0
    third = api.get_breakout_board_candidates(2026)
    assert third["candidates"][0]["breakout_score"] == 55.0
    assert len(calls) == 1


def test_board_watermark_change_invalidates_and_recomputes(monkeypatch):
    watermark = _Watermark()
    api, calls = _patch_board(monkeypatch, watermark)

    api.get_breakout_board_candidates(2026)
    api.get_breakout_board_candidates(2026)
    assert len(calls) == 1

    watermark.value = ("run-2", "grades-1", 5, "opp-1")
    api.get_breakout_board_candidates(2026)
    assert len(calls) == 2

    # The new watermark is now the cached frontier.
    api.get_breakout_board_candidates(2026)
    assert len(calls) == 2


def test_board_cache_keys_do_not_collide(monkeypatch):
    api, calls = _patch_board(monkeypatch, _Watermark())

    api.get_breakout_board_candidates(2026)                    # latest week
    api.get_breakout_board_candidates(2026, week=4)
    api.get_breakout_board_candidates(2026, week=5)
    api.get_breakout_board_candidates(2026, week="preseason")
    api.get_breakout_board_candidates(2025, week=5)
    api.get_breakout_board_candidates(2026, min_score=40.0)
    api.get_breakout_board_candidates(2026, limit=None)
    assert len(calls) == 7

    # Repeats all serve from cache, and the string spelling of a week
    # shares the int week's entry.
    api.get_breakout_board_candidates(2026)
    api.get_breakout_board_candidates(2026, week="5")
    api.get_breakout_board_candidates(2026, week=4)
    api.get_breakout_board_candidates(2026, week="preseason")
    api.get_breakout_board_candidates(2025, week=5)
    assert len(calls) == 7


def test_board_watermark_failure_bypasses_cache(monkeypatch):
    api, calls = _patch_board(monkeypatch, lambda season: None)

    api.get_breakout_board_candidates(2026)
    api.get_breakout_board_candidates(2026)

    assert len(calls) == 2
    assert api._breakout_board_cache == {}


def test_board_ttl_backstop_expires_entry(monkeypatch):
    api, calls = _patch_board(monkeypatch, _Watermark())

    api.get_breakout_board_candidates(2026)
    assert len(calls) == 1

    # Force the stored entry past its TTL: same watermark, still recomputes.
    (entry,) = api._breakout_board_cache.values()
    entry["expires"] = 0.0
    api.get_breakout_board_candidates(2026)
    assert len(calls) == 2


def _patch_track_record(monkeypatch, watermark):
    """Stub the track record's heavy inputs and count the grade read."""
    import dashboard_services.breakout_api as api
    from data_building.breakout_engine import forecasts

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: season)
    monkeypatch.setattr(api, "_breakout_data_watermark", watermark)

    counts = {"weekly_tr": 0, "board": 0}

    def weekly_tr(season):
        counts["weekly_tr"] += 1
        return {
            "scoring_version": "weekly-v6",
            "min_sample": 10,
            "overall": {"calls": 3, "graded": 3, "hit": 2},
            "groups": [],
            "score_bands": [],
            "confidence_bands": [],
            "hits": [],
            "misses": [],
            "by_week": [],
        }

    monkeypatch.setattr(forecasts, "weekly_track_record", weekly_tr)
    monkeypatch.setattr(forecasts, "season_track_record", lambda season: {
        "available": False, "overall": None, "groups": []})
    monkeypatch.setattr(api, "_reconstructed_outlook_candidates",
                        lambda season: [])

    def board(season, min_score, limit, week=None):
        counts["board"] += 1
        return {"season": season, "candidates": [], "as_of_week": 5}

    monkeypatch.setattr(api, "get_breakout_board_candidates", board)
    return api, counts


def test_track_record_caches_and_invalidates_on_watermark(monkeypatch):
    watermark = _Watermark()
    api, counts = _patch_track_record(monkeypatch, watermark)

    first = api.get_breakout_track_record(2026)
    second = api.get_breakout_track_record(2026)

    assert first == second
    assert counts == {"weekly_tr": 1, "board": 2}

    watermark.value = ("run-1", "grades-2", 5, "opp-1")
    api.get_breakout_track_record(2026)
    assert counts == {"weekly_tr": 2, "board": 4}


def test_track_record_watermark_failure_bypasses_cache(monkeypatch):
    api, counts = _patch_track_record(monkeypatch, lambda season: None)

    api.get_breakout_track_record(2026)
    api.get_breakout_track_record(2026)

    assert counts["weekly_tr"] == 2
    assert api._breakout_track_record_cache == {}
