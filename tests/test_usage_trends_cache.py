"""Tests for the get_usage_trends() TTL cache.

The waiver page and /api/weekly-trends call get_usage_trends() on every
request; the underlying query is a full table scan of player_weekly_metrics
(the 28s cold waiver load). These tests pin the caching contract: one scan
per season per TTL window, season-scoped keys, and TTL expiry.
"""
import pytest

import data_building.weekly_metrics as wm


@pytest.fixture()
def counting_compute(monkeypatch):
    calls = []

    def fake_compute(season):
        calls.append(season)
        return {"season": season, "players": {"1": {"delta": 1.0}}}

    monkeypatch.setattr(wm, "_compute_usage_trends", fake_compute)
    monkeypatch.setattr(wm, "clear_usage_trends_cache", wm.clear_usage_trends_cache)
    wm.clear_usage_trends_cache()
    yield calls
    wm.clear_usage_trends_cache()


def test_second_call_within_ttl_does_not_recompute(counting_compute):
    calls = counting_compute
    first = wm.get_usage_trends(2026)
    second = wm.get_usage_trends(2026)
    assert calls == [2026]
    assert second is first


def test_cache_is_season_scoped(counting_compute):
    calls = counting_compute
    wm.get_usage_trends(2026)
    wm.get_usage_trends(2025)
    wm.get_usage_trends(2026)
    assert calls == [2026, 2025]


def test_ttl_expiry_recomputes(counting_compute, monkeypatch):
    calls = counting_compute
    now = [1000.0]
    monkeypatch.setattr(wm.time, "monotonic", lambda: now[0])
    wm.get_usage_trends(2026)
    assert calls == [2026]
    now[0] += wm._USAGE_TRENDS_TTL - 1
    wm.get_usage_trends(2026)
    assert calls == [2026]
    now[0] += 2  # past the TTL
    wm.get_usage_trends(2026)
    assert calls == [2026, 2026]


def test_clear_usage_trends_cache_forces_recompute(counting_compute):
    calls = counting_compute
    wm.get_usage_trends(2026)
    wm.clear_usage_trends_cache()
    wm.get_usage_trends(2026)
    assert calls == [2026, 2026]
