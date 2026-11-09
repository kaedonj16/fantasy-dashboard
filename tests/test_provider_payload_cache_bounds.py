"""Provider raw-payload caches must be bounded (2026-11-08 OOM follow-up).

The Fleaflicker and MFL provider caches previously never evicted: the
TTL was only checked when the exact key was re-read, so every distinct
league a worker ever built pinned its raw payloads (~120 per
Fleaflicker league build) for the worker's whole life. These tests pin
the prune behavior: stale entries go first, then the oldest, and the
cap holds.
"""
import time

import pytest

pytest.importorskip("pandas")

from dashboard_services.providers import fleaflicker_api, mfl_api


@pytest.fixture(autouse=True)
def _clean_caches():
    fleaflicker_api._CACHE.clear()
    mfl_api._CACHE.clear()
    yield
    fleaflicker_api._CACHE.clear()
    mfl_api._CACHE.clear()


def _fill(cache, count, *, start_ts, step=1.0, id_offset=0):
    now = start_ts
    for i in range(count):
        cache[("Method", str(1000 + id_offset + i), 2026, "", True, ())] = (
            now, {"i": id_offset + i},
        )
        now += step
    return now


def test_fleaflicker_cache_prunes_to_cap_keeping_newest(monkeypatch):
    monkeypatch.setattr(fleaflicker_api, "_CACHE_MAX", 8)
    _fill(fleaflicker_api._CACHE, 20, start_ts=time.monotonic())
    fleaflicker_api._prune_payload_cache()
    assert len(fleaflicker_api._CACHE) == 8
    kept = sorted(v["i"] for _, v in fleaflicker_api._CACHE.values())
    assert kept == list(range(12, 20))


def test_fleaflicker_cache_sweeps_stale_before_oldest(monkeypatch):
    monkeypatch.setattr(fleaflicker_api, "_CACHE_MAX", 8)
    now = time.monotonic()
    # 6 stale entries (older than the stale horizon) + 6 fresh ones.
    _fill(fleaflicker_api._CACHE, 6, start_ts=now - 7200)
    _fill(fleaflicker_api._CACHE, 6, start_ts=now, id_offset=100)
    fleaflicker_api._prune_payload_cache()
    kept = sorted(v["i"] for _, v in fleaflicker_api._CACHE.values())
    # Keys collide on league id across the two fills, so assert by age:
    # nothing older than the stale horizon survives.
    assert len(fleaflicker_api._CACHE) <= 8
    assert all(
        ts > now - fleaflicker_api._CACHE_STALE_SECONDS
        for ts, _ in fleaflicker_api._CACHE.values()
    )
    assert kept  # fresh entries remain


def test_fleaflicker_cache_under_cap_untouched(monkeypatch):
    monkeypatch.setattr(fleaflicker_api, "_CACHE_MAX", 8)
    _fill(fleaflicker_api._CACHE, 5, start_ts=time.monotonic() - 7200)
    fleaflicker_api._prune_payload_cache()
    assert len(fleaflicker_api._CACHE) == 5


def test_mfl_cache_prunes_to_cap_keeping_newest(monkeypatch):
    monkeypatch.setattr(mfl_api, "_CACHE_MAX", 6)
    _fill(mfl_api._CACHE, 15, start_ts=time.monotonic())
    mfl_api._prune_payload_cache()
    assert len(mfl_api._CACHE) == 6
    kept = sorted(v["i"] for _, v in mfl_api._CACHE.values())
    assert kept == list(range(9, 15))


def test_espn_box_score_cache_evicts_oldest_over_cap(monkeypatch):
    from dashboard_services.providers import espn_api

    monkeypatch.setattr(espn_api, "_BOX_SCORE_CACHE_MAX", 6)
    espn_api._box_score_cache.clear()
    try:
        for i in range(10):
            espn_api._store_box_scores((2026, "12345", i + 1), [i])
        assert len(espn_api._box_score_cache) == 6
        weeks = sorted(k[2] for k in espn_api._box_score_cache)
        assert weeks == [5, 6, 7, 8, 9, 10]
    finally:
        espn_api._box_score_cache.clear()


def test_release_build_memory_is_safe():
    import app

    # Must never raise, on any platform.
    app._release_build_memory()
