"""Regression tests for slow-endpoint tail-latency fixes.

Two request-thread hogs are covered:

1. ``_load_rookie_rankings_for_ctx`` ran a draft-completion DB lookup plus a
   model-values scan and 5-table rankings query at the end of EVERY cold
   league-context build. It is now cached per worker (draft-year key, 6h TTL).
2. ``_prefetch_week_projections`` warms all 18 weekly projection files in
   parallel so the game-log loop hits the memo instead of paying up to 18
   sequential network fetches (20s timeout each) on the request thread.
"""
from __future__ import annotations

import threading
import time

import pytest

# app.py imports pandas (and flask/openai) at module load. The fast "lint" CI
# shard has flask but not pandas, so guard here - otherwise importing app
# raises at COLLECTION and aborts the whole run. These tests run in the
# full-dependency shard.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod


@pytest.fixture()
def rookie_cache(monkeypatch):
    """Isolated rookie-rankings cache with a stubbed fetcher."""
    monkeypatch.setattr(appmod, "_ROOKIE_CTX_RANKINGS", (None, 0.0, []))
    monkeypatch.setattr(
        "data_building.rookie_pipeline.pipeline.get_active_rookie_class",
        lambda: 2026,
    )
    calls = []

    def fake_fetch(draft_year):
        calls.append(draft_year)
        return [
            {"player_id": "p1", "name": "Rookie One", "position": "QB",
             "overall_rank": 1, "value_1qb": 10.0, "value_sf": 12.0},
        ]

    monkeypatch.setattr(appmod, "_fetch_rookie_rankings_for_ctx", fake_fetch)
    return calls


def test_rookie_rankings_fetched_once_per_worker(rookie_cache):
    """Repeated league-context builds share one rankings load per worker."""
    first = appmod._load_rookie_rankings_for_ctx()
    second = appmod._load_rookie_rankings_for_ctx()
    assert rookie_cache == [2026]
    assert first == second
    assert first[0]["name"] == "Rookie One"


def test_rookie_rankings_refetch_after_ttl(rookie_cache, monkeypatch):
    """An expired TTL entry triggers exactly one reload."""
    appmod._load_rookie_rankings_for_ctx()
    assert rookie_cache == [2026]
    stale_ts = time.time() - appmod._ROOKIE_CTX_RANKINGS_TTL - 1
    monkeypatch.setattr(
        appmod, "_ROOKIE_CTX_RANKINGS",
        (2026, stale_ts, appmod._ROOKIE_CTX_RANKINGS[2]),
    )
    appmod._load_rookie_rankings_for_ctx()
    assert rookie_cache == [2026, 2026]


def test_rookie_rankings_refetch_on_draft_year_change(rookie_cache, monkeypatch):
    """A new draft class invalidates the previous year's cache."""
    appmod._load_rookie_rankings_for_ctx()
    monkeypatch.setattr(
        "data_building.rookie_pipeline.pipeline.get_active_rookie_class",
        lambda: 2027,
    )
    appmod._load_rookie_rankings_for_ctx()
    assert rookie_cache == [2026, 2027]


def test_rookie_rankings_failure_serves_stale(rookie_cache, monkeypatch):
    """A transient fetch failure serves the stale copy, not an empty list."""
    first = appmod._load_rookie_rankings_for_ctx()
    assert first, "precondition: cache primed"

    def boom(draft_year):
        raise RuntimeError("db down")

    monkeypatch.setattr(appmod, "_fetch_rookie_rankings_for_ctx", boom)
    # Expire the entry so the loader attempts a fetch.
    stale_ts = time.time() - appmod._ROOKIE_CTX_RANKINGS_TTL - 1
    monkeypatch.setattr(
        appmod, "_ROOKIE_CTX_RANKINGS",
        (2026, stale_ts, appmod._ROOKIE_CTX_RANKINGS[2]),
    )
    assert appmod._load_rookie_rankings_for_ctx() == first


def test_rookie_rankings_results_are_copies(rookie_cache):
    """Callers cannot mutate the cached rows through the returned list."""
    first = appmod._load_rookie_rankings_for_ctx()
    first[0]["name"] = "MUTATED"
    first.append({"player_id": "x"})
    second = appmod._load_rookie_rankings_for_ctx()
    assert second[0]["name"] == "Rookie One"
    assert len(second) == 1
    assert rookie_cache == [2026]


def test_rookie_rankings_single_flight_under_threads(rookie_cache):
    """Concurrent cold builds trigger only one fetch (lock-guarded)."""
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(appmod._load_rookie_rankings_for_ctx())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rookie_cache == [2026]
    assert all(r == results[0] and r for r in results)


def test_prefetch_warms_all_18_weeks_in_parallel(monkeypatch):
    """The prefetch calls load_week_projection for weeks 1-18 concurrently."""
    import utils.utils as uu

    seen_weeks = []
    seen_threads = set()
    lock = threading.Lock()

    def fake_load(season, week):
        with lock:
            seen_weeks.append(week)
            seen_threads.add(threading.get_ident())
        time.sleep(0.01)
        return {}

    monkeypatch.setattr(uu, "load_week_projection", fake_load)
    # The endpoint imports it from utils.utils at call time; make sure the
    # helper's own import sees the stub.
    appmod._prefetch_week_projections(2026)
    assert sorted(seen_weeks) == list(range(1, 19))
    assert len(seen_threads) > 1, "expected parallel execution across threads"


def test_prefetch_never_raises(monkeypatch):
    """A broken projection loader cannot break the game-log response."""
    import utils.utils as uu

    def boom(season, week):
        raise RuntimeError("network down")

    monkeypatch.setattr(uu, "load_week_projection", boom)
    appmod._prefetch_week_projections(2026)  # must not raise
