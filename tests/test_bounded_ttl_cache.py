import threading
import time as real_time

import pytest

import dashboard_services.api as api


@pytest.fixture(autouse=True)
def isolate_ttl_cache_registry():
    """Keep per-test cache budgets independent of collection/execution order."""
    for item in api._TTL_CACHES:
        with item["lock"]:
            item["cache"].clear()
    yield
    for item in api._TTL_CACHES:
        with item["lock"]:
            item["cache"].clear()


def test_ttl_cache_evicts_lru_and_exposes_clear(monkeypatch):
    monkeypatch.setattr(api, "DASHBOARD_CACHE_MAX", 2)
    calls = []

    @api.ttl_cache(ttl=60)
    def load(key):
        calls.append(key)
        return key

    load("a")
    load("b")
    load("a")  # a is now most recently used
    load("c")
    assert len(load._cache) == 2
    assert not any(key[1] == ("b",) for key in load._cache)
    load.clear_cache()
    assert len(load._cache) == 0


def test_cache_budget_is_shared_across_decorated_functions(monkeypatch):
    monkeypatch.setattr(api, "DASHBOARD_CACHE_MAX", 2)

    @api.ttl_cache(ttl=60)
    def first(key):
        return key

    @api.ttl_cache(ttl=60)
    def second(key):
        return key

    first("a")
    second("b")
    second("c")
    assert len(first._cache) + len(second._cache) <= 2


def test_active_cache_keeps_lru_window_when_budget_shrinks(monkeypatch):
    """Previously populated caches must not evict every new active-cache row."""
    monkeypatch.setattr(api, "DASHBOARD_CACHE_MAX", 2)

    @api.ttl_cache(ttl=60)
    def old_cache(key):
        return key

    @api.ttl_cache(ttl=60)
    def active_cache(key):
        return key

    old_cache("old-a")
    old_cache("old-b")
    active_cache("new-a")
    active_cache("new-b")

    assert list(key[1] for key in active_cache._cache) == [
        ("new-a",), ("new-b",),
    ]
    assert len(old_cache._cache) + len(active_cache._cache) <= 2


def test_budget_eviction_is_safe_when_wall_clock_moves_backwards(monkeypatch):
    monkeypatch.setattr(api, "DASHBOARD_CACHE_MAX", 2)
    now = [2_000_000_000.0]
    monkeypatch.setattr(api.time, "time", lambda: now[0])
    calls = []

    @api.ttl_cache(ttl=300)
    def load(key):
        calls.append(key)
        return key

    load("old-clock")
    now[0] = 1_000.0
    assert load("new-clock") == "new-clock"
    assert load("new-clock") == "new-clock"
    assert calls.count("new-clock") == 1


def test_expired_entries_are_removed_and_stale_fallback_is_time_limited(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(api.time, "time", lambda: now[0])
    monkeypatch.setattr(api, "DASHBOARD_CACHE_STALE_TTL", 5)
    failing = [False]

    @api.ttl_cache(ttl=10)
    def load(key):
        if failing[0]:
            raise RuntimeError("upstream down")
        return "good"

    assert load("x") == "good"
    failing[0] = True
    now[0] = 112.0
    assert load("x") == "good"
    now[0] = 116.0
    with pytest.raises(RuntimeError, match="upstream down"):
        load("x")
    assert len(load._cache) == 0


def test_same_key_requests_share_one_inflight_fetch():
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    @api.ttl_cache(ttl=60)
    def load(key):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(2)
        return {"key": key}

    results = []
    threads = [threading.Thread(target=lambda: results.append(load("same"))) for _ in range(6)]
    for thread in threads:
        thread.start()
    assert entered.wait(1)
    real_time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(2)
    assert calls == 1
    assert len(results) == 6


def test_cache_max_environment_default_and_override(monkeypatch):
    assert api.DASHBOARD_CACHE_MAX >= 1
    assert api._positive_env_int("ABSENT_CACHE_SETTING", 75) == 75
    monkeypatch.setenv("CACHE_SETTING_TEST", "9")
    assert api._positive_env_int("CACHE_SETTING_TEST", 75) == 9
    monkeypatch.setenv("CACHE_SETTING_TEST", "invalid")
    assert api._positive_env_int("CACHE_SETTING_TEST", 75) == 75
