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


def test_singleflight_wait_is_bounded_and_waiter_fetches_itself(monkeypatch):
    """2026-10-27: the duplicate-key wait used to be unbounded, so one
    hung fetcher parked every waiter on its request thread forever.
    After the bounded wait the waiter falls through and fetches itself."""
    monkeypatch.setattr(api, "_TTL_SINGLEFLIGHT_WAIT_SECONDS", 0.3)
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    @api.ttl_cache(ttl=60)
    def load(key):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(3)
        return {"key": key}

    holder = []
    fetcher = threading.Thread(target=lambda: holder.append(load("same")))
    fetcher.start()
    assert entered.wait(1)
    started = real_time.time()
    try:
        result = load("same")
        # Returned via its own fetch long before the first fetcher was
        # released: it did not wait unboundedly.
        assert real_time.time() - started < 2.0
        assert result == {"key": "same"}
        assert calls == 2
    finally:
        release.set()
        fetcher.join(2)
    assert holder == [{"key": "same"}]


def test_singleflight_wait_timeout_serves_stale_entry(monkeypatch):
    """On a bounded-wait timeout with a stale entry still inside the
    stale window, the waiter serves the stale value instead of fetching
    a duplicate or waiting again."""
    monkeypatch.setattr(api, "_TTL_SINGLEFLIGHT_WAIT_SECONDS", 0.3)
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    @api.ttl_cache(ttl=1)
    def load(key):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(3)
        return {"key": key, "call": calls}

    assert load("k") == {"key": "k", "call": 1}
    real_time.sleep(1.2)  # entry is now past its TTL but inside stale grace

    holder = []
    fetcher = threading.Thread(target=lambda: holder.append(load("k")))
    fetcher.start()
    assert entered.wait(1)
    started = real_time.time()
    try:
        result = load("k")
        assert real_time.time() - started < 2.0
        assert result == {"key": "k", "call": 1}  # the stale entry
        assert calls == 2  # the waiter did not start a third fetch
    finally:
        release.set()
        fetcher.join(2)


def test_cache_max_environment_default_and_override(monkeypatch):
    assert api.DASHBOARD_CACHE_MAX >= 1
    assert api._positive_env_int("ABSENT_CACHE_SETTING", 75) == 75
    monkeypatch.setenv("CACHE_SETTING_TEST", "9")
    assert api._positive_env_int("CACHE_SETTING_TEST", 75) == 9
    monkeypatch.setenv("CACHE_SETTING_TEST", "invalid")
    assert api._positive_env_int("CACHE_SETTING_TEST", 75) == 75
