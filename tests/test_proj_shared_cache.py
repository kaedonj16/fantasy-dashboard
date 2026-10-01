"""Tests for the shared projections cache work (2026-10-01).

Covers: race-safe write_json tmp names, the per-(season, week) singleflight
+ cross-process lock on get_week_projections_cached, the force_refresh
debounce, the bounded-LRU _WEEK_PROJ_MEMO, and the Redis proj_store
roundtrip / restore path.  Hermetic: cache paths point at tmp dirs, the
Redis client is faked or absent, and the cross-process lock runs through
its flock fallback (DATABASE_URL removed).
"""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

# utils.utils imports dashboard_services.api (flask/requests/bs4 stack).
pytest.importorskip("requests")
pytest.importorskip("bs4")
pytest.importorskip("flask")

from dashboard_services import proj_store
from utils import utils

DATA = {"4984": {"ppr": 22.4, "half_ppr": 20.1, "raw_stats": {"rec": 5, "rec_yd": 60}}}


@pytest.fixture
def proj_env(tmp_path, monkeypatch):
    """Point the week-projection file cache at a tmp dir and reset state."""
    monkeypatch.setattr(
        utils, "path_week_proj",
        lambda season, week: str(tmp_path / f"projections_s{season}_w{week}.json"),
    )
    monkeypatch.setattr(utils, "get_nfl_state", lambda: {"season": 2026, "week": 1})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("PROJ_REDIS", raising=False)
    utils._WEEK_PROJ_FAIL_UNTIL.clear()
    with utils._WEEK_PROJ_MEMO_LOCK:
        utils._WEEK_PROJ_MEMO.clear()
    return tmp_path


def _cache_file(tmp_path, season=2026, week=1):
    return tmp_path / f"projections_s{season}_w{week}.json"


# ---------------------------------------------------------------------------
# (a) write_json: concurrent writers to the same path
# ---------------------------------------------------------------------------

def test_write_json_concurrent_writers_same_path(tmp_path):
    target = tmp_path / "shared.json"
    payloads = [{"writer": i, "blob": "x" * 400} for i in range(8)]
    errors = []

    def write(payload):
        try:
            utils.write_json(target, payload)
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert errors == []
    final = json.loads(target.read_text(encoding="utf-8"))
    assert final in payloads
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_write_json_failure_cleans_up_tmp(tmp_path):
    target = tmp_path / "bad.json"
    with pytest.raises(TypeError):
        utils.write_json(target, {"x": object()})
    assert not target.exists()
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


# ---------------------------------------------------------------------------
# (b) get_week_projections_cached: in-process singleflight
# ---------------------------------------------------------------------------

def test_concurrent_callers_fetch_exactly_once(proj_env):
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        time.sleep(0.3)
        return dict(DATA)

    results, errors = [], []

    def call():
        try:
            results.append(utils.get_week_projections_cached(2026, 1, fetch))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert errors == []
    assert len(calls) == 1
    assert results == [DATA, DATA]
    assert _cache_file(proj_env).exists()


# ---------------------------------------------------------------------------
# (c) force_refresh debounce
# ---------------------------------------------------------------------------

def test_force_refresh_serves_fresh_populated_file(proj_env):
    cache = _cache_file(proj_env)
    utils.write_json(cache, DATA)
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        return {"9999": {"ppr": 1.0}}

    out = utils.get_week_projections_cached(2026, 1, fetch, force_refresh=True)
    assert calls == []
    assert out == DATA


def test_force_refresh_refetches_when_stale(proj_env, monkeypatch):
    cache = _cache_file(proj_env)
    utils.write_json(cache, DATA)
    monkeypatch.setattr(utils, "_week_proj_is_stale", lambda *a, **k: True)
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        return {"9999": {"ppr": 1.0}}

    out = utils.get_week_projections_cached(2026, 1, fetch, force_refresh=True)
    assert len(calls) == 1
    assert out == {"9999": {"ppr": 1.0}}


def test_force_refresh_fetches_when_file_missing(proj_env):
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        return dict(DATA)

    out = utils.get_week_projections_cached(2026, 1, fetch, force_refresh=True)
    assert len(calls) == 1
    assert out == DATA


# ---------------------------------------------------------------------------
# (d) _WEEK_PROJ_MEMO bounded LRU
# ---------------------------------------------------------------------------

def test_week_proj_memo_evicts_oldest_only(proj_env, monkeypatch):
    monkeypatch.setattr(utils, "_WEEK_PROJ_MEMO_MAX", 3)
    for week in range(1, 6):
        utils.write_json(_cache_file(proj_env, week=week), {str(week): {"ppr": float(week)}})
        assert utils.load_week_projection(2026, week) == {str(week): {"ppr": float(week)}}

    def memo_weeks():
        return {key[1] for key in utils._WEEK_PROJ_MEMO}

    assert len(utils._WEEK_PROJ_MEMO) == 3
    assert memo_weeks() == {3, 4, 5}

    # Re-reading week 3 makes it most-recent; adding week 6 evicts week 4,
    # not everything (the old behavior cleared the whole memo).
    assert utils.load_week_projection(2026, 3) == {"3": {"ppr": 3.0}}
    utils.write_json(_cache_file(proj_env, week=6), {"6": {"ppr": 6.0}})
    assert utils.load_week_projection(2026, 6) == {"6": {"ppr": 6.0}}
    assert len(utils._WEEK_PROJ_MEMO) == 3
    assert memo_weeks() == {3, 5, 6}


# ---------------------------------------------------------------------------
# (e) proj_store: roundtrip, disabled, corrupt
# ---------------------------------------------------------------------------

class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl
        return True

    def get(self, key):
        return self.store.get(key)


def test_proj_store_roundtrip_with_fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(proj_store, "_redis_client", lambda: fake)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("PROJ_REDIS", raising=False)

    assert proj_store.enabled() is True
    assert proj_store.save(2026, 1, DATA) is True
    assert fake.ttls[proj_store.proj_key(2026, 1)] == proj_store.STORAGE_TTL_SECONDS

    hit = proj_store.load(2026, 1)
    assert hit is not None
    saved_at, data = hit
    assert data == DATA
    assert saved_at == pytest.approx(time.time(), abs=5)

    assert proj_store.load(2026, 2) is None  # missing key


def test_proj_store_corrupt_payload_returns_none(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(proj_store, "_redis_client", lambda: fake)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    key = proj_store.proj_key(2026, 1)

    fake.store[key] = b"not json at all"
    assert proj_store.load(2026, 1) is None

    fake.store[key] = json.dumps({"saved_at": "soon", "data": [1, 2]}).encode()
    assert proj_store.load(2026, 1) is None

    fake.store[key] = json.dumps({"saved_at": 123.0, "data": {"1": {}}}).encode()
    assert proj_store.load(2026, 1) == (123.0, {"1": {}})


def test_proj_store_disabled_without_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("PROJ_REDIS", raising=False)
    assert proj_store.enabled() is False
    assert proj_store.save(2026, 1, DATA) is False
    assert proj_store.load(2026, 1) is None


def test_proj_store_kill_switch(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(proj_store, "_redis_client", lambda: fake)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("PROJ_REDIS", "0")
    assert proj_store.enabled() is False
    assert proj_store.save(2026, 1, DATA) is False
    assert proj_store.load(2026, 1) is None
    assert fake.store == {}


def test_proj_store_oversize_save_skipped(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(proj_store, "_redis_client", lambda: fake)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("PROJ_REDIS_MAX_BYTES", "1024")
    big = {str(i): {"ppr": 1.0, "blob": "x" * 100} for i in range(50)}
    assert proj_store.save(2026, 1, big) is False
    assert fake.store == {}


# ---------------------------------------------------------------------------
# (f) Redis restore inside get_week_projections_cached
# ---------------------------------------------------------------------------

def test_redis_restore_serves_without_fetch(proj_env, monkeypatch):
    saved_at = time.time()
    monkeypatch.setattr(proj_store, "load", lambda s, w: (saved_at, dict(DATA)))
    saves = []
    monkeypatch.setattr(proj_store, "save", lambda *a, **k: saves.append(a))
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        return {"9999": {"ppr": 1.0}}

    out = utils.get_week_projections_cached(2026, 1, fetch)

    assert calls == []
    assert out == DATA
    cache = _cache_file(proj_env)
    assert cache.exists()
    assert json.loads(cache.read_text(encoding="utf-8")) == DATA
    # The restored file carries the stored timestamp so staleness logic
    # treats it like a fetched copy, and restoring never re-saves to Redis.
    assert os.path.getmtime(cache) == pytest.approx(saved_at, abs=0.01)
    assert saves == []


def test_redis_restore_stale_falls_through_to_fetch(proj_env, monkeypatch):
    stale_at = time.time() - 7 * 24 * 3600
    monkeypatch.setattr(proj_store, "load", lambda s, w: (stale_at, dict(DATA)))
    monkeypatch.setattr(utils, "get_nfl_state", lambda: {"season": 2026, "week": 1})
    calls = []

    def fetch(season, week):
        calls.append((season, week))
        return {"9999": {"ppr": 1.0}}

    out = utils.get_week_projections_cached(2026, 1, fetch)
    assert len(calls) == 1
    assert out == {"9999": {"ppr": 1.0}}


# ---------------------------------------------------------------------------
# resource_build_lock: name-scoped cross-process lock
# ---------------------------------------------------------------------------

def test_resource_build_lock_is_name_scoped(monkeypatch):
    from dashboard_services.league_singleflight import LeagueBuildBusy, resource_build_lock

    monkeypatch.delenv("DATABASE_URL", raising=False)
    entered = threading.Event()
    release = threading.Event()

    def owner():
        with resource_build_lock("proj-test-resource", timeout=2.0):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(1)
    try:
        with pytest.raises(LeagueBuildBusy):
            with resource_build_lock("proj-test-resource", timeout=0.1):
                pass
        # A different resource name is an independent lock.
        with resource_build_lock("proj-test-other", timeout=0.5):
            pass
    finally:
        release.set()
        thread.join(2)
    with resource_build_lock("proj-test-resource", timeout=0.5):
        pass


def test_resource_lock_key_is_stable_and_name_scoped():
    from dashboard_services.league_singleflight import resource_lock_key

    assert resource_lock_key("projections:2026:1") == resource_lock_key("projections:2026:1")
    assert resource_lock_key("projections:2026:1") != resource_lock_key("projections:2026:2")
