"""_rz_cached_collect must single-flight concurrent polls.

Regression test for the live-game 524s: the collect takes ~60s to build
(upstream PBP for every game) while the client polls every 15s and gunicorn
only has 4 request threads. Without single-flight, every poll did a full
duplicate build, starving the workers until Cloudflare timed out.

Contracts:
  * Concurrent polls for the same (platform, league, season, week) share one
    _redzone_collect build instead of each building.
  * A cache hit within the TTL never rebuilds.
  * An expired entry rebuilds on the next poll.
  * A waiter never hangs forever if the builder dies (bounded wait).
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _fresh_key(tag):
    return ("sleeper", f"lg-singleflight-{tag}", 2026, 4)


@pytest.fixture()
def app_mod(monkeypatch):
    import app

    # Isolate global single-flight state per test.
    monkeypatch.setattr(app, "_RZ_COLLECT_CACHE", {})
    monkeypatch.setattr(app, "_RZ_COLLECT_INFLIGHT", {})
    return app


def test_concurrent_polls_share_one_build(app_mod, monkeypatch):
    calls = []

    def fake_collect(platform, league_id, season, week):
        calls.append((platform, league_id, season, week))
        time.sleep(0.5)  # simulate the slow upstream build
        return {"ok": True, "n": len(calls)}

    monkeypatch.setattr(app_mod, "_redzone_collect", fake_collect)
    key = _fresh_key("shared")

    results = []
    errors = []

    def poll():
        try:
            results.append(app_mod._rz_cached_collect(*key))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=poll) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a poll hung"
    assert not errors, f"poll errors: {errors}"
    assert len(calls) == 1, f"expected 1 shared build, got {len(calls)}"
    assert len(results) == 4
    payloads = [r[0] for r in results]
    assert all(p == payloads[0] for p in payloads), "waiters got different payloads"


def test_cache_hit_within_ttl_never_rebuilds(app_mod, monkeypatch):
    calls = []

    def fake_collect(platform, league_id, season, week):
        calls.append(1)
        return {"ok": True}

    monkeypatch.setattr(app_mod, "_redzone_collect", fake_collect)
    key = _fresh_key("hit")
    first = app_mod._rz_cached_collect(*key)
    second = app_mod._rz_cached_collect(*key)
    assert len(calls) == 1
    assert first[0] == second[0]
    assert first[1] == second[1], "etag should be stable on a cache hit"


def test_expired_entry_rebuilds(app_mod, monkeypatch):
    calls = []

    def fake_collect(platform, league_id, season, week):
        calls.append(1)
        return {"ok": True, "build": len(calls)}

    monkeypatch.setattr(app_mod, "_redzone_collect", fake_collect)
    monkeypatch.setattr(app_mod, "_RZ_COLLECT_TTL", 0.05)
    key = _fresh_key("expiry")
    first = app_mod._rz_cached_collect(*key)
    time.sleep(0.08)
    second = app_mod._rz_cached_collect(*key)
    assert len(calls) == 2, "expired entry should rebuild"
    assert first[0]["build"] == 1
    assert second[0]["build"] == 2


def test_waiter_does_not_hang_when_builder_dies(app_mod, monkeypatch):
    # A builder that registers itself in-flight but never signals (simulates
    # a dead thread): the waiter must give up after the bounded wait and
    # build directly instead of hanging forever.
    monkeypatch.setattr(app_mod, "_RZ_COLLECT_WAIT_TIMEOUT", 0.2)
    real_collect_calls = []

    def fake_collect(platform, league_id, season, week):
        real_collect_calls.append(1)
        return {"ok": True}

    monkeypatch.setattr(app_mod, "_redzone_collect", fake_collect)
    key = _fresh_key("dead")
    cache_key = (str(key[0]), int(key[2]), str(key[1]), int(key[3]))

    stuck = threading.Event()
    app_mod._RZ_COLLECT_INFLIGHT[cache_key] = stuck  # never set: dead builder

    start = time.time()
    payload, _etag, _collected_at = app_mod._rz_cached_collect(*key)
    elapsed = time.time() - start
    assert payload == {"ok": True}
    assert elapsed < 30, f"waiter hung too long: {elapsed:.1f}s"
    assert real_collect_calls, "waiter should have built directly after the timeout"
