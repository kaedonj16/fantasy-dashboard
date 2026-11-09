"""ScoreZone memory bounds (2026-11-08 OOM restarts).

The web worker (2 GB) exceeded its memory limit 7 times on one game day.
The ScoreZone live path was the largest unbounded retention + concurrency
surface left: collect / user-aggregate caches held 32 entries each of
multi-MB pbp_by_game payloads, the projection cache was never pruned at
all, and builds for distinct leagues ran uncapped across all request
threads. These tests pin the bounds and the build gate.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_rz_state(monkeypatch):
    monkeypatch.setattr(app, "_RZ_COLLECT_CACHE", {})
    monkeypatch.setattr(app, "_RZ_COLLECT_INFLIGHT", {})
    monkeypatch.setattr(app, "_RZ_USER_CACHE", {})
    monkeypatch.setattr(app, "_RZ_USER_INFLIGHT", {})
    monkeypatch.setattr(app, "_RZ_PROJ_CACHE", {})
    monkeypatch.setattr(app, "_RZ_BUILD_SEM", threading.BoundedSemaphore(2))
    yield


def test_rz_collect_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(app, "_scorezone_collect", lambda *a: {"games": []})
    total = app._RZ_COLLECT_CACHE_MAX + 24
    for i in range(total):
        key = ("sleeper", 2026, f"league-{i}", 10)
        app._rz_collect_build(key, "sleeper", f"league-{i}", 2026, 10)
    assert len(app._RZ_COLLECT_CACHE) <= app._RZ_COLLECT_CACHE_MAX


def test_rz_user_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(app, "_scorezone_fetch_user", lambda *a: {"games": []})
    total = app._RZ_USER_CACHE_MAX + 24
    for i in range(total):
        key = (f"acct:{i}", "sleeper", 2026, "lg", 10)
        app._rz_user_build(key, "sleeper", "lg", 2026, 10)
    assert len(app._RZ_USER_CACHE) <= app._RZ_USER_CACHE_MAX


def test_rz_proj_cache_stays_bounded(monkeypatch):
    import utils.fantasy_scoring as scoring
    import utils.utils as uu

    monkeypatch.setattr(uu, "fetch_week_from_sleeper", lambda *a, **k: {})
    monkeypatch.setattr(uu, "load_players_index", lambda *a, **k: {})
    monkeypatch.setattr(scoring, "projection_points", lambda *a, **k: 0.0)
    total = app._RZ_PROJ_CACHE_MAX + 24
    for i in range(total):
        # A distinct scoring dict per call => a distinct cache key.
        app._rz_get_projections(2026, 10, scoring={"pts_rec": i})
    assert len(app._RZ_PROJ_CACHE) <= app._RZ_PROJ_CACHE_MAX


def test_rz_build_gate_caps_concurrent_builds(monkeypatch):
    state = {"cur": 0, "peak": 0}
    lock = threading.Lock()

    def slow_collect(*a):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        time.sleep(0.25)
        with lock:
            state["cur"] -= 1
        return {"games": []}

    monkeypatch.setattr(app, "_scorezone_collect", slow_collect)
    threads = []
    for i in range(6):
        key = ("sleeper", 2026, f"league-{i}", 10)
        t = threading.Thread(
            target=app._rz_collect_build,
            args=(key, "sleeper", f"league-{i}", 2026, 10),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert state["peak"] <= 2


def test_rz_build_gate_is_reentrant_on_one_thread():
    token = app._rz_build_acquire()
    try:
        nested = app._rz_build_acquire()
        assert nested == "nested"
        app._rz_build_release(nested)  # must not free the outer slot
        # The outer slot is still held: a fresh semaphore probe on another
        # thread is not needed -- depth is still marked on this thread.
        assert getattr(app._RZ_BUILD_LOCAL, "depth", 0) == 1
    finally:
        app._rz_build_release(token)
    assert getattr(app._RZ_BUILD_LOCAL, "depth", 0) == 0
