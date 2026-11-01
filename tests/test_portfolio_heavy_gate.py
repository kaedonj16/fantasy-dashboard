"""Portfolio heavy-gate completion: summary refresh + live-matchup rebuilds.

PR #2932 widened app.py's _BG_HEAVY_SEM (3) to every heavy background
kick it could reach, but two My Leagues mechanisms were missed:

- _kick_summary_refresh runs its builds on a private 2-thread pool with
  no aggregate gate, stacking 2 full-context walks on top of the cap.
- _schedule_live_matchup_rebuild was bounded only PER KEY: an
  11-league My Leagues load could run 11 provider matchup builds at
  once (one per card), on top of every other mechanism.

Both now acquire the same aggregate gate (non-blocking, via the lazy
app shim pattern). An over-cap summary refresh retries on its existing
ContextPending cadence (3s timer, attempt-bounded) and clears its
inflight mark; an over-cap matchup rebuild is dropped, its rebuilding
mark cleared, and the next card poll re-kicks it. These tests saturate
the aggregate semaphore and assert neither mechanism's build runs,
that the marks are cleared, and that both work again once slots free.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("numpy")

import app as appmod  # noqa: E402
import routes.user_pages_bp as pages  # noqa: E402

AGG = 3  # _BG_HEAVY_SEM size


def _wait_for(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _drain(sem, n, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = [sem.acquire(blocking=False) for _ in range(n)]
        for g in got:
            if g:
                sem.release()
        if all(got):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture()
def fresh_agg(monkeypatch):
    # Swap in a fresh aggregate semaphore (the test_aggregate_heavy_cap
    # isolation pattern): every mechanism reads the app module global at
    # call time, and an earlier test in the shard can leak a slot.
    monkeypatch.setattr(
        appmod, "_BG_HEAVY_SEM", threading.BoundedSemaphore(AGG))
    yield appmod._BG_HEAVY_SEM
    assert _drain(appmod._BG_HEAVY_SEM, AGG), "aggregate slots leaked"


def _hold_all(sem):
    held = [sem.acquire(blocking=False) for _ in range(AGG)]
    assert all(held)
    return held


def _release_all(sem, held):
    for h in held:
        if h:
            sem.release()


class _Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.cur = 0
        self.max_seen = 0
        self.calls = 0
        self.gate = threading.Event()

    def compute(self, key, platform, league_id, season, week, ctx):
        with self.lock:
            self.cur += 1
            self.calls += 1
            self.max_seen = max(self.max_seen, self.cur)
        try:
            self.gate.wait(timeout=60)
            return ([], {}, {})
        finally:
            with self.lock:
                self.cur -= 1


def test_live_matchup_rebuilds_hold_to_aggregate_cap(fresh_agg, monkeypatch):
    tracker = _Tracker()
    monkeypatch.setattr(pages, "_compute_live_matchups", tracker.compute)
    pages._LIVE_MATCHUP_CACHE.clear()
    with pages._LIVE_MATCHUP_LOCKS_GUARD:
        pages._LIVE_MATCHUP_REBUILDING.clear()
    keys = [("sleeper", f"gate-live-{i}", 2026, 9) for i in range(8)]
    try:
        for key in keys:
            pages._schedule_live_matchup_rebuild(
                key, "sleeper", key[1], 2026, 9, {})

        # Per-key dedupe alone allows all 8; the aggregate cap allows 3.
        assert _wait_for(lambda: tracker.cur == AGG)
        time.sleep(0.5)
        assert tracker.cur == AGG
        assert tracker.max_seen <= AGG
        assert tracker.calls == AGG

        tracker.gate.set()
        assert _wait_for(lambda: tracker.cur == 0)
        # Every dropped rebuild cleared its mark, so nothing is stuck.
        assert _wait_for(lambda: not pages._LIVE_MATCHUP_REBUILDING)

        # A dropped key re-kicks and actually rebuilds once slots free.
        dropped = [k for k in keys[AGG:]]
        calls_before = tracker.calls
        pages._schedule_live_matchup_rebuild(
            dropped[0], "sleeper", dropped[0][1], 2026, 9, {})
        assert _wait_for(lambda: tracker.calls > calls_before)
        assert tracker.max_seen <= AGG
    finally:
        tracker.gate.set()
        pages._LIVE_MATCHUP_CACHE.clear()
        with pages._LIVE_MATCHUP_LOCKS_GUARD:
            pages._LIVE_MATCHUP_REBUILDING.clear()


class _FakePool:
    def __init__(self):
        self.submitted = []

    def submit(self, fn):
        self.submitted.append(fn)  # never runs until the test drives it


class _FakeTimer:
    instances = []

    def __init__(self, interval, fn, args=None, kwargs=None):
        self.interval = interval
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = False
        self.started = False
        _FakeTimer.instances.append(self)

    def start(self):
        self.started = True  # records the retry; never fires it


def _membership(league_id="L1"):
    return {"platform": "sleeper", "league_id": league_id, "season": 2026,
            "name": f"League {league_id}"}


def test_summary_refresh_is_gated_and_retries(fresh_agg, monkeypatch):
    builds = []
    monkeypatch.setattr(
        "dashboard_services.portfolio_summary.build_league_summary",
        lambda account_id, membership, loader: builds.append(
            membership["league_id"]))
    pool = _FakePool()
    monkeypatch.setattr(pages, "_portfolio_refresh_pool", lambda: pool)
    _FakeTimer.instances = []
    monkeypatch.setattr(threading, "Timer", _FakeTimer)
    pages._SUMMARY_REFRESH_INFLIGHT.clear()
    try:
        # Saturate the aggregate cap: the refresh must not build.
        held = _hold_all(fresh_agg)
        try:
            pages._kick_summary_refresh(7, _membership("L1"))
            assert len(pool.submitted) == 1
            pool.submitted.pop()()
            assert builds == []
            # It deferred on the ContextPending cadence (3s, attempt + 1)
            # and cleared its inflight mark.
            assert len(_FakeTimer.instances) == 1
            timer = _FakeTimer.instances[0]
            assert timer.started and timer.interval == 3.0
            assert timer.kwargs.get("_attempt") == 1
            assert not pages._SUMMARY_REFRESH_INFLIGHT

            # With attempts exhausted an over-cap run drops cleanly: no
            # build, no further timer, mark still cleared.
            pages._kick_summary_refresh(7, _membership("L1"), _attempt=10)
            pool.submitted.pop()()
            assert builds == []
            assert len(_FakeTimer.instances) == 1
            assert not pages._SUMMARY_REFRESH_INFLIGHT
        finally:
            _release_all(fresh_agg, held)

        # Slots free: a later kick builds normally.
        pages._kick_summary_refresh(7, _membership("L1"))
        pool.submitted.pop()()
        assert builds == ["L1"]
        assert not pages._SUMMARY_REFRESH_INFLIGHT
    finally:
        pages._SUMMARY_REFRESH_INFLIGHT.clear()
