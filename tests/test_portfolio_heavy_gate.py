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
    def __init__(self, own_keys=None):
        self.lock = threading.Lock()
        self.cur = 0
        self.max_seen = 0
        self.calls = 0
        self.gate = threading.Event()
        self.own_keys = own_keys

    def compute(self, key, platform, league_id, season, week, ctx):
        if self.own_keys is not None and key not in self.own_keys:
            # A rebuild kicked by ANOTHER test's still-running
            # background work (e.g. a My Leagues page build walking its
            # cards) resolves this same module-level compute hook while
            # the monkeypatch is installed. It is not part of this
            # test's cap accounting: returning immediately lets that
            # rebuild drain at once instead of parking on this gate for
            # 60s, holding an aggregate slot, and corrupting the
            # calls/cur counts and the rebuilding-set snapshot below
            # (the 2026-11-07 full-shard cascade on PR #3067: a foreign
            # mark/build inside this test's window failed the snapshot
            # assert even though this test's own cap behavior was
            # correct).
            return ([], {}, {})
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


def _rebuilding_snapshot():
    with pages._LIVE_MATCHUP_LOCKS_GUARD:
        return set(pages._LIVE_MATCHUP_REBUILDING)


def test_live_matchup_rebuilds_hold_to_aggregate_cap(fresh_agg, monkeypatch):
    keys = [("sleeper", f"gate-live-{i}", 2026, 9) for i in range(8)]
    admitted = keys[:AGG]
    dropped = keys[AGG:]
    tracker = _Tracker(own_keys=set(keys))
    monkeypatch.setattr(pages, "_compute_live_matchups", tracker.compute)
    pages._LIVE_MATCHUP_CACHE.clear()
    with pages._LIVE_MATCHUP_LOCKS_GUARD:
        # Only this test's marks: the rebuilding set is shared session
        # state, and clearing another test's in-flight marks would let
        # a duplicate rebuild of its key kick alongside the original.
        pages._LIVE_MATCHUP_REBUILDING.difference_update(keys)
    def _wait_admit(key, want, timeout=60.0):
        # Wait for one admitted key's compute to start, RE-KICKING it
        # while waiting. A kick that lands while foreign background
        # builds (an earlier test's page builds, which read the same
        # app-global semaphore this fixture swapped in) hold every
        # aggregate slot is dropped over-cap and its mark cleared, with
        # nothing left to retry it: tracker.calls then never steps and
        # the wait times out even though the cap held (PR #3133's
        # failing Integration twin: the first admission starved for the
        # full 20s while ESPN/Yahoo builds from other tests held the
        # fresh semaphore). Re-kicking is the mechanism's own contract
        # (an over-cap rebuild is dropped and the next poll re-kicks
        # it), and the per-key rebuilding mark makes a re-kick while
        # the original kick is still starting a no-op, so this cannot
        # double-admit the key or weaken the cap assertions below.
        deadline = time.time() + timeout
        last_kick = 0.0
        while time.time() < deadline:
            if tracker.calls == want:
                return True
            if time.time() - last_kick >= 0.25:
                last_kick = time.time()
                pages._schedule_live_matchup_rebuild(
                    key, "sleeper", key[1], 2026, 9, {})
            time.sleep(0.02)
        return tracker.calls == want

    try:
        # Admit rebuilds one at a time. Each admitted build parks on
        # the tracker gate inside compute, so an admission is a stable
        # end state (tracker.calls stepping 1, 2, 3) the test can wait
        # on deterministically. Scheduling all 8 at once and polling
        # for the cur == AGG overlap flaked in the full shard: the
        # overlap is a transient scheduling artifact of eight racing
        # threads, and under shard load it was intermittently never
        # observed inside the wait window even though the cap held.
        for i, key in enumerate(admitted):
            assert _wait_admit(key, i + 1)
        # Per-key dedupe alone allows all 8; the aggregate cap parked
        # exactly these 3, each still holding its aggregate slot.
        assert tracker.cur == AGG
        assert tracker.max_seen <= AGG

        # With every aggregate slot parked, the remaining kicks are
        # over the cap: their builds never run, and each dropped thread
        # clears its rebuilding mark, leaving exactly the admitted set.
        # The snapshot is scoped to this test's keys: a foreign rebuild
        # (another test's background work) may legitimately hold a mark
        # in the shared set during this window and is not this test's
        # cap behavior to assert on.
        for key in dropped:
            pages._schedule_live_matchup_rebuild(
                key, "sleeper", key[1], 2026, 9, {})
        assert _wait_for(
            lambda: _rebuilding_snapshot() & set(keys) == set(admitted))
        assert tracker.calls == AGG

        tracker.gate.set()
        assert _wait_for(lambda: tracker.cur == 0)
        # Every dropped rebuild cleared its mark, so nothing is stuck.
        assert _wait_for(
            lambda: not (_rebuilding_snapshot() & set(keys)))

        # A dropped key re-kicks and actually rebuilds once slots free
        # (same re-kick wait as the admissions: a lone kick can still
        # lose the freed slots to foreign builds).
        calls_before = tracker.calls
        assert _wait_admit(dropped[0], calls_before + 1)
        assert tracker.max_seen <= AGG
    finally:
        # Unconditional cleanup: release the gate AND wait for the
        # parked builds to actually drain, so no thread, mark or
        # aggregate slot outlives the test even when an assertion above
        # fails mid-test. Only this test's marks are discarded; foreign
        # marks belong to their owners' finally blocks.
        tracker.gate.set()
        _wait_for(lambda: tracker.cur == 0, timeout=10.0)
        pages._LIVE_MATCHUP_CACHE.clear()
        with pages._LIVE_MATCHUP_LOCKS_GUARD:
            pages._LIVE_MATCHUP_REBUILDING.difference_update(keys)


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
