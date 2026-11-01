"""Aggregate heavy-kick cap: ONE per-worker ceiling across mechanisms.

Every heavy background mechanism has its own per-worker semaphore
(playoff sims 2, modal league warms 2, impact warms 2, league warms 2,
digests 2, portfolio actions 2, insights 1, page builds 3), but before
this fix they STACKED: a cold My Leagues load could run ~11 full-context
builds at once and OOM the 2 GB web worker. _BG_HEAVY_SEM (3) is now the
single aggregate gate every opportunistic kick also acquires
(non-blocking, drop on contention, dedupe mark cleared so a later read
re-kicks). Blocking paths (a caller waiting on its own result) are not
gated.

These tests fire simultaneous kicks across TWO mechanisms (playoff sims
+ modal league warms) whose own caps sum to 4, and assert the aggregate
holds total concurrent heavy executions at 3 -- a bound neither
per-mechanism test can see. On unfixed main the same fan-out reaches 4.
"""
from __future__ import annotations

import importlib
import threading
import time

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("numpy")

import app as appmod  # noqa: E402

AGG = 3  # _BG_HEAVY_SEM size


class _Tracker:
    """Shared concurrency counter across the sim and modal mechanisms."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cur = 0
        self.max_seen = 0
        self.sim_calls = 0
        self.modal_calls = 0
        self.gate = threading.Event()  # while unset, builds block inside

    def _enter(self, kind):
        with self.lock:
            self.cur += 1
            self.max_seen = max(self.max_seen, self.cur)
            if kind == "sim":
                self.sim_calls += 1
            else:
                self.modal_calls += 1

    def _exit(self):
        with self.lock:
            self.cur -= 1

    def sim(self, ctx, platform="sleeper", **_kw):
        self._enter("sim")
        try:
            self.gate.wait(timeout=60)
            return [{"roster_id": 1, "playoff_pct": 50.0}]
        finally:
            self._exit()

    def modal_build(self, season, scoring):
        self._enter("modal")
        try:
            self.gate.wait(timeout=60)
            return {}
        finally:
            self._exit()


def _wait_for(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _drain(sem, n, timeout=30.0):
    """Wait until all `n` slots of `sem` can be taken at once."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = [sem.acquire(blocking=False) for _ in range(n)]
        for g in got:
            if g:
                sem.release()
        if all(got):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture()
def agg_gate(monkeypatch):
    # Isolate the aggregate semaphore: tests share one process per CI
    # shard, and an earlier test can permanently hold a slot -- e.g.
    # test_activity_deferral replaces threading.Thread with a fake whose
    # start() never runs the target, so the activity build it kicks
    # acquires _BG_HEAVY_SEM and the releasing worker body never
    # executes (a pre-existing in-process leak that was invisible while
    # nothing asserted on this semaphore). Every mechanism reads the
    # module global at call time, so swapping in a fresh semaphore is
    # coherent across all of them for the duration of the test.
    monkeypatch.setattr(
        appmod, "_BG_HEAVY_SEM", threading.BoundedSemaphore(AGG))
    assert _drain(appmod._PLAYOFF_SIM_SEM, appmod._PLAYOFF_SIM_MAX_CONCURRENT), \
        "sim slots never freed"
    # Modal warm cap is env-sized at import; drain its real size.
    modal_cap = 2
    assert _drain(appmod._MODAL_WARM_SEM, modal_cap), "modal slots never freed"

    sim_mod = importlib.import_module("data_building.simulate_playoff_odds")
    svc_api = importlib.import_module("dashboard_services.api")
    plt = importlib.import_module("dashboard_services.player_league_trades")
    svc = importlib.import_module("dashboard_services.service")

    monkeypatch.setattr(appmod, "_playoff_sim_sig", lambda ctx, platform: "test-sig")
    monkeypatch.setitem(appmod.app.config, "TESTING", False)

    tracker = _Tracker()
    monkeypatch.setattr(sim_mod, "simulate_playoff_odds", tracker.sim)
    monkeypatch.setattr(appmod, "_load_season_weekly_points", tracker.modal_build)
    monkeypatch.setattr(appmod, "_season_rank_rows", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_season_weekstat_points", lambda *a, **k: None)
    monkeypatch.setattr(
        svc_api, "get_league",
        lambda lid, *a, **k: {"scoring_settings": {"rec": 0.5, "pass_yd": 0.04}},
    )
    monkeypatch.setattr(
        svc_api, "build_league_history_map", lambda *a, **k: {2026: str(a[1])})
    monkeypatch.setattr(svc, "get_transactions_by_week", lambda *a, **k: None)
    monkeypatch.setattr(plt, "_roster_names", lambda *a, **k: None)
    monkeypatch.setattr(plt, "build_draft_resolution_map", lambda *a, **k: None)
    monkeypatch.setattr(plt, "_slot_map_for_league", lambda *a, **k: None)

    appmod._PLAYOFF_SIM_CACHE.clear()
    with appmod._PLAYOFF_SIM_BUILDING_LOCK:
        appmod._PLAYOFF_SIM_BUILDING.clear()
    appmod._MODAL_LEAGUE_WARMED.clear()
    yield tracker
    tracker.gate.set()
    _wait_for(lambda: tracker.cur == 0)
    appmod._PLAYOFF_SIM_CACHE.clear()
    appmod._MODAL_LEAGUE_WARMED.clear()


def _sim_ctx(league_id):
    return {
        "league_id": league_id,
        "season": 2026,
        "current_week": 8,
        "rosters": [
            {"roster_id": 1, "players": ["p1", "p2"]},
            {"roster_id": 2, "players": ["p3", "p4"]},
        ],
    }


def test_aggregate_cap_holds_across_sim_and_modal_kicks(agg_gate):
    tracker = agg_gate
    sim_ids = [f"agg-sim-{i}" for i in range(4)]
    modal_ids = [f"agg-modal-{i}" for i in range(4)]

    # Fire both fan-outs at once: per-mechanism caps alone allow 2 + 2.
    for lid in sim_ids:
        appmod._playoff_sim_cached(_sim_ctx(lid), "sleeper", block=False)
    for lid in modal_ids:
        appmod._kick_modal_league_warm(
            "sleeper", lid, 2026, {"scoring_settings": {"rec": 0.5, "pass_yd": 0.04}})

    # Total concurrent heavy executions reach the aggregate cap and stop.
    assert _wait_for(lambda: tracker.cur == AGG)
    time.sleep(0.5)
    assert tracker.cur == AGG
    assert tracker.max_seen <= AGG
    started = tracker.sim_calls + tracker.modal_calls
    assert started == AGG

    # Dropped kicks left no dedupe mark behind: at least one sim league is
    # out of the BUILDING set and at least one modal league is un-marked,
    # so a later read can re-kick them.
    with appmod._PLAYOFF_SIM_BUILDING_LOCK:
        building = set(appmod._PLAYOFF_SIM_BUILDING)
    assert len(building) < len(sim_ids)
    marked = {key[1] for key in appmod._MODAL_LEAGUE_WARMED}
    assert len(marked) < len(modal_ids)

    # Release everything; the running jobs finish and free their slots.
    tracker.gate.set()
    assert _wait_for(lambda: tracker.cur == 0)
    assert _wait_for(lambda: _drain(appmod._BG_HEAVY_SEM, AGG, timeout=10))

    # A dropped kick of each mechanism re-kicks and actually executes.
    dropped_sim = [lid for lid in sim_ids
                   if ("sleeper", lid, 2026, "v3") not in building]
    sim_before = tracker.sim_calls
    appmod._playoff_sim_cached(_sim_ctx(dropped_sim[0]), "sleeper", block=False)
    assert _wait_for(lambda: tracker.sim_calls > sim_before)

    dropped_modal = [lid for lid in modal_ids if lid not in marked]
    modal_before = tracker.modal_calls
    appmod._kick_modal_league_warm(
        "sleeper", dropped_modal[0], 2026,
        {"scoring_settings": {"rec": 0.5, "pass_yd": 0.04}})
    assert _wait_for(lambda: tracker.modal_calls > modal_before)
    assert tracker.max_seen <= AGG


def test_kick_from_inside_a_heavy_job_passes_through(agg_gate):
    """Nesting: get_league_ctx_from_cache kicks modal/sim work from inside
    heavy jobs (league warms, page builds, portfolio scans). A kick made
    on a thread already executing a heavy job must NOT consume a second
    aggregate slot or be dropped just because the cap is full -- the
    outermost kick owns the slot."""
    tracker = agg_gate
    # Saturate the aggregate cap from the outside.
    held = [appmod._BG_HEAVY_SEM.acquire(blocking=False) for _ in range(AGG)]
    assert all(held)
    try:
        # Simulate "this thread is inside a heavy job" the way the worker
        # wrappers mark it, then kick a sim: it must still run.
        prev = appmod._heavy_job_start("slot")
        try:
            appmod._playoff_sim_cached(_sim_ctx("agg-nested-1"), "sleeper", block=False)
        finally:
            appmod._heavy_job_end("nested", prev)
        assert _wait_for(lambda: tracker.sim_calls == 1)
        tracker.gate.set()
        assert _wait_for(lambda: tracker.cur == 0)
    finally:
        for h in held:
            if h:
                appmod._BG_HEAVY_SEM.release()


def test_blocking_sim_path_is_not_gated(agg_gate):
    """A caller that waits for its own result (block=True) never takes the
    aggregate gate: it runs even with every aggregate slot held."""
    tracker = agg_gate
    tracker.gate.set()  # do not block the synchronous sim
    held = [appmod._BG_HEAVY_SEM.acquire(blocking=False) for _ in range(AGG)]
    assert all(held)
    try:
        rows = appmod._playoff_sim_cached(_sim_ctx("agg-block-1"), "sleeper", block=True)
        assert rows and rows[0]["playoff_pct"] == 50.0
        assert tracker.sim_calls == 1
    finally:
        for h in held:
            if h:
                appmod._BG_HEAVY_SEM.release()
