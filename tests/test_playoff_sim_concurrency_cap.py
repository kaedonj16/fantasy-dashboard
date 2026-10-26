"""Background playoff sims must be concurrency-capped per process.

_playoff_sim_cached(block=False) kicks a 10k-iteration Monte Carlo sim on a
cache miss. Single-league surfaces kick one at a time, but My Leagues fans
out to every linked league at once (summary builds, per-league cards and
the actions scan in one page load) -- uncapped, that storm OOM-restarted
the web instance seconds after the first portfolio load on the v25 deploy
(2026-10-25). Excess kicks are dropped, not queued, and the dropped league
leaves the BUILDING set so a later read can re-kick it.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("numpy")


def _ctx(league_id):
    return {
        "league_id": league_id,
        "season": 2026,
        "current_week": 8,
        "rosters": [
            {"roster_id": 1, "players": ["p1", "p2"]},
            {"roster_id": 2, "players": ["p3", "p4"]},
        ],
    }


def _drain_sim_slots(app, timeout=60.0):
    """Wait until every background-sim slot is free.

    Tests share one process per CI shard, and an earlier test may have
    kicked REAL sims that still hold slots (a 10k run on a loaded CI box
    takes seconds). Without the drain, this test's kicks could all be
    dropped by the cap before the fake sim is even installed.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = [
            app._PLAYOFF_SIM_SEM.acquire(blocking=False)
            for _ in range(app._PLAYOFF_SIM_MAX_CONCURRENT)
        ]
        for g in got:
            if g:
                app._PLAYOFF_SIM_SEM.release()
        if all(got):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture()
def sim_gate(monkeypatch):
    """Replace the Monte Carlo sim with a gated fake that records overlap."""
    import app
    import data_building.simulate_playoff_odds as sim_mod

    assert _drain_sim_slots(app), "background sim slots never freed"
    # The cache signature normally fingerprints the published schedule,
    # which can do slow provider/DB I/O; it is irrelevant to the cap, so
    # pin it to keep this test about concurrency only.
    monkeypatch.setattr(app, "_playoff_sim_sig", lambda ctx, platform: "test-sig")

    gate = threading.Event()
    state = {"started": 0, "running": 0, "max_running": 0}
    lock = threading.Lock()

    def fake_sim(ctx, platform="sleeper", **_kw):
        with lock:
            state["started"] += 1
            state["running"] += 1
            state["max_running"] = max(state["max_running"], state["running"])
        try:
            gate.wait(timeout=120)
            return [{"roster_id": 1, "playoff_pct": 50.0}]
        finally:
            with lock:
                state["running"] -= 1

    monkeypatch.setattr(sim_mod, "simulate_playoff_odds", fake_sim)
    app._PLAYOFF_SIM_CACHE.clear()
    with app._PLAYOFF_SIM_BUILDING_LOCK:
        app._PLAYOFF_SIM_BUILDING.clear()
    yield app, state, gate
    gate.set()
    # Let any kicked threads drain so the semaphore is fully released for
    # other tests sharing this process.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with app._PLAYOFF_SIM_BUILDING_LOCK:
            if not app._PLAYOFF_SIM_BUILDING:
                break
        time.sleep(0.05)
    app._PLAYOFF_SIM_CACHE.clear()


def _wait_until(pred, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_background_sim_kicks_are_capped_and_dropped_kicks_can_rekick(sim_gate):
    app, state, gate = sim_gate
    cap = app._PLAYOFF_SIM_MAX_CONCURRENT
    assert cap >= 1

    # A portfolio-style fan-out: many leagues miss the sim cache at once.
    league_ids = [f"cap-fanout-{i}" for i in range(cap + 4)]
    results = {}

    def _kick(lid):
        results[lid] = app._playoff_sim_cached(_ctx(lid), "sleeper", block=False)

    threads = [threading.Thread(target=_kick, args=(lid,)) for lid in league_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(not t.is_alive() for t in threads)
    assert all(v == [] for v in results.values())

    # Exactly `cap` sims started and are held by the gate; the rest were
    # dropped, not queued (started can never pass cap while slots are held).
    assert _wait_until(lambda: state["started"] == cap)
    time.sleep(0.3)
    assert state["started"] == cap
    assert state["running"] == cap
    assert state["max_running"] <= cap

    # Dropped leagues are not stuck in the BUILDING set: they can re-kick
    # once a slot frees (this is what lets their odds appear on a later
    # portfolio load instead of never).
    dropped = league_ids[cap:]
    for lid in dropped:
        key = ("sleeper", lid, 2026, "v3")
        with app._PLAYOFF_SIM_BUILDING_LOCK:
            assert key not in app._PLAYOFF_SIM_BUILDING

    gate.set()
    assert _wait_until(lambda: state["running"] == 0)

    def _my_keys_cleared():
        with app._PLAYOFF_SIM_BUILDING_LOCK:
            return all(
                ("sleeper", lid, 2026, "v3") not in app._PLAYOFF_SIM_BUILDING
                for lid in league_ids[:cap]
            )

    assert _wait_until(_my_keys_cleared)

    # The finished sims populated the cache (served without a new sim run).
    for lid in league_ids[:cap]:
        rows = app._playoff_sim_cached(_ctx(lid), "sleeper", block=False)
        assert rows and rows[0]["playoff_pct"] == 50.0
    assert state["started"] == cap

    # A dropped league re-kicks now that slots are free, and lands in cache.
    app._playoff_sim_cached(_ctx(dropped[0]), "sleeper", block=False)
    assert _wait_until(lambda: state["started"] == cap + 1)
    assert _wait_until(
        lambda: bool(app._playoff_sim_cached(_ctx(dropped[0]), "sleeper", block=False))
    )
