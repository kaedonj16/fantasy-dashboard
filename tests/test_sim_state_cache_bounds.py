"""Sim-state cache memory bounds: small caps, freeze-free cached states.

The playoff sim stack's in-process caches were a prime suspect in the
2026-11-15/16 web OOM restarts: up to 64 cached states, each able to pin a
several-MB ``_mc_freeze`` because ``_ensure_freeze`` mutates the shared
cached dict. These guards lock in the bounded shape: a 12-entry state cap,
cached states that never carry a freeze, a tiny separate freeze store the
impact warm registers into, and on-demand freeze regeneration (same seed,
identical numbers) for everyone else.
"""
from __future__ import annotations

import time

import pytest

np = pytest.importorskip("numpy")

from data_building import simulate_playoff_odds as spo


def _league_state(n_teams=12, n_weeks=14, seed=1234):
    """A minimal buildable sim state (same shape as test_playoff_sim_swaps)."""
    teams = [{
        "roster_id": i,
        "name": f"T{i}",
        "wins": 0, "losses": 0, "ties": 0,
        "avg": 110.0 + (i - 6) * 1.5,
        "std": 22.0,
        "pf": 0.0,
    } for i in range(1, n_teams + 1)]
    rids = [t["roster_id"] for t in teams]
    matchups = {}
    for w in range(1, n_weeks + 1):
        rotated = rids[1:]
        rot = rotated[(w - 1) % len(rotated):] + rotated[:(w - 1) % len(rotated)]
        order = [rids[0]] + rot
        matchups[w] = [(order[i], order[i + 1]) for i in range(0, n_teams, 2)]
    lost = np.array([4.0] * 9, dtype=np.float32)
    haz = np.array([0.05] * 9, dtype=np.float32)
    week_profiles = {}
    for w in range(1, n_weeks + 1):
        week_profiles[w] = {
            t["roster_id"]: {
                "mean": t["avg"], "std": t["std"], "lost": lost, "haz": haz,
            }
            for t in teams
        }
    return {
        "teams": teams,
        "matchups": matchups,
        "week_profiles": week_profiles,
        "playoff_teams": 6,
        "seed": seed,
        "roster_pid_map": {t["roster_id"]: [str(t["roster_id"])] for t in teams},
        "ppg_map": {},
        "pos_map": {},
        "roster_positions": [],
        "blend": 1.0,
        "week_ppg_maps": {},
        "hist_avg_by_rid": {t["roster_id"]: 0.0 for t in teams},
        "hist_std_by_rid": {t["roster_id"]: 0.0 for t in teams},
    }


def _ctx(league_id):
    return {
        "league_id": str(league_id),
        "season": 2026,
        "current_week": 10,
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "rosters": [
            {"roster_id": 1, "players": ["1"]},
            {"roster_id": 2, "players": ["2"]},
        ],
    }


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch):
    spo._SIM_STATE_CACHE.clear()
    spo._SIM_FREEZE_CACHE.clear()
    # Hermetic ctx signatures: no provider schedule fetch in tests.
    monkeypatch.setattr(spo, "playoff_schedule_sig", lambda ctx, platform: "sched")
    monkeypatch.setattr(
        spo, "_build_sim_state_impl",
        lambda ctx, platform="sleeper": _league_state(),
    )
    yield
    spo._SIM_STATE_CACHE.clear()
    spo._SIM_FREEZE_CACHE.clear()


def test_state_cache_cap_evicts_oldest():
    assert spo._SIM_STATE_MAX == 12
    for i in range(13):
        spo.build_sim_state(_ctx(f"L{i}"))
    assert len(spo._SIM_STATE_CACHE) == 12
    # The oldest league was evicted; the newest is served.
    assert spo.peek_sim_state(_ctx("L0")) is None
    assert spo.peek_sim_state(_ctx("L12")) is not None


def test_cached_state_never_carries_a_freeze():
    ctx = _ctx("L1")
    state = spo.build_sim_state(ctx)
    # The impact warm path: draw the baseline on the caller's copy.
    spo.run_base_simulation(state, n_sims=2000)
    assert state["_mc_freeze"]["n_sims"] == 2000
    # The cache entry itself stayed freeze-free...
    stored = next(iter(spo._SIM_STATE_CACHE.values()))[1]
    assert "_mc_freeze" not in stored
    # ...and so do later build hits and peeks (nothing registered yet).
    assert "_mc_freeze" not in spo.build_sim_state(ctx)
    assert "_mc_freeze" not in spo.peek_sim_state(ctx)


def test_registered_freeze_is_served_by_peek_and_swaps_match_direct():
    ctx = _ctx("L1")
    state = spo.build_sim_state(ctx)
    spo.run_base_simulation(state, n_sims=2000)
    spo.store_sim_freeze(ctx, "sleeper", state.get("_mc_freeze"))

    peeked = spo.peek_sim_state(ctx)
    assert peeked["_mc_freeze"]["n_sims"] == 2000
    # Still not inside the cache entry itself.
    stored = next(iter(spo._SIM_STATE_CACHE.values()))[1]
    assert "_mc_freeze" not in stored

    via_peek = spo.simulate_swap_impact(peeked, 1, ["1"], ["2"], n_sims=2000)
    direct_state = _league_state()
    spo.run_base_simulation(direct_state, n_sims=2000)
    direct = spo.simulate_swap_impact(direct_state, 1, ["1"], ["2"], n_sims=2000)
    assert via_peek["before"] == direct["before"]
    assert via_peek["after"] == direct["after"]
    assert via_peek["delta"] == direct["delta"]


def test_swap_on_freezeless_peek_regenerates_identical_results():
    ctx = _ctx("L1")
    spo.build_sim_state(ctx)
    peeked = spo.peek_sim_state(ctx)
    assert "_mc_freeze" not in peeked

    via_peek = spo.simulate_swap_impact(peeked, 1, ["1"], ["2"], n_sims=2000)
    direct_state = _league_state()
    direct = spo.simulate_swap_impact(direct_state, 1, ["1"], ["2"], n_sims=2000)
    assert via_peek["before"] == direct["before"]
    assert via_peek["after"] == direct["after"]
    assert via_peek["delta"] == direct["delta"]
    # Regeneration attached the freeze to the caller's copy only.
    assert "_mc_freeze" in peeked
    stored = next(iter(spo._SIM_STATE_CACHE.values()))[1]
    assert "_mc_freeze" not in stored


def test_expired_entries_are_swept_on_store():
    old = time.time() - spo._SIM_STATE_TTL - 1
    spo._SIM_STATE_CACHE["stale-sig"] = (old, {"teams": []})
    spo._SIM_FREEZE_CACHE["stale-sig"] = (old, {"n_sims": 2000})
    spo.build_sim_state(_ctx("L1"))
    assert "stale-sig" not in spo._SIM_STATE_CACHE
    assert len(spo._SIM_STATE_CACHE) == 1
    # A state store sweeps expired freezes too.
    assert "stale-sig" not in spo._SIM_FREEZE_CACHE


def test_freeze_store_cap_evicts_oldest():
    assert spo._SIM_FREEZE_MAX == 4
    for i in range(5):
        spo.store_sim_freeze(_ctx(f"F{i}"), "sleeper", {"n_sims": 2000})
    assert len(spo._SIM_FREEZE_CACHE) == 4
    # The oldest league's freeze is gone: its peeked state has none.
    spo.build_sim_state(_ctx("F0"))
    assert "_mc_freeze" not in spo.peek_sim_state(_ctx("F0"))
    spo.build_sim_state(_ctx("F4"))
    assert spo.peek_sim_state(_ctx("F4"))["_mc_freeze"]["n_sims"] == 2000
