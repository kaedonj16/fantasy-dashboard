"""Frozen-opponent swap sims: no-op is exact, upgrades raise odds, freeze is reused.

The suggestion engine used to replay a full-league 10k Monte Carlo for every
package. These guards lock in the freeze + 1–2-team overlay path that replaced
that, without pinning exact playoff percentages (the RNG stream is per-team).
"""
from __future__ import annotations

import copy

import pytest

np = pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import (
    _ensure_freeze,
    _run_mc,
    _simulate_week_scores,
    run_base_simulation,
    simulate_with_swap,
)


def _league(n_teams=12, n_weeks=14, n_starters=9, seed=1234):
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
    lost = np.array([4.0] * n_starters, dtype=np.float32)
    haz = np.array([0.05] * n_starters, dtype=np.float32)
    week_profiles = {}
    for w in range(1, n_weeks + 1):
        week_profiles[w] = {
            t["roster_id"]: {
                "mean": t["avg"], "std": t["std"], "lost": lost, "haz": haz,
            }
            for t in teams
        }
    # Each team "owns" a dummy player id equal to its roster id so swaps can
    # move a token without needing a real PPG map.
    roster_pid_map = {t["roster_id"]: [str(t["roster_id"])] for t in teams}
    return {
        "teams": teams,
        "matchups": matchups,
        "week_profiles": week_profiles,
        "playoff_teams": 6,
        "seed": seed,
        "roster_pid_map": roster_pid_map,
        "ppg_map": {},
        "pos_map": {},
        "roster_positions": [],
        "blend": 1.0,
        "week_ppg_maps": {},
        "hist_avg_by_rid": {t["roster_id"]: 0.0 for t in teams},
        "hist_std_by_rid": {t["roster_id"]: 0.0 for t in teams},
    }


def test_stronger_teams_have_higher_playoff_odds():
    lg = _league()
    rows = _run_mc(
        lg["teams"], lg["matchups"], lg["week_profiles"],
        lg["playoff_teams"], 2000, lg["seed"],
    )
    by_id = {r["roster_id"]: r["playoff_pct"] for r in rows}
    assert by_id[12] > by_id[1]
    assert by_id[10] > by_id[3]


def test_noop_swap_is_exact_zero_delta():
    lg = _league()
    base = run_base_simulation(lg, n_sims=2000)
    after_pct, new_avg = simulate_with_swap(
        lg, 1, lg["roster_pid_map"][1], n_sims=2000,
    )
    assert after_pct == base[1]
    # No week_ppg_maps → new_avg falls back to 0, which is fine; the playoff
    # % is the CRN contract we care about.
    assert new_avg == 0.0 or isinstance(new_avg, float)


def test_boosting_viewer_mean_raises_playoff_odds():
    lg = _league()
    base = run_base_simulation(lg, n_sims=2000)
    # Lift team 1's weekly mean without touching anyone else.
    boosted = copy.deepcopy(lg)
    for w, wp in boosted["week_profiles"].items():
        row = dict(wp[1])
        row["mean"] = row["mean"] + 15.0
        boosted["week_profiles"][w] = dict(wp)
        boosted["week_profiles"][w][1] = row
    # Drop any freeze copied from the original so the boosted profiles are used.
    boosted.pop("_mc_freeze", None)
    boosted_odds = run_base_simulation(boosted, n_sims=2000)
    assert boosted_odds[1] > base[1]


def test_swap_overlay_does_not_replay_the_league(monkeypatch):
    """The second simulate_with_swap must not call _simulate_week_scores again."""
    lg = _league()
    run_base_simulation(lg, n_sims=2000)
    calls = {"n": 0}
    real = _simulate_week_scores

    def _count(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "data_building.simulate_playoff_odds._simulate_week_scores", _count,
    )
    # Move player "2" onto team 1 (and team 2 receives "1") — a real two-team overlay.
    simulate_with_swap(lg, 1, ["1", "2"], n_sims=2000)
    simulate_with_swap(lg, 1, ["1", "3"], n_sims=2000)
    assert calls["n"] == 0, "freeze should have been reused; league was re-simulated"


def test_freeze_is_keyed_by_n_sims():
    lg = _league()
    a = _ensure_freeze(lg, 2000)
    assert a["n_sims"] == 2000
    b = _ensure_freeze(lg, 2000)
    assert a is b
    c = _ensure_freeze(lg, 500)
    assert c["n_sims"] == 500
    assert c is not a
