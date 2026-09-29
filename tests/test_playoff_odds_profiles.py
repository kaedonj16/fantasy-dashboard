"""Tests for the playoff-odds integration of the shared player distribution model.

Phase 1: per-player profile stds replace the generic position formula inside
_team_std_from_starters, threaded through _lineup_with_replacements ->
_team_week_profile -> _compute_week_profiles. All data access is monkeypatched.
"""

import pytest

# simulate_playoff_odds imports numpy at module load. The fast "lint" CI shard
# has flask but not numpy, so guard on numpy first — otherwise importing the
# sim module here raises at COLLECTION and aborts the whole run. The full
# integration shard has numpy and runs these tests normally.
pytest.importorskip("numpy")

import data_building.player_distributions as pdist
import data_building.simulate_playoff_odds as spo


@pytest.fixture
def fake_profiles(monkeypatch):
    """build_profiles returns a fixed std per player id."""
    def fake(requests, season, week, ctx=None):
        return {
            r["player_id"]: {
                "player_id": r["player_id"],
                "pos": r["pos"],
                "mean": r["mean"],
                "std": {"p1": 9.0, "p2": 2.0}.get(r["player_id"], 5.0),
                "skew_alpha": 2.0, "dud_risk": 0.0, "n_games": 5.0, "factors": {},
            }
            for r in requests
        }
    monkeypatch.setattr(pdist, "build_profiles", fake)
    monkeypatch.setattr(pdist, "profile_inputs_signature",
                        lambda season, week: f"sig-{season}-{week}")
    return fake


def _ppg_map():
    return {
        "p1": {"ppg": 15.0, "pos": "WR"},
        "p2": {"ppg": 12.0, "pos": "RB"},
        "p3": {"ppg": 8.0, "pos": "WR"},
    }


def test_lineup_with_replacements_carries_pids():
    total, starters, repls = spo._lineup_with_replacements(
        ["p1", "p2", "p3"], _ppg_map(), {}, ["WR", "RB", "BN", "BN"])
    pids = {s[2] for s in starters}
    assert pids <= {"p1", "p2", "p3"}
    assert len(pids) == len(starters)  # no duplicates
    assert total == pytest.approx(sum(s[1] for s in starters))


def test_team_std_uses_profile_stds(fake_profiles):
    starters = [("WR", 15.0, "p1"), ("RB", 12.0, "p2")]
    pmap = spo._profile_std_map(["p1", "p2"], _ppg_map(), {}, 2026, 4)
    assert pmap == {"p1": 9.0, "p2": 2.0}
    with_profiles = spo._team_std_from_starters(starters, pmap)
    without = spo._team_std_from_starters(starters, None)
    # Position formula: WR 0.50*15+2=9.5, RB 0.45*12+2=7.4 -> sqrt(9.5^2+7.4^2)
    assert without == pytest.approx((9.5 ** 2 + 7.4 ** 2) ** 0.5)
    # Profiles: 9.0 and 2.0 -> sqrt(81+4)
    assert with_profiles == pytest.approx((9.0 ** 2 + 2.0 ** 2) ** 0.5)
    assert with_profiles < without


def test_team_std_accepts_legacy_2tuples():
    starters = [("WR", 15.0), ("RB", 12.0)]
    assert spo._team_std_from_starters(starters) == pytest.approx(
        (9.5 ** 2 + 7.4 ** 2) ** 0.5)


def test_team_week_profile_threads_map(fake_profiles):
    ppg_map = _ppg_map()
    pmap = spo._profile_std_map(["p1", "p2", "p3"], ppg_map, {}, 2026, 4)
    prof = spo._team_week_profile(
        ["p1", "p2", "p3"], ppg_map, {}, ["WR", "RB", "BN", "BN"],
        hist_avg=0.0, hist_std=0.0, projection_weight=1.0,
        profile_std_map=pmap)
    no_prof = spo._team_week_profile(
        ["p1", "p2", "p3"], ppg_map, {}, ["WR", "RB", "BN", "BN"],
        hist_avg=0.0, hist_std=0.0, projection_weight=1.0)
    assert prof["std"] != no_prof["std"]
    assert prof["mean"] == no_prof["mean"]  # means untouched by profiles


def test_profile_std_map_never_raises(monkeypatch):
    def boom(requests, season, week, ctx=None):
        raise RuntimeError("nope")
    monkeypatch.setattr(pdist, "build_profiles", boom)
    assert spo._profile_std_map(["p1"], _ppg_map(), {}, 2026, 4) is None


def test_ctx_signature_includes_profile_inputs(fake_profiles):
    ctx = {"league_id": "1", "season": 2026, "current_week": 4,
           "league_settings": {}, "rosters": []}
    sig = spo._ctx_signature(ctx, "sleeper")
    assert sig.endswith(":sig-2026-4")
