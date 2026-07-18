"""Guards the roster-aware consolidation/distribution tiers in archetype_engine.

The tier a consolidation aims for depends on what the viewer already has at that
position: a team with only flex-worthy pieces gets steered to a pure starter,
never an unrealistic reach for an elite; a team that already rosters pure
starters can consolidate them into an elite. Both boundaries are matched to
existing app definitions:
  - elite   = the ELITE chip's per-position value-rank cutoffs
  - starter = starter-caliber value (same thresholds as the depth warnings)

Pure module-level functions, so this runs in the base suite (no Flask/pandas).
"""
from dashboard_services.archetype_engine import (
    _consolidate_target_allowed,
    _pos_category,
)
from utils.roster_strength import STARTER_THRESHOLD, derive_league_thresholds
from utils.tier_thresholds import ELITE_RANK_CUTOFFS

# 12-team, standard 1QB starting lineup (QB, 2RB, 2WR, TE, 2FLEX).
_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]
_THR, _ = derive_league_thresholds(_LINEUP, 12)


def test_elite_matches_the_chip_cutoffs():
    # A player at the chip's positional value-rank cutoff is elite; one past it
    # is not (it falls to starter/flex by value).
    for pos, cutoff in ELITE_RANK_CUTOFFS.items():
        big = STARTER_THRESHOLD[pos] * 3  # clearly elite-level value
        assert _pos_category(pos, cutoff, big, _THR) == "elite"
        assert _pos_category(pos, cutoff + 1, big, _THR) != "elite"


def test_starter_matches_the_value_threshold():
    thr = _THR["WR"]
    rank_below_elite = ELITE_RANK_CUTOFFS["WR"] + 5
    # At/above the starter-caliber value => pure starter.
    assert _pos_category("WR", rank_below_elite, thr, _THR) == "starter"
    assert _pos_category("WR", rank_below_elite, thr + 100, _THR) == "starter"
    # Below it => flex/depth.
    assert _pos_category("WR", rank_below_elite, thr - 1, _THR) == "flex"


def test_thresholds_scale_with_league_size():
    # Larger leagues dilute talent, so the starter bar drops.
    thr_16, _ = derive_league_thresholds(_LINEUP, 16)
    thr_8, _ = derive_league_thresholds(_LINEUP, 8)
    assert thr_16["WR"] < thr_8["WR"]


def test_unrostered_or_valueless_is_depth():
    assert _pos_category("WR", None, 999, _THR) == "depth"


def test_flex_only_team_cannot_reach_an_elite():
    assert _consolidate_target_allowed("elite", "flex") is False
    assert _consolidate_target_allowed("elite", "depth") is False


def test_team_with_a_starter_can_consolidate_into_an_elite():
    assert _consolidate_target_allowed("elite", "starter") is True
    assert _consolidate_target_allowed("elite", "elite") is True


def test_pure_starter_targets_are_always_allowed():
    for best in ("depth", "flex", "starter", "elite"):
        assert _consolidate_target_allowed("starter", best) is True
        assert _consolidate_target_allowed("flex", best) is True
