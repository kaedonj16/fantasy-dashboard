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
    _positional_ranks,
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


def test_positional_ranks_are_per_position_by_value():
    vals = {
        "w1": {"position": "WR", "value": 900},
        "w2": {"position": "WR", "value": 700},
        "w3": {"position": "WR", "value": 500},
        "r1": {"position": "RB", "value": 800},
        "r2": {"position": "RB", "value": 600},
        "k1": {"position": "K", "value": 50},   # non-skill, ignored
    }
    ranks = _positional_ranks(vals)
    assert ranks["w1"] == 1 and ranks["w2"] == 2 and ranks["w3"] == 3
    assert ranks["r1"] == 1 and ranks["r2"] == 2
    assert "k1" not in ranks


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


# ── Starter-gap need model ─────────────────────────────────────────────────────
from utils.player_tiers import (
    ceiling_needs,
    roster_position_counts,
    startable_surplus,
    starter_gap_needs,
)

# 12-team league starter bars (350/350 for RB/WR at 12 teams).
_FLOOR = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}


def test_counts_split_startable_from_bench_depth():
    # Only players clearing the value bar count as starters; the rest still add to
    # the total. Three WRs but only two are startable; one flex-tier RB.
    counts = roster_position_counts(
        [("WR", 900), ("WR", 500), ("WR", 100), ("RB", 400), ("QB", 0)],
        _THR,
    )
    assert counts["WR"] == {"total": 3, "starters": 2}
    assert counts["RB"] == {"total": 1, "starters": 1}
    assert counts["QB"] == {"total": 1, "starters": 0}   # below the QB bar
    assert counts["TE"] == {"total": 0, "starters": 0}


def test_starter_gap_is_a_hole_not_low_total():
    # Flex depth never masks a hole: three below-bar WRs are still zero starters,
    # so WR is a need; two startable RBs exactly fill the RB floor (no need).
    counts = roster_position_counts(
        [("WR", 100), ("WR", 120), ("WR", 90), ("RB", 500), ("RB", 400)],
        _THR,
    )
    needs = starter_gap_needs(counts, _FLOOR)
    assert "WR" in needs
    assert "RB" not in needs


def test_needs_are_ordered_by_gap_size():
    # WR gap of 2 (floor 2, zero starters) outranks the QB gap of 1.
    counts = roster_position_counts([("RB", 500), ("RB", 450)], _THR)
    needs = starter_gap_needs(counts, _FLOOR)
    assert needs[0] == "WR" and needs.index("WR") < needs.index("QB")


def test_surplus_needs_a_startable_beyond_the_slots():
    # Two startable RBs == the floor (not surplus); a third makes it tradeable.
    two = roster_position_counts([("RB", 500), ("RB", 450)], _THR)
    assert "RB" not in startable_surplus(two, _FLOOR)
    three = roster_position_counts([("RB", 500), ("RB", 450), ("RB", 400)], _THR)
    assert "RB" in startable_surplus(three, _FLOOR)


def test_ceiling_need_is_a_filled_slot_without_an_elite():
    # WR slot filled (two startable) but the best is only a starter -> ceiling gap;
    # an elite best is no gap; an unfilled slot is a hole, not a ceiling need.
    counts = roster_position_counts([("WR", 500), ("WR", 450), ("RB", 900)], _THR)
    assert "WR" in ceiling_needs(counts, {"WR": "starter", "RB": "elite"}, _FLOOR)
    assert "WR" not in ceiling_needs(counts, {"WR": "elite", "RB": "elite"}, _FLOOR)
    thin = roster_position_counts([("WR", 500)], _THR)  # only 1 startable, floor 2
    assert "WR" not in ceiling_needs(thin, {"WR": "starter"}, _FLOOR)
