"""Unit tests for utils.lineup_slots — provider slot aliases and waiver-need math."""
import pytest

from utils.lineup_slots import (
    canonicalize_slot,
    canonicalize_slots,
    count_lineup_slots,
    is_superflex_lineup,
    starter_need_counts,
    start_sit_groups,
    start_sit_pos,
)
from utils.optimal_lineup import compute_optimal_lineup
from utils.roster_strength import derive_league_thresholds


@pytest.mark.parametrize("raw, canon", [
    ("FLEX", "FLEX"),
    ("WRRB_FLEX", "FLEX"),
    ("WR/RB/TE", "FLEX"),
    ("W/R/T", "FLEX"),
    ("REC_FLEX", "FLEX"),
    ("SUPER_FLEX", "SUPER_FLEX"),
    ("SFLEX", "SUPER_FLEX"),
    ("OP", "SUPER_FLEX"),
    ("Q/W/R/T", "SUPER_FLEX"),
    ("DST", "DEF"),
    ("D/ST", "DEF"),
    ("D-ST", "DEF"),
    ("RB/WR/TE", "FLEX"),
    ("QB/RB/WR/TE", "SUPER_FLEX"),
    ("RB/WR/TE/QB", "SUPER_FLEX"),
    ("BE", "BN"),
    ("QB", "QB"),
])
def test_canonicalize_provider_aliases(raw, canon):
    assert canonicalize_slot(raw) == canon


def test_count_lineup_slots_collapses_aliases():
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "WRRB_FLEX", "OP", "DST", "BN"]
    counts = count_lineup_slots(slots)
    assert counts["QB"] == 1
    assert counts["RB"] == 2
    assert counts["FLEX"] == 1
    assert counts["SUPER_FLEX"] == 1
    assert counts["DEF"] == 1
    assert counts["BN"] == 1
    assert is_superflex_lineup(slots) is True


def test_standard_league_is_not_superflex():
    assert is_superflex_lineup(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]) is False


def test_starter_need_does_not_double_count_flex():
    # 1QB / 2RB / 2WR / 1TE / 1FLEX: flex is split (WR gets the odd slot), plus
    # one depth player at each position. Superflex is NOT added to RB/WR.
    std = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"])
    assert std == {"QB": 2, "RB": 3, "WR": 4, "TE": 2}

    # Old waiver math added the full flex count to BOTH RB and WR (+1 each) and
    # Superflex to RB/WR too, producing RB=4 / WR=4 even in a 1-flex 1QB league.
    sf = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"])
    assert sf["QB"] == 3          # 1 QB + 1 SF + 1 depth
    assert sf["RB"] == 3          # not inflated by the superflex slot
    assert sf["WR"] == 4
    assert sf["TE"] == 2


def test_starter_need_espn_op_alias_counts_as_superflex():
    espn = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "OP"])
    sleeper = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"])
    assert espn == sleeper


def test_optimal_lineup_op_alias_matches_super_flex():
    pts = {"qb1": 28, "qb2": 22, "rb1": 18}
    pos = {"qb1": "QB", "qb2": "QB", "rb1": "RB"}
    a = compute_optimal_lineup(pts, pos, ["QB", "SUPER_FLEX"], list(pts))
    b = compute_optimal_lineup(pts, pos, ["QB", "OP"], list(pts))
    assert a == b
    assert a[0] == {"qb1", "qb2"}


def test_optimal_lineup_wrrb_flex_alias():
    pts = {"rb1": 30, "rb2": 25, "wr1": 20, "te1": 10}
    pos = {"rb1": "RB", "rb2": "RB", "wr1": "WR", "te1": "TE"}
    a = compute_optimal_lineup(pts, pos, ["RB", "WR", "FLEX"], list(pts))
    b = compute_optimal_lineup(pts, pos, ["RB", "WR", "WRRB_FLEX"], list(pts))
    assert a == b
    assert a[0] == {"rb1", "wr1", "rb2"}


def test_optimal_lineup_dst_alias_fills_def_slot():
    pts = {"d1": 12, "d2": 4, "qb": 20}
    pos = {"d1": "DST", "d2": "DEF", "qb": "QB"}
    starters, total = compute_optimal_lineup(pts, pos, ["QB", "DST"], list(pts))
    assert starters == {"qb", "d1"}
    assert total == 32.0


def test_canonicalize_slots_drops_empty():
    assert canonicalize_slots(["QB", "", None, "flex"]) == ["QB", "FLEX"]


def test_derive_thresholds_treats_op_as_superflex():
    a = derive_league_thresholds(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "OP"], 12)
    b = derive_league_thresholds(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"], 12)
    assert a == b
    # Superflex raises the QB depth floor by 1 vs a 1QB league.
    _, floor_1qb = derive_league_thresholds(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"], 12)
    _, floor_sf = b
    assert floor_sf["QB"] == floor_1qb["QB"] + 1


def test_start_sit_pos_maps_dst_to_def():
    assert start_sit_pos("DST") == "DEF"
    assert start_sit_pos("D/ST") == "DEF"
    assert start_sit_pos("K") == "K"
    assert start_sit_pos("WR") == "WR"
    assert start_sit_pos("LB") == ""


def test_start_sit_groups_include_k_and_def_when_started():
    skill = start_sit_groups(roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"])
    assert skill == ["QB", "RB", "WR", "TE"]
    espn = start_sit_groups(roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "D/ST", "BN"])
    assert espn == ["QB", "RB", "WR", "TE", "K", "DEF"]
