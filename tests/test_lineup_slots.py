"""Unit tests for utils.lineup_slots — provider slot aliases and waiver-need math."""
import pytest

from utils.lineup_slots import (
    canonicalize_slot,
    canonicalize_slots,
    count_lineup_slots,
    flex_count,
    is_superflex_lineup,
    slot_eligible_positions,
    starter_need_counts,
    start_sit_groups,
    start_sit_pos,
)
from utils.optimal_lineup import compute_optimal_lineup
from utils.roster_strength import derive_league_thresholds


@pytest.mark.parametrize("raw, canon", [
    ("FLEX", "FLEX"),
    ("WRRB_FLEX", "RB_WR"),
    ("WR/RB/TE", "FLEX"),
    ("W/R/T", "FLEX"),
    ("W/R", "RB_WR"),
    ("REC_FLEX", "WR_TE"),
    ("SUPER_FLEX", "SUPER_FLEX"),
    ("SFLEX", "SUPER_FLEX"),
    ("OP", "SUPER_FLEX"),
    ("Q/W/R/T", "SUPER_FLEX"),
    ("DST", "DEF"),
    ("D/ST", "DEF"),
    ("D-ST", "DEF"),
    ("RB/WR/TE", "FLEX"),
    ("W/T", "WR_TE"),
    ("R/T", "RB_TE"),
    ("RB/WR", "RB_WR"),
    ("WR/TE", "WR_TE"),
    ("RB+WR", "RB_WR"),
    ("QB/RB/WR/TE", "SUPER_FLEX"),
    ("RB/WR/TE/QB", "SUPER_FLEX"),
    ("BE", "BN"),
    ("QB", "QB"),
])
def test_canonicalize_provider_aliases(raw, canon):
    assert canonicalize_slot(raw) == canon


def test_count_roster_positions_delegates_to_canonical_counts():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "utils" / "utils.py").read_text(encoding="utf-8")
    assert "count_lineup_slots(positions)" in src


def test_count_lineup_slots_collapses_aliases():
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "WRRB_FLEX", "OP", "DST", "BN"]
    counts = count_lineup_slots(slots)
    assert counts["QB"] == 1
    assert counts["RB"] == 2
    assert counts.get("FLEX", 0) == 0
    assert counts["RB_WR"] == 1
    assert counts["SUPER_FLEX"] == 1
    assert counts["DEF"] == 1
    assert counts["BN"] == 1
    assert is_superflex_lineup(slots) is True
    assert flex_count(slots) == 0


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


def test_optimal_lineup_wrrb_flex_excludes_te():
    pts = {"rb1": 30, "wr1": 20, "te1": 28, "rb2": 15}
    pos = {"rb1": "RB", "wr1": "WR", "te1": "TE", "rb2": "RB"}
    flex = compute_optimal_lineup(pts, pos, ["RB", "WR", "FLEX"], list(pts))
    wrrb = compute_optimal_lineup(pts, pos, ["RB", "WR", "WRRB_FLEX"], list(pts))
    assert flex[0] == {"rb1", "wr1", "te1"}
    assert wrrb[0] == {"rb1", "wr1", "rb2"}
    assert "te1" not in wrrb[0]


def test_optimal_lineup_rec_flex_excludes_rb():
    pts = {"wr1": 22, "te1": 18, "rb1": 30}
    pos = {"wr1": "WR", "te1": "TE", "rb1": "RB"}
    rec = compute_optimal_lineup(pts, pos, ["WR", "REC_FLEX"], list(pts))
    assert rec[0] == {"wr1", "te1"}
    assert "rb1" not in rec[0]


def test_slot_eligible_positions_restricted_flex():
    assert slot_eligible_positions("WRRB_FLEX") == frozenset({"RB", "WR"})
    assert slot_eligible_positions("W/R") == frozenset({"RB", "WR"})
    assert slot_eligible_positions("REC_FLEX") == frozenset({"WR", "TE"})
    assert slot_eligible_positions("W/T") == frozenset({"WR", "TE"})
    assert slot_eligible_positions("R/T") == frozenset({"RB", "TE"})
    assert slot_eligible_positions("FLEX") == frozenset({"RB", "WR", "TE"})


def test_starter_need_restricted_flex_splits_eligible_positions():
    wrrb = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "WRRB_FLEX"])
    flex = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"])
    assert wrrb["RB"] == flex["RB"]
    assert wrrb["WR"] == flex["WR"]
    assert wrrb["TE"] == flex["TE"]
    rec = starter_need_counts(["QB", "RB", "RB", "WR", "WR", "TE", "REC_FLEX"])
    assert rec["TE"] == flex["TE"] + 1


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
