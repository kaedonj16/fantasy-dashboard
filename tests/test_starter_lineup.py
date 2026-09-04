"""Slot-legal starter derivation for hosts without live lineup flags."""
from utils.starter_lineup import derive_starters_from_slots, starter_slots


def test_starter_slots_drop_bench_and_map_superflex():
    assert starter_slots(["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "IR"]) == [
        "QB", "RB", "WR", "TE", "FLEX", "SF",
    ]


def test_derive_starters_fills_most_restrictive_slots():
    ids = ["qb1", "rb1", "rb2", "wr1", "wr2", "te1", "bn1"]
    pos = {
        "qb1": "QB", "rb1": "RB", "rb2": "RB", "wr1": "WR",
        "wr2": "WR", "te1": "TE", "bn1": "WR",
    }
    out = derive_starters_from_slots(ids, ["QB", "RB", "WR", "TE", "BN"], pos)
    assert out == ["qb1", "rb1", "wr1", "te1"]
    assert "bn1" not in out


def test_derive_starters_empty_without_ids_or_slots():
    assert derive_starters_from_slots([], ["QB"], {"1": "QB"}) == []
    assert derive_starters_from_slots(["1"], [], {"1": "QB"}) == []
