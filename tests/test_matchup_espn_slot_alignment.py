"""ESPN matchup-board lineup alignment: slot order + short starter lists.

Follow-up to #2077 (which only preserved blank-slot PLACEHOLDERS). Kaedon's
screenshot proved two remaining bugs on the ESPN "My Matchup" board:

1. ESPN's adapter appends only real players to the starters list, so an
   empty slot yields a SHORTER list with no placeholder. Pairing by index
   then renders every player below the gap one row too low.
2. ESPN reports slots in slot-id order
   (QB, RB, RB, WR, WR, TE, DEF, K, BN.., IR, FLEX), so the FLEX chip
   rendered as D/ST and a phantom IR slot shifted the K/DEF rows.

Covers _canonical_slot_order, _realign_starters_to_slots and _slot_chip in
dashboard_services.matchups.
"""
from itertools import zip_longest

import dashboard_services.matchups as matchups


def _p(name, pos):
    return {"pid": name, "name": name, "pos": pos, "nfl": "FA", "pts": 0.0}


# The exact ESPN expansion for a standard 1-QB league:
# lineupSlotCounts in slot-id order -> mapped names.
ESPN_SLOTS = [
    "QB", "RB", "RB", "WR", "WR", "TE", "DEF", "K",
    "BN", "BN", "BN", "BN", "BN", "BN", "BN", "IR", "FLEX",
]


def test_canonical_slot_order_espn():
    assert matchups._canonical_slot_order(ESPN_SLOTS) == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
    ]


def test_canonical_slot_order_sleeper_unchanged():
    # Providers that already report display order must be untouched: the
    # sort is stable, so equal-rank entries keep their relative order.
    sleeper = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
               "BN", "BN", "BN", "BN", "BN", "BN"]
    assert matchups._canonical_slot_order(sleeper) == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
    ]


def test_canonical_slot_order_empty():
    assert matchups._canonical_slot_order(None) == []
    assert matchups._canonical_slot_order([]) == []


def test_realign_screenshot_scenario():
    # Kaedon's screenshot: left team has an empty RB2 slot, so ESPN sent 8
    # starters for 9 slots. Adams must land on a WR row, not the RB row.
    slots = matchups._canonical_slot_order(ESPN_SLOTS)
    starters = [
        _p("Bryce Young", "QB"), _p("Derrick Henry", "RB"),
        _p("Davante Adams", "WR"), _p("Rome Odunze", "WR"),
        _p("Hunter Henry", "TE"), _p("Michael Wilson", "WR"),
        _p("Ka'imi Fairbairn", "K"), _p("Los Angeles Rams", "DEF"),
    ]
    got = matchups._realign_starters_to_slots(starters, slots)
    names = [s["name"] if s else None for s in got]
    assert names == [
        "Bryce Young", "Derrick Henry", None, "Davante Adams",
        "Rome Odunze", "Hunter Henry", "Michael Wilson",
        "Ka'imi Fairbairn", "Los Angeles Rams",
    ]


def test_realign_full_list_untouched():
    # Full (or #2077 placeholder) lists already cover the slots: pass through.
    slots = matchups._canonical_slot_order(ESPN_SLOTS)
    starters = [
        _p("QB1", "QB"), _p("RB1", "RB"), _p("RB2", "RB"),
        _p("WR1", "WR"), _p("WR2", "WR"), _p("TE1", "TE"),
        _p("FX1", "WR"), _p("K1", "K"), _p("DEF1", "DEF"),
    ]
    got = matchups._realign_starters_to_slots(starters, slots)
    assert got == starters
    placeholders = list(starters)
    placeholders[2] = None
    assert matchups._realign_starters_to_slots(placeholders, slots) == placeholders


def test_realign_superflex_qb_eligible():
    slots = ["QB", "SUPER_FLEX", "RB", "WR", "TE", "K", "DEF"]
    starters = [_p("QB1", "QB"), _p("QB2", "QB"), _p("RB1", "RB")]
    got = matchups._realign_starters_to_slots(starters, slots)
    names = [s["name"] if s else None for s in got]
    assert names == ["QB1", "QB2", "RB1", None, None, None, None]


def test_chips_for_screenshot_rows():
    # End-to-end of the board pairing: realign both sides, pair rows, and
    # assert the centre chips read the slot order, including FLEX.
    slots = matchups._canonical_slot_order(ESPN_SLOTS)
    left = [
        _p("Bryce Young", "QB"), _p("Derrick Henry", "RB"),
        _p("Davante Adams", "WR"), _p("Rome Odunze", "WR"),
        _p("Hunter Henry", "TE"), _p("Michael Wilson", "WR"),
        _p("Ka'imi Fairbairn", "K"), _p("Los Angeles Rams", "DEF"),
    ]
    right = [
        _p("Deshaun Watson", "QB"), _p("James Cook", "RB"),
        _p("Chase Brown", "RB"), _p("Rashee Rice", "WR"),
        _p("Christian Watson", "WR"), _p("Tyler Warren", "TE"),
        _p("Luther Burden", "WR"), _p("Tyler Loop", "K"),
        _p("Detroit Lions", "DEF"),
    ]
    rl = matchups._realign_starters_to_slots(left, slots)
    rr = matchups._realign_starters_to_slots(right, slots)
    chips = []
    for i, (L, R) in enumerate(zip_longest(rl, rr, fillvalue=None)):
        slot = slots[i] if i < len(slots) else ""
        lp = (L or {}).get("pos", "")
        rp = (R or {}).get("pos", "")
        chips.append(matchups._slot_chip(slot, lp, rp))
    assert [c for c, _ in chips] == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
    ]
    assert [label for _, label in chips] == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "D/ST",
    ]
    # Adams now sits on a WR chip row, and the empty RB2 row pairs
    # (None, Chase Brown) on the RB chip row.
    row_names = [
        ((L or {}).get("name"), (R or {}).get("name")) for L, R
        in zip_longest(rl, rr, fillvalue=None)
    ]
    assert row_names[2] == (None, "Chase Brown")
    assert row_names[3] == ("Davante Adams", "Rashee Rice")
    assert row_names[6] == ("Michael Wilson", "Luther Burden")
