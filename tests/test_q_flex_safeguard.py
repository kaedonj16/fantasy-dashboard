"""Q-flex safeguard: Questionable positional starters should be suggested at FLEX.

If a Q player is ruled out, a FLEX slot can be filled by any RB/WR/TE from the
bench, while a positional slot only accepts that one position. So when a Q
player earns a positional start and a non-Q player of the same position is
starting in FLEX, their slots swap. The starter SET never changes, only the
badges move.
"""

import pytest

pytest.importorskip("pandas")

from app import _apply_q_flex_safeguard


def _p(name, pos, *, start=False, flex_start=False, on_bye=False, injury=""):
    return {
        "player_id": name,
        "name": name,
        "start": start,
        "flex_start": flex_start,
        "flex_eligible": True,
        "on_bye": on_bye,
        "injury_status": injury,
        "_score": 10.0,
    }


def _lineup():
    # 2 WR + 1 FLEX: Q WR1 starts at WR, healthy WR3 starts at FLEX.
    return {
        "WR": [
            _p("wr1", "WR", start=True, injury="Q"),
            _p("wr2", "WR", start=True),
            _p("wr3", "WR", start=True, flex_start=True),
            _p("wr4", "WR"),
        ],
        "RB": [_p("rb1", "RB", start=True), _p("rb2", "RB", start=True)],
    }


def test_q_positional_starter_swaps_into_flex():
    positions = _lineup()
    swaps = _apply_q_flex_safeguard(positions, flex_slots=1)
    assert swaps == 1
    wr1, wr3 = positions["WR"][0], positions["WR"][2]
    assert wr1["flex_start"] is True
    assert wr3["flex_start"] is False
    # Both still start: the starter set is unchanged, only slots moved.
    assert wr1["start"] is True and wr3["start"] is True
    assert wr1["flex_safeguard"] == {"from_pos": "WR", "via": "wr3"}


def test_no_flex_slots_is_noop():
    positions = _lineup()
    assert _apply_q_flex_safeguard(positions, flex_slots=0) == 0
    assert positions["WR"][0]["flex_start"] is False
    assert "flex_safeguard" not in positions["WR"][0]


def test_q_already_in_flex_is_noop():
    positions = _lineup()
    positions["WR"][0]["flex_start"] = True  # Q player already safeguarded
    positions["WR"][2]["flex_start"] = False
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 0


def test_no_same_position_trade_partner_is_noop():
    positions = _lineup()
    # The only WR in FLEX is also Q: no healthy same-position partner to trade with.
    positions["WR"][2]["injury_status"] = "Questionable"
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 0
    assert positions["WR"][0]["flex_start"] is False


def test_doubtful_is_not_questionable():
    positions = _lineup()
    positions["WR"][0]["injury_status"] = "D"
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 0


def test_gtd_counts_as_questionable():
    positions = _lineup()
    positions["WR"][0]["injury_status"] = "GTD"
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 1
    assert positions["WR"][0]["flex_start"] is True


def test_bye_q_player_untouched():
    positions = _lineup()
    positions["WR"][0]["on_bye"] = True
    positions["WR"][0]["start"] = False
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 0


def test_healthy_positional_starter_untouched():
    positions = {
        "WR": [
            _p("wr1", "WR", start=True),
            _p("wr2", "WR", start=True),
            _p("wr3", "WR", start=True, flex_start=True),
        ],
    }
    assert _apply_q_flex_safeguard(positions, flex_slots=1) == 0
    assert all("flex_safeguard" not in p for p in positions["WR"])
