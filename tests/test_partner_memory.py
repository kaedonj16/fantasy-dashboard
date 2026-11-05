"""Frontier 4: per-manager acceptance memory (utils/partner_memory.py).

Priors come from the league's own completed trades only, are
sample-gated, and are bounded. Under the gate, or with no history,
the layer is exactly silent: acceptance runs as it did in Wave 2.
"""
from utils.partner_memory import (
    MIN_SAMPLE,
    completed_trades,
    partner_acceptance_adj,
    partner_profile,
)


def _trade(sides, status="complete"):
    """sides: {roster_id: (gets, sends)} in Sleeper transaction shape."""
    adds, drops = {}, {}
    for rid, (gets, sends) in sides.items():
        for pid in gets:
            adds[pid] = rid
        for pid in sends:
            drops[pid] = rid
    return {"type": "trade", "status": status,
            "roster_ids": list(sides), "adds": adds, "drops": drops,
            "draft_picks": []}


def _history():
    """Partner 2 in four completed trades; partner 3 in two."""
    return [
        # 2 receives a QB (q1), sends a WR.
        _trade({2: (["q1"], ["w1"]), 5: (["w1"], ["q1"])}),
        # 2 receives a QB again in a 2-for-1 he took (sent 1, got 2).
        _trade({2: (["q2", "r9"], ["w2"]), 6: (["w2"], ["q2", "r9"])}),
        # 2 sends a QB this time, receives two pieces.
        _trade({2: (["w3", "t1"], ["q3"]), 7: (["q3"], ["w3", "t1"])}),
        # 2 in a straight 1-for-1, no QB.
        _trade({2: (["r1"], ["w4"]), 8: (["w4"], ["r1"])}),
        _trade({3: (["r2"], ["w5"]), 9: (["w5"], ["r2"])}),
        _trade({3: (["r3"], ["w6"]), 10: (["w6"], ["r3"])}),
        # Not completed / not a trade: must not count.
        _trade({2: (["q9"], ["w9"]), 4: (["w9"], ["q9"])},
               status="failed"),
        {"type": "waiver", "status": "complete", "adds": {"x": 2}},
    ]


POS = {"q1": "QB", "q2": "QB", "q3": "QB", "w1": "WR", "w2": "WR",
       "w3": "WR", "w4": "WR", "r1": "RB", "r9": "RB", "t1": "TE",
       "r2": "RB", "r3": "RB", "w5": "WR", "w6": "WR"}


def test_completed_trades_skips_failed_and_non_trades():
    trades = completed_trades(_history())
    assert len(trades) == 6
    assert all(len(t["sides"]) == 2 for t in trades)


def test_profile_is_sample_gated():
    assert partner_profile(_history(), 3, positions_by_id=POS) is None
    profile = partner_profile(_history(), 2, positions_by_id=POS)
    assert profile["sample"] == 4
    assert profile["qb_received_share"] == 0.5    # 2 of 4 brought a QB
    assert profile["uneven_taken_share"] == 0.5   # two 2-for-1s taken
    assert profile["value_tilt"] is None          # no values supplied


def test_value_tilt_uses_current_values_and_says_so():
    values = {pid: 100.0 for pid in POS}
    values.update({"q1": 300.0, "q2": 300.0, "r9": 100.0})
    profile = partner_profile(_history(), 2, values_by_id=values,
                              positions_by_id=POS)
    # He received 300/100/200/100 and sent 100/100/300/100 -> tilt < 1.
    assert profile["value_tilt"] is not None
    assert profile["value_tilt"] < 0.9
    assert profile["value_basis"] == "current_values"


def test_adj_qb_appetite_and_shape_history():
    profile = partner_profile(_history(), 2, positions_by_id=POS)
    points, notes = partner_acceptance_adj(
        profile, partner_receives_positions=["QB"],
        partner_sends_count=1, partner_receives_count=1)
    assert points == 4
    assert any("QB" in n for n in notes)
    # 2-for-1 against him: he has taken the shape -> +3, stacked.
    points, notes = partner_acceptance_adj(
        profile, partner_receives_positions=["RB", "WR"],
        partner_sends_count=1, partner_receives_count=2)
    assert points == 3
    assert any("2-for-1" in n for n in notes)


def test_adj_never_took_2for1_costs_points():
    history = [
        _trade({2: (["r1"], ["w1"]), 5: (["w1"], ["r1"])})
        for _ in range(4)
    ]
    profile = partner_profile(history, 2, positions_by_id=POS)
    points, notes = partner_acceptance_adj(
        profile, partner_receives_positions=["RB", "WR"],
        partner_sends_count=1, partner_receives_count=2)
    assert points == -4
    assert any("never taken a 2-for-1" in n for n in notes)


def test_adj_silent_without_profile():
    assert partner_acceptance_adj(None) == (0, [])
    assert partner_acceptance_adj(
        {"sample": MIN_SAMPLE, "qb_received_share": None,
         "uneven_taken_share": 0.0, "value_tilt": None},
        partner_receives_positions=["QB"],
        partner_sends_count=2, partner_receives_count=1) == (0, [])
