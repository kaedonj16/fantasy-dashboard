"""Unit tests for utils.waiver_lineup — roster-aware, lineup-based waiver
evaluation (item 1). Pure logic reusing the shared optimal-lineup solver."""
import pytest

from utils.waiver_lineup import (
    MIN_MEANINGFUL_GAIN,
    evaluate_pickup,
    evaluate_pickup_week,
    optimal_starters_and_points,
)


def _wk(pts_map, covered=True, bye=False):
    return {"pts_map": pts_map, "covered": covered, "bye": bye}


# A 1QB starting lineup with one FLEX and two bench spots.
_1QB_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN"]
_SF_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN"]

_POS = {
    "QB1": "QB", "QB2": "QB",
    "RB1": "RB", "RB2": "RB", "RB3": "RB", "RBscrub": "RB",
    "WR1": "WR", "WR2": "WR", "WR3": "WR", "WRscrub": "WR",
    "TE1": "TE", "TE2": "TE",
}

# Not full: 7 players in a 7-starter + 2-bench league.
_ROSTER = ["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3"]
_PTS = {"QB1": 20, "RB1": 12, "RB2": 10, "WR1": 11, "WR2": 9, "TE1": 6, "RB3": 7}


# ---- 1QB: a FLEX upgrade beats a second QB ---------------------------------

def test_1qb_flex_upgrade_beats_second_qb():
    pos = dict(_POS)
    # A backup QB can't start in a 1QB lineup (FLEX doesn't take QB) -> no gain.
    qb = evaluate_pickup("QB2", "QB", _ROSTER, [_wk({**_PTS, "QB2": 15})],
                         pos, _1QB_SLOTS)
    assert qb.week_gain == pytest.approx(0.0)
    assert qb.outcome in ("hold", "stash")
    assert qb.starts_now is False

    # A WR who out-scores the current FLEX (RB3=7) is a real lineup upgrade.
    wr = evaluate_pickup("WR3", "WR", _ROSTER, [_wk({**_PTS, "WR3": 13})],
                         pos, _1QB_SLOTS)
    assert wr.week_gain == pytest.approx(6.0)   # 13 into FLEX, RB3 (7) benched
    assert wr.outcome == "add"                   # open bench slot, no drop needed
    assert wr.starts_now is True
    assert wr.replaces_pid == "RB3"
    assert wr.week_gain > qb.week_gain


# ---- Superflex: a second QB IS valuable ------------------------------------

def test_superflex_second_qb_is_valuable():
    pos = dict(_POS)
    # Same backup QB (15) now has a Superflex slot to start in -> real gain.
    qb = evaluate_pickup("QB2", "QB", _ROSTER, [_wk({**_PTS, "QB2": 15})],
                         pos, _SF_SLOTS)
    assert qb.week_gain > MIN_MEANINGFUL_GAIN
    assert qb.starts_now is True
    assert qb.outcome == "add"


# ---- TE eligibility / TE-premium scoring flows through FLEX ----------------

def test_te_premium_points_can_win_flex_slot():
    pos = dict(_POS)
    # TE points already reflect TE-premium scoring (applied upstream). A 12-pt TE
    # beats the 7-pt RB3 currently in FLEX.
    te = evaluate_pickup("TE2", "TE", _ROSTER, [_wk({**_PTS, "TE2": 12})],
                         pos, _1QB_SLOTS)
    # TE2 (12) takes the TE slot, bumping TE1 (6); RB3 (7) stays in FLEX.
    assert te.week_gain == pytest.approx(6.0)
    assert te.starts_now is True
    assert te.replaces_pid == "TE1"


# ---- Full roster requires a drop; bench scrub cut, weak starter replaced ----

def test_full_roster_add_drop_separates_drop_from_replaced():
    pos = dict(_POS)
    full_roster = ["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3", "WRscrub", "RBscrub"]
    pts = {**_PTS, "WRscrub": 2, "RBscrub": 1, "WR3": 13}
    res = evaluate_pickup("WR3", "WR", full_roster, [_wk(pts)], pos, _1QB_SLOTS)
    assert res.outcome == "add_drop"
    assert res.week_gain == pytest.approx(6.0)      # WR3 (13) for RB3 (7) in FLEX
    # We cut the least productive spare, not the starter we bumped.
    assert res.drop_pid == "RBscrub"
    assert res.replaces_pid == "RB3"
    assert res.drop_pid != res.replaces_pid
    assert res.drop_points == pytest.approx(1.0)


# ---- Bench production is never a starting-lineup gain -----------------------

def test_bench_production_not_counted_as_gain():
    pos = dict(_POS)
    # A player who scores real points but below every starter goes to the bench;
    # his production does not count as a lineup gain.
    res = evaluate_pickup("WR3", "WR", _ROSTER, [_wk({**_PTS, "WR3": 8})],
                          pos, _1QB_SLOTS)
    # WR3 (8) < RB3 (7)? No, 8 > 7, so it WOULD start. Use a clearly-benched score.
    res2 = evaluate_pickup("WR3", "WR", _ROSTER, [_wk({**_PTS, "WR3": 5})],
                           pos, _1QB_SLOTS)
    assert res2.week_gain == pytest.approx(0.0)
    assert res2.starts_now is False
    assert res2.outcome == "hold"


# ---- Protected / locked players are never dropped --------------------------

def test_protected_players_excluded_from_drop_pool():
    pos = dict(_POS)
    full_roster = ["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3", "WRscrub", "RBscrub"]
    pts = {**_PTS, "WRscrub": 2, "RBscrub": 1, "WR3": 13}
    # Protect the scrubs (e.g. dynasty taxi / do-not-drop): the only eligible cut
    # left is RB3, the current FLEX starter.
    res = evaluate_pickup_week("WR3", "WR", full_roster, pts, pos, _1QB_SLOTS,
                               droppable_pids=["RB3"])
    assert res["drop_pid"] == "RB3"


def test_no_eligible_drop_yields_cannot_evaluate():
    pos = dict(_POS)
    full_roster = ["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3", "WRscrub", "RBscrub"]
    pts = {**_PTS, "WRscrub": 2, "RBscrub": 1, "WR3": 13}
    # Everything protected -> no legal drop -> cannot safely evaluate.
    res = evaluate_pickup("WR3", "WR", full_roster, [_wk(pts)], pos, _1QB_SLOTS,
                          droppable_pids=[])
    assert res.outcome == "cannot_evaluate"


# ---- Injuries / byes flow through as ~0 points -----------------------------

def test_injured_starter_makes_pickup_a_real_upgrade():
    pos = dict(_POS)
    # RB1 is injured this week (projects ~0), so a 10-pt add now cracks the lineup.
    hurt = {**_PTS, "RB1": 0.0, "WR3": 10}
    res = evaluate_pickup("WR3", "WR", _ROSTER, [_wk(hurt)], pos, _1QB_SLOTS)
    assert res.starts_now is True
    assert res.week_gain > MIN_MEANINGFUL_GAIN


# ---- Horizon coverage: missing weeks reported, byes excluded ---------------

def test_missing_weeks_reported_not_zeroed():
    pos = dict(_POS)
    weekly = [
        _wk({**_PTS, "WR3": 13}),                 # week 0 covered
        _wk({**_PTS, "WR3": 12}),                 # week 1 covered
        _wk({}, covered=False),                    # week 2 no projection -> missing
        _wk({**_PTS, "WR3": 0}, bye=True),         # week 3 bye -> excluded
    ]
    res = evaluate_pickup("WR3", "WR", _ROSTER, weekly, pos, _1QB_SLOTS)
    assert res.horizon_weeks_missing == 1
    # Covered weeks: week0 + week1 (bye and missing excluded).
    assert res.horizon_weeks_covered == 2
    assert res.horizon_gain > 0


def test_stash_when_upside_but_no_lineup_gain():
    pos = dict(_POS)
    # A low scorer now (benched) but flagged high speculative upside -> stash,
    # never claiming a lineup improvement it doesn't produce.
    res = evaluate_pickup("WR3", "WR", _ROSTER, [_wk({**_PTS, "WR3": 3})],
                          pos, _1QB_SLOTS, speculative_upside=0.8)
    assert res.outcome == "stash"
    assert res.week_gain == pytest.approx(0.0)
    assert res.starts_now is False


def test_locked_starter_stays_in_lineup():
    pos = dict(_POS)
    # A locked, already-played low-scoring starter (TE1=6) must remain even though
    # a higher-scoring bench TE exists; the optimizer can't bench a lock.
    pts = {**_PTS, "TE2": 20}
    starters, total = optimal_starters_and_points(
        pts, pos, _1QB_SLOTS, _ROSTER + ["TE2"], locked_starter_pids=["TE1"])
    assert "TE1" in starters
    # TE2 (20) still starts via FLEX (beats RB3=7), TE1 stays in the TE slot.
    assert "TE2" in starters


def test_cannot_evaluate_without_week0_projection():
    res = evaluate_pickup("WR3", "WR", _ROSTER, [_wk({})], dict(_POS), _1QB_SLOTS)
    assert res.outcome == "cannot_evaluate"
