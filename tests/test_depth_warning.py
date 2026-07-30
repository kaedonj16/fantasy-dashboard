"""Roster depth-warning behavior.

Guards the pieces that were previously broken or missing:
  * league-derived thresholds are actually USED (not the 12-team 1QB defaults),
  * "starter-caliber" is judged by REDRAFT value (current-season), so a
    productive vet counts and a hype rookie doesn't -- the opposite of what
    dynasty value would say,
  * superflex raises the QB depth floor,
  * the soft ramp / rounding never invents a "you'll have no starter" alert at a
    position you were already thin at.
"""
from dashboard_services.ai.context_builders import calculate_roster_depth_warning
from utils.roster_strength import derive_league_thresholds, dedicated_starter_counts

_SF_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"]
_STD_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]


def _lookup(te_dynasty, te_redraft):
    return {
        "rb1": {"position": "RB", "value": 600, "redraft_value_sf": 600},
        "rb2": {"position": "RB", "value": 500, "redraft_value_sf": 500},
        "te1": {"position": "TE", "value": te_dynasty, "redraft_value_sf": te_redraft, "name": "TE1"},
        "wrIn": {"position": "WR", "value": 900, "redraft_value_sf": 900, "name": "WR-in"},
    }


def _trade_away_te(lookup):
    return calculate_roster_depth_warning(
        {"players": ["rb1", "rb2", "te1"]},
        lookup,
        sending_assets=[{"id": "te1", "position": "TE"}],
        receiving_assets=[{"id": "wrIn", "position": "WR"}],
        roster_positions=_SF_LINEUP,
        num_teams=12,
        is_sf=True,
    )


def test_productive_vet_counts_as_starter():
    # Low dynasty (outlook) but high redraft (producing now) -> losing him is a
    # real starter loss. Dynasty value (300 < 400) would have missed this.
    res = _trade_away_te(_lookup(te_dynasty=300, te_redraft=700))
    assert "TE" in res and res["TE"]["severity"] == "danger"
    assert res["TE"]["before"] == 1 and res["TE"]["after"] == 0


def test_hype_rookie_is_not_a_starter():
    # High dynasty (outlook) but low redraft (no role yet) -> trading him is a
    # non-event. Dynasty value (800 >= 400) would have wrongly flagged it.
    assert _trade_away_te(_lookup(te_dynasty=800, te_redraft=120)) == {}


def test_borderline_does_not_invent_no_starter_alert():
    # Only a sub-threshold TE to begin with -> pre-existing thinness, stay quiet.
    assert _trade_away_te(_lookup(te_dynasty=360, te_redraft=360)) == {}


def test_flex_is_fungible_lateral_swap_is_quiet():
    # Nico + CeeDee + London, trade Nico for a startable RB. Two WR left cover
    # the two dedicated WR slots, so no WR alert (the FLEX can be the new RB).
    lookup = {
        "nico": {"position": "WR", "value": 700, "redraft_value_1qb": 800},
        "ceedee": {"position": "WR", "value": 800, "redraft_value_1qb": 850},
        "london": {"position": "WR", "value": 650, "redraft_value_1qb": 700},
        "rbIn": {"position": "RB", "value": 600, "redraft_value_1qb": 650, "name": "RB-in"},
    }
    res = calculate_roster_depth_warning(
        {"players": ["nico", "ceedee", "london"]},
        lookup,
        sending_assets=[{"id": "nico", "position": "WR"}],
        receiving_assets=[{"id": "rbIn", "position": "RB"}],
        roster_positions=_STD_LINEUP,
        num_teams=12,
        is_sf=False,
    )
    assert res == {}


def test_dropping_below_dedicated_slots_warns():
    # Only two startable WR, trade one away -> 1 left for 2 dedicated WR slots.
    lookup = {
        "wr1": {"position": "WR", "value": 800, "redraft_value_1qb": 800},
        "wr2": {"position": "WR", "value": 700, "redraft_value_1qb": 700},
        "rbIn": {"position": "RB", "value": 600, "redraft_value_1qb": 650, "name": "RB-in"},
    }
    res = calculate_roster_depth_warning(
        {"players": ["wr1", "wr2"]},
        lookup,
        sending_assets=[{"id": "wr1", "position": "WR"}],
        receiving_assets=[{"id": "rbIn", "position": "RB"}],
        roster_positions=_STD_LINEUP,
        num_teams=12,
        is_sf=False,
    )
    assert "WR" in res and res["WR"]["after"] == 1 and res["WR"]["need"] == 2


def test_dedicated_counts_treat_flex_as_fungible():
    d = dedicated_starter_counts(_STD_LINEUP)
    assert d == {"QB": 1, "RB": 2, "WR": 2, "TE": 1}   # FLEX not attributed
    d_sf = dedicated_starter_counts(_SF_LINEUP)
    assert d_sf["QB"] == 2                              # superflex -> QB need


def test_superflex_raises_qb_floor():
    thr, _ = derive_league_thresholds(_SF_LINEUP, 12, is_sf=True)
    assert dedicated_starter_counts(_SF_LINEUP)["QB"] == 2   # 1 QB slot + 1 superflex
    assert thr["QB"] > 500                                   # SF QB bar lifted above 1QB
    # Trading QB2 in SF should warn you're under the 2-QB floor.
    lookup = {
        "qb1": {"position": "QB", "value": 900, "redraft_value_sf": 950},
        "qb2": {"position": "QB", "value": 700, "redraft_value_sf": 900},
        "rb1": {"position": "RB", "value": 600, "redraft_value_sf": 600},
        "rbIn": {"position": "RB", "value": 650, "redraft_value_sf": 650, "name": "rbIn"},
    }
    res = calculate_roster_depth_warning(
        {"players": ["qb1", "qb2", "rb1"]},
        lookup,
        sending_assets=[{"id": "qb2", "position": "QB"}],
        receiving_assets=[{"id": "rbIn", "position": "RB"}],
        roster_positions=_SF_LINEUP,
        num_teams=12,
        is_sf=True,
    )
    assert "QB" in res and res["QB"]["need"] == 2 and "2 starting slots" in res["QB"]["warning"]


def test_league_size_scales_thresholds():
    # Bigger league -> talent spread thinner -> lower starter bar.
    thr_12, _ = derive_league_thresholds(_SF_LINEUP, 12)
    thr_16, _ = derive_league_thresholds(_SF_LINEUP, 16)
    assert thr_16["RB"] < thr_12["RB"]


def test_dynasty_fallback_when_no_redraft_coverage():
    # No redraft field anywhere -> degrade to dynasty value instead of flagging
    # the whole roster as zero-starter.
    lookup = {
        "rb1": {"position": "RB", "value": 600},
        "rb2": {"position": "RB", "value": 500},
        "te1": {"position": "TE", "value": 700, "name": "TE1"},
        "wrIn": {"position": "WR", "value": 900, "name": "WR-in"},
    }
    res = calculate_roster_depth_warning(
        {"players": ["rb1", "rb2", "te1"]},
        lookup,
        sending_assets=[{"id": "te1", "position": "TE"}],
        receiving_assets=[{"id": "wrIn", "position": "WR"}],
        roster_positions=_SF_LINEUP,
        num_teams=12,
        is_sf=True,
    )
    assert "TE" in res  # dynasty value 700 >= TE threshold, so the loss registers
