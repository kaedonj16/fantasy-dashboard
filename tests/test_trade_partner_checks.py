"""Wave 2 PR 3: shared trade-partner checks (spec 3.2 checks 1,2,3,5,6
and the conditioned acceptance of 3.3).

One fixture per Nov 4 failing case:
- Burrow: SF partner with exactly 2 startable QBs cannot sell one.
- Caleb: Watson-only package orphans Goodfellas; the QB-back repair
  (Watson + the viewer's spare QB2) makes it viable and fair.
- Moore + Dobbins: two non-starting sweeteners fail headline caliber;
  the same value as one starting piece + filler passes.
- Contention: same roster at 2-6 vs 6-2 against a fixture 8-team rules
  record moves gettability/acceptance in opposite directions.
- Bye: a target whose bye is the viewer's crunch week is flagged;
  a non-crunch target is not.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services.trade_partner_checks import (
    acceptance_fit_restricted,
    bye_collision_flag,
    classify_contention,
    conditioned_acceptance,
    contention_acceptance_adj,
    contention_gettability_adj,
    evaluate_package,
    fairness_ratio,
    find_repair_player,
    surplus_ok,
)

SF_REQ = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}
BARS = {"QB": 400.0, "RB": 300.0, "WR": 300.0, "TE": 180.0}


def _p(pid, pos, value, **kw):
    d = {"player_id": pid, "id": pid, "position": pos, "value": float(value),
         "name": pid}
    d.update(kw)
    return d


# --- Burrow (check 1: surplus-by-format) ------------------------------------

def _burrow_partner():
    # American Dream, Superflex: exactly 2 startable QBs.
    return [
        _p("burrow", "QB", 900), _p("rodgers", "QB", 500),
        _p("rb-a", "RB", 600), _p("rb-b", "RB", 400),
        _p("wr-a", "WR", 700), _p("wr-b", "WR", 500),
        _p("te-a", "TE", 300),
    ]


def test_burrow_no_surplus_no_package():
    partner = _burrow_partner()
    target = partner[0]
    assert surplus_ok(partner_players=partner, target=target,
                      requirements=SF_REQ, value_bars=BARS) is False
    ev = evaluate_package(
        partner_players=partner, viewer_players=[],
        send=[_p("wr-x", "WR", 850)], receive=[target],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["viable"] is False
    assert ev["shortfalls"].get("QB") == {"have": 1, "need": 2}
    assert "Leaves them without a second QB" in ev["notes"]


def test_burrow_pitchable_when_package_returns_startable_qb():
    partner = _burrow_partner()
    ev = evaluate_package(
        partner_players=partner, viewer_players=[],
        send=[_p("qb-back", "QB", 600), _p("wr-x", "WR", 300)],
        receive=[partner[0]],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["viable"] is True
    assert ev["shortfalls"] == {}


# --- Caleb (check 2: viability + QB-back repair) ------------------------------

def _goodfellas():
    # Maye + Caleb are the two starting QBs; the rest of the QB room is
    # rookie-grade (below the startable bar).
    return [
        _p("maye", "QB", 950), _p("caleb", "QB", 800),
        _p("ward", "QB", 150), _p("mendoza", "QB", 120),
        _p("rb-a", "RB", 600), _p("rb-b", "RB", 450),
        _p("wr-a", "WR", 650), _p("wr-b", "WR", 500),
        _p("te-a", "TE", 320),
    ]


def _viewer_qb_room():
    # Viewer starts QB1 + QB2 and holds a third startable QB to spare.
    return [
        _p("my-qb1", "QB", 700), _p("my-qb2", "QB", 550),
        _p("mccarthy", "QB", 450),
        _p("watson", "WR", 420),
        _p("wr-b", "WR", 380), _p("rb-a", "RB", 500),
    ]


def test_caleb_watson_only_not_viable():
    partner = _goodfellas()
    ev = evaluate_package(
        partner_players=partner, viewer_players=_viewer_qb_room(),
        send=[_p("watson", "WR", 420)], receive=[partner[1]],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["viable"] is False
    assert ev["shortfalls"].get("QB") == {"have": 1, "need": 2}


def test_caleb_repair_adds_viewer_spare_qb_and_stays_fair():
    partner = _goodfellas()
    viewer = _viewer_qb_room()
    send = [_p("watson", "WR", 420)]
    repair = find_repair_player(
        viewer_players=viewer, send=send, position="QB",
        viewer_requirements=SF_REQ, value_bars=BARS)
    assert repair is not None and repair["player_id"] == "mccarthy"
    repaired = send + [repair]
    ev = evaluate_package(
        partner_players=partner, viewer_players=viewer,
        send=repaired, receive=[partner[1]],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["viable"] is True
    send_val = sum(a["value"] for a in repaired)
    assert fairness_ratio(send_val, 800) >= 0.80


def test_repair_none_when_viewer_has_no_spare():
    viewer = [_p("my-qb1", "QB", 700), _p("my-qb2", "QB", 550),
              _p("watson", "WR", 420)]
    repair = find_repair_player(
        viewer_players=viewer, send=[viewer[2]], position="QB",
        viewer_requirements=SF_REQ, value_bars=BARS)
    assert repair is None


# --- Headline caliber (check 3) ----------------------------------------------

def _flash_partner():
    # Partner's starting WRs/RBs are strong; Moore would be their WR4
    # and Dobbins their RB4 — neither cracks the post-trade lineup.
    return [
        _p("love", "QB", 850), _p("qb2", "QB", 500),
        _p("wr1", "WR", 900), _p("wr2", "WR", 800),
        _p("rb1", "RB", 850), _p("rb2", "RB", 750),
        _p("te1", "TE", 350),
    ]


def test_moore_dobbins_headline_fails():
    partner = _flash_partner()
    ev = evaluate_package(
        partner_players=partner, viewer_players=[],
        send=[_p("moore", "WR", 430), _p("dobbins", "RB", 420)],
        receive=[partner[0]],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["headline_starts"] is False
    assert ev["headline_ok"] is False
    assert "headline_not_starting_caliber" in ev["reasons"]
    assert ev["starting_pieces"] == []


def test_single_starting_piece_plus_filler_passes():
    partner = _flash_partner()
    # One piece that starts for them (a WR better than their WR2) plus
    # a small filler, same total value as Moore + Dobbins.
    ev = evaluate_package(
        partner_players=partner, viewer_players=[],
        send=[_p("stud-wr", "WR", 820), _p("filler", "RB", 60)],
        receive=[partner[0]],
        requirements=SF_REQ, value_bars=BARS)
    assert ev["headline_ok"] is True
    assert ev["starting_pieces"] == ["stud-wr"]


def test_fit_bonus_only_for_starting_pieces():
    pkg = [_p("moore", "WR", 430), _p("stud-wr", "WR", 820)]
    fit_none_start = acceptance_fit_restricted(
        pkg, need={"WR"}, stacked=set(), starting_ids=[])
    fit_one_starts = acceptance_fit_restricted(
        pkg, need={"WR"}, stacked=set(), starting_ids=["stud-wr"])
    assert fit_none_start == 0
    assert fit_one_starts == 6


# --- Contention (check 5) ------------------------------------------------------

def test_contention_moves_opposite_directions_with_true_line():
    # Same roster/PF profile; the record is what changes. 8-team field
    # from a user-confirmed rules record.
    seller = classify_contention(
        record="2-6", pf=900, pf_percentile=0.25, seed=11,
        games_behind_cut=2.5, playoff_teams=8,
        rules_source="user_confirmed")
    contender = classify_contention(
        record="6-2", pf=900, pf_percentile=0.25, seed=3,
        games_behind_cut=0.0, playoff_teams=8,
        rules_source="user_confirmed")
    assert seller["class"] == "seller"
    assert contender["class"] == "contender"
    assert seller["basis"] == "playoff_line"
    assert contention_gettability_adj(seller) > 0
    assert contention_gettability_adj(contender) < 0
    assert contention_acceptance_adj(seller) > 0
    assert contention_acceptance_adj(contender) < 0


def test_contention_unverified_never_uses_fabricated_field():
    # default_unverified: seed/games-behind against an invented field
    # must be ignored — record + PF percentile only.
    c = classify_contention(
        record="6-2", pf=900, pf_percentile=0.8, seed=11,
        games_behind_cut=4.0, playoff_teams=6,
        rules_source="default_unverified")
    assert c["basis"] == "record_pf"
    assert c["class"] == "contender"  # from 6-2 + top PF, not the seed


# --- Bye collision (check 6) ---------------------------------------------------

def test_bye_collision_flag_and_non_crunch_control():
    viewer = [
        _p("v-qb", "QB", 600, team="GB"),
        _p("v-rb1", "RB", 600, team="LAR"),
        _p("v-rb2", "RB", 500, team="SEA"),
        _p("v-wr1", "WR", 700, team="CIN"),
        _p("v-wr2", "WR", 600, team="MIA"),
        _p("v-te", "TE", 300, team="KC"),
    ]
    bye_by_team = {"GB": 5, "LAR": 11, "SEA": 11, "CIN": 10, "MIA": 9,
                   "KC": 8, "NYJ": 12}
    # Week 11: both viewer RBs on bye -> RB crunch week.
    target = _p("love", "QB", 850, team="NYJ", bye_week=11)
    # build_bye_outlook crunch is position-tight; Love is a QB, so use
    # an RB target for the collision and Love as the control.
    rb_target = _p("rb-t", "RB", 700, team="NYJ", bye_week=11)
    flag = bye_collision_flag(rb_target, viewer, bye_by_team, SF_REQ)
    assert flag is not None
    assert flag["week"] == 11
    assert "Week 11" in flag["label"]
    assert bye_collision_flag(target, viewer, bye_by_team, SF_REQ) is None
    other = _p("rb-o", "RB", 700, team="CIN", bye_week=10)
    assert bye_collision_flag(other, viewer, bye_by_team, SF_REQ) is None


# --- Conditioned acceptance (3.3) ----------------------------------------------

def test_acceptance_prior_demoted_and_basis_honest():
    prob_empty, basis_empty = conditioned_acceptance(
        800, 800, fit=0, trades_like_this=0, total_trades=0)
    prob_prior, basis_prior = conditioned_acceptance(
        800, 800, fit=0, trades_like_this=50, total_trades=100)
    assert basis_empty == "ratio_fit"
    assert basis_prior == "ratio_fit+prior"
    # The prior can only add its capped +12, never dominate.
    assert 0 <= prob_prior - prob_empty <= 12


def test_acceptance_surplus_and_contention_terms():
    ev_surplus = {"target_surplus": True}
    ev_exact = {"target_surplus": False}
    p_surplus, _ = conditioned_acceptance(800, 800, evaluation=ev_surplus)
    p_exact, _ = conditioned_acceptance(800, 800, evaluation=ev_exact)
    assert p_surplus > p_exact
    p_seller, _ = conditioned_acceptance(
        800, 800, evaluation=ev_exact, contention={"class": "seller"})
    p_contender, _ = conditioned_acceptance(
        800, 800, evaluation=ev_exact, contention={"class": "contender"})
    assert p_seller > p_contender
