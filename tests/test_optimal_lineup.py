"""Unit tests for utils.optimal_lineup.compute_optimal_lineup.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.optimal_lineup import analyze_lineup, compute_optimal_lineup
from dashboard_services.pages.optimal_page import verified_completed_weeks


def test_single_position_slots_pick_top_scorers():
    pts = {"a": 20, "b": 5, "c": 15, "d": 1}
    pos = {"a": "QB", "b": "QB", "c": "RB", "d": "RB"}
    roster = ["QB", "RB"]
    starters, total = compute_optimal_lineup(pts, pos, roster, list(pts))
    assert starters == {"a", "c"}
    assert total == 35.0


def test_flex_takes_best_remaining_rb_wr_te():
    # One RB, one WR slot, plus a FLEX. FLEX should grab the best leftover.
    pts = {"rb1": 30, "rb2": 25, "wr1": 20, "te1": 10}
    pos = {"rb1": "RB", "rb2": "RB", "wr1": "WR", "te1": "TE"}
    roster = ["RB", "WR", "FLEX"]
    starters, total = compute_optimal_lineup(pts, pos, roster, list(pts))
    # RB slot -> rb1, WR slot -> wr1, FLEX -> rb2 (25 > te1 10)
    assert starters == {"rb1", "wr1", "rb2"}
    assert total == 75.0


def test_super_flex_can_take_a_qb():
    pts = {"qb1": 28, "qb2": 22, "rb1": 18}
    pos = {"qb1": "QB", "qb2": "QB", "rb1": "RB"}
    roster = ["QB", "SUPER_FLEX"]
    starters, total = compute_optimal_lineup(pts, pos, roster, list(pts))
    # QB slot -> qb1, SUPER_FLEX -> qb2 (22 > rb1 18)
    assert starters == {"qb1", "qb2"}
    assert total == 50.0


def test_sflex_alias_equivalent_to_super_flex():
    pts = {"qb1": 28, "qb2": 22, "rb1": 18}
    pos = {"qb1": "QB", "qb2": "QB", "rb1": "RB"}
    a = compute_optimal_lineup(pts, pos, ["QB", "SUPER_FLEX"], list(pts))
    b = compute_optimal_lineup(pts, pos, ["QB", "SFLEX"], list(pts))
    assert a == b


def test_missing_points_treated_as_zero():
    pts = {"a": 10}  # b has no entry
    pos = {"a": "RB", "b": "RB"}
    starters, total = compute_optimal_lineup(pts, pos, ["RB"], ["a", "b"])
    assert starters == {"a"}
    assert total == 10.0


def test_not_enough_players_fills_what_it_can():
    pts = {"a": 10}
    pos = {"a": "QB"}
    starters, total = compute_optimal_lineup(pts, pos, ["QB", "RB", "WR"], ["a"])
    assert starters == {"a"}
    assert total == 10.0


def test_no_double_counting_across_slots():
    # A player eligible for both RB and FLEX must not be counted twice.
    pts = {"rb1": 30}
    pos = {"rb1": "RB"}
    starters, total = compute_optimal_lineup(pts, pos, ["RB", "FLEX"], ["rb1"])
    assert starters == {"rb1"}
    assert total == 30.0


def test_empty_roster_returns_empty():
    starters, total = compute_optimal_lineup({"a": 10}, {"a": "RB"}, [], ["a"])
    assert starters == set()
    assert total == 0.0


def test_case_insensitive_positions_and_slots():
    pts = {"a": 20, "b": 15}
    pos = {"a": "qb", "b": "rb"}
    starters, total = compute_optimal_lineup(pts, pos, ["qb", "flex"], list(pts))
    assert starters == {"a", "b"}
    assert total == 35.0


def test_complete_actual_is_optimizer_candidate_and_adjustment_is_separate():
    result = analyze_lineup({"a": 7.2, "b": 18.4}, {"a": "WR", "b": "WR"},
                            ["WR"], ["a", "b"], ["a"], official_total=8.2)
    assert result["complete"] is True
    assert (result["actual"], result["optimal"], result["missed"]) == (7.2, 18.4, 11.2)
    assert result["official_total"] == 8.2
    assert sum(group["gain"] for group in result["groups"]) == result["missed"]


def test_missing_score_is_unknown_but_zero_is_complete():
    missing = analyze_lineup({"a": 0.0}, {"a": "RB", "b": "RB"}, ["RB"], ["a", "b"], ["a"])
    assert missing["complete"] is False
    assert missing["missing_scores"] == ["b"]
    zero = analyze_lineup({"a": 0.0}, {"a": "RB"}, ["RB"], ["a"], ["a"])
    assert zero["complete"] is True
    assert zero["actual"] == zero["optimal"] == 0.0


def test_negative_starter_and_empty_slot_are_supported():
    result = analyze_lineup({"bad": -3.0}, {"bad": "K"}, ["K"], ["bad"], ["bad"])
    assert result["complete"] is True
    assert result["actual"] == -3.0
    assert result["optimal"] == 0.0
    assert result["optimal_assignment"] == [None]


def test_superflex_chain_reconciles_as_one_group():
    scores = {"qb1": 20, "qb2": 19, "rb1": 18, "rb2": 5}
    pos = {"qb1": "QB", "qb2": "QB", "rb1": "RB", "rb2": "RB"}
    result = analyze_lineup(scores, pos, ["QB", "RB", "SUPER_FLEX"], list(scores),
                            ["qb1", "rb2", "rb1"])
    assert result["optimal"] >= result["actual"]
    assert len(result["groups"]) == 1
    assert result["groups"][0]["gain"] == result["missed"]


def test_equal_score_tie_retains_actual_starter():
    result = analyze_lineup({"a": 10, "b": 10}, {"a": "WR", "b": "WR"},
                            ["WR"], ["b", "a"], ["a"])
    assert result["optimal_assignment"] == ["a"]
    assert result["groups"] == []


def test_completed_week_selection_uses_final_status_and_keeps_playoffs():
    import pandas as pd
    current = pd.DataFrame([{"week": 1, "finalized": True}, {"week": 2, "finalized": False}])
    assert verified_completed_weeks(current) == [1]
    historical = pd.DataFrame([{"week": 14, "finalized": True}, {"week": 15, "finalized": True}])
    assert verified_completed_weeks(historical) == [14, 15]
    assert verified_completed_weeks(None, season_complete=True,
                                    matchups_by_week={1: [], 17: []}) == [1, 17]
