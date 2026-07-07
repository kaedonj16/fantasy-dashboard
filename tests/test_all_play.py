"""Tests for utils.all_play."""
from utils.all_play import all_play_analysis, luck_label


def test_empty():
    assert all_play_analysis({}, {}) == {}


def test_all_play_record_basic():
    # 3 teams, 2 weeks. A always highest, C always lowest.
    scores = {
        1: {"A": 100, "B": 90, "C": 80},
        2: {"A": 110, "B": 95, "C": 85},
    }
    res = all_play_analysis(scores, {"A": 2, "B": 1, "C": 0})
    # A beats B and C each week -> 4-0 all-play over 2 weeks.
    assert res["A"]["all_play_wins"] == 4.0
    assert res["A"]["all_play_losses"] == 0.0
    assert res["A"]["all_play_pct"] == 1.0
    # C loses to both each week -> 0-4.
    assert res["C"]["all_play_wins"] == 0.0
    assert res["C"]["all_play_pct"] == 0.0
    # B is 2-2 (beats C, loses to A, each week).
    assert res["B"]["all_play_wins"] == 2.0
    assert res["B"]["all_play_losses"] == 2.0


def test_expected_wins_and_luck_delta():
    scores = {
        1: {"A": 100, "B": 90, "C": 80},
        2: {"A": 110, "B": 95, "C": 85},
    }
    # B "deserves" 0.5 * 2 games = 1 win. If B actually won 2, that's +1 lucky.
    res = all_play_analysis(scores, {"A": 2, "B": 2, "C": 0})
    assert res["B"]["expected_wins"] == 1.0
    assert res["B"]["luck_delta"] == 1.0  # 2 actual - 1 expected
    # A deserved 2, won 2 -> neutral.
    assert res["A"]["luck_delta"] == 0.0


def test_expected_seed_matches_all_play_strength():
    scores = {
        1: {"A": 100, "B": 90, "C": 80},
        2: {"A": 110, "B": 95, "C": 85},
    }
    res = all_play_analysis(scores, {"A": 0, "B": 3, "C": 3})  # actual wins scrambled
    assert res["A"]["expected_seed"] == 1
    assert res["B"]["expected_seed"] == 2
    assert res["C"]["expected_seed"] == 3


def test_ties_split():
    scores = {1: {"A": 100, "B": 100, "C": 80}}
    res = all_play_analysis(scores, {"A": 1, "B": 0, "C": 0})
    # A vs B tie (0.5/0.5), A beats C -> 1.5 wins, 0.5 losses.
    assert res["A"]["all_play_wins"] == 1.5
    assert res["A"]["all_play_losses"] == 0.5
    assert res["B"]["all_play_wins"] == 1.5


def test_unequal_weeks_played():
    # C only appears in week 1 (e.g. joined late / bye handling).
    scores = {
        1: {"A": 100, "B": 90, "C": 80},
        2: {"A": 110, "B": 95},
    }
    res = all_play_analysis(scores, {"A": 2, "B": 1, "C": 0})
    assert res["C"]["games"] == 1
    assert res["A"]["games"] == 2


def test_luck_label():
    assert luck_label(2.0) == "Lucky"
    assert luck_label(-2.0) == "Unlucky"
    assert luck_label(0.5) == ""
    assert luck_label(-0.5) == ""


def test_luck_delta_signs_are_consistent():
    # A team that scores well but loses close games should read unlucky.
    scores = {
        1: {"A": 100, "B": 99, "C": 50},  # A top all-play
        2: {"A": 100, "B": 99, "C": 50},
    }
    # But A actually lost both head-to-head (bad luck / tough schedule).
    res = all_play_analysis(scores, {"A": 0, "B": 2, "C": 2})
    assert res["A"]["luck_delta"] < 0   # deserved wins, didn't get them
    assert res["B"]["luck_delta"] > 0   # over-performed
