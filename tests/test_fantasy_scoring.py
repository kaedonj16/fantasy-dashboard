"""Unit tests for utils.fantasy_scoring.score_stats.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.fantasy_scoring import score_stats


def test_empty_inputs_score_zero():
    assert score_stats({}, {}) == 0.0
    assert score_stats(None, None) == 0.0


def test_default_ppr_passing_line():
    # 300 pass yds * .04 + 2 pass TD * 4 - 1 INT * 2 = 12 + 8 - 2 = 18,
    # plus the 300-yard bonus is only added if the league defines one (here 0).
    s = {"pass_yd": 300, "pass_td": 2, "pass_int": 1}
    assert score_stats(s, {}) == pytest.approx(18.0)


def test_default_rushing_and_receiving():
    # rush: 100*.1 + 1*6 = 16 ; rec: 5 rec (0 pts default) + 80*.1 + 1*6 = 14
    s = {"rush_yd": 100, "rush_td": 1, "rec": 5, "rec_yd": 80, "rec_td": 1}
    # 100-yard rush bonus is 0 unless defined; 100-yard rec bonus 0 unless defined.
    assert score_stats(s, {}) == pytest.approx(30.0)


def test_custom_reception_scoring_is_applied():
    s = {"rec": 6}
    assert score_stats(s, {"rec": 1.0}) == pytest.approx(6.0)
    assert score_stats(s, {"rec": 0.5}) == pytest.approx(3.0)


def test_fumble_lost_penalty():
    assert score_stats({"fum_lost": 2}, {}) == pytest.approx(-4.0)
    assert score_stats({"fum_lost": 1}, {"fum_lost": -1.0}) == pytest.approx(-1.0)


def test_yardage_bonuses_thresholds():
    ss = {
        "bonus_pass_yd_300": 2, "bonus_pass_yd_400": 5,
        "bonus_rush_yd_100": 3, "bonus_rush_yd_200": 7,
        "bonus_rec_yd_100": 3, "bonus_rec_yd_200": 7,
    }
    # 400+ pass takes the 400 bonus (not the 300 one).
    assert score_stats({"pass_yd": 400}, ss) == pytest.approx(400 * 0.04 + 5)
    # 350 pass takes the 300 bonus.
    assert score_stats({"pass_yd": 350}, ss) == pytest.approx(350 * 0.04 + 2)


def test_combined_rush_rec_bonus():
    ss = {"bonus_rush_rec_yd_100": 4, "bonus_rush_rec_yd_200": 9}
    # 60 rush + 60 rec = 120 combined -> the 100 combined bonus (+4).
    out = score_stats({"rush_yd": 60, "rec_yd": 60}, ss)
    base = 60 * 0.1 + 60 * 0.1
    assert out == pytest.approx(base + 4)


def test_missing_stat_keys_default_zero():
    # Only a TD present; everything else absent.
    assert score_stats({"rush_td": 1}, {}) == pytest.approx(6.0)
