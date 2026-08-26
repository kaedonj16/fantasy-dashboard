"""Unit tests for utils.fantasy_scoring.score_stats.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.fantasy_scoring import (
    projection_points,
    score_stats,
    weekly_projection_points,
)


def _wk_entry():
    return {
        "6786": {
            "ppr": 15.76, "half_ppr": 13.07, "std": 10.38,
            "raw_stats": {"rec": 6, "rec_yd": 80,
                          "pts_ppr": 17.15, "pts_half_ppr": 14.46, "pts_std": 11.77},
        }
    }


def test_weekly_projection_points_trusts_published_for_standard():
    wk = _wk_entry()
    assert weekly_projection_points(wk, "6786", {"rec": 1.0}, "WR") == 17.15
    assert weekly_projection_points(wk, "6786", {"rec": 0.5}, "WR") == 14.46
    assert weekly_projection_points(wk, "6786", {"rec": 0.0}, "WR") == 11.77


def test_weekly_projection_points_recomputes_for_custom():
    wk = _wk_entry()
    # 0.75-PPR: not standard -> recompute from raw (6*0.75 + 80*0.1 = 12.5).
    assert weekly_projection_points(wk, "6786", {"rec": 0.75, "rec_yd": 0.1}, "WR") == 12.5


def test_weekly_projection_points_absent_and_scalar():
    wk = _wk_entry()
    assert weekly_projection_points(wk, "9999", {"rec": 1.0}, "WR") is None
    assert weekly_projection_points({"x": 12.3}, "x", {"rec": 1.0}) == 12.3
    assert weekly_projection_points(None, "x", {"rec": 1.0}) is None


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


def test_explicit_zero_overrides_standard_default():
    assert score_stats({"pass_td": 2}, {"pass_td": 0}) == 0.0


def test_matching_custom_categories_are_scored():
    stats = {"pass_fd": 12, "rush_fd": 3, "two_pt": 1}
    settings = {"pass_fd": 0.5, "rush_fd": 1.0, "two_pt": 2.0}
    assert score_stats(stats, settings) == pytest.approx(11.0)


def test_kicker_and_defense_categories_are_scored_generically():
    stats = {"fgm_40_49": 2, "xpm": 3, "sack": 4, "int": 1}
    settings = {"fgm_40_49": 4, "xpm": 1, "sack": 1, "int": 2}
    assert score_stats(stats, settings) == 17.0


def test_te_reception_premium_uses_exact_league_rate():
    assert score_stats({"rec": 4}, {"rec": 0.5, "bonus_rec_te": 0.75}, "TE") == 5.0


def test_cached_raw_stats_use_exact_custom_scoring():
    entry = {
        "raw_stats": {"rec": 8, "rec_yd": 100},
        "ppr": 18.0, "half_ppr": 14.0,
    }
    assert projection_points(entry, {"rec": 0.75, "rec_yd": 0.1}, "WR") == 16.0


def test_legacy_cache_without_raw_stats_still_uses_variant():
    entry = {"ppr": 18.0, "half_ppr": 14.0}
    assert projection_points(entry, {"rec": 0.5}, "WR") == 14.0


def test_standard_league_uses_sleeper_precomputed_total():
    # Standard PPR/half/std: show Sleeper's own pts_*, not our recompute.
    entry = {"raw_stats": {"rec": 6, "rec_yd": 80, "rec_td": 1,
                           "pts_ppr": 21.7, "pts_half_ppr": 18.7, "pts_std": 15.7}}
    assert projection_points(entry, {"rec": 1.0}, "WR") == 21.7
    assert projection_points(entry, {"rec": 0.5}, "WR") == 18.7
    assert projection_points(entry, {"rec": 0.0}, "WR") == 15.7


def test_custom_reception_ignores_precomputed_total():
    # A 0.75-PPR league is not standard: recompute from the raw line.
    entry = {"raw_stats": {"rec": 8, "rec_yd": 100, "pts_ppr": 99.0}}
    assert projection_points(entry, {"rec": 0.75, "rec_yd": 0.1}, "WR") == 16.0


def test_bonus_or_6pt_league_ignores_precomputed_total():
    # Yardage-bonus league: Sleeper's standard pts_ppr can't reflect it → recompute.
    entry = {"raw_stats": {"pass_yd": 350, "pts_ppr": 99.0}}
    assert projection_points(entry, {"rec": 1.0, "bonus_pass_yd_300": 3}, "QB") == pytest.approx(350 * 0.04 + 3)
    # 6pt passing TD is custom too.
    entry2 = {"raw_stats": {"pass_td": 3, "pts_ppr": 99.0}}
    assert projection_points(entry2, {"rec": 1.0, "pass_td": 6}, "QB") == pytest.approx(18.0)


def test_custom_interception_rate_ignores_standard_precomputed_total():
    entry = {
        "raw_stats": {
            "pass_yd": 250, "pass_td": 2, "pass_int": 1,
            "pts_ppr": 99.0,
        }
    }
    settings = {"rec": 1.0, "pass_yd": 0.04, "pass_td": 4, "pass_int": -1}
    assert projection_points(entry, settings, "QB") == pytest.approx(17.0)
