"""Pure-logic tests for historical analytics definitions (slim CI, no pandas)."""
from datetime import date

from dashboard_services.historical.definitions import (
    ABSOLUTE_BUST_OUTSIDE,
    AGE_BUCKETS,
    CAREER_STAGE_ORDER,
    COMP_BOARD_TIERS,
    COMP_RELAXATION_ORDER,
    DEFAULT_BAYES_PRIOR_N,
    MIN_COMP_CELL_N,
    POSITION_TIER_WIDTH,
    PRIOR_FINISH_NONE,
    RELIABLE_SEASON_FLOOR,
    SCORING_FORMATS,
    SNAP_RELIABLE_FLOOR,
    FTN_SEASON_FLOOR,
    NGS_SEASON_FLOOR,
    TARGET_SHARE_BUCKETS,
    TIER_CUTOFFS,
    age_as_of_season_start,
    age_bucket,
    career_stage,
    confidence_label,
    display_percent,
    draft_capital_bucket,
    empirical_bayes,
    integer_age,
    is_absolute_bust,
    parse_birth_date,
    positional_tier_label,
    prior_finish_bucket,
    tier_flags,
    value_bucket,
    years_experience_before_season,
)


def test_reliable_floor_is_2016_not_a_uniform_2012():
    assert RELIABLE_SEASON_FLOOR == 2016
    assert SCORING_FORMATS == ("ppr", "half_ppr", "standard")
    assert TIER_CUTOFFS == {
        "top_3": 3,
        "top_5": 5,
        "top_6": 6,
        "top_12": 12,
        "top_24": 24,
        "top_36": 36,
    }
    assert POSITION_TIER_WIDTH == 12
    assert NGS_SEASON_FLOOR == 2016
    assert FTN_SEASON_FLOOR == 2022
    assert SNAP_RELIABLE_FLOOR == 2022


def test_parse_birth_date_accepts_sleeper_and_iso():
    assert parse_birth_date("9/4/1996") == date(1996, 9, 4)
    assert parse_birth_date("1996-09-04") == date(1996, 9, 4)
    assert parse_birth_date("1996-09-04T00:00:00") == date(1996, 9, 4)
    assert parse_birth_date(date(1996, 9, 4)) == date(1996, 9, 4)
    assert parse_birth_date(None) is None
    assert parse_birth_date("") is None
    assert parse_birth_date("not-a-date") is None


def test_age_as_of_sept_1_truncates_not_rounds():
    # Born 1996-09-04, as of 2018-09-01 is just short of 22.
    age = age_as_of_season_start("1996-09-04", 2018)
    assert age == 21.9
    # Same player, 2024 season-start.
    assert age_as_of_season_start("2/14/2002", 2024) == 22.5
    assert age_as_of_season_start(None, 2024) is None
    assert age_as_of_season_start("1996-09-04", None) is None
    # Missing birth date is None, never a fake 0.0.
    assert age_as_of_season_start("", 2024) is None


def test_age_bucket_uses_floor_and_position_map():
    assert age_bucket("RB", 22.9) == "<=22"
    assert age_bucket("RB", 23.0) == "23-24"
    assert age_bucket("RB", 31) == "31+"
    assert age_bucket("WR", 26.4) == "25-27"
    assert age_bucket("WR", 33) == "33+"
    assert age_bucket("TE", 23.1) == "<=23"
    assert age_bucket("QB", 36.0) == "36+"
    assert age_bucket("QB", None) is None
    assert age_bucket("K", 28) is None
    assert set(AGE_BUCKETS) == {"RB", "WR", "TE", "QB"}
    assert integer_age(22.9) == 22
    assert integer_age(None) is None
    assert integer_age("") is None


def test_career_stage_rookie_is_zero_missing_is_none():
    assert career_stage(0) == "rookie"
    assert career_stage(1) == "year_2"
    assert career_stage(2) == "year_3"
    assert career_stage(3) == "year_4"
    assert career_stage(4) == "year_5"
    assert career_stage(5) == "year_6_plus"
    assert career_stage(12) == "year_6_plus"
    assert career_stage(None) is None
    assert career_stage("") is None
    assert CAREER_STAGE_ORDER[0] == "rookie"


def test_years_experience_rookie_is_zero_missing_is_none():
    assert years_experience_before_season(2023, draft_year=2023) == 0
    assert years_experience_before_season(2025, draft_year=2023) == 2
    assert years_experience_before_season(2025, draft_year=None) is None
    assert years_experience_before_season(2025, draft_year=None, first_season=2024) == 1
    # Do not invent a 0 for a veteran with no draft year.
    assert years_experience_before_season(2019, draft_year=None, first_season=None) is None


def test_draft_capital_bucket_does_not_infer_undrafted():
    assert draft_capital_bucket(1, 3) == "round_1"
    assert draft_capital_bucket(2, 40) == "day_2"
    assert draft_capital_bucket(3) == "day_2"
    assert draft_capital_bucket(4) == "day_3"
    assert draft_capital_bucket(7, 250) == "day_3"
    assert draft_capital_bucket(None, 15) == "round_1"  # pick 15 → round 1
    assert draft_capital_bucket(None, None) is None
    assert draft_capital_bucket(None, None, undrafted=True) == "undrafted"
    assert draft_capital_bucket(0) == "undrafted"


def test_positional_tier_label_and_flags():
    assert positional_tier_label("RB", 1) == "RB1"
    assert positional_tier_label("RB", 12) == "RB1"
    assert positional_tier_label("RB", 13) == "RB2"
    assert positional_tier_label("WR", 24) == "WR2"
    assert positional_tier_label("TE", 36) == "TE3"
    assert positional_tier_label("RB", None) is None
    flags = tier_flags(12)
    assert flags["top_12"] is True and flags["top_6"] is False and flags["top_24"] is True
    assert flags["top_5"] is False
    assert tier_flags(5)["top_5"] is True and tier_flags(5)["top_3"] is False
    assert tier_flags(None) == {k: False for k in TIER_CUTOFFS}


def test_confidence_smoothing_and_display_percent():
    assert confidence_label(0) == "low"
    assert confidence_label(14) == "low"
    assert confidence_label(15) == "moderate"
    assert confidence_label(39) == "moderate"
    assert confidence_label(40) == "good"
    assert confidence_label(99) == "good"
    assert confidence_label(100) == "strong"
    assert confidence_label(None) is None
    # 3/10 raw = 0.30; with prior 20/100 and prior_n=10 → (3+2)/(10+10) = 0.25
    smoothed = empirical_bayes(3, 10, prior_successes=2, prior_n=10)
    assert abs(smoothed - 0.25) < 1e-12
    assert empirical_bayes(None, 10, 1, 10) is None
    assert DEFAULT_BAYES_PRIOR_N == 10
    assert display_percent(0.374) == 37
    assert display_percent(None) is None


def test_absolute_bust_none_when_finish_missing():
    assert ABSOLUTE_BUST_OUTSIDE["RB"] == 24
    assert is_absolute_bust("RB", 30) is True
    assert is_absolute_bust("RB", 12) is False
    assert is_absolute_bust("RB", None) is None
    assert is_absolute_bust("K", 40) is None


def test_value_bucket_skips_missing_and_uses_exclusive_hi():
    assert value_bucket(None, TARGET_SHARE_BUCKETS) is None
    assert value_bucket(0.09, TARGET_SHARE_BUCKETS) == "<10%"
    assert value_bucket(0.10, TARGET_SHARE_BUCKETS) == "10-15%"
    assert value_bucket(0.25, TARGET_SHARE_BUCKETS) == "25%+"
    assert value_bucket("", TARGET_SHARE_BUCKETS) is None


def test_prior_finish_bucket_rookie_none_veteran_missing_omitted():
    assert prior_finish_bucket(None, years_experience=0) == PRIOR_FINISH_NONE
    assert prior_finish_bucket(None, years_experience=3) is None
    assert prior_finish_bucket(None, years_experience=None) is None
    assert prior_finish_bucket(5) == "top_5"
    assert prior_finish_bucket(6) == "top_12"
    assert prior_finish_bucket(12) == "top_12"
    assert prior_finish_bucket(13) == "top_24"
    assert prior_finish_bucket(24) == "top_24"
    assert prior_finish_bucket(36) == "top_36"
    assert prior_finish_bucket(37) == "outside_36"
    assert MIN_COMP_CELL_N == 15
    assert COMP_BOARD_TIERS == ("top_5", "top_12", "top_24")
    assert COMP_RELAXATION_ORDER[0] == "target_share"
    assert "position" not in COMP_RELAXATION_ORDER

