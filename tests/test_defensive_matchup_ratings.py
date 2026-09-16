from utils.defensive_matchup_ratings import (
    blend_value,
    rank_team_schedules,
    rank_values,
    rating_cache_key,
    season_weights,
)


def test_before_week_one_uses_prior_baseline():
    assert season_weights(0) == (1.0, 0.0)
    assert blend_value(20, 40, 0) == (20.0, "blended")


def test_weeks_one_through_five_use_configured_progression():
    expected = {1: .25, 2: .40, 3: .55, 4: .70, 5: .85}
    for week, current_weight in expected.items():
        prior_weight, actual_current = season_weights(week)
        assert actual_current == current_weight
        value, source = blend_value(10, 20, week)
        assert value == prior_weight * 10 + current_weight * 20
        assert source == "blended"


def test_week_six_onward_is_current_only_and_historical_disables_blend():
    assert season_weights(6) == (0.0, 1.0)
    assert season_weights(18) == (0.0, 1.0)
    assert season_weights(2, blend=False) == (0.0, 1.0)


def test_missing_data_is_never_zero_filled():
    assert blend_value(None, 14, 2) == (14, "current")
    assert blend_value(18, None, 2) == (18, "prior")
    assert blend_value(None, None, 2) == (None, "unavailable")


def test_higher_points_allowed_is_easier_rank():
    ranks, total = rank_values({"A": 12, "B": 24, "C": None})
    assert (ranks, total) == ({"B": 1, "A": 2}, 2)


def test_sos_uses_selected_non_bye_values_not_ordinal_average():
    values = {"D1": 30, "D2": 10, "D3": 19}
    schedules = {
        "A": ["D1", None],       # bye excluded; mean 30
        "B": ["D2", "D3"],      # selected range only; mean 14.5
    }
    result = rank_team_schedules(schedules, values)
    assert result["A"] == (1, 2, 30.0)
    assert result["B"] == (2, 2, 14.5)


def test_cache_key_isolates_season_week_position_and_full_scoring_profile():
    base = rating_cache_key(2026, 2, {"rec": 1, "pass_td": 4}, "QB")
    assert base != rating_cache_key(2025, 2, {"rec": 1, "pass_td": 4}, "QB")
    assert base != rating_cache_key(2026, 3, {"rec": 1, "pass_td": 4}, "QB")
    assert base != rating_cache_key(2026, 2, {"rec": .5, "pass_td": 4}, "QB")
    assert base != rating_cache_key(2026, 2, {"rec": 1, "pass_td": 4}, "RB")

