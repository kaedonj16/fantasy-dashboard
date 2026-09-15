from datetime import date

from utils.nfl_context import (
    calendar_nfl_season, current_sample_weight, normalize_nfl_state,
    season_cache_key,
)


def test_2026_regular_season_resolves_from_provider():
    state = normalize_nfl_state(
        {"season": "2026", "week": 2, "season_type": "reg"},
        on_date=date(2026, 9, 15),
    )
    assert (state["season"], state["week"], state["season_type"]) == (2026, 2, "reg")
    assert state["freshness"]["classification"] == "live"


def test_calendar_fallback_handles_postseason_and_future_years():
    assert calendar_nfl_season(date(2027, 1, 10)) == 2026
    assert calendar_nfl_season(date(2027, 9, 10)) == 2027
    state = normalize_nfl_state({}, on_date=date(2026, 9, 15))
    assert state["season"] == 2026
    assert state["freshness"]["classification"] == "fallback"
    assert state["freshness"]["fallback_reason"]


def test_cache_keys_cannot_cross_seasons_or_scoring_contexts():
    base = dict(namespace="player", week=2, league_id="123", provider="sleeper")
    assert season_cache_key(**base, season=2025) != season_cache_key(**base, season=2026)
    assert season_cache_key(**base, season=2026, scoring="ppr") != season_cache_key(
        **base, season=2026, scoring="half_ppr")


def test_early_season_blend_moves_monotonically_to_current_data():
    assert current_sample_weight(0) == 0.0
    assert current_sample_weight(2) == 2 / 6
    assert current_sample_weight(6) == 1.0
    assert current_sample_weight(20) == 1.0
