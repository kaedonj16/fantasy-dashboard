import json

from dashboard_services.ai import weekly_recap
from dashboard_services import matchups


def test_cached_gotw_is_scoped_to_platform_league_season_and_target_week(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly_recap, "AI_CACHE_DIR", tmp_path)
    key = weekly_recap._recap_cache_key("league-1", 2025, 1)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "content": "cached recap",
        "metadata": {"gotw_selection": {
            "platform": "sleeper", "league_id": "league-1", "season": "2025",
            "source_week": 1, "target_week": 2, "matchup_id": 7,
            "roster_ids": ["10", "20"],
        }},
    }))
    selected = weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 2)
    assert selected and selected["matchup_id"] == 7
    assert weekly_recap.get_cached_gotw_selection("espn", "league-1", 2025, 2) is None
    assert weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 3) is None


def test_gotw_matches_team_pair_in_either_order_and_renders_fire_badges():
    selection = {"matchup_id": "7", "roster_ids": ["10", "20"]}
    matchup = {"matchup_id": 7, "left": {"roster_id": 20}, "right": {"roster_id": 10}}
    assert matchups.matchup_matches_gotw(matchup, selection)
    carousel = matchups.render_matchup_carousel_weeks(
        {2: "<div class='m-slide'></div>"}, False, active_week=2,
        gotw_selection=selection,
    )
    assert "Game of the Week" in carousel
    assert "fa-solid fa-fire" in carousel
