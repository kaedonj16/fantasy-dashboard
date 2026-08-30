"""GM memo context must tolerate production standings_map seed ints.

Production ``build_standings_map`` returns ``{roster_id: seed:int}``. Treating
those values as record dicts 500'd ``/api/gm-memo`` with:
AttributeError: 'int' object has no attribute 'get'.
"""
from dashboard_services.ai.context_builders import (
    build_power_rankings_context,
    build_team_gm_context,
)


def test_build_team_gm_context_accepts_seed_int_standings_map():
    ctx = {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": {},
        "current_season": 2026,
        "current_week": 1,
        "roster_positions": ["QB", "RB", "WR", "TE"],
        "rosters": [{
            "roster_id": 1,
            "players": ["p1"],
            "settings": {
                "wins": 0, "losses": 0, "ties": 0,
                "fpts": 12, "fpts_decimal": 50,
                "fpts_against": 8, "fpts_against_decimal": 0,
            },
        }],
        "roster_map": {"1": "Team A"},
        "standings_map": {1: 4},  # seed int — production shape
        "picks_by_roster": {},
        "players_index": {"p1": {"full_name": "Star RB", "position": "RB", "age": 24}},
        "players_map": {},
        "model_value_table": [{
            "id": "p1", "name": "Star RB", "position": "RB",
            "value": 200, "redraft_value_1qb": 880, "redraft_value_sf": 880,
        }],
    }
    team_ctx = build_team_gm_context(ctx, "1")
    assert team_ctx is not None
    assert team_ctx["record"] == "0-0"
    assert team_ctx["points_for"] == 12.5
    assert team_ctx["points_against"] == 8.0
    assert team_ctx["season_phase"] == "preseason"


def test_build_power_rankings_context_accepts_seed_int_standings_map():
    ctx = {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": {"playoff_teams": 6},
        "current_season": 2026,
        "current_week": 1,
        "rosters": [
            {"roster_id": 1, "players": ["p1"], "settings": {"wins": 0, "losses": 0, "fpts": 10}},
            {"roster_id": 2, "players": ["p2"], "settings": {"wins": 0, "losses": 0, "fpts": 8}},
        ],
        "roster_map": {"1": "Team A", "2": "Team B"},
        "standings_map": {1: 1, 2: 2},
        "picks_by_roster": {},
        "players_index": {
            "p1": {"full_name": "Star RB", "position": "RB", "age": 24},
            "p2": {"full_name": "Other WR", "position": "WR", "age": 25},
        },
        "players_map": {},
        "model_value_table": [
            {"id": "p1", "position": "RB", "value": 200, "redraft_value_1qb": 800},
            {"id": "p2", "position": "WR", "value": 150, "redraft_value_1qb": 600},
        ],
        "matchups_by_week": {},
    }
    out = build_power_rankings_context(ctx)
    assert out is not None
    assert len(out.get("teams") or []) == 2
