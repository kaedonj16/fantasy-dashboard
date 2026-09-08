"""My Leagues and the Teams page must rank the same rooms the same way.

The Teams page used to rank by positional_strength_profile composite while
My Leagues used weighted_pos_strength. Those formulas can flip adjacent
places (WR 6th vs WR 5th) on the same roster.
"""
from pathlib import Path

import pytest

from dashboard_services.ai.context_builders import league_format_value_lookup
from utils.roster_strength import (
    positional_strength_profile, rank_rosters_by_position, weighted_pos_strength,
)
from utils.trade_value import player_trade_value, snap_league_size

ROOT = Path(__file__).resolve().parents[1]


def test_weighted_and_composite_can_flip_adjacent_places():
    """The bug: starter-heavy WR room ranks above a deeper room on weighted
    strength, and below it on the composite the Teams page used to sort by."""
    slots = {"WR": 2, "FLEX": 1}
    elite = [110, 90, 5]
    depth = [95, 88, 70, 50, 40]
    assert weighted_pos_strength(elite, "WR", slots) > weighted_pos_strength(
        depth, "WR", slots,
    )
    assert (
        positional_strength_profile(elite, "WR", slots)["composite"]
        < positional_strength_profile(depth, "WR", slots)["composite"]
    )


def test_shared_ranker_follows_weighted_not_composite():
    slots = {"WR": 2, "FLEX": 1}
    filler = {"QB": [50], "RB": [50], "TE": [50]}
    team_pos = {}
    for i in range(1, 5):
        team_pos[i] = {**filler, "WR": [200 - 10 * i, 80, 10]}
    team_pos[5] = {**filler, "WR": [110, 90, 5]}
    team_pos[6] = {**filler, "WR": [95, 88, 70, 50, 40]}
    _strengths, ranks = rank_rosters_by_position(team_pos, slots)
    assert ranks["WR"][5] == 5
    assert ranks["WR"][6] == 6


def test_league_format_value_lookup_uses_sf_and_te_premium():
    ctx = {
        "platform": "sleeper",
        "roster_positions": ["QB", "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"],
        "scoring_settings": {"bonus_rec_te": 1.0, "rec": 1.0},
        "league_settings": {"type": 2},
        "total_rosters": 10,
        "model_value_table": [
            {"id": "qb", "position": "QB", "value": 100, "sf_value": 180},
            {"id": "te", "position": "TE", "value": 100, "sf_value": 100},
        ],
    }
    lookup = league_format_value_lookup(ctx)
    assert lookup["qb"]["value"] == 180
    assert lookup["te"]["value"] == pytest.approx(120.0)


def test_league_format_value_lookup_matches_player_modal_not_size_overlay():
    """12-team Superflex still uses the modal's 10-team sf_value, not sf_value_12."""
    row = {"id": "qb", "position": "QB", "value": 100, "sf_value": 180, "sf_value_12": 240}
    ctx = {
        "platform": "sleeper",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"],
        "scoring_settings": {"rec": 1.0},
        "league_settings": {"type": 2},
        "total_rosters": 12,
        "model_value_table": [row],
    }
    lookup = league_format_value_lookup(ctx)
    modal = player_trade_value(
        row, league_type="sf", league_size=10, scoring_type="dynasty",
    )
    overlay = player_trade_value(
        row, league_type="sf", league_size=12, scoring_type="dynasty",
    )
    assert lookup["qb"]["value"] == modal == 180
    assert overlay == 240
    assert lookup["qb"]["value"] != overlay


def test_league_format_value_lookup_missing_rec_defaults_to_ppr():
    """Player modal treats missing rec as PPR, not standard."""
    ctx = {
        "platform": "sleeper",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "scoring_settings": {},
        "league_settings": {"type": 2},
        "total_rosters": 10,
        "model_value_table": [
            {"id": "wr", "position": "WR", "value": 100},
        ],
    }
    lookup = league_format_value_lookup(ctx)
    assert lookup["wr"]["value"] == 100


def test_league_format_value_lookup_half_ppr_scales_like_modal():
    ctx = {
        "platform": "sleeper",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "scoring_settings": {"rec": 0.5},
        "league_settings": {"type": 2},
        "total_rosters": 10,
        "model_value_table": [
            {"id": "wr", "position": "WR", "value": 100},
        ],
    }
    lookup = league_format_value_lookup(ctx)
    assert lookup["wr"]["value"] == 97.0


def test_league_format_value_lookup_uses_redraft_not_dynasty():
    ctx = {
        "platform": "espn",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "scoring_settings": {"rec": 1.0},
        "total_rosters": 10,
        "model_value_table": [
            {"id": "wr", "position": "WR", "value": 400, "redraft_value_1qb": 40},
        ],
    }
    lookup = league_format_value_lookup(ctx)
    assert lookup["wr"]["value"] == 40


def test_snap_league_size_nearest_supported_bucket():
    assert snap_league_size(10) == 10
    assert snap_league_size(12) == 12
    assert snap_league_size(11) == 10
    assert snap_league_size(13) == 12
    assert snap_league_size(16) == 14
    assert snap_league_size(None) == 10


def test_my_leagues_and_teams_page_share_ranker_and_values():
    teams = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text()
    portfolio = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = portfolio.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "rank_rosters_by_position" in teams
    assert "rank_rosters_by_position" in summary
    assert "league_format_value_lookup" in teams
    assert "league_format_value_lookup" in summary
    assert "player_trade_value" in (ROOT / "dashboard_services" / "ai" / "context_builders.py").read_text()
    # Composite profile is detail-only on Teams; it must not assign the #N place.
    assert 'profile["composite"]' not in teams
    assert "rank by z-score" not in teams
    assert "_weighted_pos_strength" not in summary
