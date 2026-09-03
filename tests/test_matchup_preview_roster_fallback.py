"""Matchup preview must backfill starters from rosters when platform rows are empty."""
from unittest import mock

import pytest

pytest.importorskip("flask")
pytest.importorskip("requests")

from dashboard_services.matchups import build_matchup_preview


def test_yahoo_empty_matchup_row_backfills_from_roster():
    players = [str(i) for i in range(12)]
    rosters = [
        {
            "roster_id": 1,
            "owner_id": "owner-a",
            "players": players,
            "starters": [],
            "reserve": list(players),
            "settings": {"wins": 0, "losses": 0},
        },
        {
            "roster_id": 2,
            "owner_id": "owner-b",
            "players": players,
            "starters": players[:9],
            "reserve": players[9:],
            "settings": {"wins": 0, "losses": 0},
        },
    ]
    matchups = [
        {"matchup_id": 1, "roster_id": 1, "points": 0.0, "starters": [], "players": [], "players_points": {}},
        {"matchup_id": 1, "roster_id": 2, "points": 88.8, "starters": [], "players": [], "players_points": {}},
    ]
    users = [
        {"user_id": "owner-a", "roster_id": 1, "display_name": "Lucas"},
        {"user_id": "owner-b", "roster_id": 2, "display_name": "Zach"},
    ]

    with mock.patch("dashboard_services.matchups.get_matchups", return_value=matchups), mock.patch(
        "dashboard_services.matchups.get_rosters", return_value=rosters
    ), mock.patch("dashboard_services.matchups.get_users", return_value=users), mock.patch(
        "dashboard_services.matchups.get_league_settings", return_value={}
    ), mock.patch(
        "dashboard_services.matchups.team_avatar", return_value=""
    ):
        preview = build_matchup_preview(
            league_id="1307110",
            week=1,
            roster_map={"1": "Free win", "2": "Red Zone Zach"},
            players_map={pid: {"name": f"P{pid}", "pos": "RB", "team": "KC"} for pid in players},
            season="2026",
            platform="yahoo",
        )

    assert len(preview) == 1
    left = preview[0]["left"]
    right = preview[0]["right"]
    assert len(left["starters"]) == 9
    assert len(right["starters"]) == 9


def test_yahoo_empty_scoreboard_does_not_invent_round_robin_pairings():
    rosters = [
        {"roster_id": i, "owner_id": f"o{i}", "players": ["1"], "starters": ["1"],
         "reserve": [], "settings": {"wins": 0, "losses": 0}}
        for i in range(1, 5)
    ]
    users = [{"user_id": f"o{i}", "roster_id": i, "display_name": f"U{i}"} for i in range(1, 5)]
    with mock.patch("dashboard_services.matchups.get_matchups", return_value=[]), mock.patch(
        "dashboard_services.matchups.get_rosters", return_value=rosters
    ), mock.patch("dashboard_services.matchups.get_users", return_value=users), mock.patch(
        "dashboard_services.matchups.get_league_settings", return_value={}
    ), mock.patch(
        "dashboard_services.matchups.team_avatar", return_value=""
    ):
        preview = build_matchup_preview(
            league_id="1307110",
            week=1,
            roster_map={str(i): f"Team {i}" for i in range(1, 5)},
            players_map={"1": {"name": "P", "pos": "RB", "team": "KC"}},
            season="2026",
            platform="yahoo",
        )
    assert preview == []


def test_sleeper_empty_matchups_still_synthesize_pairings():
    rosters = [
        {"roster_id": i, "owner_id": f"o{i}", "players": ["1"], "starters": ["1"],
         "reserve": [], "settings": {"wins": 0, "losses": 0}}
        for i in range(1, 5)
    ]
    users = [{"user_id": f"o{i}", "roster_id": i, "display_name": f"U{i}"} for i in range(1, 5)]
    with mock.patch("dashboard_services.matchups.get_matchups", return_value=[]), mock.patch(
        "dashboard_services.matchups.get_rosters", return_value=rosters
    ), mock.patch("dashboard_services.matchups.get_users", return_value=users), mock.patch(
        "dashboard_services.matchups.get_league_settings", return_value={}
    ), mock.patch(
        "dashboard_services.matchups.team_avatar", return_value=""
    ):
        preview = build_matchup_preview(
            league_id="1",
            week=1,
            roster_map={str(i): f"Team {i}" for i in range(1, 5)},
            players_map={"1": {"name": "P", "pos": "RB", "team": "KC"}},
            season="2026",
            platform="sleeper",
        )
    assert len(preview) == 2
    paired = {
        tuple(sorted((m["left"]["roster_id"], m["right"]["roster_id"])))
        for m in preview
    }
    assert paired == {("1", "2"), ("3", "4")}
