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
        {"matchup_id": 1, "roster_id": 1, "points": 0.0, "projected_points": 118.4,
         "starters": [], "players": [], "players_points": {}},
        {"matchup_id": 1, "roster_id": 2, "points": 88.8, "projected_points": 104.1,
         "starters": [], "players": [], "players_points": {}},
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
    assert left["proj_total"] == pytest.approx(118.4)
    assert right["proj_total"] == pytest.approx(104.1)
    assert [s["pid"] for s in left["starters"]] == players[:9]
    assert [s["pid"] for s in right["starters"]] == players[:9]


def test_yahoo_matchup_uses_scoreboard_roster_starters_not_current_roster():
    """Week-N starters already on the matchup row must win over get_rosters."""
    matchups = [
        {
            "matchup_id": 1, "roster_id": 1, "points": 0.0,
            "starters": ["4984"], "players": ["4984", "99"], "players_points": {},
        },
        {
            "matchup_id": 1, "roster_id": 2, "points": 0.0,
            "starters": ["11564"], "players": ["11564"], "players_points": {},
        },
    ]
    rosters = [
        {"roster_id": 1, "owner_id": "a", "players": ["99"], "starters": ["99"],
         "reserve": [], "settings": {"wins": 0, "losses": 0}},
        {"roster_id": 2, "owner_id": "b", "players": ["88"], "starters": ["88"],
         "reserve": [], "settings": {"wins": 0, "losses": 0}},
    ]
    users = [
        {"user_id": "a", "roster_id": 1, "display_name": "A"},
        {"user_id": "b", "roster_id": 2, "display_name": "B"},
    ]
    with mock.patch("dashboard_services.matchups.get_matchups", return_value=matchups), mock.patch(
        "dashboard_services.matchups.get_rosters", return_value=rosters
    ), mock.patch("dashboard_services.matchups.get_users", return_value=users), mock.patch(
        "dashboard_services.matchups.get_league_settings", return_value={}
    ), mock.patch(
        "dashboard_services.matchups.team_avatar", return_value=""
    ):
        preview = build_matchup_preview(
            league_id="1307110", week=1,
            roster_map={"1": "A", "2": "B"},
            players_map={
                "4984": {"name": "Josh Allen", "pos": "QB", "team": "BUF"},
                "11564": {"name": "Drake Maye", "pos": "QB", "team": "NE"},
                "99": {"name": "Wrong", "pos": "RB", "team": "KC"},
                "88": {"name": "Also wrong", "pos": "RB", "team": "SF"},
            },
            season="2026", platform="yahoo",
        )
    assert preview[0]["left"]["starters"][0]["pid"] == "4984"
    assert preview[0]["right"]["starters"][0]["pid"] == "11564"


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
