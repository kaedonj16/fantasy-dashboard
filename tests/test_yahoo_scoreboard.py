"""Yahoo scoreboard shapes that previously dropped every matchup."""
import pytest

pytest.importorskip("flask")
pytest.importorskip("bs4")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")


def _team(team_id, points="0.00"):
    return {
        "team": [
            {"team_id": str(team_id)},
            {"team_key": f"461.l.1307110.t.{team_id}"},
            {"team_points": {"total": points}},
        ]
    }


def test_flatten_matchup_list_fragments():
    entry = {
        "matchup": [
            {"week": "1", "status": "predraft", "is_tied": 0},
            {"teams": {"count": 2, "0": _team(1), "1": _team(2, "12.4")}},
        ]
    }
    flat = yahoo_api._flatten_yahoo_matchup(entry)
    assert "teams" in flat
    assert flat["week"] == "1"


def test_flatten_matchup_promotes_teams_under_numeric_wrapper():
    entry = {
        "matchup": [
            {"week": "1", "status": "preevent"},
            {"0": {"teams": {"count": 2, "0": _team(5), "1": _team(8)}}},
        ]
    }
    flat = yahoo_api._flatten_yahoo_matchup(entry)
    assert "teams" in flat
    assert flat["teams"]["count"] == 2


def test_scoreboard_as_list_of_fragments():
    raw = {
        "fantasy_content": {
            "league": [
                {"league_key": "461.l.1307110"},
                {"scoreboard": [
                    {"week": "1"},
                    {"matchups": {
                        "count": 1,
                        "0": {"matchup": [
                            {"week": "1"},
                            {"teams": {"0": _team(3), "1": _team(4)}},
                        ]},
                    }},
                ]},
            ]
        }
    }
    board = yahoo_api._yahoo_scoreboard_dict(raw)
    assert "matchups" in board


def test_get_matchups_parses_list_shaped_yahoo_scoreboard(monkeypatch):
    payload = {
        "fantasy_content": {
            "league": [
                {"league_key": "461.l.1307110"},
                {"scoreboard": {
                    "week": 1,
                    "matchups": {
                        "count": 2,
                        "0": {"matchup": [
                            {"week": "1"},
                            {"teams": {
                                "count": 2,
                                "0": _team(1, "0"),
                                "1": _team(2, "0"),
                            }},
                        ]},
                        "1": {"matchup": [
                            {"week": "1"},
                            {"teams": {
                                "count": 2,
                                "0": _team(3, "0"),
                                "1": _team(4, "0"),
                            }},
                        ]},
                    },
                }},
            ]
        }
    }
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "461.l.1307110")
    rows = yahoo_api.get_matchups(2026, "1307110", 1, access_token="tok")
    assert len(rows) == 4
    assert {r["roster_id"] for r in rows} == {1, 2, 3, 4}
    assert {r["matchup_id"] for r in rows} == {1, 2}


def test_get_matchups_falls_back_to_default_scoreboard(monkeypatch):
    calls = []

    def _fake_get(token, path, params=None):
        calls.append(path)
        if "week=" in path:
            return {"fantasy_content": {"league": [{"league_key": "461.l.1"}, {"scoreboard": {"week": 1}}]}}
        return {
            "fantasy_content": {
                "league": [
                    {"league_key": "461.l.1"},
                    {"scoreboard": {
                        "week": "1",
                        "0": {"matchups": {
                            "count": 1,
                            "0": {"matchup": [
                                {"week": "1"},
                                {"0": {"teams": {
                                    "count": 2,
                                    "0": _team(1, "0"),
                                    "1": _team(2, "0"),
                                }}},
                            ]},
                        }},
                    }},
                ]
            }
        }

    monkeypatch.setattr(yahoo_api, "_yahoo_get", _fake_get)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "461.l.1")
    rows = yahoo_api.get_matchups(2026, "1", 1, access_token="tok")
    assert any("week=" in p for p in calls)
    assert any(p.endswith("/scoreboard") for p in calls)
    assert {r["roster_id"] for r in rows} == {1, 2}


def test_get_matchups_returns_empty_on_http_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("404 scoreboard")
    monkeypatch.setattr(yahoo_api, "_yahoo_get", _boom)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "461.l.1307110")
    assert yahoo_api.get_matchups(2026, "1307110", 1, access_token="tok") == []


def test_build_matchup_preview_does_not_synthesize_when_scoreboard_raises():
    from unittest import mock
    from dashboard_services.matchups import build_matchup_preview

    rosters = [
        {"roster_id": 1, "owner_id": "a", "players": ["11"], "starters": ["11"],
         "settings": {"wins": 0, "losses": 0}},
        {"roster_id": 2, "owner_id": "b", "players": ["22"], "starters": ["22"],
         "settings": {"wins": 0, "losses": 0}},
    ]
    users = [
        {"user_id": "a", "roster_id": 1, "display_name": "A"},
        {"user_id": "b", "roster_id": 2, "display_name": "B"},
    ]
    with mock.patch("dashboard_services.matchups.get_matchups", side_effect=RuntimeError("yahoo")), \
            mock.patch("dashboard_services.matchups.get_rosters", return_value=rosters), \
            mock.patch("dashboard_services.matchups.get_users", return_value=users), \
            mock.patch("dashboard_services.matchups.get_league_settings", return_value={}), \
            mock.patch("dashboard_services.matchups.team_avatar", return_value=""):
        preview = build_matchup_preview(
            league_id="1307110", week=1,
            roster_map={"1": "A", "2": "B"},
            players_map={"11": {"name": "P1", "pos": "QB", "team": "KC"},
                         "22": {"name": "P2", "pos": "RB", "team": "SF"}},
            season="2026", platform="yahoo",
        )
    # Yahoo must not invent round-robin pairings when the scoreboard is missing.
    assert preview == []
