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
    calls = []

    def _boom(token, path, params=None):
        calls.append(path)
        raise RuntimeError("403 scoreboard")

    monkeypatch.setattr(yahoo_api, "_yahoo_get", _boom)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "461.l.1307110")
    assert yahoo_api.get_matchups(2026, "1307110", 1, access_token="tok") == []
    assert len(calls) == 1
    assert "week=" in calls[0]


class _FakeYahooResp:
    def __init__(self, status, text, payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = __import__("requests").HTTPError(
                f"{self.status_code} Client Error: Forbidden for url: x"
            )
            err.response = self
            raise err


_NOT_IN_LEAGUE = (
    '{"error":{"description":"You are not allowed to view this page '
    'because you are not in this league.","detail":""}}'
)


def test_parse_nfl_game_keys_pairs_list_of_fragments():
    raw = {
        "fantasy_content": {
            "games": [
                {"game_key": "470"},
                {"season": "2026"},
                {"game_key": "461"},
                {"season": "2025"},
            ]
        }
    }
    parsed = yahoo_api._parse_nfl_game_keys(raw)
    assert parsed[2026] == "470"
    assert parsed[2025] == "461"


def test_parse_nfl_game_keys_merges_wrapped_game_lists():
    raw = {
        "fantasy_content": {
            "games": {
                "count": 2,
                "0": {"game": [{"game_key": "470"}, {"season": "2026"}]},
                "1": {"game": [{"game_key": "461"}, {"season": "2025"}]},
            }
        }
    }
    parsed = yahoo_api._parse_nfl_game_keys(raw)
    assert parsed[2026] == "470"
    assert parsed[2025] == "461"


def test_league_key_for_2026_uses_470_when_games_collection_empty(monkeypatch):
    yahoo_api._clear_yahoo_request_state()
    monkeypatch.setattr(yahoo_api, "_nfl_game_keys", lambda token: [])
    assert yahoo_api._league_key_for_season("1307110", 2026, "tok") == "470.l.1307110"
    assert yahoo_api._league_key_for_season("1307110", 2025, "tok") == "461.l.1307110"


def test_yahoo_get_retries_2026_game_key_when_461_says_not_in_league(monkeypatch):
    """A 2026 member 403s on last year's 461.l.<id>; same token can read 470."""
    yahoo_api._clear_yahoo_request_state()
    urls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        urls.append(url)
        if "470.l.1307110" in url:
            return _FakeYahooResp(200, "{}", {"fantasy_content": {"ok": True}})
        return _FakeYahooResp(403, _NOT_IN_LEAGUE)

    monkeypatch.setattr(yahoo_api.requests, "get", fake_get)
    monkeypatch.setattr(yahoo_api, "get_league_token", lambda *a, **k: None)
    data = yahoo_api._yahoo_get("member-tok", "league/461.l.1307110/scoreboard;week=1")
    assert data["fantasy_content"]["ok"] is True
    assert any("461.l.1307110" in u for u in urls)
    assert any("470.l.1307110" in u for u in urls)
    assert yahoo_api._season_key_map.get(("1307110", 2026)) == "470.l.1307110"

    n = len(urls)
    yahoo_api._yahoo_get("member-tok", "league/461.l.1307110/scoreboard;week=2")
    assert any("470.l.1307110" in u and "week=2" in u for u in urls[n:])
    assert not any("461.l.1307110" in u and "week=2" in u for u in urls[n:])


def test_yahoo_get_retries_owner_token_when_session_not_in_league(monkeypatch):
    yahoo_api._clear_yahoo_request_state()
    auths = []

    def fake_get(url, params=None, headers=None, timeout=None):
        auth = (headers or {}).get("Authorization", "")
        auths.append(auth)
        if "session-tok" in auth:
            return _FakeYahooResp(403, _NOT_IN_LEAGUE)
        return _FakeYahooResp(200, "{}", {"fantasy_content": {"ok": True}})

    monkeypatch.setattr(yahoo_api.requests, "get", fake_get)
    monkeypatch.setattr(yahoo_api, "get_league_token", lambda lid, season: "owner-tok")
    data = yahoo_api._yahoo_get("session-tok", "league/461.l.1307110/scoreboard;week=1")
    assert data["fantasy_content"]["ok"] is True
    assert auths[0] == "Bearer session-tok"
    assert "Bearer owner-tok" in auths
    # Same token is tried on 461 then nfl / 470 / 449 before the owner account.
    session_first = auths.count("Bearer session-tok")
    assert session_first >= 2

    # Later weeks skip every session game-key that already 403'd.
    yahoo_api._yahoo_get("session-tok", "league/461.l.1307110/scoreboard;week=2")
    assert auths.count("Bearer session-tok") == session_first
    assert auths.count("Bearer owner-tok") >= 2


def test_yahoo_get_short_circuits_after_membership_403(monkeypatch):
    yahoo_api._clear_yahoo_request_state()
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _FakeYahooResp(403, _NOT_IN_LEAGUE)

    monkeypatch.setattr(yahoo_api.requests, "get", fake_get)
    monkeypatch.setattr(yahoo_api, "get_league_token", lambda *a, **k: None)
    with pytest.raises(yahoo_api.YahooLeagueAccessDenied):
        yahoo_api._yahoo_get("session-tok", "league/461.l.1307110/scoreboard;week=5")
    first_wave = len(calls)
    assert first_wave >= 2
    with pytest.raises(yahoo_api.YahooLeagueAccessDenied):
        yahoo_api._yahoo_get("session-tok", "league/461.l.1307110/scoreboard;week=6")
    assert len(calls) == first_wave


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
