from unittest.mock import Mock, patch

import pytest

from dashboard_services.providers.base import (
    ProviderAuthenticationError, ProviderUnavailableError, UnsupportedCapabilityError,
)
from dashboard_services.providers.fleaflicker_api import FleaflickerProvider, _CACHE, login


def response(payload, status=200):
    result = Mock(status_code=status, headers={}, cookies={})
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    result.text = ""
    return result


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()


def test_normalizes_league_users_rosters_and_matchups(monkeypatch):
    provider = FleaflickerProvider()
    payloads = {
        "FetchLeagueStandings": {
            "league": {"id": 14153, "name": "Dynasty", "size": 2},
            "divisions": [{"teams": [
                {"id": 1, "name": "Owls", "owners": [{"id": 9, "displayName": "Ada"}]},
                {"id": 2, "name": "Bears", "owners": [{"id": 8, "displayName": "Bea"}]},
            ]}],
        },
        "FetchLeagueRules": {
            "rosterPositions": [{"label": "QB", "start": 1}, {"label": "RB", "start": 2}],
            "groups": [{"scoringRules": [{"category": {"abbreviation": "TD"}, "points": {"value": 6}}]}],
        },
        "FetchLeagueRosters": {
            "rosters": [{"team": {"id": 1}, "players": [
                {"proPlayer": {"id": 9, "nameFull": "Known Player", "position": "QB"}},
                {"proPlayer": {"id": 404, "nameFull": "Unknown", "position": "WR"}},
            ]}],
        },
        "FetchLeagueScoreboard": {
            "games": [{
                "id": "g1",
                "home": {"id": 1},
                "away": {"id": 2},
                "homeScore": {"score": {"value": 101.5}},
                "awayScore": {"score": {"value": 99}},
            }],
        },
    }
    monkeypatch.setattr(provider, "_call", lambda method, *a, **k: payloads[method])
    monkeypatch.setattr(provider, "_canonical_map", lambda *a, **k: {"9": "canon-9"})
    league = provider.get_league("14153", 2026)
    assert league["total_rosters"] == 2
    assert league["roster_positions"] == ["QB", "RB", "RB"]
    assert provider.get_users("14153", 2026)[0]["user_id"] == "9"
    roster = provider.get_rosters("14153", 2026)[0]
    assert roster["players"] == ["canon-9"]
    assert roster["metadata"]["unmapped_player_count"] == 1
    assert provider.get_matchups("14153", 2026, 1)[0]["points"] == 101.5


def test_bracket_is_unsupported():
    with pytest.raises(UnsupportedCapabilityError):
        FleaflickerProvider().get_bracket("1", 2026, "winners")


@patch("dashboard_services.providers.fleaflicker_api._request_get")
def test_upstream_timeout_is_safe(mock_get):
    mock_get.side_effect = ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        FleaflickerProvider().get_league("14153", 2026)


@patch("dashboard_services.providers.fleaflicker_api._request_post")
def test_login_returns_token_without_exposing_password(mock_post):
    mock_post.return_value = response({"user": {"token": "abc123"}})
    assert login("a@b.com", "secret") == "abc123"
    body = mock_post.call_args.kwargs.get("json") or {}
    assert body["password"] == "secret"
    # Callers must never persist the password; only the token is returned.
    assert "password" not in {"token": login("a@b.com", "secret")}


@patch("dashboard_services.providers.fleaflicker_api._request_post")
def test_login_failure_is_auth_error(mock_post):
    mock_post.return_value = response({"failure": "LOGIN_INVALID_PASSWORD"})
    with pytest.raises(ProviderAuthenticationError):
        login("a@b.com", "bad")
