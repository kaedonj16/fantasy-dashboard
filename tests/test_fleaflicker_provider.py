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
    user = provider.get_users("14153", 2026)[0]
    assert user["user_id"] == "1"
    assert user["metadata"]["flea_owner_id"] == "9"
    roster = provider.get_rosters("14153", 2026)[0]
    assert roster["owner_id"] == "1"
    assert roster["metadata"]["team_name"] == "Owls"
    assert roster["players"] == ["canon-9"]
    assert roster["metadata"]["unmapped_player_count"] == 1
    assert provider.get_matchups("14153", 2026, 1)[0]["points"] == 101.5


def test_resolve_fleaflicker_team_id_matches_owner_or_team():
    from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
    users = [
        {"user_id": "1020439", "roster_id": 1020439,
         "metadata": {"team_name": "East Bay Biters", "flea_owner_id": "532417"}},
    ]
    assert resolve_fleaflicker_team_id(users, team_id="1020439") == "1020439"
    assert resolve_fleaflicker_team_id(users, flea_user_id="532417") == "1020439"
    assert resolve_fleaflicker_team_id(users, flea_user_id="999") is None


def test_build_roster_map_uses_fleaflicker_team_names():
    from utils.league_payload import build_roster_map
    users = [
        {"user_id": "1", "display_name": "Ada",
         "metadata": {"team_name": "Owls", "flea_owner_id": "9"}},
    ]
    rosters = [{"roster_id": 1, "owner_id": "1", "metadata": {"team_name": "Owls"}}]
    assert build_roster_map(users, rosters) == {"1": "Owls"}
    with pytest.raises(UnsupportedCapabilityError):
        FleaflickerProvider().get_bracket("1", 2026, "winners")


@patch("dashboard_services.providers.fleaflicker_api._request_get")
def test_upstream_timeout_is_safe(mock_get):
    mock_get.side_effect = ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        FleaflickerProvider().get_league("14153", 2026)


@patch("dashboard_services.providers.fleaflicker_api._request_post")
def test_login_returns_token_without_exposing_password(mock_post):
    mock_post.return_value = response({"user": {"token": "abc123", "id": 532417}})
    session = login("a@b.com", "secret")
    assert session == {"token": "abc123", "user_id": "532417"}
    body = mock_post.call_args.kwargs.get("json") or {}
    assert body == {"loginId": "a@b.com", "password": "secret"}
    assert "email" not in body
    params = mock_post.call_args.kwargs.get("params") or {}
    assert params.get("sport") == "NFL"
    # Callers must never persist the password; only the token is returned.
    assert "password" not in session


@patch("dashboard_services.providers.fleaflicker_api._request_post")
def test_login_failure_is_auth_error(mock_post):
    mock_post.return_value = response({"failure": "LOGIN_INVALID_PASSWORD"})
    with pytest.raises(ProviderAuthenticationError):
        login("a@b.com", "bad")


@patch("dashboard_services.providers.fleaflicker_api._request_post")
def test_login_html_400_from_wrong_field_is_unavailable(mock_post):
    """Regression: posting ``email`` instead of ``loginId`` returns HTML 400."""
    html = response("<!DOCTYPE html><html>Error 400</html>", status=400)
    html.headers = {"Content-Type": "text/html;charset=utf-8"}
    html.text = "<!DOCTYPE html><html>Error 400</html>"
    html.json.side_effect = ValueError("no json")
    mock_post.return_value = html
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        login("a@b.com", "secret")


@patch("dashboard_services.providers.fleaflicker_api._request_get")
def test_rules_request_omits_season_by_default(mock_get):
    """FetchLeagueRules OpenAPI has no season; passing one returns HTML 400 upstream."""
    mock_get.return_value = response({"rosterPositions": []})
    FleaflickerProvider()._call("FetchLeagueRules", "92916", 2026)
    params = mock_get.call_args.kwargs["params"]
    assert "season" not in params
    assert params["league_id"] == 92916


@patch("dashboard_services.providers.fleaflicker_api._request_get")
def test_html_forbidden_is_unavailable_not_private_auth(mock_get):
    html = response("<html>Forbidden</html>", status=403)
    html.headers = {"Content-Type": "text/html"}
    html.text = "<html>Forbidden</html>"
    html.json.side_effect = ValueError("no json")
    mock_get.return_value = html
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        FleaflickerProvider()._call("FetchLeagueStandings", "14153", 2026)


def test_get_league_survives_rules_failure(monkeypatch):
    provider = FleaflickerProvider()

    def fake_call(method, *a, **k):
        if method == "FetchLeagueRules":
            raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
        return {
            "league": {"id": 14153, "name": "Dynasty", "size": 2},
            "divisions": [{"teams": [{"id": 1, "name": "Owls", "owners": [{"id": 9}]}]}],
            "season": 2026,
        }

    monkeypatch.setattr(provider, "_call", fake_call)
    league = provider.get_league("14153", 2026)
    assert league["name"] == "Dynasty"
    assert league["roster_positions"] == []


@patch("dashboard_services.providers.fleaflicker_api._request_get")
def test_preview_survives_rules_season_html_400(mock_get):
    """Regression: rules+season HTML 400 used to 503 the whole /preview path."""
    standings = response({
        "league": {"id": 92916, "name": "All American All Star League", "size": 12},
        "divisions": [{"teams": [{"id": 1, "name": "East Bay Biters", "owners": [{"id": 9}]}]}],
        "season": 2026,
    })
    rules_ok = response({"rosterPositions": [{"label": "QB", "start": 1}]})
    rules_bad = response("<!DOCTYPE html><html>Error 400</html>", status=400)
    rules_bad.headers = {"Content-Type": "text/html;charset=utf-8"}
    rules_bad.text = "<!DOCTYPE html><html>Error 400</html>"
    rules_bad.json.side_effect = ValueError("no json")

    def side_effect(url, **kwargs):
        params = kwargs.get("params") or {}
        if "FetchLeagueStandings" in url:
            return standings
        if "FetchLeagueRules" in url:
            if "season" in params:
                return rules_bad
            return rules_ok
        raise AssertionError(f"unexpected url {url}")

    mock_get.side_effect = side_effect
    league = FleaflickerProvider().get_league("92916", 2026)
    assert league["name"] == "All American All Star League"
    assert league["roster_positions"] == ["QB"]


def test_get_league_exposes_scheduled_draft_day(monkeypatch):
    provider = FleaflickerProvider()
    draft_ms = 1_735_689_600_000  # Aug 2025-ish
    monkeypatch.setattr(
        provider,
        "_call",
        lambda method, *a, **k: {
            "league": {
                "id": 92916,
                "name": "All American",
                "size": 12,
                "draft_status": "NOT_YET_DRAFTED",
                "draft_live_time_epoch_milli": str(draft_ms),
            },
            "divisions": [{"teams": [{"id": 1, "name": "Owls", "owners": [{"id": 9}]}]}],
            "season": 2026,
        } if method == "FetchLeagueStandings" else {"groups": []},
    )
    league = provider.get_league("92916", 2026)
    assert league["draft_day"] == draft_ms
    assert league["settings"]["draft_status"] == "NOT_YET_DRAFTED"


def test_get_drafts_reports_upcoming_start_time_and_status(monkeypatch):
    provider = FleaflickerProvider()
    draft_ms = 1_735_689_600_000

    def fake_call(method, *a, **k):
        if method == "FetchLeagueStandings":
            return {
                "league": {
                    "id": 92916,
                    "draft_status": "NOT_YET_DRAFTED",
                    "draft_live_time_epoch_milli": str(draft_ms),
                },
            }
        if method == "FetchLeagueDraftBoard":
            return {"rows": [], "is_in_progress": False}
        raise AssertionError(method)

    monkeypatch.setattr(provider, "_call", fake_call)
    drafts = provider.get_drafts("92916", 2026)
    assert len(drafts) == 1
    assert drafts[0]["start_time"] == draft_ms
    assert drafts[0]["status"] == "pre_draft"
    assert drafts[0]["picks"] == []


def test_get_drafts_marks_post_draft_complete(monkeypatch):
    provider = FleaflickerProvider()
    draft_ms = 1_735_689_600_000

    def fake_call(method, *a, **k):
        if method == "FetchLeagueStandings":
            return {
                "league": {
                    "id": 92916,
                    "draft_status": "POST_DRAFT",
                    "draft_live_time_epoch_milli": str(draft_ms),
                },
            }
        if method == "FetchLeagueDraftBoard":
            return {
                "rows": [{
                    "round": 1,
                    "cells": [{
                        "team": {"id": 1},
                        "player": {"proPlayer": {"id": 99, "nameFull": "Player", "position": "QB"}},
                    }],
                }],
                "is_in_progress": False,
            }
        raise AssertionError(method)

    monkeypatch.setattr(provider, "_call", fake_call)
    drafts = provider.get_drafts("92916", 2026)
    assert drafts[0]["status"] == "complete"
    assert len(drafts[0]["picks"]) == 1
