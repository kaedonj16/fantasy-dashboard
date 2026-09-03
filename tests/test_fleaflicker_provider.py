from unittest.mock import Mock, patch

import pytest

from dashboard_services.providers.base import (
    ProviderAuthenticationError, ProviderUnavailableError, UnsupportedCapabilityError,
)
from dashboard_services.providers.fleaflicker_api import (
    FleaflickerProvider, _CACHE, _flea_pro_team, _fleaflicker_draft_status,
    _fleaflicker_sleeper_league_type, _name_index_from_players,
    _normalize_fleaflicker_draft_status, _pick_canonical, login,
)


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
        "FetchRoster": {
            "groups": [{
                "group": "START",
                "slots": [{
                    "position": {"label": "QB", "group": "START"},
                    "leaguePlayer": {
                        "proPlayer": {"id": 9, "nameFull": "Known Player", "position": "QB"},
                    },
                }],
            }, {
                "group": "BENCH",
                "slots": [{
                    "position": {"label": "BN", "group": "BENCH"},
                    "leaguePlayer": {
                        "proPlayer": {"id": 404, "nameFull": "Unknown", "position": "WR"},
                    },
                }],
            }],
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
    assert league["settings"]["type"] == 0
    assert league["settings"]["league_type"] == "redraft"
    assert league["roster_positions"] == ["QB", "RB", "RB"]
    user = provider.get_users("14153", 2026)[0]
    assert user["user_id"] == "1"
    assert user["metadata"]["flea_owner_id"] == "9"
    roster = provider.get_rosters("14153", 2026)[0]
    assert roster["owner_id"] == "1"
    assert roster["metadata"]["team_name"] == "Owls"
    assert roster["players"] == ["canon-9"]
    assert roster["starters"] == ["canon-9"]
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


@pytest.mark.parametrize(
    ("max_keepers", "teams", "sleeper_type", "label"),
    [
        (0, 12, 0, "redraft"),
        (2, 12, 1, "keeper"),
        (25, 12, 2, "dynasty"),
    ],
)
def test_fleaflicker_sleeper_league_type_mapping(max_keepers, teams, sleeper_type, label):
    assert _fleaflicker_sleeper_league_type(max_keepers, teams) == (sleeper_type, label)


def test_get_league_detects_keeper_league(monkeypatch):
    provider = FleaflickerProvider()
    monkeypatch.setattr(
        provider,
        "_call",
        lambda method, *a, **k: {
            "league": {
                "id": 92916,
                "name": "Keeper League",
                "size": 12,
                "maxKeepers": 3,
            },
            "divisions": [{"teams": [{"id": 1, "name": "Owls", "owners": [{"id": 9}]}]}],
            "season": 2026,
        } if method == "FetchLeagueStandings" else {"groups": []},
    )
    league = provider.get_league("92916", 2026)
    assert league["settings"]["type"] == 1
    assert league["settings"]["league_type"] == "keeper"
    assert league["settings"]["max_keepers"] == 3
    # Protobuf omits the default NOT_YET_DRAFTED enum; persist it so keeper
    # rosters from last year are not treated as an already-run draft.
    assert league["settings"]["draft_status"] == "NOT_YET_DRAFTED"


def test_omitted_draft_status_is_not_yet_drafted():
    assert _fleaflicker_draft_status({}) == "NOT_YET_DRAFTED"
    assert _fleaflicker_draft_status({"name": "X"}) == "NOT_YET_DRAFTED"
    assert _normalize_fleaflicker_draft_status(None, pick_count=12) == "pre_draft"
    assert _normalize_fleaflicker_draft_status("", pick_count=12) == "pre_draft"


def test_get_drafts_omitted_status_ignores_last_year_board(monkeypatch):
    """Keeper leagues roll last year's board; omitted status must stay pre-draft."""
    provider = FleaflickerProvider()
    draft_ms = 1_735_689_600_000

    def fake_call(method, *a, **k):
        if method == "FetchLeagueStandings":
            return {
                "league": {
                    "id": 92916,
                    # draft_status omitted — protobuf default NOT_YET_DRAFTED
                    "maxKeepers": 3,
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


def test_positions_canonicalizes_fleaflicker_flex_sf_and_dst():
    """Fleaflicker labels must become Sleeper slots or draft-room drops them."""
    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "QB", "group": "START", "start": 1},
            {"label": "RB", "group": "START", "start": 2},
            {"label": "WR", "group": "START", "start": 2},
            {"label": "TE", "group": "START", "start": 1},
            {"label": "RB/WR/TE", "group": "START", "start": 1,
             "eligibility": ["RB", "WR", "TE"]},
            {"label": "QB/RB/WR/TE", "group": "START", "start": 1,
             "eligibility": ["QB", "RB", "WR", "TE"]},
            {"label": "K", "group": "START", "start": 1},
            {"label": "D/ST", "group": "START", "start": 1},
            {"label": "BN", "group": "BENCH", "max": 6},
            {"label": "IR", "group": "INJURED", "max": 2},
        ],
    })
    assert slots == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF",
    ]


def test_positions_skips_bench_even_when_start_is_set():
    slots = FleaflickerProvider._positions({
        "roster_positions": [
            {"label": "QB", "group": "START", "start": 1},
            {"label": "BN", "group": "BENCH", "start": 6},
        ],
    })
    assert slots == ["QB"]


def test_positions_treats_start_group_with_omitted_start_as_one_slot():
    # Protobuf omits zero-valued ints; START rows with no ``start`` are one slot.
    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "QB", "group": "START"},
            {"label": "RB/WR/TE", "group": "START"},
            {"label": "BN", "group": "BENCH"},
        ],
    })
    assert slots == ["QB", "FLEX"]


def test_positions_restricted_flex_labels_stay_distinct():
    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "RB/WR", "group": "START", "start": 1,
             "eligibility": ["RB", "WR"]},
            {"label": "WR/TE", "group": "START", "start": 1,
             "eligibility": ["WR", "TE"]},
            {"label": "RB/TE", "group": "START", "start": 1,
             "eligibility": ["RB", "TE"]},
        ],
    })
    assert slots == ["RB_WR", "WR_TE", "RB_TE"]


def test_positions_generic_flex_uses_eligibility_for_wr_rb_only():
    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "FLEX", "group": "START", "start": 1,
             "eligibility": ["RB", "WR"]},
            {"label": "", "group": "START", "start": 1,
             "eligibility": ["WR", "TE"]},
        ],
    })
    assert slots == ["RB_WR", "WR_TE"]


def test_positions_flex_with_qb_eligibility_is_superflex():
    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "FLEX", "group": "START", "start": 1,
             "eligibility": ["QB", "RB", "WR", "TE"]},
        ],
    })
    assert slots == ["SUPER_FLEX"]


def test_positions_falls_back_to_standings_roster_requirements():
    slots = FleaflickerProvider._positions(
        {},
        league={
            "rosterRequirements": {
                "positions": [
                    {"label": "QB", "group": "START", "start": 1},
                    {"label": "QB/RB/WR/TE", "group": "START", "start": 1},
                    {"label": "D/ST", "group": "START", "start": 1},
                ],
            },
        },
    )
    assert slots == ["QB", "SUPER_FLEX", "DEF"]


def test_get_league_uses_standings_slots_when_rules_fail(monkeypatch):
    provider = FleaflickerProvider()

    def fake_call(method, *a, **k):
        if method == "FetchLeagueRules":
            raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
        return {
            "league": {
                "id": 14153, "name": "Dynasty", "size": 2,
                "rosterRequirements": {
                    "positions": [
                        {"label": "QB", "group": "START", "start": 1},
                        {"label": "RB/WR/TE", "group": "START", "start": 1},
                    ],
                },
            },
            "divisions": [{"teams": [{"id": 1, "name": "Owls", "owners": [{"id": 9}]}]}],
            "season": 2026,
        }

    monkeypatch.setattr(provider, "_call", fake_call)
    league = provider.get_league("14153", 2026)
    assert league["roster_positions"] == ["QB", "FLEX"]


def test_get_rosters_uses_fetch_roster_starters_when_bulk_list_exists(monkeypatch):
    provider = FleaflickerProvider()

    def fake_call(method, *args, **kwargs):
        if method == "FetchLeagueRosters":
            return {
                "rosters": [{
                    "team": {"id": 1, "name": "Owls"},
                    "players": [
                        {"proPlayer": {"id": 9, "nameFull": "Starter QB", "position": "QB"}},
                        {"proPlayer": {"id": 10, "nameFull": "Bench RB", "position": "RB"}},
                    ],
                }],
            }
        if method == "FetchLeagueStandings":
            return {
                "league": {"id": 14153, "name": "Dynasty", "size": 1},
                "divisions": [{"teams": [{"id": 1, "name": "Owls"}]}],
            }
        if method == "FetchRoster":
            return {
                "groups": [
                    {
                        "group": "START",
                        "slots": [{
                            "position": {"label": "QB", "group": "START"},
                            "leaguePlayer": {
                                "proPlayer": {"id": 9, "nameFull": "Starter QB", "position": "QB"},
                            },
                        }],
                    },
                    {
                        "group": "BENCH",
                        "slots": [{
                            "position": {"label": "RB", "group": "BENCH"},
                            "leaguePlayer": {
                                "proPlayer": {"id": 10, "nameFull": "Bench RB", "position": "RB"},
                            },
                        }],
                    },
                ],
            }
        raise AssertionError(method)

    monkeypatch.setattr(provider, "_call", fake_call)
    monkeypatch.setattr(provider, "_canonical_map", lambda *a, **k: {"9": "canon-qb", "10": "canon-rb"})
    roster = provider.get_rosters("14153", 2026)[0]
    assert roster["players"] == ["canon-qb", "canon-rb"]
    assert roster["starters"] == ["canon-qb"]
    assert "canon-rb" not in roster["starters"]


def test_name_index_reads_pos_and_keeps_both_lamars():
    def norm(name):
        return (name or "").strip().lower()

    index = {
        "4881": {"name": "Lamar Jackson", "pos": "QB", "team": "BAL"},
        "6994": {"name": "Lamar Jackson", "pos": "CB", "team": "ATL"},
        "7525": {"name": "DeVonta Smith", "pos": "WR", "team": "PHI"},
        "13977": {"name": "DeVonta Smith", "position": "CB", "team": "CAR"},
    }
    by_name = _name_index_from_players(index, norm)
    assert _pick_canonical(by_name, "lamar jackson", "QB") == "4881"
    assert _pick_canonical(by_name, "lamar jackson", "CB") == "6994"
    assert _pick_canonical(by_name, "devonta smith", "WR") == "7525"
    # Fleaflicker QB/WR must not fall through to the IDP namesake.
    assert _pick_canonical(by_name, "lamar jackson", "") == "4881"
    assert _pick_canonical(by_name, "devonta smith", "") == "7525"


def test_pick_canonical_legacy_tuple_map_still_works():
    by_name = {("lamar jackson", "QB"): "4881", ("lamar jackson", "CB"): "6994"}
    assert _pick_canonical(by_name, "lamar jackson", "QB") == "4881"
    assert _pick_canonical(by_name, "lamar jackson", "CB") == "6994"


def test_pick_canonical_uses_nfl_team_for_namesakes():
    def norm(name):
        return (name or "").strip().lower()

    index = {
        "4881": {"name": "Lamar Jackson", "pos": "QB", "team": "BAL"},
        "6994": {"name": "Lamar Jackson", "pos": "CB", "team": "ATL"},
        "7525": {"name": "DeVonta Smith", "pos": "WR", "team": "PHI"},
        "13977": {"name": "DeVonta Smith", "position": "CB", "team": "CAR"},
        "4984": {"name": "Josh Allen", "pos": "QB", "team": "BUF"},
        "5846": {"name": "Josh Allen", "pos": "DE", "team": "JAX"},
        "1": {"name": "Terry McLaurin", "pos": "WR", "team": "WAS"},
    }
    by_name = _name_index_from_players(index, norm)
    # Team alone (Fleaflicker sometimes omits position on boxscore rows).
    assert _pick_canonical(by_name, "lamar jackson", "", "BAL") == "4881"
    assert _pick_canonical(by_name, "lamar jackson", "", "ATL") == "6994"
    assert _pick_canonical(by_name, "devonta smith", "", "PHI") == "7525"
    assert _pick_canonical(by_name, "devonta smith", "", "CAR") == "13977"
    assert _pick_canonical(by_name, "josh allen", "QB", "BUF") == "4984"
    assert _pick_canonical(by_name, "josh allen", "DE", "JAX") == "5846"
    # WAS/WSH schedule alias.
    assert _pick_canonical(by_name, "terry mclaurin", "WR", "WSH") == "1"


def test_flea_pro_team_reads_abbrev_and_nested():
    assert _flea_pro_team({"proTeamAbbreviation": "BAL"}) == "BAL"
    assert _flea_pro_team({"proTeam": {"abbreviation": "PHI"}}) == "PHI"
    assert _flea_pro_team({"pro_team": {"abbreviation": "car"}}) == "CAR"
    assert _flea_pro_team({}) == ""


def test_fleaflicker_scoring_maps_group_abbreviations():
    rules = {
        "groups": [
            {"label": "Passing", "scoringRules": [
                {
                    "category": {"abbreviation": "Yd", "nameSingular": "Passing Yard"},
                    "points": {"value": 1}, "pointsPer": {"value": 0.04}, "forEvery": 25,
                },
                {
                    "category": {"abbreviation": "TD", "nameSingular": "Passing TD"},
                    "points": {"value": 4}, "forEvery": 1,
                },
                {
                    "category": {"abbreviation": "INT", "nameSingular": "Interception"},
                    "points": {"value": -2}, "forEvery": 1,
                },
            ]},
            {"label": "Rushing", "scoringRules": [
                {
                    "category": {"abbreviation": "Yd", "nameSingular": "Rushing Yard"},
                    "points": {"value": 1}, "pointsPer": {"value": 0.1}, "forEvery": 10,
                },
                {
                    "category": {"abbreviation": "TD", "nameSingular": "Rushing TD"},
                    "points": {"value": 6}, "forEvery": 1,
                },
            ]},
            {"label": "Receiving", "scoringRules": [
                {
                    "category": {"abbreviation": "Rec", "nameSingular": "Catch"},
                    "points": {"value": 1}, "pointsPer": {"value": 1.0}, "forEvery": 1,
                },
                {
                    "category": {"abbreviation": "Yd", "nameSingular": "Receiving Yard"},
                    "points": {"value": 1}, "pointsPer": {"value": 0.1}, "forEvery": 10,
                },
                {
                    "category": {"abbreviation": "TD", "nameSingular": "Receiving TD"},
                    "points": {"value": 6}, "forEvery": 1,
                },
            ]},
            {"label": "Misc", "scoringRules": [
                {
                    "category": {"abbreviation": "Fum", "nameSingular": "Fumble"},
                    "points": {"value": -2}, "forEvery": 1,
                },
            ]},
        ]
    }
    out = FleaflickerProvider._scoring(rules)
    assert out["pass_yd"] == 0.04
    assert out["pass_td"] == 4.0
    assert out["pass_int"] == -2.0
    assert out["rush_yd"] == 0.1
    assert out["rush_td"] == 6.0
    assert out["rec"] == 1.0
    assert out["rec_yd"] == 0.1
    assert out["rec_td"] == 6.0
    assert out["fum_lost"] == -2.0
    assert "TD" not in out
    assert "Yd" not in out

    from utils.fantasy_scoring import projection_points
    pts = projection_points(
        {"raw_stats": {
            "pass_yd": 250, "pass_td": 2, "rush_yd": 20,
            "pts_ppr": 20.0, "pts_std": 18.0,
        }},
        out, "QB",
    )
    assert pts == 20.0


def test_fleaflicker_scoring_divides_points_by_for_every():
    rules = {"groups": [{"label": "Passing", "scoring_rules": [
        {"category": {"abbreviation": "Yd"}, "points": {"value": 1}, "for_every": 25},
    ]}]}
    assert FleaflickerProvider._scoring(rules)["pass_yd"] == 0.04


def _standard_flea_rules_with_bonuses():
    """All American All Star League (92916) shape: standard scoring + milestones."""
    return {"groups": [
        {"label": "Passing", "scoringRules": [
            {
                "category": {"abbreviation": "Cmp", "nameSingular": "Passing Completion"},
                "points": {"value": 2}, "boundLower": 25,
                "description": "2 extra points when total Passing Completions is greater than or equal to 25",
            },
            {
                "category": {"abbreviation": "Yd", "nameSingular": "Passing Yard"},
                "points": {"value": 1}, "pointsPer": {"value": 0.05}, "forEvery": 20,
                "description": "1 point for every 20 Passing Yards (0.05 per)",
            },
            {
                "category": {"abbreviation": "Yd", "nameSingular": "Passing Yard"},
                "points": {"value": 1}, "boundLower": 300,
                "description": "1 extra point when total Passing Yards is greater than or equal to 300",
            },
            {
                "category": {"abbreviation": "TD", "nameSingular": "Passing TD"},
                "points": {"value": 4}, "forEvery": 1,
            },
            {
                "category": {"abbreviation": "INT", "nameSingular": "Interception"},
                "points": {"value": -2}, "forEvery": 1,
            },
        ]},
        {"label": "Rushing", "scoringRules": [
            {
                "category": {"abbreviation": "Yd", "nameSingular": "Rushing Yard"},
                "points": {"value": 1}, "pointsPer": {"value": 0.1}, "forEvery": 10,
            },
            {
                "category": {"abbreviation": "TD", "nameSingular": "Rushing TD"},
                "points": {"value": 6}, "forEvery": 1,
            },
        ]},
        {"label": "Receiving", "scoringRules": [
            {
                "category": {"abbreviation": "Rec", "nameSingular": "Catch"},
                "points": {"value": 2}, "boundLower": 9,
                "description": "2 extra points when total Catches is greater than or equal to 9",
                "template": "${points} extra points when total ${categoryPlural} is greater than or equal to ${boundLower}",
            },
            {
                "category": {"abbreviation": "Yd", "nameSingular": "Receiving Yard"},
                "points": {"value": 1}, "pointsPer": {"value": 0.1}, "forEvery": 10,
            },
            {
                "category": {"abbreviation": "Yd", "nameSingular": "Receiving Yard"},
                "points": {"value": 1}, "boundLower": 150,
                "description": "1 extra point when total Receiving Yards is greater than or equal to 150",
            },
            {
                "category": {"abbreviation": "TD", "nameSingular": "Receiving TD"},
                "points": {"value": 6}, "forEvery": 1,
            },
        ]},
        {"label": "Defense", "scoringRules": [
            {
                "category": {"abbreviation": "INT", "nameSingular": "Interception"},
                "points": {"value": 2}, "forEvery": 1,
            },
        ]},
    ]}


def test_fleaflicker_threshold_bonuses_are_not_per_stat_rates():
    out = FleaflickerProvider._scoring(_standard_flea_rules_with_bonuses())
    assert out["pass_yd"] == 0.05
    assert out["pass_td"] == 4.0
    assert out["pass_int"] == -2.0
    assert out["rush_yd"] == 0.1
    assert out["rec_yd"] == 0.1
    assert out["rec_td"] == 6.0
    assert "rec" not in out
    assert out.get("def_int") == 2.0


def test_fleaflicker_standard_league_is_not_ppr():
    from utils.league_scoring import normalize_league_scoring

    raw = FleaflickerProvider._scoring(_standard_flea_rules_with_bonuses())
    norm = normalize_league_scoring("fleaflicker", raw)
    assert norm["rec"] == 0.0


def test_fleaflicker_standard_weekly_proj_is_not_one_point_per_yard():
    from utils.fantasy_scoring import projection_points
    from utils.league_scoring import normalize_league_scoring

    ss = normalize_league_scoring(
        "fleaflicker", FleaflickerProvider._scoring(_standard_flea_rules_with_bonuses()),
    )
    pts = projection_points(
        {"raw_stats": {
            "rec": 6.0, "rec_yd": 83.0, "rec_td": 0.5,
            "pts_ppr": 17.3, "pts_half_ppr": 14.3, "pts_std": 11.3,
        }},
        ss, "WR",
    )
    assert pts < 20
    assert pts != 83.0
    assert pts == pytest.approx(11.3, abs=0.2)


def test_get_matchups_uses_boxscore_slot_order(monkeypatch):
    provider = FleaflickerProvider()
    payloads = {
        "FetchLeagueScoreboard": {
            "games": [{
                "id": 77,
                "home": {"id": 1},
                "away": {"id": 2},
                "homeScore": {"score": {"value": 0}},
                "awayScore": {"score": {"value": 0}},
            }],
        },
        "FetchLeagueBoxscore": {
            "lineups": [{
                "group": "START",
                "slots": [
                    {
                        "position": {"label": "QB", "group": "START"},
                        "home": {
                            "proPlayer": {"id": 9, "nameFull": "Bo Nix", "position": "QB"},
                            "viewingActualPoints": {"value": 0},
                        },
                        "away": {
                            "proPlayer": {
                                "id": 8, "nameFull": "Lamar Jackson", "position": "QB",
                            },
                            "viewingActualPoints": {"value": 0},
                        },
                    },
                    {
                        "position": {"label": "RB", "group": "START"},
                        "home": {
                            "proPlayer": {
                                "id": 10, "nameFull": "Bijan Robinson", "position": "RB",
                            },
                        },
                        "away": {
                            "proPlayer": {
                                "id": 11, "nameFull": "Breece Hall", "position": "RB",
                            },
                        },
                    },
                ],
            }],
        },
    }
    monkeypatch.setattr(provider, "_call", lambda method, *a, **k: payloads[method])
    monkeypatch.setattr(
        provider, "_canonical_map",
        lambda *a, **k: {"9": "nix", "8": "lamar", "10": "bijan", "11": "breece"},
    )
    monkeypatch.setattr(provider, "_build_name_index", lambda: {})
    rows = provider.get_matchups("14153", 2026, 1)
    home = next(r for r in rows if r["roster_id"] == 1)
    away = next(r for r in rows if r["roster_id"] == 2)
    assert home["starters"] == ["nix", "bijan"]
    assert away["starters"] == ["lamar", "breece"]
    assert home["matchup_id"] == away["matchup_id"]
