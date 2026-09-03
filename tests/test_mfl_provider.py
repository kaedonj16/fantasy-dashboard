from unittest.mock import Mock, patch

import pytest

from dashboard_services.providers.base import ProviderUnavailableError
from dashboard_services.providers.mfl_api import MFLProvider, _CACHE


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()


def test_normalizes_league_users_rosters_matchups_and_picks(monkeypatch):
    provider = MFLProvider()
    payloads = {
        "league": {"league": {"id": "123", "name": "Dynasty", "size": "2",
                    "lastRegularSeasonWeek": "14", "starters": "QB,RB,WR",
                    "franchises": {"franchise": [{"id": "0001", "name": "Owls", "owner_name": "Ada"}]}}},
        "players": {"players": {"player": [{"id": "9", "name": "Known Player", "position": "QB"}]}},
        "rosters": {"rosters": {"franchise": [{"id": "0001", "player": [{"id": "9"}, {"id": "404"}]}]}},
        "weeklyResults": {"weeklyResults": {"matchup": [{"franchise": [{"id": "0001", "score": "101.5"}, {"id": "0002", "score": "99"}]}]}},
        "futureDraftPicks": {"futureDraftPicks": {"futureDraftPick": [{"year": "2027", "round": "1", "originalPickFor": "0001", "currentPickFor": "0002"}]}},
    }
    monkeypatch.setattr(provider, "_export", lambda kind, *a, **k: payloads[kind])
    monkeypatch.setattr(provider, "_canonical_map", lambda *a: {"9": "canon-9"})
    assert provider.get_league("123", 2026)["total_rosters"] == 2
    assert provider.get_users("123", 2026)[0]["user_id"] == "0001"
    roster = provider.get_rosters("123", 2026)[0]
    assert roster["players"] == ["canon-9"]
    assert roster["metadata"]["unmapped_player_count"] == 1
    assert provider.get_matchups("123", 2026, 1)[0]["points"] == 101.5
    assert provider.get_traded_picks("123", 2026)[0]["owner_id"] == 2


def test_normalizes_transactions_and_draft_results(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "_export", lambda kind, *a, **k: {
        "transactions": {"transactions": {"transaction": [{"id": "tx1", "type": "TRADE", "franchise": "0001,0002", "timestamp": "10"}]}},
        "draftResults": {"draftResults": {"draftUnit": [{"round": "1", "pick": "1", "franchise": "0001", "player": "9"}]}},
    }[kind])
    assert provider.get_transactions("123", 2026, 1)[0]["type"] == "trade"
    draft = provider.get_drafts("123", 2026)[0]
    assert draft["picks"][0]["pick_no"] == 1
    assert draft["status"] == "complete"


def test_mfl_empty_draft_results_are_predraft(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "_export", lambda *a, **k: {"draftResults": {}})
    draft = provider.get_drafts("123", 2026)[0]
    assert draft["status"] == "pre_draft"
    assert draft["picks"] == []


def test_mfl_unpicked_slots_are_predraft(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "_export", lambda *a, **k: {
        "draftResults": {"draftUnit": [
            {"round": "1", "pick": "1", "franchise": "0001", "player": ""},
            {"round": "1", "pick": "2", "franchise": "0002"},
        ]},
    })
    draft = provider.get_drafts("123", 2026)[0]
    assert draft["status"] == "pre_draft"
    assert draft["picks"] == []


def test_mfl_partial_draft_is_drafting(monkeypatch):
    provider = MFLProvider()
    monkeypatch.setattr(provider, "_export", lambda *a, **k: {
        "draftResults": {"draftUnit": [
            {"round": "1", "pick": "1", "franchise": "0001", "player": "9"},
            {"round": "1", "pick": "2", "franchise": "0002", "player": ""},
        ]},
    })
    draft = provider.get_drafts("123", 2026)[0]
    assert draft["status"] == "drafting"
    assert len(draft["picks"]) == 1


@patch("dashboard_services.providers.mfl_api._request_get")
def test_upstream_timeout_is_safe(mock_get):
    mock_get.side_effect = ProviderUnavailableError("MyFantasyLeague is temporarily unavailable.")
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        MFLProvider().get_league("123", 2026)


@patch("dashboard_services.providers.mfl_api._request_get")
def test_private_export_sends_cookie_and_apikey(mock_get):
    mock_get.return_value = response({"league": {"id": "123", "name": "Private", "size": "1",
                                                  "franchises": {"franchise": []}}})
    MFLProvider().get_league("123", 2026, cookie="MFL_USER_ID=abc", apikey="key-1")
    kwargs = mock_get.call_args.kwargs
    assert kwargs["cookies"] == {"MFL_USER_ID": "abc"}
    assert kwargs["params"]["APIKEY"] == "key-1"


@patch("dashboard_services.providers.mfl_api._request_post")
def test_mfl_login_returns_cookie(mock_post):
    from dashboard_services.providers.mfl_api import login
    result = Mock(status_code=200, headers={}, text="", cookies={"MFL_USER_ID": "cookie-value"})
    result.raise_for_status.return_value = None
    mock_post.return_value = result
    assert login("user", "pass", 2026) == "cookie-value"
