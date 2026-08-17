"""DraftKings season-long futures provider (free, unofficial).

The fixture mirrors a real NFL "Regular Season Receiving TDs" response
(subcategory 17315): the player is on the event, the line is in the Over
selection's label ("Over 6.5"), the side is outcomeType, and negative odds use a
Unicode minus. No network or requests dependency (a fake session is injected).
"""
from dashboard_services.market_intelligence.draftkings import (
    DraftKingsClient,
    _american_from_selection,
    _parse_market_map,
    season_records_from_payload,
)


class _FakeResp:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeSession:
    def __init__(self, data):
        self._data = data
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _FakeResp(self._data)


# Trimmed from a real US-SB response for subcategory 17315.
_PAYLOAD = {
    "events": [{
        "id": "evt-kelce",
        "name": "NFL 2026/27 - Travis Kelce",
        "startEventDate": "2026-09-13T18:00:00.0000000Z",
        "participants": [
            {"id": "18379", "name": "KC Chiefs", "metadata": {"shortName": "KC"}},
            {"name": "Travis Kelce", "type": "Team"},
        ],
    }],
    "markets": [{
        "id": "m-kelce", "eventId": "evt-kelce", "subcategoryId": "17315",
        "name": "NFL 2026/27 – Travis Kelce Regular Season Receiving TDs",
    }],
    "selections": [
        {"marketId": "m-kelce", "outcomeType": "Over", "label": "Over 6.5",
         "displayOdds": {"american": "−105"}},
        {"marketId": "m-kelce", "outcomeType": "Under", "label": "Under 6.5",
         "displayOdds": {"american": "−120"}},
    ],
}


def test_parse_market_map_keeps_known_stats_only():
    got = _parse_market_map("receiving_touchdowns=17315, rushing_yards=4502, bogus=9, passing_yards=")
    assert got == {"receiving_touchdowns": "17315", "rushing_yards": "4502"}


def test_american_normalizes_unicode_minus():
    assert _american_from_selection({"displayOdds": {"american": "−105"}}) == -105.0
    assert _american_from_selection({"displayOdds": {"american": "+115"}}) == 115.0


def test_season_records_from_real_payload_shape():
    records = season_records_from_payload(_PAYLOAD, "receiving_touchdowns")
    assert len(records) == 1
    r = records[0]
    assert r.stat_type == "receiving_touchdowns"
    assert r.line == 6.5                              # line read from the "Over 6.5" label
    assert r.context == "season"
    assert r.sportsbook == "draftkings"
    assert r.provider_player_id == "dk:Travis Kelce"  # player from the EVENT, not the market name
    assert r.over_price == -105.0 and r.under_price == -120.0  # Unicode minus normalized


def test_season_records_skips_market_without_over():
    payload = {"events": [{"id": "e", "name": "NFL 2026/27 - Nobody"}],
               "markets": [{"id": "m", "eventId": "e"}],
               "selections": [{"marketId": "m", "outcomeType": "Under", "label": "Under 5.5",
                               "displayOdds": {"american": "−110"}}]}
    assert season_records_from_payload(payload, "receiving_touchdowns") == []


def test_client_on_by_default_with_baked_in_ids(monkeypatch):
    for key in ("DRAFTKINGS_NFL_LEAGUE_ID", "DRAFTKINGS_NFL_SEASON_MARKETS",
                "DRAFTKINGS_SEASON_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    client = DraftKingsClient(session=_FakeSession({}))
    assert client.configured is True               # works with zero config
    assert client.league_id == "88808"
    # All seven Player Stats O/U tabs are baked in.
    assert client.market_map["receiving_touchdowns"] == "17315"
    assert client.market_map["passing_yards"] == "17147"
    assert client.market_map["receptions"] == "20168"
    assert len(client.market_map) == 7


def test_enabled_flag_kills_the_source(monkeypatch):
    monkeypatch.setenv("DRAFTKINGS_SEASON_ENABLED", "0")
    assert DraftKingsClient(session=_FakeSession({})).configured is False


def test_env_map_overrides_baked_in_defaults(monkeypatch):
    monkeypatch.setenv("DRAFTKINGS_NFL_SEASON_MARKETS", "receiving_touchdowns=17315")
    client = DraftKingsClient(session=_FakeSession(_PAYLOAD))
    assert client.market_map == {"receiving_touchdowns": "17315"}
    got = list(client.iter_season_markets())
    assert len(got) == 1 and got[0][0] == "receiving_touchdowns"
    params = client.session.calls[0]["params"]
    assert params["templateVars"] == "88808,17315"   # league + subcategory, comma-joined
    assert "17315" in params["marketsQuery"]
