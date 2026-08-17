"""DraftKings season-long futures provider (free, unofficial).

Exercises the payload parser, the stat->subcategory config, and the off-by-default
gating without any network or requests dependency (a fake session is injected).
Field mapping is validated against the confirmed sportscontent response shape.
"""
from dashboard_services.market_intelligence.draftkings import (
    DraftKingsClient,
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


_PAYLOAD = {
    "markets": [{"id": "m1", "name": "Patrick Mahomes"}],
    "selections": [
        {"marketId": "m1", "label": "Over", "points": 4500.5, "displayOdds": {"american": "-110"}},
        {"marketId": "m1", "label": "Under", "points": 4500.5, "displayOdds": {"american": "-110"}},
    ],
}


def test_parse_market_map_keeps_known_stats_only():
    got = _parse_market_map("passing_yards=4501, rushing_yards=4502, bogus=9, receiving_yards=")
    assert got == {"passing_yards": "4501", "rushing_yards": "4502"}


def test_season_records_from_payload_maps_confirmed_fields():
    records = season_records_from_payload(_PAYLOAD, "passing_yards", "evt-1")
    assert len(records) == 1
    r = records[0]
    assert r.stat_type == "passing_yards"
    assert r.line == 4500.5
    assert r.context == "season"
    assert r.sportsbook == "draftkings"
    assert r.provider_player_id == "dk:Patrick Mahomes"
    assert r.over_price == -110.0 and r.under_price == -110.0


def test_line_parsed_from_label_when_no_points_field():
    # The controldata feed may omit a dedicated line field; recover it from the
    # "Over 3999.5" label.
    payload = {"markets": [{"id": "m1", "name": "Josh Allen Passing Yards"}],
               "selections": [
                   {"marketId": "m1", "label": "Over 3999.5", "displayOdds": {"american": "-115"}},
                   {"marketId": "m1", "label": "Under 3999.5", "displayOdds": {"american": "-105"}}]}
    records = season_records_from_payload(payload, "passing_yards", "evt")
    assert len(records) == 1
    assert records[0].line == 3999.5


def test_season_records_skips_market_without_over_or_line():
    payload = {"markets": [{"id": "m1", "name": "Nobody"}],
               "selections": [{"marketId": "m1", "label": "Under", "points": 10, "displayOdds": {"american": "-110"}}]}
    assert season_records_from_payload(payload, "passing_yards", "evt") == []


def test_client_off_by_default(monkeypatch):
    for key in ("DRAFTKINGS_NFL_LEAGUE_ID", "DRAFTKINGS_NFL_SEASON_MARKETS"):
        monkeypatch.delenv(key, raising=False)
    assert DraftKingsClient(session=_FakeSession({})).configured is False


def test_client_iterates_configured_markets(monkeypatch):
    monkeypatch.setenv("DRAFTKINGS_NFL_LEAGUE_ID", "88808")
    monkeypatch.setenv("DRAFTKINGS_NFL_SEASON_MARKETS", "passing_yards=4501")
    client = DraftKingsClient(session=_FakeSession(_PAYLOAD))
    assert client.configured is True
    got = list(client.iter_season_markets())
    assert len(got) == 1 and got[0][0] == "passing_yards"
    # The OData request carries the configured league + subcategory id.
    params = client.session.calls[0]["params"]
    assert params["templateVars"] == "88808"
    assert "4501" in params["marketsQuery"]
