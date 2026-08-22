import json

from dashboard_services.market_intelligence.client import SportsGameOddsClient
from dashboard_services.market_intelligence.provider_debug import (
    MAX_BOOK_ROWS, MAX_EVENTS, MAX_PLAYER_ODDS, MAX_TEAM_FAILURES, MAX_TEAM_ODDS,
    SportsGameOddsDebug, sanitize_identity,
)
from dashboard_services.market_intelligence.team import build_team_environments


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *_args, **_kwargs):
        return _Response(self.payload)


def _payload(event_count=5, odds_per_event=20):
    events = []
    for event_index in range(event_count):
        odds = []
        for odd_index in range(odds_per_event):
            player = odd_index % 2 == 0
            odds.append({
                "oddID": f"odd-{event_index}-{odd_index}",
                "statID": "passing_yards" if player else "points",
                "statEntityID": "player" if player else "all", "periodID": "game",
                "betTypeID": "ou", "sideID": "over", "bookOverUnder": 47.5,
                "byBookmaker": {"book": {"available": True, "odds": -110,
                                             "overUnder": 47.5, "secret": "remove-me"}},
            })
        events.append({
            "eventID": f"event-{event_index}", "leagueID": "NFL", "eventType": "match",
            "teams": {"opaque-home": {"teamID": "opaque-home", "name": "Kansas City Chiefs",
                                        "abbreviation": "KC", "alignment": "home",
                                        "huge": list(range(1000))}},
            "status": {"startsAt": "2030-01-01T00:00:00Z", "accessToken": "remove-me"},
            "odds": odds, "players": [{"huge": "x" * 10000}],
        })
    return {"data": events, "nextCursor": None, "apiKey": "payload-secret"}


def test_provider_debug_is_off_by_default(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("MARKET_DEBUG_PROVIDER_RESPONSES", raising=False)
    client = SportsGameOddsClient("api-secret", _Session(_payload(1, 1)), cache_dir=tmp_path)
    assert len(list(client.iter_nfl_events(starts_after="a", starts_before="b"))) == 1
    assert "market-debug" not in capsys.readouterr().out


def test_debug_output_is_sanitized_and_strictly_bounded(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MARKET_DEBUG_PROVIDER_RESPONSES", "1")
    snapshot = tmp_path / "sample.json"
    client = SportsGameOddsClient("api-secret", _Session(_payload()), cache_dir=tmp_path / "cache")
    client.debug.snapshot_path = snapshot
    assert len(list(client.iter_nfl_events(starts_after="a", starts_before="b"))) == 5
    output = capsys.readouterr().out
    assert output.count("eventID: ") == MAX_EVENTS
    assert output.count("player odd sample=") == MAX_PLAYER_ODDS
    assert output.count("team odd sample=") == MAX_TEAM_ODDS
    assert output.count("bookmaker row sample=") == MAX_BOOK_ROWS
    assert "api-secret" not in output and "payload-secret" not in output and "remove-me" not in output
    assert len(output) < 30_000
    saved = snapshot.read_text()
    assert len(json.loads(saved)["events"]) == MAX_EVENTS
    assert "accessToken" not in saved and "players" not in saved and "remove-me" not in saved


def test_identity_sanitizer_retains_nested_team_shape_but_drops_large_fields():
    safe = sanitize_identity({
        "teams": {"opaque": {"teamID": "opaque", "shortName": "Chiefs",
                               "abbreviation": "KC", "alignment": "home", "odds": [1] * 1000}},
        "authorization": "secret", "odds": [1] * 1000,
    })
    assert safe["teams"]["opaque"]["teamID"] == "opaque"
    assert safe["teams"]["opaque"]["abbreviation"] == "KC"
    assert "odds" not in safe["teams"]["opaque"]
    assert "authorization" not in safe


def test_team_resolution_debug_and_identity_counters_are_bounded(capsys):
    debug = SportsGameOddsDebug(enabled=True)
    events = [{"eventID": str(i), "status": {"startsAt": "2030-01-01T00:00:00Z"},
               "participants": {"unexpected": {"name": "Unknown Club"}}, "odds": []}
              for i in range(10)]
    diagnostics = {}
    assert build_team_environments(events, diagnostics, debug=debug) == {}
    output = capsys.readouterr().out
    assert output.count("event team resolution failed") == MAX_TEAM_FAILURES
    assert diagnostics["events_team_container_unrecognized_shape"] == 10
    assert diagnostics["events_missing_home"] == 10
    assert diagnostics["events_missing_away"] == 10
    assert diagnostics["events_team_identity_resolved"] == 0
