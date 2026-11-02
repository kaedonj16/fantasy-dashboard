"""Trade + nav-search league-players views must stamp espnHeadshot from
the players_index espnID, like the full and rankings views already do.

The base payload only carries the stored espnHeadshot string, and newer
players_index entries (rookie classes added with an espnID only, via
add_new_players_from_tank01.py) have no stored string. Without the stamp
those players ship blank in ?view=trade / ?view=search and their trade
chips render an empty face even though ESPN has a photo for them.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import utils.utils as uu

FAKE_INDEX = {
    # 2026 rookie: espnID present, no stored espnHeadshot string.
    "13287": {"name": "Jeremiyah Love", "espnID": "4870808"},
    # Vet with the stored string already populated.
    "9221": {
        "name": "Jahmyr Gibbs",
        "espnID": "4429795",
        "espnHeadshot": "https://a.espncdn.com/i/headshots/nfl/players/full/4429795.png",
    },
    # No ESPN id at all: stays blank, never a fabricated URL.
    "99999": {"name": "No Photo"},
}

ROOKIE_URL = "https://a.espncdn.com/i/headshots/nfl/players/full/4870808.png"
GIBBS_URL = "https://a.espncdn.com/i/headshots/nfl/players/full/4429795.png"


@pytest.fixture
def fake_index(monkeypatch):
    monkeypatch.setattr(uu, "load_players_index", lambda: dict(FAKE_INDEX))
    return FAKE_INDEX


def _payload():
    return {
        "players": [
            {"id": "13287", "name": "Jeremiyah Love", "position": "RB",
             "team": "ARI", "value": 900.0, "espnHeadshot": ""},
            {"id": "9221", "name": "Jahmyr Gibbs", "position": "RB",
             "team": "DET", "value": 1100.0, "espnHeadshot": GIBBS_URL},
            {"id": "99999", "name": "No Photo", "position": "WR",
             "team": "FA", "value": 10.0, "espnHeadshot": ""},
        ]
    }


def test_stamp_fills_rookie_from_espnid_and_never_mutates_input(fake_index):
    import app

    payload = _payload()
    out = app._stamp_espn_headshot(payload)
    by_id = {p["id"]: p for p in out["players"]}

    assert by_id["13287"]["espnHeadshot"] == ROOKIE_URL
    # Stored string wins untouched.
    assert by_id["9221"]["espnHeadshot"] == GIBBS_URL
    # No espnID in the index: no invented URL.
    assert not by_id["99999"]["espnHeadshot"]
    # The memoized base payload is never mutated.
    assert payload["players"][0]["espnHeadshot"] == ""


def test_trade_view_ships_stamped_rookie_headshot(fake_index):
    import json
    import uuid

    import app

    with app.app.app_context():
        resp = app._trade_league_players_response(
            _payload(),
            overlay_key=f"test-{uuid.uuid4().hex}",
            is_sf=False,
            league_type="1qb",
            league_size=10,
        )
    body = json.loads(resp.get_data(as_text=True))
    by_id = {p["id"]: p for p in body["players"]}

    assert by_id["13287"]["espnHeadshot"] == ROOKIE_URL
    assert by_id["9221"]["espnHeadshot"] == GIBBS_URL
    assert not by_id["99999"].get("espnHeadshot")


def test_search_view_slim_keeps_stamped_rookie_headshot(fake_index):
    import app
    from dashboard_services.league_players_board import slim_search_payload

    slim = slim_search_payload(app._stamp_espn_headshot(_payload()))
    by_id = {p["id"]: p for p in slim["players"]}

    assert by_id["13287"]["espnHeadshot"] == ROOKIE_URL
    assert by_id["9221"]["espnHeadshot"] == GIBBS_URL


def test_search_route_stamps_before_slimming():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text()
    assert "slim_search_payload(_stamp_espn_headshot(payload))" in src
    assert "slim_trade_payload(_stamp_espn_headshot(payload)" in src
