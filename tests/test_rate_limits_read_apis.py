"""Rate limits on the heavy read APIs (pre-deployment checklist follow-up).

/api/players, /api/league-players, /api/teams, /api/league-rosters,
/api/value-movers, /api/player-deltas, and the stacked redzone-data /
scorezone-data view fan out to league context builds, value tables, or
play data per call and were uncapped. Each now carries a generous
"300 per minute" per-IP limit that legitimate use (a page load fans out
to a handful of these calls) can never trip.

Behavioral tests only: hit real endpoints from an isolated REMOTE_ADDR
until a 429 appears. No flask-limiter private attributes are inspected
(that introspection broke before on a limiter upgrade). The three
endpoints exercised end to end are the cheapest shapes of their views
(empty league id short-circuits, no-match search), so 300 requests stay
fast. A source-level test covers all eight routes, including the shared
redzone/scorezone decorator.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("flask_limiter")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")

READ_LIMIT = 300

ROUTES = [
    "/api/players",
    "/api/league-players",
    "/api/teams",
    "/api/league-rosters",
    "/api/value-movers",
    "/api/player-deltas",
    "/api/<platform>/<int:season>/<league_id>/redzone-data",
    "/api/<platform>/<int:season>/<league_id>/scorezone-data",
]


def test_all_eight_read_routes_carry_the_300_per_minute_limit():
    for route in ROUTES:
        marker = f'@app.route("{route}")'
        assert APP_PY.count(marker) == 1, route
        after = APP_PY.split(marker, 1)[1]
        decorator_block = after.split("def ", 1)[0]
        assert '@limiter.limit("300 per minute")' in decorator_block, route


def _assert_limited(client, url, remote_addr):
    env = {"REMOTE_ADDR": remote_addr}
    for i in range(READ_LIMIT):
        resp = client.get(url, environ_overrides=env)
        assert resp.status_code != 429, f"{url} limited early at request {i + 1}"
        assert resp.status_code == 200, f"{url} request {i + 1} -> {resp.status_code}"
    resp = client.get(url, environ_overrides=env)
    assert resp.status_code == 429, f"{url} request {READ_LIMIT + 1} -> {resp.status_code}"


def test_api_teams_is_rate_limited(offline_client):
    # No league_id: the view returns [] immediately, so this is the cheap
    # shape of the endpoint while still exercising the real limiter.
    _assert_limited(offline_client, "/api/teams", "10.20.0.1")


def test_api_league_rosters_is_rate_limited(offline_client):
    # No league_id: returns {"teams": [], "viewer_roster_id": ""} at once.
    _assert_limited(offline_client, "/api/league-rosters", "10.20.0.2")


def test_api_players_is_rate_limited(offline_client):
    # A query that matches no player keeps each response tiny.
    _assert_limited(
        offline_client, "/api/players?q=zzznomatchzz&limit=1", "10.20.0.3"
    )
