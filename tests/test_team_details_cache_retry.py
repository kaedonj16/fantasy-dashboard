"""Team modal "will not load" hardening.

Live check 2026-10-26: a team whose name contains an emoji ("Super Cena -
✋🏾", League of Greatness roster 1) loads fine; the payload carries the
name intact. What actually stranded the modal was the /api/team-details
build being slow on a cold worker (15-30s) and intermittently 502ing at
the gateway, for every team in the league. Two guards:

- the endpoint caches its built payload briefly per roster, so repeat
  opens (and retries) do not rebuild it;
- fetchTeamDetails retries a transient 502/503/504 or dropped connection
  once before showing the error panel.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = (_ROOT / "app.py").read_text(encoding="utf-8")
_APP_JS = (_ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _handler_body(start_marker, end_marker):
    start = _APP_PY.find(start_marker)
    assert start != -1
    end = _APP_PY.find(end_marker, start)
    assert end != -1
    return _APP_PY[start:end]


def test_team_details_route_caches_built_payload():
    body = _handler_body('@app.route("/api/team-details/<roster_id>")',
                         '@app.route("/api/player-league-trades/<player_id>")')
    assert "_TEAM_DETAILS_CACHE.get(_td_key)" in body
    assert "_TEAM_DETAILS_CACHE[_td_key] = (time.time(), cleaned_response)" in body
    assert "_TEAM_DETAILS_TTL = 120" in _APP_PY
    # The cache lookup must happen before the expensive league build.
    assert body.index("_TEAM_DETAILS_CACHE.get(_td_key)") < body.index("get_league(platform, league_id, season)")


def test_fetch_team_details_retries_transient_gateway_errors_once():
    start = _APP_JS.find("async function fetchTeamDetails")
    assert start != -1
    body = _APP_JS[start:start + 6000]
    assert "attempt < 2" in body
    assert "[502, 503, 504].includes(response.status)" in body
