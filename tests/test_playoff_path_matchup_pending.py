"""Your Matchup card intermittently missing on the Playoff Path Overview.

Kaedon (2026-11-10, Week 10, games in progress): "sometimes I get a
matchup card and sometimes I don't on overview" -- same league, same week.

Root cause: the card is built from the live matchup bundle
(routes.user_pages_bp._build_live_matchups), which is
stale-while-revalidate: on a cold key it kicks a background rebuild and
returns None, never building on the request thread. The scenarios
endpoint translated that None into a silently omitted card -- a response
indistinguishable from the legitimate absences (no viewer team, bye
week) -- and the client rendered once and locked the card-less render
into the panel's 5-minute TTL. Whether the card appeared depended
entirely on whether something else (the Matchups page, a portfolio
card) had already warmed that league/week bundle.

Fix pinned here: the endpoint flags that one case as
your_matchup_pending (the same pending contract the league-scores
endpoint returns), and initPlayoffPath polls the scenarios endpoint
every 3s, bounded, rendering with the first response that carries the
card. Legitimate absences keep pending=False and the card stays hidden.
"""
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()

CTX = {
    "league_settings": {},
    "current_week": 10,
    "rosters": [],
    "users": [],
    "resolved_league_id": "L1",
}

STATE = {
    "playoff_teams": 2,
    "teams": [
        {"roster_id": 1, "name": "Couch", "wins": 7, "losses": 2,
         "ties": 0, "pf": 1427.6},
        {"roster_id": 2, "name": "Pilots", "wins": 8, "losses": 1,
         "ties": 0, "pf": 1407.7},
        {"roster_id": 3, "name": "Fire Riley", "wins": 7, "losses": 2,
         "ties": 0, "pf": 1349.4},
        {"roster_id": 4, "name": "Nunky", "wins": 5, "losses": 4,
         "ties": 0, "pf": 1313.1},
    ],
    "matchups": {10: [(1, 2), (3, 4)]},
}

BUNDLE = (
    [{"left": {"roster_id": 1, "name": "Couch"},
      "right": {"roster_id": 2, "name": "Pilots"}},
     {"left": {"roster_id": 3, "name": "Fire Riley"},
      "right": {"roster_id": 4, "name": "Nunky"}}],
    {},
    {},
)

URL = ("/api/playoff-scenarios?platform=sleeper&league_id=L1"
       "&season=2026&roster_id=1")


@pytest.fixture()
def client(monkeypatch):
    import app as appmod
    import data_building.simulate_playoff_odds as simmod
    import dashboard_services.matchups as mumod

    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache", lambda *a, **k: CTX)
    monkeypatch.setattr(simmod, "build_sim_state", lambda ctx, platform: STATE)
    monkeypatch.setattr(
        mumod, "team_live_totals", lambda team, s, p: (0.0, 158.8))
    monkeypatch.setattr(mumod, "compute_win_prob", lambda *a, **k: 0.52)
    return appmod.app.test_client()


def _set_bundle(monkeypatch, value):
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "_build_live_matchups", lambda *a, **k: value)


def test_cold_bundle_flags_pending_instead_of_silent_omission(client, monkeypatch):
    # The intermittent case: viewer has a Week 10 game, but the live
    # bundle has not been built yet, so _build_live_matchups returns
    # None while its background rebuild lands.
    _set_bundle(monkeypatch, None)
    body = client.get(URL).get_json()
    assert body["show"] is True
    assert body["your_matchup"] is None
    assert body["your_matchup_pending"] is True


def test_warm_bundle_builds_card_and_clears_pending(client, monkeypatch):
    _set_bundle(monkeypatch, BUNDLE)
    body = client.get(URL).get_json()
    assert body["your_matchup_pending"] is False
    mu = body["your_matchup"]
    assert mu["week"] == 10
    assert mu["viewer_roster_id"] == 1
    assert mu["opp_roster_id"] == 2
    assert mu["opp_name"] == "Pilots"
    assert mu["viewer_proj"] == 158.8
    assert mu["viewer_win_prob"] == 52.0


def test_pending_resolves_once_the_rebuild_lands(client, monkeypatch):
    # First poll: bundle still cold. Second poll (what the client's
    # retry fetches): rebuild has landed, card present.
    _set_bundle(monkeypatch, None)
    first = client.get(URL).get_json()
    assert first["your_matchup"] is None
    assert first["your_matchup_pending"] is True
    _set_bundle(monkeypatch, BUNDLE)
    second = client.get(URL).get_json()
    assert second["your_matchup"] is not None
    assert second["your_matchup_pending"] is False


def test_viewer_with_no_game_is_not_pending(client, monkeypatch):
    # Legitimate absence: roster 99 is in no remaining pair (bye / no
    # game), so there is nothing to wait for and nothing to show.
    _set_bundle(monkeypatch, BUNDLE)
    body = client.get(URL.replace("roster_id=1", "roster_id=99")).get_json()
    assert body["your_matchup"] is None
    assert body["your_matchup_pending"] is False


def test_no_viewer_roster_is_not_pending(client, monkeypatch):
    # Legitimate absence: no roster_id (guest / no team in the league).
    _set_bundle(monkeypatch, None)
    body = client.get(URL.replace("&roster_id=1", "")).get_json()
    assert body["your_matchup"] is None
    assert body["your_matchup_pending"] is False


# ── Client contract ──────────────────────────────────────────────────────

def _path_loader_block() -> str:
    start = APP_JS.index("function initPlayoffPath")
    return APP_JS[start:APP_JS.index("function switchPathTab", start)]


def test_client_polls_scenarios_while_matchup_pending():
    block = _path_loader_block()
    # The pending branch is gated on the flag being set AND the card
    # still missing, inside the in-window (scen.show) render path.
    assert "scen.your_matchup_pending && !scen.your_matchup" in block
    # It re-asks for the scenarios payload on the league-scores cadence,
    # bounded, and renders from the polled response via the same finish
    # path as the first response.
    assert "fetch(`/api/playoff-scenarios${scenQ}`)" in block
    assert "setTimeout(_pollScen, 3000)" in block
    assert "_muAttempts >= 5" in block
    assert "finishPath()" in block
    # A panel that left the DOM stops the poll instead of rendering
    # into a detached node.
    assert "document.body.contains(panel)" in block
