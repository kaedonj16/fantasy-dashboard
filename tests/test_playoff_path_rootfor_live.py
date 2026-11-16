"""Playoff Path live game status (Kaedon, 2026-11-16).

"I want to be able to see the live status of games that I need to
root for in playoff impact" + "handle the transition from proj to
live scores". The scenarios endpoint now also returns games_live:
every game in the first remaining week built by
build_league_scores_list over the same live matchup bundle as the
Your Matchup card (actual score, live projected total, pre/in/final
status), each side extended with a yet-to-play starter count. The
viewer's own card gains viewer_score / opp_score / status / yet
fields from its games_live entry while its proj fields stay exactly
as they were. A cold bundle flags games_live_pending (the client
polls on it, same contract as your_matchup_pending); no roster_id
means no games_live and no pending.
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

URL = ("/api/playoff-scenarios?platform=sleeper&league_id=L1"
       "&season=2026&roster_id=1")


def _bundle(status_by_pid):
    matchups = [
        {"left": {"roster_id": 1, "name": "Couch",
                  "starters": [{"pid": "a1"}, {"pid": "a2"}]},
         "right": {"roster_id": 2, "name": "Pilots",
                   "starters": [{"pid": "b1"}, {"pid": "b2"}]}},
        {"left": {"roster_id": 3, "name": "Fire Riley",
                  "starters": [{"pid": "c1"}]},
         "right": {"roster_id": 4, "name": "Nunky",
                   "starters": [{"pid": "d1"}]}},
    ]
    return (matchups, status_by_pid, {})


@pytest.fixture()
def client(monkeypatch):
    import app as appmod
    import data_building.simulate_playoff_odds as simmod
    import dashboard_services.matchups as mumod

    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache", lambda *a, **k: CTX)
    monkeypatch.setattr(simmod, "build_sim_state", lambda ctx, platform: STATE)
    # Same patched totals the matchup-pending suite uses: every team
    # totals (actual 0.0, proj 158.8); win prob 52% for the left side.
    monkeypatch.setattr(
        mumod, "team_live_totals", lambda team, s, p, **k: (0.0, 158.8))
    monkeypatch.setattr(mumod, "compute_win_prob", lambda *a, **k: 0.52)
    return appmod.app.test_client()


def _set_bundle(monkeypatch, value):
    import routes.user_pages_bp as pages
    monkeypatch.setattr(pages, "_build_live_matchups", lambda *a, **k: value)


def _game(body, rid):
    for entry in body["games_live"]:
        if entry["left"]["roster_id"] == str(rid):
            return entry
        if entry["right"] and entry["right"]["roster_id"] == str(rid):
            return entry
    raise AssertionError(f"no games_live entry for roster {rid}")


def _side(entry, rid):
    if entry["left"]["roster_id"] == str(rid):
        return entry["left"]
    return entry["right"]


# ── games_live statuses / yet counts / win prob ─────────────────────

def test_games_live_pre_game(client, monkeypatch):
    _set_bundle(monkeypatch, _bundle({}))
    body = client.get(URL).get_json()
    assert body["games_live_pending"] is False
    assert len(body["games_live"]) == 2
    g = _game(body, 1)
    assert g["status"] == "pre"
    assert g["win_prob"] == 52.0
    assert _side(g, 1)["yet"] == 2
    assert _side(g, 2)["yet"] == 2
    assert _side(g, 1)["score"] == 0.0
    assert _side(g, 1)["proj"] == 158.8


def test_games_live_in_progress(client, monkeypatch):
    _set_bundle(monkeypatch, _bundle(
        {"a1": "in_progress", "a2": "final", "b1": "not_started"}))
    body = client.get(URL).get_json()
    g = _game(body, 1)
    assert g["status"] == "in"
    assert _side(g, 1)["yet"] == 0
    # b1 is explicitly not_started and b2 has no status entry at all,
    # which also means not started: both are yet to play.
    assert _side(g, 2)["yet"] == 2
    # The other game has no started players at all: still pre.
    assert _game(body, 3)["status"] == "pre"


def test_games_live_final(client, monkeypatch):
    _set_bundle(monkeypatch, _bundle(
        {"a1": "final", "a2": "final", "b1": "final", "b2": "final",
         "c1": "final", "d1": "final"}))
    body = client.get(URL).get_json()
    g = _game(body, 1)
    assert g["status"] == "final"
    assert _side(g, 1)["yet"] == 0
    assert _side(g, 2)["yet"] == 0


def test_games_live_yet_counts_accept_str_pid_keys(client, monkeypatch):
    # status_by_pid may be keyed by str(pid) while starters carry the
    # raw pid; the count must consult both, like team_live_totals.
    bundle = _bundle({})
    bundle[0][0]["left"]["starters"] = [{"pid": 101}, {"pid": 102}]
    _set_bundle(monkeypatch, (bundle[0], {"101": "final"}, {}))
    body = client.get(URL).get_json()
    assert _side(_game(body, 1), 1)["yet"] == 1


# ── pending contract ────────────────────────────────────────────────

def test_games_live_pending_when_bundle_cold(client, monkeypatch):
    _set_bundle(monkeypatch, None)
    body = client.get(URL).get_json()
    assert body["games_live"] == []
    assert body["games_live_pending"] is True


def test_games_live_not_pending_without_roster_id(client, monkeypatch):
    _set_bundle(monkeypatch, None)
    body = client.get(URL.replace("&roster_id=1", "")).get_json()
    assert body["games_live"] == []
    assert body["games_live_pending"] is False


# ── your_matchup live fields ────────────────────────────────────────

def test_your_matchup_carries_live_fields_and_keeps_proj(client, monkeypatch):
    _set_bundle(monkeypatch, _bundle(
        {"a1": "in_progress", "a2": "final", "b1": "not_started"}))
    body = client.get(URL).get_json()
    mu = body["your_matchup"]
    assert mu["status"] == "in"
    assert mu["viewer_score"] == 0.0
    assert mu["opp_score"] == 0.0
    assert mu["viewer_yet"] == 0
    assert mu["opp_yet"] == 2
    # Proj fields unchanged: already live projected totals + win prob.
    assert mu["viewer_proj"] == 158.8
    assert mu["opp_proj"] == 158.8
    assert mu["viewer_win_prob"] == 52.0


def test_your_matchup_pre_game_status(client, monkeypatch):
    _set_bundle(monkeypatch, _bundle({}))
    body = client.get(URL).get_json()
    mu = body["your_matchup"]
    assert mu["status"] == "pre"
    assert mu["viewer_yet"] == 2
    assert mu["opp_yet"] == 2


# ── Client contract ─────────────────────────────────────────────────

def test_client_renders_root_for_live_lines():
    assert "scen.games_live" in APP_JS
    assert "_liveFor" in APP_JS
    assert "path-live-line" in APP_JS
    assert "Rooted team won" in APP_JS
    assert "Rooted team lost" in APP_JS


def test_client_matchup_card_is_status_aware():
    assert "mu.status" in APP_JS
    assert "Live proj" in APP_JS
    assert "path-live-chip" in APP_JS
    assert "path-final-chip" in APP_JS


def test_client_polls_while_games_live():
    start = APP_JS.index("function initPlayoffPath")
    block = APP_JS[start:APP_JS.index("function switchPathTab", start)]
    assert "games_live_pending" in block
    assert "_startPathLivePoll" in block
    assert "setTimeout(_pollLive, 60000)" in block
    assert "_livePolls < 240" in block
    assert "panel.dataset.loadedAt" in block
