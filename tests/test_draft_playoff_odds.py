"""Draft recap playoff odds must use the live league when one exists.

Standings already runs ``simulate_playoff_odds`` on the real league context
(settings, current rosters, published schedule). The recap used to rebuild a
guessed preseason room from the draft board (4-vs-6 playoff spots, week-15
start, PPR-only, no league id). These tests lock the live path to the same
cache as Standings, and the slot mapping that paints chips on the board.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _helpers():
    pytest.importorskip("flask")
    from routes import draft_api_bp as bp
    return bp
DRAFT_JS = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
ADMIN = (ROOT / "routes" / "draft_api_bp.py").read_text(encoding="utf-8")


def _odds(rid, pct, name=""):
    return {
        "roster_id": rid,
        "team_name": name,
        "playoff_pct": pct,
        "bye_pct": pct / 2,
        "first_seed_pct": pct / 4,
        "avg_final_wins": 8.0,
        "avg_final_losses": 6.0,
    }


def test_map_odds_prefers_posted_roster_id():
    hp = _helpers()
    odds = [_odds(7, 61.4, "Alpha"), _odds(3, 22.2, "Beta")]
    teams = [
        {"slot": 1, "roster_id": 7, "name": "Alpha"},
        {"slot": 2, "roster_id": 3, "name": "Beta"},
        {"slot": 0, "roster_id": 7, "name": "You"},
    ]
    mapped = hp._map_playoff_odds_to_slots(odds, teams)
    by_slot = {row["slot"]: row["playoff_pct"] for row in mapped}
    assert by_slot[1] == 61.4
    assert by_slot[2] == 22.2
    assert by_slot[0] == 61.4


def test_map_odds_falls_back_to_slot_equals_roster_id():
    hp = _helpers()
    odds = [_odds(2, 40.0)]
    mapped = hp._map_playoff_odds_to_slots(odds, [{"slot": 2, "name": "Seat 2"}])
    assert mapped[0]["slot"] == 2
    assert mapped[0]["playoff_pct"] == 40.0


def test_map_odds_matches_team_name_when_ids_differ():
    hp = _helpers()
    odds = [_odds(99, 55.5, "Night Owls")]
    mapped = hp._map_playoff_odds_to_slots(
        odds, [{"slot": 4, "roster_id": 1, "name": "Night Owls"}]
    )
    assert mapped[0]["slot"] == 4
    assert mapped[0]["playoff_pct"] == 55.5


def test_ctx_has_roster_players_requires_two_filled_teams():
    hp = _helpers()
    assert hp._ctx_has_roster_players({"rosters": []}) is False
    assert hp._ctx_has_roster_players({"rosters": [{"players": ["1"]}]}) is False
    assert hp._ctx_has_roster_players({
        "rosters": [{"players": ["1"]}, {"players": []}, {"players": ["2"]}],
    }) is True


def test_overlay_league_settings_copies_playoff_format_and_scoring():
    hp = _helpers()
    ctx = hp._synthetic_draft_ctx(
        {"ppr": 1, "season": 2026, "playoff_teams": 6, "roster": {"QB": 1, "RB": 2}},
        [
            {"slot": 1, "roster_id": 1, "players": ["a"]},
            {"slot": 2, "roster_id": 2, "players": ["b"]},
        ],
    )
    hp._overlay_league_settings(ctx, {
        "league_id": "lg9",
        "league_settings": {
            "playoff_teams": 4,
            "playoff_week_start": 14,
            "divisions": 2,
        },
        "scoring_settings": {"rec": 0.5},
        "raw_scoring_settings": {"rec": 0.5, "bonus_rec_te": 0.5},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    })
    assert ctx["league_id"] == "lg9"
    assert ctx["league_settings"]["playoff_teams"] == 4
    assert ctx["league_settings"]["playoff_week_start"] == 14
    assert ctx["league_settings"]["divisions"] == 2
    assert ctx["scoring_settings"]["rec"] == 0.5
    assert ctx["raw_scoring_settings"]["bonus_rec_te"] == 0.5
    assert "SUPER_FLEX" not in ctx["roster_positions"]
    assert ctx["roster_positions"][0] == "QB"


def test_route_uses_standings_cache_for_live_league():
    fn = ADMIN[ADMIN.index("def api_draft_playoff_odds"):]
    assert "_load_league_ctx" in fn
    assert "_run_league_playoff_sim" in fn
    assert "use_league" in fn
    assert 'source": "league"' in fn
    assert "def _load_league_ctx" in ADMIN
    assert "get_league_ctx_from_cache" in ADMIN
    assert "_playoff_sim_cached" in ADMIN


def test_draft_room_posts_league_identity_on_live_recap():
    refresh = DRAFT_JS[DRAFT_JS.index("function refreshServerPlayoffOdds"):]
    refresh = refresh[: refresh.index("\n  function playoffOddsSource")]
    assert "use_league:" in refresh
    assert "cfg.leagueId" in refresh
    assert "state.mode === 'live'" in refresh
    assert "roster_id:" in refresh
    assert "_slotRosterId" in refresh
    assert "function _slotToRosterFromLive" in DRAFT_JS
    assert "slotToRosterId: _slotToRosterFromLive(d)" in DRAFT_JS
    assert "function _poFmt" in DRAFT_JS


def test_live_league_endpoint_returns_standings_rows(monkeypatch):
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    from routes.draft_api_bp import draft_api_bp
    app.register_blueprint(draft_api_bp)

    league_ctx = {
        "league_id": "lg1",
        "season": 2026,
        "league_settings": {"playoff_teams": 4, "playoff_week_start": 14},
        "rosters": [
            {"roster_id": 10, "players": ["111"]},
            {"roster_id": 11, "players": ["222"]},
        ],
    }
    sim_rows = [
        _odds(10, 71.2, "A"),
        _odds(11, 28.8, "B"),
    ]

    monkeypatch.setattr(
        "routes.draft_api_bp._load_league_ctx",
        lambda platform, league_id, season: league_ctx,
    )
    monkeypatch.setattr(
        "routes.draft_api_bp._run_league_playoff_sim",
        lambda ctx, platform: sim_rows,
    )

    client = app.test_client()
    resp = client.post("/api/draft-playoff-odds", json={
        "use_league": True,
        "platform": "sleeper",
        "league_id": "lg1",
        "season": 2026,
        "teams": [
            {"slot": 1, "roster_id": 10, "name": "A"},
            {"slot": 2, "roster_id": 11, "name": "B"},
            {"slot": 0, "roster_id": 10, "name": "You"},
        ],
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source"] == "league"
    assert data["playoff_teams"] == 4
    by_slot = {row["slot"]: row["playoff_pct"] for row in data["odds"]}
    assert by_slot[1] == 71.2
    assert by_slot[2] == 28.8
    assert by_slot[0] == 71.2
