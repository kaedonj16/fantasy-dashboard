"""API tests for the K/DST tabs of /api/waiver-candidates.

Kickers and team defenses are scored by the shared streaming rankers (Vegas
implied totals), not the skill-position composite, but presented in the same
row shape so the waiver list renders one unified UI.
"""
import pytest

pytest.importorskip("flask")

K_ROWS = [
    {"player_id": "k2", "name": "Elite Kicker", "position": "K", "team": "BUF",
     "opponent": "MIA", "matchup": "vs MIA", "own_implied": 27.5,
     "stream_score": 134.0},
    {"player_id": "k3", "name": "Mid Kicker", "position": "K", "team": "MIA",
     "opponent": "BUF", "matchup": "@ BUF", "own_implied": 20.0,
     "stream_score": 74.0},
]

DEF_ROWS = [
    {"player_id": "KC", "name": "KC D/ST", "position": "DEF", "team": "KC",
     "opponent": "LV", "matchup": "vs LV", "opp_implied": 16.5,
     "stream_score": 126.0},
    {"player_id": "NE", "name": "NE D/ST", "position": "DEF", "team": "NE",
     "opponent": "NYJ", "matchup": "vs NYJ", "opp_implied": 19.0,
     "stream_score": 106.0},
]

# 8 active players, 8 slots (QB/RB/WR/TE/K/DEF/BN/BN): roster is full, so an add
# must name a drop.
CTX = {
    "current_week": 4,
    "league": {"settings": {"waiver_type": 0}},
    "rosters": [{
        "roster_id": 1,
        "players": ["k1", "qb1", "rb1", "wr1", "te1", "dst1", "b1", "b2"],
        "reserve": [], "taxi": [], "settings": {},
    }],
    "users": [],
    "players_index": {
        "k1": {"name": "My Kicker", "pos": "K", "team": "KC"},
        "qb1": {"name": "My QB", "pos": "QB", "team": "BUF"},
        "rb1": {"name": "My RB", "pos": "RB", "team": "DAL"},
        "wr1": {"name": "My WR", "pos": "WR", "team": "PHI"},
        "te1": {"name": "My TE", "pos": "TE", "team": "SF"},
        "dst1": {"name": "My D/ST", "pos": "DEF", "team": "BAL"},
        "b1": {"name": "Bench One", "pos": "RB", "team": "GB"},
        "b2": {"name": "Bench Two", "pos": "WR", "team": "DET"},
    },
    "roster_positions": ["QB", "RB", "WR", "TE", "K", "DEF", "BN", "BN"],
    "raw_scoring_settings": {},
}

FWD = {"k1": 5.0, "k2": 9.0, "k3": 4.0, "qb1": 20.0, "rb1": 12.0,
       "wr1": 11.0, "te1": 8.0, "dst1": 7.0, "b1": 3.0, "b2": 2.0}


def _client(monkeypatch, ctx=None):
    from flask import Flask
    import routes.waiver_api_bp as waiver

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_api_bp)
    monkeypatch.setattr(waiver, "get_nfl_state", lambda: {"season": 2026, "week": 4})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: ctx or CTX)
    monkeypatch.setattr(
        waiver, "_streaming_targets",
        lambda c, season, players_index=None: {
            "defense": [dict(r) for r in DEF_ROWS],
            "kicker": [dict(r) for r in K_ROWS],
            "in_season": True, "uses_k": True, "uses_def": True})
    monkeypatch.setattr(waiver, "_forward_ppg_map",
                        lambda season, raw, entries: dict(FWD))
    monkeypatch.setattr(waiver, "_sleeper_trending_adds", lambda limit=50: [])
    return app.test_client()


def _get(client, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return client.get(f"/api/waiver-candidates?{qs}")


def test_k_tab_unified_row_shape_unlinked(monkeypatch):
    d = _get(_client(monkeypatch), platform="sleeper", league_id="L",
             season=2026, position="K").json
    assert d["position"] == "K"
    assert d["personalized"] is False
    assert d["league_uses_k"] is True and d["league_uses_def"] is True
    assert [c["player_id"] for c in d["candidates"]] == ["k2", "k3"]
    row = d["candidates"][0]
    # Same composite scale as skill rows, matchup signal, FAAB guidance.
    assert row["composite_score"] == 134.0
    assert "28 implied total" in row["signal"] and "Strong offense" in row["signal"]
    assert row["faab_mode"] in ("faab", "priority", "waiver_priority")
    assert row["ros_ppg"] == 9.0
    # Unlinked: no lineup math, no drop (roster ownership is unknown).
    assert row["outcome"] is None and row["lineup_gain"] is None
    assert row["drop"] is None
    assert row["replaces"] is None


def test_k_tab_linked_has_lineup_gain_and_drop(monkeypatch):
    d = _get(_client(monkeypatch), platform="sleeper", league_id="L",
             season=2026, position="K", rid=1).json
    assert d["personalized"] is True
    row = d["candidates"][0]
    # k2 (9.0) beats k1 (5.0) in the K slot; roster is full so it's add/drop.
    assert row["outcome"] == "add_drop"
    assert row["lineup_gain"] == 4.0
    assert row["replaces"]["player_id"] == "k1"
    # The solver cuts the weakest bench player, not the replaced starter.
    assert row["drop"]["player_id"] == "b2"
    # A worse kicker than the incumbent: no gain, no add.
    worse = d["candidates"][1]
    assert worse["player_id"] == "k3"
    assert worse["outcome"] in (None, "hold", "cannot_evaluate")
    assert worse["lineup_gain"] is None


def test_def_tab_rows_and_implied_signal(monkeypatch):
    d = _get(_client(monkeypatch), platform="sleeper", league_id="L",
             season=2026, position="DEF", rid=1).json
    assert d["position"] == "DEF"
    assert d["personalized"] is True
    assert [c["player_id"] for c in d["candidates"]] == ["KC", "NE"]
    row = d["candidates"][0]
    assert row["composite_score"] == 126.0
    assert "16" in row["signal"] and "Weak opposing offense" in row["signal"]
    assert row["opp_implied"] == 16.5
    # Team defenses carry no forward projection: no lineup math, no drop.
    assert row["ros_ppg"] is None
    assert row["outcome"] is None and row["drop"] is None


def test_def_aliases_normalize(monkeypatch):
    client = _client(monkeypatch)
    for alias in ("DST", "D/ST", "D-ST", "def"):
        d = _get(client, platform="sleeper", league_id="L", season=2026,
                 position=alias).json
        assert d["position"] == "DEF", alias
        assert [c["player_id"] for c in d["candidates"]] == ["KC", "NE"]


def test_unresolvable_rid_is_not_personalized(monkeypatch):
    d = _get(_client(monkeypatch), platform="sleeper", league_id="L",
             season=2026, position="K", rid=999).json
    assert d["personalized"] is False
    assert all(c["outcome"] is None for c in d["candidates"])


def test_kdef_hidden_when_league_starts_neither(monkeypatch):
    ctx = dict(CTX, roster_positions=["QB", "RB", "WR", "TE", "BN", "BN"])
    d = _get(_client(monkeypatch, ctx), platform="sleeper", league_id="L",
             season=2026, position="K").json
    assert d["league_uses_k"] is False and d["league_uses_def"] is False
