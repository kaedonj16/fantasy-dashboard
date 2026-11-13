"""Iteration 2, item 7a: zero-projection non-starters must not
surface as waiver Best Moves. The Nov 12 list for league 92916
carried Joe Fagnano (dropped by the viewer Nov 5, 0.00 projection)
and Kenny Pickett (0.00 projection): both clear the value floor on
stale value while the live projections feed projects them for
nothing. A candidate whose forward projection resolves to exactly
0.0 on a live feed is dropped before the list is built; a QB with
a real projection stays.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

CTX = {
    "current_week": 10,
    "league": {"settings": {"waiver_type": 0}},
    "rosters": [{
        "roster_id": 1,
        "players": ["qb_own", "rb_own"],
        "reserve": [], "taxi": [], "settings": {},
    }],
    "users": [],
    "players_index": {
        "qb_own": {"name": "My QB", "pos": "QB", "team": "BUF"},
        "rb_own": {"name": "My RB", "pos": "RB", "team": "DAL"},
        "fagnano": {"name": "Joe Fagnano", "pos": "QB", "team": "WAS"},
        "pickett": {"name": "Kenny Pickett", "pos": "QB", "team": "CLE"},
        "realqb": {"name": "Real Option", "pos": "QB", "team": "MIA"},
    },
    "roster_positions": ["QB", "RB", "WR", "TE", "BN", "BN"],
    "raw_scoring_settings": {"pass_yd": 0.04, "pass_td": 4},
}


def _row(pid, name, team, val):
    return {"player_id": pid, "id": pid, "name": name, "position": "QB",
            "team": team, "value": val, "sf_value": val,
            "redraft_value_1qb": val, "redraft_value_sf": val,
            "rank_change_7d": 0}


TABLE = [
    _row("fagnano", "Joe Fagnano", "WAS", 620),
    _row("pickett", "Kenny Pickett", "CLE", 610),
    _row("realqb", "Real Option", "MIA", 600),
]

# Live feed: only realqb projects. Fagnano and Pickett are absent
# from every future week, so their forward PPG resolves to 0.0.
RAW_PROJ = {"realqb": {"pass_yd": 280, "pass_td": 2}}
GAMES = [{"home": "MIA", "away": "BUF"}, {"home": "WAS", "away": "DAL"},
         {"home": "CLE", "away": "NYJ"}]


def _client(monkeypatch):
    from flask import Flask
    import routes.waiver_bp as waiver
    import utils.utils as utils_mod

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_bp)
    monkeypatch.setattr(waiver, "get_nfl_state",
                        lambda: {"season": 2026, "week": 10,
                                 "season_type": "reg"})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache",
                        lambda *a: CTX)
    monkeypatch.setattr(waiver, "get_model_value_table_cached",
                        lambda *a, **k: list(TABLE))
    monkeypatch.setattr(waiver, "get_players_global",
                        lambda: {pid: dict(meta, full_name=meta["name"],
                                           position=meta["pos"])
                                 for pid, meta in
                                 CTX["players_index"].items()})
    monkeypatch.setattr(waiver, "_sleeper_trending_adds", lambda limit=50: [])
    monkeypatch.setattr(waiver, "_forward_ppg_map",
                        lambda season, raw, entries: {})
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: dict(RAW_PROJ))
    monkeypatch.setattr(utils_mod, "load_week_schedule",
                        lambda season, week: list(GAMES))
    # Deterministic scoring of the raw map (the pin is the gate, not
    # the scoring math): a player in the map projects, absent is None.
    import utils.fantasy_scoring as fs_mod
    monkeypatch.setattr(
        fs_mod, "weekly_projection_points",
        lambda week_map, pid, scoring, pos="":
            19.2 if str(pid) in (week_map or {}) else None)
    return app.test_client()


def test_zero_projection_qbs_do_not_surface(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get(
        "/api/waiver-candidates?platform=sleeper&league_id=L"
        "&season=2026&position=QB")
    assert resp.status_code == 200
    ids = [c["player_id"] for c in resp.json["candidates"]]
    assert "realqb" in ids
    assert "fagnano" not in ids
    assert "pickett" not in ids


def test_dead_feed_never_gates(monkeypatch):
    # When the projections feed itself is unavailable the forward
    # PPG is unknown (None), not zero: the gate must not empty the
    # list on a data outage.
    import utils.utils as utils_mod
    client = _client(monkeypatch)
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: {})
    resp = client.get(
        "/api/waiver-candidates?platform=sleeper&league_id=L2"
        "&season=2026&position=QB")
    assert resp.status_code == 200
    ids = [c["player_id"] for c in resp.json["candidates"]]
    assert "fagnano" in ids and "pickett" in ids and "realqb" in ids
