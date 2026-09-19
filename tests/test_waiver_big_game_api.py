"""League-context curation for the unexpected-performance strip."""
from dashboard_services.waiver_discovery_curation import curate_big_game_discoveries


def _row(pid, pos, surprise, sustain, absolute, value=500, confirmed=True):
    return {"player_id": pid, "position": pos, "performance_surprise": surprise,
            "role_sustainability": sustain, "absolute_score": absolute,
            "role_confirmed": confirmed, "value": value, "cautions": []}


def test_one_qb_results_are_not_dominated_by_replacement_passers():
    rows = [_row(f"q{i}", "QB", .7, .45, .5) for i in range(5)]
    rows += [_row("wr", "WR", .55, .8, .45), _row("te", "TE", .6, .7, .6)]
    result = curate_big_game_discoveries(rows, superflex=False)
    assert [r["position"] for r in result].count("QB") <= 1
    assert {r["player_id"] for r in result} >= {"wr", "te"}


def test_superflex_preserves_starting_qb_relevance():
    rows = [_row("q1", "QB", .7, .7, .6), _row("q2", "QB", .65, .7, .55),
            _row("wr", "WR", .6, .6, .5)]
    result = curate_big_game_discoveries(rows, superflex=True)
    assert {"q1", "q2"}.issubset({r["player_id"] for r in result})


def test_curated_order_rewards_sustainability_not_raw_surprise_alone():
    fluke = _row("fluke", "WR", .95, .1, .9, confirmed=False)
    fluke["cautions"] = ["td_dependent", "role_unconfirmed"]
    real = _row("real", "TE", .62, .9, .6, value=2500)
    result = curate_big_game_discoveries([fluke, real])
    assert result[0]["player_id"] == "real"


def _big_game_client(monkeypatch, roster_context):
    import sys
    import types
    from flask import Flask
    import routes.waiver_api_bp as waiver

    app = Flask(__name__)
    app.register_blueprint(waiver.waiver_api_bp)
    monkeypatch.setattr(waiver, "get_nfl_state", lambda: {"season": 2026, "week": 4})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: roster_context)
    monkeypatch.setattr(waiver, "get_model_value_table_cached", lambda: [
        {"id": "free", "name": "Free Player", "position": "WR", "team": "BUF", "value": 500},
        {"id": "reserve", "name": "Reserve Player", "position": "WR", "team": "BUF", "value": 500},
        {"id": "taxi", "name": "Taxi Player", "position": "RB", "team": "BUF", "value": 500},
    ])
    monkeypatch.setattr(waiver, "_waiver_value_keys", lambda ctx: ("value", "value"))
    monkeypatch.setattr(waiver, "get_viewer_session_for_league", lambda *a: {})
    fake = types.ModuleType("dashboard_services.waiver_discoveries")
    fake.get_week_discoveries = lambda *a: [
        _row("free", "WR", .8, .8, .8),
        _row("reserve", "WR", .8, .8, .8),
        _row("taxi", "RB", .8, .8, .8),
    ]
    monkeypatch.setitem(sys.modules, "dashboard_services.waiver_discoveries", fake)
    return app.test_client()


def test_failed_roster_loading_cannot_confirm_unrostered(monkeypatch):
    import routes.waiver_api_bp as waiver
    client = _big_game_client(monkeypatch, {})
    monkeypatch.setattr(waiver, "get_league_ctx_from_cache", lambda *a: (_ for _ in ()).throw(TimeoutError()))
    response = client.get("/api/waiver-big-games?league_id=L&season=2026&week=3")
    assert response.status_code == 503
    assert response.json["availability"] == "unavailable"
    assert response.json["discoveries"] == []


def test_reserve_and_taxi_players_are_not_pickup_discoveries(monkeypatch):
    context = {
        "rosters": [{"roster_id": 1, "players": ["active"], "reserve": ["reserve"], "taxi": ["taxi"]}],
        "players_index": {
            "free": {"name": "Free Player", "pos": "WR", "team": "BUF"},
            "reserve": {"name": "Reserve Player", "pos": "WR", "team": "BUF"},
            "taxi": {"name": "Taxi Player", "pos": "RB", "team": "BUF"},
        },
        "roster_positions": ["QB", "RB", "WR"], "users": [],
    }
    response = _big_game_client(monkeypatch, context).get(
        "/api/waiver-big-games?league_id=L&season=2026&week=3")
    assert response.status_code == 200
    assert response.json["availability"] == "verified"
    assert [row["player_id"] for row in response.json["discoveries"]] == ["free"]
    assert response.json["discoveries"][0]["availability"] == "unrostered"
