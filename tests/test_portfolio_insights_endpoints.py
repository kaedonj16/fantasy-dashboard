"""Endpoint tests for the split Portfolio insights APIs (speed fold-in):

- /api/portfolio-bye serves Bye Watch from cached contexts only -- it must
  never touch the provider matchup API, and cold leagues are skipped.
- /api/portfolio-insights (Rooting Guide) caches per viewer for its TTL and
  assembles parallel per-league results in the original league order.
- The client (static/app.js) loads bye immediately and defers the rooting
  scan until the league cards settle (or a bounded fallback).
"""
import pathlib

import pytest

pytest.importorskip("flask")

ROOT = pathlib.Path(__file__).resolve().parents[1]

_PLAYERS = {
    "s1": {"full_name": "Starter One", "position": "RB", "team": "BUF",
           "bye_week": 8},
    "s2": {"full_name": "Starter Two", "position": "WR", "team": "BUF",
           "bye_week": 3},
    "b1": {"full_name": "Bench One", "position": "RB", "team": "MIA",
           "bye_week": 5},
    "fa1": {"full_name": "Free Agent", "position": "RB", "team": "NE",
            "bye_week": 5},
}
_MODEL_ROWS = [{"id": "fa1", "position": "RB", "value": 500.0,
                "name": "Free Agent"}]
_LEAGUES = [
    {"league_id": "L1", "platform": "sleeper", "season": 2026,
     "name": "Alpha"},
    {"league_id": "L2", "platform": "sleeper", "season": 2026,
     "name": "Beta"},
]


def _ctx(starter, league_name):
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1", "starters": [starter],
            "players": [starter, "b1"], "reserve": [], "taxi": [],
        }],
        "league": {"name": league_name, "settings": {}},
        "users": [],
        "roster_positions": ["RB", "BN"],
        "proj_by_week": {7: {"projections": {starter: 12.0}}},
    }


def _client(monkeypatch, ctx_by_lid, leagues=None, sched=None):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.platform_api as platform_api
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    bp_mod._INSIGHTS_CACHE.clear()
    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(
        bp_mod, "get_league_ctx_from_cache",
        lambda plat, lid, season, allow_build=False: ctx_by_lid.get(lid, {}))
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: _MODEL_ROWS)
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (leagues or _LEAGUES, 2026))
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: dict(_PLAYERS))
    monkeypatch.setattr(uu, "load_week_schedule",
                        lambda season, week: sched or [])
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client, bp_mod, platform_api


def _sun_1pm_et_epoch():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(2026, 10, 25, 13, tzinfo=ZoneInfo("America/New_York")).timestamp()


# ── Bye endpoint ────────────────────────────────────────────────────────────

def test_bye_endpoint_needs_no_matchups_and_skips_cold(monkeypatch):
    """get_matchups is rigged to raise: the bye payload must still be
    complete for the warm league, including the Make-it-a-move fix, and
    the cold league must be skipped in the scan block."""
    client, bp_mod, platform_api = _client(
        monkeypatch, {"L1": _ctx("s1", "Alpha")})  # L2 cold: no ctx

    def _boom(*a, **k):
        raise AssertionError("bye endpoint must not call get_matchups")

    monkeypatch.setattr(platform_api, "get_matchups", _boom)
    resp = client.get("/api/portfolio-bye")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["bye"]["week"] == 8
    assert data["bye"]["total_out"] == 1
    rows = data["bye"]["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Alpha"
    assert row["out_count"] == 1
    assert row["out"][0]["name"] == "Starter One"
    assert row["verdict"].startswith("Hole")
    # Bench RB is a stash (no value), so the fix promotes the top FA.
    assert row["fix"]["move"]["pid"] == "fa1"
    assert data["scan"] == {"total": 2, "scanned": 1, "incomplete": True}


def test_bye_endpoint_all_cold_is_honest_empty(monkeypatch):
    client, _bp, platform_api = _client(monkeypatch, {})
    monkeypatch.setattr(platform_api, "get_matchups",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    data = client.get("/api/portfolio-bye").get_json()
    assert data["bye"]["rows"] == []
    assert data["scan"]["scanned"] == 0
    assert data["scan"]["incomplete"] is True


# ── Insights endpoint: cache + parallel order ───────────────────────────────

def _matchup_rows(plat, lid, week, season):
    return [
        {"roster_id": 1, "matchup_id": 1, "players_points": {"s1": 10.0, "s2": 9.0}},
        {"roster_id": 2, "matchup_id": 1, "players_points": {"o1": 7.0}},
    ]


def test_insights_second_call_within_ttl_is_cached(monkeypatch):
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)

    calls = {"n": 0}
    orig = bp_mod.get_league_ctx_from_cache

    def _counting(plat, lid, season, allow_build=False):
        calls["n"] += 1
        return orig(plat, lid, season, allow_build=allow_build)

    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", _counting)
    first = client.get("/api/portfolio-insights").get_json()
    assert first["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert calls["n"] == 2  # one ctx read per league on the cold call
    second = client.get("/api/portfolio-insights").get_json()
    assert calls["n"] == 2  # TTL cache: no recomputation at all
    assert second["rooting"] == first["rooting"]
    assert second["updated"] == first["updated"]  # honest computed-at time
    # Bye no longer rides the insights response.
    assert second["bye"]["rows"] == []


def test_insights_parallel_assembly_preserves_league_order(monkeypatch):
    """Both leagues' starters are on BUF in the same game; the home side
    must list them in league order (L1 before L2) even though the
    per-league work runs on a thread pool."""
    sched = [{"home": "BUF", "away": "NYJ",
              "gameTime_epoch": _sun_1pm_et_epoch()}]
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, _bp, platform_api = _client(monkeypatch, ctxs, sched=sched)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    data = client.get("/api/portfolio-insights").get_json()
    windows = data["rooting"]["windows"]
    assert len(windows) == 1
    card = windows[0]["games"][0]
    assert card["home"] == "BUF"
    names = [s["name"] for s in card["sides"]["home"]]
    assert names == ["Starter One", "Starter Two"]


# ── Client wiring ───────────────────────────────────────────────────────────

def test_client_loads_bye_immediately_and_defers_rooting():
    js = (ROOT / "static" / "app.js").read_text()
    assert "fetcher('/api/portfolio-bye'" in js
    assert "function loadBye()" in js
    rooting = js[js.index("function loadInsights()"):]
    rooting = rooting[:rooting.index("function loadBye()")]
    assert "renderBye(" not in rooting  # bye no longer rides insights
    assert "window.addEventListener('br:portfolio-primary-settled', kickInsights" in js
    assert "setTimeout(kickInsights, 6000);" in js
    assert "insightsKicked" in js
