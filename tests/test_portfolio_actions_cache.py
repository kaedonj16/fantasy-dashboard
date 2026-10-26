"""Response cache + single-flight for /api/portfolio-actions (2026-10-26).

The warm scan measured 12.9s for just 4 Sleeper leagues locally; across
all 11 linked leagues in production it overran the client's 15s fetch
timeout, so the moves board showed "Could not load your moves" on every
load, and every retry rescanned from zero. Now complete scans are
cached per viewer for 60s, and while one request computes, concurrent
requests answer with the honest incomplete payload immediately so the
client's retry ladder lands on the finished cache.
"""
import time

import pytest

pytest.importorskip("flask")

_LEAGUES = [{"league_id": "L1", "platform": "sleeper", "season": 2026,
             "name": "Alpha"}]


def _ctx():
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1", "starters": [],
            "players": [], "reserve": [], "taxi": [],
        }],
        "league": {"name": "Alpha", "settings": {}},
        "users": [{"user_id": "owner-1", "display_name": "tester"}],
        "roster_positions": ["QB", "BN"],
        "proj_by_week": {},
    }


def _client(monkeypatch):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.subscriptions as subs
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_INFLIGHT.clear()
    calls = {"ctx": 0}

    def _ctx_cached(plat, lid, season, allow_build=True):
        calls["ctx"] += 1
        return _ctx()

    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", _ctx_cached)
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (list(_LEAGUES), 2026))
    monkeypatch.setattr(subs, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: {})
    monkeypatch.setattr(uu, "load_week_schedule", lambda season, week: [])
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client, bp_mod, calls


def test_complete_scan_is_cached_per_viewer(monkeypatch):
    client, bp_mod, calls = _client(monkeypatch)
    first = client.get("/api/portfolio-actions").get_json()
    assert first["scan"]["incomplete"] is False
    assert first["scan"]["scanned"] == 1
    ctx_calls_after_first = calls["ctx"]
    assert ctx_calls_after_first >= 1
    second = client.get("/api/portfolio-actions").get_json()
    assert calls["ctx"] == ctx_calls_after_first  # served from cache
    assert second == first


def test_inflight_compute_defers_with_incomplete_payload(monkeypatch):
    client, bp_mod, calls = _client(monkeypatch)
    key = ("viewer:owner-1", 2026, 7)
    bp_mod._ACTIONS_INFLIGHT[key] = time.time()
    data = client.get("/api/portfolio-actions").get_json()
    assert data["scan"]["incomplete"] is True
    assert data["scan"]["total"] == 1
    assert data["actions"] == []
    assert calls["ctx"] == 0  # no duplicate scan was started


def test_client_retries_moves_failures_before_showing_error():
    import pathlib
    js = (pathlib.Path(__file__).resolve().parents[1]
          / "static" / "app.js").read_text()
    fn = js[js.index("function __pfMovesGo(attempt)"):]
    fn = fn[:fn.index("window.__pfBoardInit")]
    # A timeout/network failure retries on the ladder; the error copy
    # only renders once the ladder is exhausted.
    catch = fn[fn.index(".catch(function ()"):]
    assert "attempt < PF_RETRY.length" in catch
    assert "__pfMovesGo(attempt + 1)" in catch
    assert "45000" in fn  # longer leash than the old 15s abort
