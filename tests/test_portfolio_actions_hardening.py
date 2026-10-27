"""Hardening for /api/portfolio-actions (2026-10-26, second failure round).

The #2814 cache + single-flight shipped, yet the moves board still
showed "Could not load your moves" in production. Trace of the shipped
code found three holes:

1. Two per-league sections were NOT best-effort like the rest: the
   viewer-roster match (match_viewer_roster / sleeper_owner_id_for_
   account) and the lineup-issues scan (find_lineup_issues /
   lineup_actions_from_issues) ran unwrapped, so one league's bad
   data raised out of the handler as a 500 and aborted every league.
2. The single-flight mark was only cleared on the success path, so
   that exception leaked _ACTIONS_INFLIGHT for 120s: later requests
   deferred to a compute that was not running, and the cache never
   populated (a permanent-failure loop for that viewer).
3. The compute ran inside a request thread with no internal deadline
   (measured 12.9s for 4 leagues locally, ~35s at 11 leagues), so on
   the 1-worker x 4-thread box a loaded moment pushed it past the
   client's 45s abort; the client then rendered the error panel after
   its ladder, and a server 500 envelope (JSON, no scan block) was
   rendered as a false "all caught up" board.

Now: per-league sections degrade individually, the scan runs once per
viewer on a capped background thread and lands in the cache for the
client's retry ladder, the in-flight mark is cleared in a finally,
failures answer with the honest incomplete payload (never a cached
failure, never a bare 500), and the client routes scan-less error
envelopes through the retry ladder.
"""
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_LEAGUES = [{"league_id": "L1", "platform": "sleeper", "season": 2026,
             "name": "Alpha"}]


def _ctx(starters=()):
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1",
            "starters": list(starters), "players": list(starters),
            "reserve": [], "taxi": [], "settings": {},
        }],
        "league": {"name": "Alpha", "settings": {}},
        "users": [{"user_id": "owner-1", "display_name": "tester"}],
        "roster_positions": ["QB", "BN"],
        "proj_by_week": {},
        "players_index": {},
        "model_value_table": [],
    }


def _client(monkeypatch, leagues=None, ctx=None, players=None):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.subscriptions as subs
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_INFLIGHT.clear()
    bp_mod._ACTIONS_PENDING.clear()
    bp_mod._ACTIONS_FRAGMENTS.clear()
    bp_mod._ACTIONS_LATEST.clear()
    calls = {"ctx": 0}
    the_ctx = ctx if ctx is not None else _ctx()

    def _ctx_cached(plat, lid, season, allow_build=True):
        calls["ctx"] += 1
        return the_ctx

    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", _ctx_cached)
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (list(leagues or _LEAGUES), 2026))
    monkeypatch.setattr(subs, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players",
                        lambda *a, **k: players or {})
    monkeypatch.setattr(uu, "load_week_schedule", lambda season, week: [])
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client, bp_mod, calls


def _wait_for_idle(bp_mod, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not bp_mod._ACTIONS_PENDING and not bp_mod._ACTIONS_INFLIGHT:
            return True
        time.sleep(0.05)
    return False


def test_lineup_exception_degrades_instead_of_aborting(monkeypatch):
    import utils.lineup_issues as li

    def _boom(*a, **k):
        raise RuntimeError("bad starter shape")

    monkeypatch.setattr(li, "find_lineup_issues", _boom)
    players = {"p1": {"full_name": "P One", "team": "KC",
                      "injury_status": ""}}
    client, bp_mod, calls = _client(
        monkeypatch, ctx=_ctx(starters=["p1"]), players=players)

    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200  # was a scan-aborting 500 pre-fix
    # 2026-10-27: the request defers while the scan runs in the
    # background; the degraded-but-complete scan lands in the cache and
    # the next GET serves it.
    assert resp.get_json()["scan"]["incomplete"] is True
    assert _wait_for_idle(bp_mod)
    # The degraded-but-complete scan is cached like any complete scan.
    assert ("viewer:owner-1", 2026, 7) in bp_mod._ACTIONS_CACHE
    data = client.get("/api/portfolio-actions").get_json()
    assert data["scan"]["scanned"] == 1  # the league still counts
    assert data["scan"]["incomplete"] is False


def test_roster_match_exception_counts_unresolved_and_recovers(monkeypatch):
    import utils.scorezone_user as szu

    real_match = szu.match_viewer_roster
    state = {"fail": True}

    def _flaky(*a, **k):
        if state["fail"]:
            raise RuntimeError("identity row shape changed")
        return real_match(*a, **k)

    monkeypatch.setattr(szu, "match_viewer_roster", _flaky)
    client, bp_mod, calls = _client(monkeypatch)

    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200
    # The request defers (2026-10-27 no-wait dispatch); the background
    # scan is where the roster-match failure is absorbed.
    assert resp.get_json()["scan"]["incomplete"] is True
    assert _wait_for_idle(bp_mod)
    # The failure is NOT cached as complete, and no mark leaks: the
    # next request recomputes instead of deferring to a dead compute.
    assert bp_mod._ACTIONS_CACHE == {}
    ctx_calls = calls["ctx"]
    state["fail"] = False
    second = client.get("/api/portfolio-actions").get_json()
    assert _wait_for_idle(bp_mod)
    assert calls["ctx"] > ctx_calls
    second = client.get("/api/portfolio-actions").get_json()
    assert second["scan"]["scanned"] == 1
    assert second["scan"]["incomplete"] is False


def test_unexpected_compute_failure_is_honest_incomplete_not_500(monkeypatch):
    # A starters value with a non-iterable shape raises outside every
    # per-section guard: the endpoint must still answer JSON with the
    # honest incomplete payload, clear the single-flight mark, and
    # cache nothing.
    bad = _ctx()
    bad["rosters"][0]["starters"] = 12345
    bad["rosters"][0]["players"] = ["p1"]
    players = {"p1": {"full_name": "P One", "team": "KC",
                      "injury_status": ""}}
    client, bp_mod, calls = _client(monkeypatch, ctx=bad, players=players)

    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["scan"]["incomplete"] is True
    assert data["actions"] == []
    assert _wait_for_idle(bp_mod)
    assert bp_mod._ACTIONS_CACHE == {}


def test_ranking_failure_serves_unranked_actions(monkeypatch):
    import utils.cross_league_actions as cla
    import utils.lineup_issues as li

    def _one_issue(*a, **k):
        return [{"type": "bye", "player_id": "p1"}]

    def _one_action(issues, **k):
        return [{"kind": "lineup", "title": "Fix the bye",
                 "severity": 0.5, "platform": k.get("platform"),
                 "league_id": k.get("league_id")}]

    def _boom_rank(*a, **k):
        raise RuntimeError("ranker blew up")

    monkeypatch.setattr(li, "find_lineup_issues", _one_issue)
    monkeypatch.setattr(cla, "lineup_actions_from_issues", _one_action)
    monkeypatch.setattr(cla, "rank_cross_league_actions", _boom_rank)
    players = {"p1": {"full_name": "P One", "team": "KC",
                      "injury_status": ""}}
    client, bp_mod, calls = _client(
        monkeypatch, ctx=_ctx(starters=["p1"]), players=players)

    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200
    # Deferred while the (unranked-fallback) scan computes in the
    # background; the landed scan is complete.
    assert resp.get_json()["scan"]["incomplete"] is True
    assert _wait_for_idle(bp_mod)
    data = client.get("/api/portfolio-actions").get_json()
    assert data["scan"]["incomplete"] is False


def test_slow_scan_defers_fast_then_lands_in_cache(monkeypatch):
    import dashboard_services.portfolio_board as pb

    monkeypatch.setattr(
        "routes.user_pages_bp._ACTIONS_SYNC_WAIT", 0.2)
    real_odds = pb.odds_for_ctx

    def _slow_odds(*a, **k):
        time.sleep(1.0)
        return real_odds(*a, **k)

    monkeypatch.setattr(pb, "odds_for_ctx", _slow_odds)
    client, bp_mod, calls = _client(monkeypatch)

    t0 = time.time()
    first = client.get("/api/portfolio-actions")
    elapsed = time.time() - t0
    assert first.status_code == 200
    # The request answers inside the bounded grace even though the
    # scan takes ~1s: it defers instead of riding the request thread.
    assert elapsed < 0.9
    assert first.get_json()["scan"]["incomplete"] is True

    data = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        data = client.get("/api/portfolio-actions").get_json()
        if data["scan"]["incomplete"] is False:
            break
        time.sleep(0.15)
    assert data is not None and data["scan"]["incomplete"] is False
    assert data["scan"]["scanned"] == 1
    assert _wait_for_idle(bp_mod)
    assert ("viewer:owner-1", 2026, 7) in bp_mod._ACTIONS_CACHE


def test_thread_spawn_failure_falls_back_to_inline_compute(monkeypatch):
    import routes.user_pages_bp as bp_mod

    def _no_threads(*a, **k):
        raise RuntimeError("can't start thread")

    monkeypatch.setattr(bp_mod.threading, "Thread", _no_threads)
    client, bp_mod, calls = _client(monkeypatch)
    resp = client.get("/api/portfolio-actions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["scan"]["scanned"] == 1
    assert data["scan"]["incomplete"] is False
    assert bp_mod._ACTIONS_INFLIGHT == {}
    assert bp_mod._ACTIONS_PENDING == {}


def test_client_routes_scanless_error_envelope_to_retry_ladder():
    import pathlib
    js = (pathlib.Path(__file__).resolve().parents[1]
          / "static" / "app.js").read_text()
    fn = js[js.index("function __pfMovesGo(attempt)"):]
    fn = fn[:fn.index("window.__pfBoardInit")]
    # A response with no scan block is an error envelope: it must throw
    # into the ladder (which ends at the error panel) instead of being
    # rendered as an empty "all caught up" board.
    assert "if (!scan)" in fn
    assert "no scan payload" in fn
    assert fn.index("if (!scan)") > fn.index("res.d.paywall")
