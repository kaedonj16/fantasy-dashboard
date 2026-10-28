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


@pytest.fixture(autouse=True)
def _isolate_actions_scan_stores():
    """Keep the process-global actions-scan stores from leaking into
    or out of these tests.

    The /api/portfolio-actions scan path consults five module-global
    stores in routes.user_pages_bp (_ACTIONS_CACHE, _ACTIONS_LATEST,
    _ACTIONS_FRAGMENTS, _ACTIONS_INFLIGHT, _ACTIONS_PENDING), all of
    which outlive monkeypatch. _client() below clears four of them at
    setup, but not _ACTIONS_PENDING, and nothing clears any of them on
    teardown, so a completed scan one test leaves behind (the sibling
    suites assert ("viewer:owner-1", 2026, 7) warm in _ACTIONS_CACHE
    at their end) is exactly the state the next test in the process
    inherits. Clear all five around every test so each one starts
    from the cold-scan state it asserts. Production semantics are
    unchanged: a warm cache still serves its TTL within a process.
    """
    try:
        import routes.user_pages_bp as bp_mod
    except ImportError:
        yield
        return
    stores = (bp_mod._ACTIONS_CACHE, bp_mod._ACTIONS_LATEST,
              bp_mod._ACTIONS_FRAGMENTS, bp_mod._ACTIONS_INFLIGHT,
              bp_mod._ACTIONS_PENDING)
    for store in stores:
        store.clear()
    yield
    for store in stores:
        store.clear()


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
    bp_mod._ACTIONS_FRAGMENTS.clear()
    bp_mod._ACTIONS_LATEST.clear()
    calls = {"ctx": 0}

    def _ctx_cached(plat, lid, season, allow_build=True):
        calls["ctx"] += 1
        # The production scan takes seconds (module docstring: 12.9s
        # warm for 4 leagues), so the first GET's kick-and-defer is
        # never close. An instant fake makes it a thread-scheduling
        # race: in a warm process the background scan can finish
        # before the request thread reaches the post-spawn holder /
        # latest-pass checks, the "first" response is then the
        # completed scan, and the cold-scan assert below flakes
        # (seen in CI on PR #2869's head). A short deterministic
        # build time models the real scan and pins the contract;
        # the sibling hardening suite does the same with a 1s odds
        # stub in test_slow_scan_defers_fast_then_lands_in_cache.
        time.sleep(0.25)
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
    # 2026-10-27: the request thread no longer waits for the scan. The
    # first GET kicks the background compute and defers; the complete
    # scan lands in the per-viewer cache and later GETs serve it.
    first = client.get("/api/portfolio-actions").get_json()
    assert first["scan"]["incomplete"] is True
    complete = first
    deadline = time.time() + 5.0
    while time.time() < deadline:
        complete = client.get("/api/portfolio-actions").get_json()
        if complete["scan"]["incomplete"] is False:
            break
        time.sleep(0.1)
    assert complete["scan"]["incomplete"] is False
    assert complete["scan"]["scanned"] == 1
    ctx_calls_after_first = calls["ctx"]
    assert ctx_calls_after_first >= 1
    second = client.get("/api/portfolio-actions").get_json()
    assert calls["ctx"] == ctx_calls_after_first  # served from cache
    assert second == complete


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
