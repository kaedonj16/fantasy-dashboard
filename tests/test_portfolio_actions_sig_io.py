"""Regression: the portfolio actions scan must not pay provider I/O for
playoff-odds signature checks (2026-10-28).

Production failure: a Google-linked account viewer with 11 leagues
(Sleeper + Yahoo + Fleaflicker) saw "Could not check 11 of 11 leagues
yet. Your leagues are still loading - check back in a moment." on the
portfolio page permanently -- every poll of /api/portfolio-actions
answered the hardcoded deferred payload (scanned 0).

Root cause: the scan's per-league board-meta step attaches playoff odds
via portfolio_board.odds_for_ctx, documented as cache-only
(_playoff_sim_cached(block=False)). But _playoff_sim_cached computed
the FULL cache signature on the calling thread before consulting the
cache, and the signature's schedule component
(simulate_playoff_odds.playoff_schedule_sig) fetches the published
slate from the provider -- one get_matchups call per regular-season
week (14), per league, per pass. On the production box those
synchronous fetches (Yahoo OAuth round-trips included) stalled the
single background scan pass inside the FIRST scanned league's odds
attach, ahead of that league's fragment write and the pass-end cache
store, so no pass ever banked progress and every poll re-served
scanned=0. A stalled provider call wedged the pass outright.

Fix: block=False validates a warm sim-cache entry on the locally
computable signature parts (roster fingerprint + current week, both
pure functions of the ctx) and never computes the schedule fingerprint
on the calling thread; the background sim thread computes the full
signature when it stores its result. The blocking path is unchanged.

These tests pin the contract two ways:
- unit: _playoff_sim_cached(block=False) never runs the schedule
  signature on the caller's thread (cold or warm), while roster/week
  changes still invalidate and block=True still validates fully;
- endpoint: the production-shaped scan (account viewer, 11 leagues
  across 3 platforms, real ctx cache/warm machinery) completes with
  ZERO provider matchup fetches on the scan thread.
"""
import threading
import time
import traceback

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as app_mod  # noqa: E402
import dashboard_services.accounts as accounts  # noqa: E402
import dashboard_services.api as ds_api  # noqa: E402
import dashboard_services.platform_api as platform_api  # noqa: E402
import dashboard_services.subscriptions as subs  # noqa: E402
import data_building.simulate_playoff_odds as sim_mod  # noqa: E402
import routes.user_pages_bp as bp_mod  # noqa: E402
import utils.utils as uu  # noqa: E402
from flask import Flask  # noqa: E402


def _wait_until(pred, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


# ── Unit: _playoff_sim_cached signature contract ─────────────────────


def _sim_ctx(league_id="sig-league", week=8, players=("p1", "p2")):
    return {
        "league_id": league_id,
        "season": 2026,
        "current_week": week,
        "rosters": [
            {"roster_id": 1, "players": list(players)},
            {"roster_id": 2, "players": ["p3", "p4"]},
        ],
    }


@pytest.fixture()
def sig_gate(monkeypatch):
    """Fake sim + a schedule-signature fake that records its caller
    thread. Drains background sim slots first so kicks are not dropped
    by slots leaked from sibling tests in this process."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        got = [
            app_mod._PLAYOFF_SIM_SEM.acquire(blocking=False)
            for _ in range(app_mod._PLAYOFF_SIM_MAX_CONCURRENT)
        ]
        for g in got:
            if g:
                app_mod._PLAYOFF_SIM_SEM.release()
        if all(got):
            break
        time.sleep(0.1)
    else:
        raise AssertionError("background sim slots never freed")

    sig_calls = []
    main_ident = threading.get_ident()

    def fake_schedule_sig(ctx, platform="sleeper"):
        sig_calls.append(threading.get_ident())
        return "sched-fake"

    monkeypatch.setattr(sim_mod, "playoff_schedule_sig", fake_schedule_sig)
    monkeypatch.setattr(
        sim_mod, "simulate_playoff_odds",
        lambda ctx, platform="sleeper", **_kw: [
            {"roster_id": 1, "playoff_pct": 50.0}],
    )
    app_mod._PLAYOFF_SIM_CACHE.clear()
    with app_mod._PLAYOFF_SIM_BUILDING_LOCK:
        app_mod._PLAYOFF_SIM_BUILDING.clear()
    yield sig_calls, main_ident
    # Drain kicked sims so no slot or BUILDING mark leaks to siblings.
    _wait_until(
        lambda: not app_mod._PLAYOFF_SIM_BUILDING, timeout=15)
    app_mod._PLAYOFF_SIM_CACHE.clear()


def test_block_false_never_signs_on_the_caller_thread(sig_gate):
    sig_calls, main_ident = sig_gate
    key = ("sleeper", "sig-league", 2026, "v3")

    # Cold cache: returns [] immediately and kicks the sim; the schedule
    # signature must NOT be computed on this thread.
    assert app_mod._playoff_sim_cached(_sim_ctx(), "sleeper",
                                       block=False) == []
    assert main_ident not in sig_calls

    # The background sim stores its result (full sig computed there).
    assert _wait_until(lambda: key in app_mod._PLAYOFF_SIM_CACHE)
    entry = app_mod._PLAYOFF_SIM_CACHE[key]
    assert entry["sig"].endswith("|sched-fake")
    assert entry["roster_sig"] == app_mod._roster_sig(_sim_ctx())
    assert entry["week"] == 8

    # Warm cache: served with no signature work on this thread at all.
    rows = app_mod._playoff_sim_cached(_sim_ctx(), "sleeper", block=False)
    assert rows and rows[0]["playoff_pct"] == 50.0
    assert main_ident not in sig_calls

    # A roster change still invalidates on the cheap local check and
    # re-kicks -- again without signing on this thread.
    changed = _sim_ctx(players=("p1", "p9"))
    assert app_mod._playoff_sim_cached(changed, "sleeper",
                                       block=False) == []
    assert main_ident not in sig_calls
    assert _wait_until(
        lambda: app_mod._PLAYOFF_SIM_CACHE[key]["roster_sig"]
        == app_mod._roster_sig(changed))

    # A week rollover invalidates the same way.
    next_week = _sim_ctx(week=9, players=("p1", "p9"))
    assert app_mod._playoff_sim_cached(next_week, "sleeper",
                                       block=False) == []
    assert main_ident not in sig_calls


def test_block_true_still_validates_the_full_signature(sig_gate):
    sig_calls, main_ident = sig_gate
    rows = app_mod._playoff_sim_cached(_sim_ctx(), "sleeper", block=True)
    assert rows and rows[0]["playoff_pct"] == 50.0
    # The blocking path keeps its contract: full signature (schedule
    # included) computed on the caller thread before serving/storing.
    assert main_ident in sig_calls
    # A repeat blocking call revalidates against the full signature and
    # serves the stored rows.
    rows2 = app_mod._playoff_sim_cached(_sim_ctx(), "sleeper", block=True)
    assert rows2 == rows


# ── Endpoint: production-shaped scan pays no provider I/O ────────────

_LEAGUES = (
    [{"league_id": f"S{i}", "platform": "sleeper", "season": 2026,
      "name": f"Sleeper {i}"} for i in range(1, 7)]
    + [{"league_id": f"Y{i}", "platform": "yahoo", "season": 2026,
        "name": f"Yahoo {i}"} for i in range(1, 4)]
    + [{"league_id": f"F{i}", "platform": "fleaflicker", "season": 2026,
        "name": f"Flea {i}"} for i in range(1, 3)]
)


def _ctx_for(platform, lid, season):
    return {
        "platform": platform, "league_id": lid, "season": season,
        "league": {"name": f"League {lid}", "settings": {}},
        "rosters": [
            {"roster_id": 3, "owner_id": "her-id",
             "players": ["p1"], "starters": ["p1"], "reserve": [],
             "taxi": [], "settings": {"wins": 5, "losses": 3, "ties": 0,
                                       "fpts": 900.0}},
            {"roster_id": 4, "owner_id": "other-id",
             "players": [], "starters": [], "reserve": [], "taxi": [],
             "settings": {"wins": 4, "losses": 4, "ties": 0, "fpts": 800.0}},
        ],
        "users": [{"user_id": "her-id", "display_name": "her"}],
        "roster_positions": ["QB", "BN"],
        "proj_by_week": {},
        "players_index": {},
        "model_value_table": [],
    }


@pytest.fixture()
def account_viewer_env(monkeypatch):
    app_mod.DASHBOARD_CACHE.clear()
    app_mod._PLAYOFF_SIM_CACHE.clear()
    with app_mod._PLAYOFF_SIM_BUILDING_LOCK:
        app_mod._PLAYOFF_SIM_BUILDING.clear()
    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_INFLIGHT.clear()
    bp_mod._ACTIONS_PENDING.clear()
    bp_mod._ACTIONS_FRAGMENTS.clear()
    bp_mod._ACTIONS_LATEST.clear()

    def _fake_build(platform, league_id, season):
        time.sleep(0.2)
        return _ctx_for(platform, league_id, season)

    monkeypatch.setattr(app_mod, "build_league_context", _fake_build)
    monkeypatch.setattr(app_mod, "get_viewer_session_for_league",
                        lambda *a, **k: {})
    monkeypatch.setattr(app_mod, "_kick_player_modal_warm",
                        lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "_kick_modal_league_warm",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 8, "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "get_model_value_table_cached",
                        lambda *a, **k: [])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season, **k: (list(_LEAGUES), 2026))
    monkeypatch.setattr(
        accounts, "resolve_account_viewer_for_league",
        lambda aid, plat, lid, season, users, rosters:
            {"viewer_roster_id": "3"})
    monkeypatch.setattr(subs, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: {})
    monkeypatch.setattr(uu, "load_week_schedule",
                        lambda season, week: [])

    # Provider matchup fetches carry finite latency and are classified
    # by calling stack: any call whose stack runs through the actions
    # compute is a scan-path fetch (the bug). Background sim threads
    # legitimately fetch when they compute their signature.
    calls = {"compute": 0, "other": 0}
    lock = threading.Lock()

    def _slow_get_matchups(platform, league_id, week, season=None, **_kw):
        frames = [f.name for f in traceback.extract_stack()]
        with lock:
            if "_run_actions_compute" in frames:
                calls["compute"] += 1
            else:
                calls["other"] += 1
        time.sleep(0.05)
        return []

    monkeypatch.setattr(platform_api, "get_matchups", _slow_get_matchups)
    # The Monte Carlo itself is not what this test exercises; make it
    # instant so kicked sims drain. Their signature computation (the
    # suspect path) stays real.
    monkeypatch.setattr(
        sim_mod, "simulate_playoff_odds",
        lambda ctx, platform="sleeper", **_kw: [])

    flask_app = Flask(__name__)
    flask_app.secret_key = "test"
    flask_app.register_blueprint(bp_mod.user_pages_bp)
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["account_id"] = 42
    yield client, calls
    # Let kicked sims drain so slots/marks don't leak to sibling tests.
    _wait_until(lambda: not app_mod._PLAYOFF_SIM_BUILDING, timeout=15)
    app_mod._PLAYOFF_SIM_CACHE.clear()


def _wait_actions_idle(timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not bp_mod._ACTIONS_PENDING and not bp_mod._ACTIONS_INFLIGHT:
            return True
        time.sleep(0.05)
    return False


def test_account_viewer_scan_completes_without_scan_thread_io(
        account_viewer_env):
    client, calls = account_viewer_env
    final = None
    for _cycle in range(60):
        data = client.get("/api/portfolio-actions").get_json()
        _wait_actions_idle(timeout=15.0)
        final = data["scan"]
        if data["scan"]["incomplete"] is False:
            break
        time.sleep(0.2)
    assert final is not None and final["scanned"] == 11, (
        f"scan never completed: {final}")
    assert final["total"] == 11
    assert calls["compute"] == 0, (
        f"actions scan thread paid {calls['compute']} provider matchup "
        "fetches (schedule-signature fan-out on the scan path)")
