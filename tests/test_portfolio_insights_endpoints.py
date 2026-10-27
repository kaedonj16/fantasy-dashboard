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

# Real Sleeper shape: /players/nfl rows carry NO bye field at all, so
# these fixtures deliberately omit bye_week. Byes must be derived from
# the player's team via the schedule-derived team bye map (stubbed
# below as _BYE_MAP); a fixture that hand-writes bye_week onto the
# players is exactly how the production bug shipped green.
_PLAYERS = {
    "s1": {"full_name": "Starter One", "position": "RB", "team": "BUF"},
    "s2": {"full_name": "Starter Two", "position": "WR", "team": "BUF"},
    "s3": {"full_name": "Texan Starter", "position": "WR", "team": "HOU"},
    "b1": {"full_name": "Bench One", "position": "RB", "team": "MIA"},
    "fa1": {"full_name": "Free Agent", "position": "RB", "team": "NE"},
}
_BYE_MAP = {"BUF": 8, "MIA": 5, "NE": 5, "HOU": 8, "WAS": 9}
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


def _client(monkeypatch, ctx_by_lid, leagues=None, sched=None, week=7):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.platform_api as platform_api
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    bp_mod._INSIGHTS_CACHE.clear()
    bp_mod._INSIGHTS_INFLIGHT.clear()
    bp_mod._BYE_CACHE.clear()
    bp_mod._BYE_INFLIGHT.clear()
    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": week, "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "_team_bye_map", lambda season: dict(_BYE_MAP))
    monkeypatch.setattr(
        bp_mod, "get_league_ctx_from_cache",
        lambda plat, lid, season, allow_build=False: ctx_by_lid.get(lid, {}))
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: _MODEL_ROWS)
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (leagues or _LEAGUES, 2026))
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: dict(_PLAYERS))
    # The insights compute overlays the week's games with the LIVE ESPN
    # scoreboard for the schedule's dates (get_nfl_scores_for_date /
    # build_team_game_lookup, imported from ds_api at call time). Left
    # unstubbed, the order test below passes or fails with whatever the
    # real scoreboard says for 2026-10-25 -- green while the real slate
    # was upcoming/live, red (windows == 0) once it went final. Pin the
    # overlay to empty so the stubbed schedule drives the games.
    monkeypatch.setattr(ds_api, "get_nfl_scores_for_date", lambda d, **k: {})
    monkeypatch.setattr(ds_api, "build_team_game_lookup", lambda body: {})
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


def test_players_with_byes_derives_from_team_map(monkeypatch):
    """Unit-pin the derivation itself: a row with no bye_week gets its
    team's bye (schedule aliases canonicalized), a row that already
    carries one keeps it, and the input rows are never mutated."""
    import routes.user_pages_bp as bp_mod

    monkeypatch.setattr(bp_mod, "_team_bye_map", lambda season: dict(_BYE_MAP))
    raw = {
        "a": {"full_name": "A", "team": "BUF"},
        "b": {"full_name": "B", "team": "WSH"},  # alias of WAS in the map
        "c": {"full_name": "C", "team": "BUF", "bye_week": 3},
        "d": {"full_name": "D", "team": "FA"},
    }
    out = bp_mod._pf_players_with_byes(raw, 2026)
    assert out["a"]["bye_week"] == 8
    assert out["b"]["bye_week"] == 9
    assert out["c"]["bye_week"] == 3
    assert "bye_week" not in out["d"]
    assert "bye_week" not in raw["a"]  # shared cache rows untouched


def test_bye_targets_current_week_before_first_kickoff(monkeypatch):
    """Tuesday-before-the-week case (the reported screenshot): Sleeper
    state has already advanced to Week 8 but no Week 8 game has kicked
    off, so Bye Watch must cover Week 8 itself, not skip to Week 9.
    Schedule epochs are strings in the real cache shape."""
    import time as _time

    future = str(_time.time() + 3 * 86400)
    sched = [{"home": "HOU", "away": "IND", "gameTime_epoch": future}]
    client, _bp, _pa = _client(
        monkeypatch, {"L1": _ctx("s3", "Alpha")}, sched=sched, week=8)
    data = client.get("/api/portfolio-bye").get_json()
    assert data["bye"]["week"] == 8
    assert data["bye"]["total_out"] == 1
    row = data["bye"]["rows"][0]
    assert row["out"][0]["name"] == "Texan Starter"
    assert row["verdict"].startswith("Hole")


def test_bye_targets_next_week_once_kickoff_passes(monkeypatch):
    """Once the current week's first game has kicked off, the target
    rolls to the following week (the in-progress week's byes are no
    longer actionable)."""
    import time as _time

    past = str(_time.time() - 86400)
    sched = [{"home": "HOU", "away": "IND", "gameTime_epoch": past}]
    client, _bp, _pa = _client(
        monkeypatch, {"L1": _ctx("s3", "Alpha")}, sched=sched, week=8)
    data = client.get("/api/portfolio-bye").get_json()
    assert data["bye"]["week"] == 9
    assert data["bye"]["total_out"] == 0
    assert data["bye"]["rows"][0]["verdict"] == "Full strength"


# ── Bye endpoint: response cache + stale-while-revalidate ───────────────────

def test_bye_second_call_within_ttl_does_not_recompute(monkeypatch):
    """The bye payload is cached per viewer under the light key: a
    second call inside the TTL must not pay the _pf_viewer_leagues
    preamble (league resolution, players index, model table) at all."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, _pa = _client(monkeypatch, ctxs)
    calls = {"n": 0}
    orig = bp_mod._pf_viewer_leagues

    def _counting(identity=None):
        calls["n"] += 1
        return orig(identity)

    monkeypatch.setattr(bp_mod, "_pf_viewer_leagues", _counting)
    first = client.get("/api/portfolio-bye").get_json()
    assert first["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert calls["n"] == 1
    second = client.get("/api/portfolio-bye").get_json()
    assert calls["n"] == 1  # cache hit: zero recomputation
    assert second == first  # same payload, original updated stamp


def test_bye_stale_payload_served_while_one_background_refresh_runs(
        monkeypatch):
    """Past the fresh TTL (but inside the stale grace) the endpoint
    serves the stale payload immediately -- original ``updated`` and
    all -- and kicks exactly one background refresh; a second stale
    call while it runs does not kick another."""
    import threading
    import time as _time

    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, _pa = _client(monkeypatch, ctxs)
    first = client.get("/api/portfolio-bye").get_json()
    assert first["scan"]["scanned"] == 2
    key = next(iter(bp_mod._BYE_CACHE))
    stale_payload = bp_mod._BYE_CACHE[key][1]
    bp_mod._BYE_CACHE[key] = (
        _time.monotonic() - bp_mod._BYE_TTL - 1.0, stale_payload)

    calls = {"n": 0}
    orig_compute = bp_mod._compute_portfolio_bye
    entered = threading.Event()
    release = threading.Event()

    def _blocking_compute(base):
        calls["n"] += 1
        entered.set()
        release.wait(timeout=5)
        return orig_compute(base)

    monkeypatch.setattr(bp_mod, "_compute_portfolio_bye", _blocking_compute)
    try:
        served = client.get("/api/portfolio-bye").get_json()
        assert served == first  # stale payload, honest updated stamp
        assert entered.wait(timeout=2.0)  # the one refresh is running
        served2 = client.get("/api/portfolio-bye").get_json()
        assert served2 == first
        assert calls["n"] == 1  # still exactly one refresh in flight
    finally:
        release.set()
    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        if not bp_mod._BYE_INFLIGHT and bp_mod._bye_cache_get(key) is not None:
            break
        _time.sleep(0.02)
    assert bp_mod._BYE_INFLIGHT == {}
    refreshed = bp_mod._bye_cache_get(key)
    assert refreshed is not None
    assert refreshed["updated"] != first["updated"]  # a real recompute landed


# ── Insights endpoint: cache + parallel order ───────────────────────────────

def _poll_insights(client, bp_mod, timeout=5.0):
    """Poll the insights endpoint until the background compute lands in
    the cache (and its single-flight mark clears), like the client's
    retry ladder does."""
    import time as _time

    deadline = _time.time() + timeout
    data = None
    while _time.time() < deadline:
        data = client.get("/api/portfolio-insights").get_json()
        if (data and not (data.get("scan") or {}).get("incomplete")
                and not bp_mod._INSIGHTS_INFLIGHT):
            return data
        _time.sleep(0.02)
    return data


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
    # Cold load (2026-10-27 speed pass): the request thread no longer
    # computes; it answers deferred and the compute lands in background.
    assert first["scan"]["incomplete"] is True
    landed = _poll_insights(client, bp_mod)
    assert landed["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert calls["n"] == 2  # one ctx read per league on the landed compute
    second = client.get("/api/portfolio-insights").get_json()
    assert calls["n"] == 2  # TTL cache: no recomputation at all
    assert second["rooting"] == landed["rooting"]
    assert second["updated"] == landed["updated"]  # honest computed-at time
    # Bye no longer rides the insights response.
    assert second["bye"]["rows"] == []


def test_insights_parallel_assembly_preserves_league_order(monkeypatch):
    """Both leagues' starters are on BUF in the same game; the home side
    must list them in league order (L1 before L2) even though the
    per-league work runs on a thread pool."""
    sched = [{"home": "BUF", "away": "NYJ",
              "gameTime_epoch": _sun_1pm_et_epoch()}]
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs, sched=sched)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    data = _poll_insights(client, bp_mod)
    windows = data["rooting"]["windows"]
    assert len(windows) == 1
    card = windows[0]["games"][0]
    assert card["home"] == "BUF"
    names = [s["name"] for s in card["sides"]["home"]]
    assert names == ["Starter One", "Starter Two"]


def test_insights_cold_call_defers_to_background_compute(monkeypatch):
    """Non-blocking cold load (2026-10-27 speed pass): NO request thread
    runs the per-league provider sweep anymore, not even the first one.
    A cold call answers immediately with the honest incomplete payload
    while exactly one background compute runs; a concurrent call for
    the same viewer defers to that compute instead of duplicating it,
    and the landed compute is what later calls serve."""
    import threading
    import time as _time

    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs)
    # A second client on a fresh app over the same blueprint state, with
    # the same viewer session: the single-flight key must match.
    client2, _bp2, _pa2 = _client(monkeypatch, ctxs)

    entered = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _blocking_rows(plat, lid, week, season):
        calls["n"] += 1
        if calls["n"] <= 2:  # the background compute's two league fetches
            entered.set()
            release.wait(timeout=5)
        return _matchup_rows(plat, lid, week, season)

    monkeypatch.setattr(platform_api, "get_matchups", _blocking_rows)
    bp_mod._INSIGHTS_INFLIGHT.clear()
    try:
        started = _time.time()
        first = client.get("/api/portfolio-insights").get_json()
        # Answered immediately even though the compute it kicked is
        # parked inside get_matchups: the request thread never computed.
        assert _time.time() - started < 2.0
        assert first["scan"]["scanned"] == 0
        assert first["scan"]["incomplete"] is True
        assert entered.wait(timeout=2.0)  # background compute is in flight
        deadline = _time.time() + 2.0
        while calls["n"] < 2 and _time.time() < deadline:
            _time.sleep(0.02)
        assert calls["n"] == 2  # both of the compute's fetches are parked
        deferred = client2.get("/api/portfolio-insights").get_json()
        # Concurrent call: deferred too, no duplicate compute kicked.
        assert calls["n"] == 2
        assert deferred["scan"]["scanned"] == 0
        assert deferred["scan"]["incomplete"] is True
    finally:
        release.set()
    # The background compute landed and cached; the next call serves it.
    served = _poll_insights(client, bp_mod)
    assert served["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert bp_mod._INSIGHTS_INFLIGHT == {}


def test_insights_skips_live_overlay_before_first_kickoff(monkeypatch):
    """A fully upcoming slate has no scores to overlay, and the ESPN
    scoreboard only discards its wrong-date answer for future dates:
    the compute must not pay for those calls at all. A recorder stands
    in for the scores feed; the windows must still build from the
    schedule alone."""
    import time as _time

    import dashboard_services.api as ds_api

    calls = []

    def _recorder(d, **k):
        calls.append(d)
        return {}

    monkeypatch.setattr(ds_api, "get_nfl_scores_for_date", _recorder)
    future = _time.time() + 2 * 86400
    sched = [{"home": "BUF", "away": "NYJ", "gameTime_epoch": future}]
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs, sched=sched)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    data = _poll_insights(client, bp_mod)
    assert calls == []
    windows = data["rooting"]["windows"]
    assert len(windows) == 1
    assert windows[0]["games"][0]["home"] == "BUF"


# ── Client wiring ───────────────────────────────────────────────────────────

def test_client_loads_bye_immediately_and_defers_rooting():
    js = (ROOT / "static" / "app.js").read_text()
    assert "fetcher('/api/portfolio-bye'" in js
    assert "function loadBye()" in js
    rooting = js[js.index("function loadInsights(attempt)"):]
    rooting = rooting[:rooting.index("function loadBye()")]
    assert "renderBye(" not in rooting  # bye no longer rides insights
    # The rooting load retries on the moves-board ladder instead of
    # painting its error after a single short fetch.
    assert "INS_RETRY" in rooting
    assert "45000" in rooting
    assert "loadInsights(attempt + 1)" in rooting
    assert "window.addEventListener('br:portfolio-primary-settled', kickInsights" in js
    assert "setTimeout(kickInsights, 6000);" in js
    assert "insightsKicked" in js
    # Speed pass (2026-10-27): the server answers deferred instantly
    # (background compute), so the first retry is shorter, and the
    # Rooting card also kicks when scrolled into view.
    assert "var INS_RETRY = [3000, 15000, 30000];" in js
    assert "IntersectionObserver" in js
    assert "getElementById('pfSecRooting')" in js
