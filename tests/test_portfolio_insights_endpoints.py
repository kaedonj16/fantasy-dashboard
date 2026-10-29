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
    bp_mod._PF_RESOLUTION_CACHE.clear()
    # Disk tier isolation: each test gets its own throwaway disk-cache
    # file so payloads persisted by one test can never serve another.
    import tempfile
    bp_mod._PF_DISK_PATH_OVERRIDE = str(
        pathlib.Path(tempfile.mkdtemp(prefix="pf-disk-"))
        / "pf_insights_cache.json")
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
    # Light resolver seam (2026-10-28): the bye/insights computes read
    # league meta/rosters/users from the cached provider layer and the
    # current week's projections from the memoized per-week builder,
    # so the ctx fixtures are exploded into provider stubs over the
    # same data. A league absent from ctx_by_lid is "cold": its
    # provider reads come back empty and the scan reports it
    # unscanned -- exactly what a missing ctx used to mean. (The ctx
    # stub above stays for the surfaces that still read contexts.)
    import dashboard_services.portfolio_light as pf_light

    def _stub_league(plat, lid, season):
        ctx = ctx_by_lid.get(lid) or {}
        league = dict(ctx.get("league") or {})
        if ctx:
            league.setdefault("roster_positions",
                              ctx.get("roster_positions") or [])
            league.setdefault("scoring_settings", {})
        return league

    monkeypatch.setattr(platform_api, "get_league", _stub_league)
    monkeypatch.setattr(
        platform_api, "get_rosters",
        lambda plat, lid, season: list(
            (ctx_by_lid.get(lid) or {}).get("rosters") or []))
    monkeypatch.setattr(
        platform_api, "get_users",
        lambda plat, lid, season: list(
            (ctx_by_lid.get(lid) or {}).get("users") or []))
    _proj = {}
    for _c in ctx_by_lid.values():
        for _w, _bundle in (_c.get("proj_by_week") or {}).items():
            _entry = _proj.setdefault(_w, {"projections": {}})
            _entry["projections"].update(
                (_bundle or {}).get("projections") or {})
    monkeypatch.setattr(
        pf_light, "_projections_by_week",
        lambda season, weeks, scoring: dict(_proj))
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
    second call inside the TTL must not pay the _pf_light_base
    preamble (league resolution, players index, model table) at all."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, _pa = _client(monkeypatch, ctxs)
    calls = {"n": 0}
    orig = bp_mod._pf_light_base

    def _counting(identity=None, **kw):
        calls["n"] += 1
        return orig(identity, **kw)

    monkeypatch.setattr(bp_mod, "_pf_light_base", _counting)
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

    # Light resolver (2026-10-28): the landed compute reads each
    # league's rosters from the provider layer exactly once and never
    # touches a league context at all.
    calls = {"ctx": 0, "rosters": 0}
    orig_ctx = bp_mod.get_league_ctx_from_cache

    def _counting_ctx(plat, lid, season, allow_build=False):
        calls["ctx"] += 1
        return orig_ctx(plat, lid, season, allow_build=allow_build)

    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", _counting_ctx)
    orig_rosters = platform_api.get_rosters

    def _counting_rosters(plat, lid, season):
        calls["rosters"] += 1
        return orig_rosters(plat, lid, season)

    monkeypatch.setattr(platform_api, "get_rosters", _counting_rosters)
    first = client.get("/api/portfolio-insights").get_json()
    # Cold load (2026-10-27 speed pass): the request thread no longer
    # computes; it answers deferred and the compute lands in background.
    assert first["scan"]["incomplete"] is True
    landed = _poll_insights(client, bp_mod)
    assert landed["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert calls == {"ctx": 0, "rosters": 2}
    second = client.get("/api/portfolio-insights").get_json()
    assert calls == {"ctx": 0, "rosters": 2}  # TTL cache: no recompute
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
        if calls["n"] <= 1:  # the background compute's first league fetch
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
        # Sequential compute (light resolver): the second league's
        # fetch has not started while the first is parked.
        assert calls["n"] == 1
        deferred = client2.get("/api/portfolio-insights").get_json()
        # Concurrent call: deferred too, no duplicate compute kicked.
        assert calls["n"] == 1
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


# ── Disk tier: payloads survive a worker recycle ────────────────────────

def test_pf_disk_tier_roundtrip_and_expiry(monkeypatch):
    """The disk tier stores the finished payload with its true compute
    time: a read inside the grace returns it with its real age, a read
    past the grace returns nothing (TTL enforced on read)."""
    import time as _time

    import routes.user_pages_bp as bp_mod

    key = ("acct-1", 2026, 7)
    payload = {"ok": True, "scan": {"total": 2, "scanned": 2,
                                    "incomplete": False},
               "updated": "2026-10-25T17:00:00+00:00"}
    bp_mod._pf_disk_put("insights", key, payload)
    got = bp_mod._pf_disk_get("insights", key, 1800.0)
    assert got is not None
    served, age = got
    assert served == payload  # updated stamp untouched
    assert 0 <= age < 60
    # Backdate the stored compute time past the grace: expired.
    import json as _json
    path = bp_mod._pf_disk_path()
    blob = _json.loads(pathlib.Path(path).read_text())
    store_key = bp_mod._pf_disk_store_key("insights", key)
    blob[store_key]["computed_at"] = _time.time() - 1801
    pathlib.Path(path).write_text(_json.dumps(blob))
    assert bp_mod._pf_disk_get("insights", key, 1800.0) is None


def test_insights_disk_tier_serves_after_inprocess_wipe(monkeypatch):
    """Simulated worker recycle: the finished compute persisted to
    disk, so after BOTH in-process caches are cleared the endpoint
    serves the disk payload instantly -- no preamble, no provider
    fetch, original ``updated`` stamp."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    landed = _poll_insights(client, bp_mod)
    assert landed["scan"] == {"total": 2, "scanned": 2, "incomplete": False}

    calls = {"preamble": 0, "matchups": 0}
    orig_preamble = bp_mod._pf_light_base

    def _counting_preamble(identity=None, **kw):
        calls["preamble"] += 1
        return orig_preamble(identity, **kw)

    def _counting_matchups(*a, **k):
        calls["matchups"] += 1
        return _matchup_rows(*a, **k)

    monkeypatch.setattr(bp_mod, "_pf_viewer_leagues", _counting_preamble)
    monkeypatch.setattr(platform_api, "get_matchups", _counting_matchups)
    bp_mod._INSIGHTS_CACHE.clear()   # the recycle wipes both dicts
    bp_mod._INSIGHTS_INFLIGHT.clear()
    served = client.get("/api/portfolio-insights").get_json()
    assert served["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert served["rooting"] == landed["rooting"]
    assert served["updated"] == landed["updated"]  # never restamped
    assert calls == {"preamble": 0, "matchups": 0}


def test_bye_disk_tier_serves_after_inprocess_wipe(monkeypatch):
    """Bye Watch after a recycle: the disk payload serves before the
    endpoint pays the _pf_light_base preamble at all."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, _pa = _client(monkeypatch, ctxs)
    first = client.get("/api/portfolio-bye").get_json()
    assert first["scan"] == {"total": 2, "scanned": 2, "incomplete": False}

    calls = {"n": 0}
    orig = bp_mod._pf_light_base

    def _counting(identity=None, **kw):
        calls["n"] += 1
        return orig(identity, **kw)

    monkeypatch.setattr(bp_mod, "_pf_viewer_leagues", _counting)
    bp_mod._BYE_CACHE.clear()   # the recycle wipes both dicts
    bp_mod._BYE_INFLIGHT.clear()
    served = client.get("/api/portfolio-bye").get_json()
    assert served == first  # same payload, same updated stamp
    assert calls["n"] == 0  # disk answered before the preamble


def test_insights_kick_not_starved_by_actions_semaphore(monkeypatch):
    """Starvation regression: the insights compute used to share
    _ACTIONS_COMPUTE_SEM(2) with the long moves-board scans, so a busy
    board silently dropped its kick. With both actions slots held, a
    cold insights call must STILL land its compute (dedicated
    _INSIGHTS_COMPUTE_SEM(1))."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    sem = bp_mod._ACTIONS_COMPUTE_SEM
    assert sem.acquire(blocking=False)
    assert sem.acquire(blocking=False)
    try:
        first = client.get("/api/portfolio-insights").get_json()
        assert first["scan"]["incomplete"] is True  # deferred as usual
        landed = _poll_insights(client, bp_mod)
    finally:
        sem.release()
        sem.release()
    assert landed["scan"] == {"total": 2, "scanned": 2, "incomplete": False}


# ── Shared snapshot: one resolution serves every endpoint ───────────────

def test_snapshot_resolution_serves_bye_and_insights_preambles(monkeypatch):
    """The page build stores its league resolution into the shared
    snapshot; after that, BOTH endpoint preambles (the bye endpoint's
    synchronous one and the insights background worker's) must reuse
    it instead of paying resolve_my_leagues again."""
    import dashboard_services.accounts as accounts

    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)
    # What the page build does with the resolution it already paid for:
    bp_mod._pf_resolution_put("owner-1", None, 2026, _LEAGUES, 2026)

    calls = {"n": 0}
    orig_resolver = accounts.resolve_my_leagues

    def _counting_resolver(*a, **k):
        calls["n"] += 1
        return orig_resolver(*a, **k)

    monkeypatch.setattr(accounts, "resolve_my_leagues", _counting_resolver)
    bye = client.get("/api/portfolio-bye").get_json()
    assert bye["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    landed = _poll_insights(client, bp_mod)
    assert landed["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    assert calls["n"] == 0  # one resolution, stored once, served both


def test_embed_payloads_compute_in_page_pass_and_store(monkeypatch):
    """The page build's embed pass: bye (cache-only) and the pregame
    rooting payload are computed from the data the pass already holds
    -- with get_matchups rigged to raise, proving rooting's embed
    shape makes NO provider call -- and both land in the snapshot
    caches (+ disk for bye) for the endpoints to serve. A second
    embed pass reuses them without recomputing."""
    import time as _time

    future = _time.time() + 2 * 86400
    sched = [{"home": "BUF", "away": "NYJ", "gameTime_epoch": future}]
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    client, bp_mod, platform_api = _client(
        monkeypatch, ctxs, sched=sched, week=8)

    def _boom(*a, **k):
        raise AssertionError("embed rooting must not call get_matchups")

    monkeypatch.setattr(platform_api, "get_matchups", _boom)
    # The page build's embed pass now resolves through the light
    # preamble with the resolution it already paid for.
    base = bp_mod._pf_light_base(
        identity={"viewer_username": "tester",
                  "viewer_user_id": "owner-1", "account_id": None},
        league_inputs=list(_LEAGUES), season=2026, week=8)
    assert base["cache_key"] == ("viewer:owner-1", 2026, 8)
    bye_calls = {"n": 0}
    orig_bye = bp_mod._compute_portfolio_bye

    def _counting_bye(b):
        bye_calls["n"] += 1
        return orig_bye(b)

    monkeypatch.setattr(bp_mod, "_compute_portfolio_bye", _counting_bye)
    bye_payload, rooting_payload = bp_mod._pf_embed_payloads(base)
    # Bye: Week 8 target (kickoff still ahead), both BUF starters out.
    assert bye_payload["bye"]["week"] == 8
    assert bye_payload["scan"]["scanned"] == 2
    assert bye_payload["bye"]["total_out"] == 2
    # Rooting: windows built from schedule + starters + projections,
    # points honestly None pregame, and no provider call happened.
    assert rooting_payload["scan"] == {
        "total": 2, "scanned": 2, "incomplete": False}
    windows = rooting_payload["rooting"]["windows"]
    assert len(windows) == 1
    starters = windows[0]["games"][0]["sides"]["home"]
    assert {s["name"] for s in starters} == {"Starter One", "Starter Two"}
    assert all(s["points"] is None for s in starters)
    # Stored for the endpoints: in-process caches and the bye disk tier.
    key = base["cache_key"]
    assert bp_mod._bye_cache_get(key) == bye_payload
    assert bp_mod._insights_cache_get(key) == rooting_payload
    assert bp_mod._pf_disk_get("bye", key, 300.0) is not None
    # Second pass (the next page load) reuses, never recomputes.
    bye2, rooting2 = bp_mod._pf_embed_payloads(base)
    assert bye_calls["n"] == 1
    assert bye2 == bye_payload and rooting2 == rooting_payload


# ── Client wiring ───────────────────────────────────────────────────────────

def test_client_hydrates_insights_lazily_on_first_row_open():
    js = (ROOT / "static" / "app.js").read_text()
    assert "fetcher('/api/portfolio-bye'" in js
    assert "function loadBye(attempt)" in js
    rooting = js[js.index("function loadInsights(attempt, startedAt)"):]
    rooting = rooting[:rooting.index("function loadBye(attempt)")]
    assert "renderBye(" not in rooting  # bye no longer rides insights
    # The rooting load retries on the moves-board ladder instead of
    # painting its error after a single short fetch.
    assert "INS_RETRY" in rooting
    assert "45000" in rooting
    assert "loadInsights(attempt + 1, t0)" in rooting
    # Direction C (2026-10-29): neither loader runs on page load. The
    # old page-load kicks (primary-settled listener, 6s fallback timer,
    # scroll-into-view observer) are gone; the accordion's first open
    # dispatches through __pfLoadSection, each section loads once, and
    # the whole-page refresh hooks re-run only already-loaded sections.
    assert "kickInsights" not in js
    assert "window.__pfLoadSection = function (id)" in js
    assert "if (id === 'pfSecRooting')" in js
    assert ("if (insightsLoaded || !document.getElementById('pfRootingBody')) return;"
            in js)
    assert ("if (byeLoaded || !document.getElementById('pfByeBody')) return;"
            in js)
    assert ("window.__pfRefreshInsights = function () "
            "{ if (insightsLoaded) loadInsights(); };") in js
    assert ("window.__pfRefreshBye = function () "
            "{ if (byeLoaded) loadBye(); };") in js
    # Speed pass (2026-10-27): the server answers deferred instantly
    # (background compute), so the first retry is shorter.
    assert "var INS_RETRY = [3000, 15000, 30000];" in js
    # Cold-load fix (2026-10-27): the old ladder burned out at ~48s,
    # before a cold production compute lands, and painted the
    # no-starters empty state for a scan that never completed. The
    # ladder now polls politely for ~3 minutes, and the budget-expired
    # incomplete state is an honest still-loading message.
    assert "var INS_POLL_MS = 10000;" in rooting
    assert "var INS_BUDGET_MS = 180000;" in rooting
    assert "Still loading your games, taking longer than usual." in rooting
    assert "rb.isConnected" in rooting
    # Bye Watch client: it used to be one 20s shot with no retry at
    # all. It now has its own bounded ladder for failures and for
    # incomplete scans that produced no rows yet.
    bye = js[js.index("function loadBye(attempt)"):]
    bye = bye[:bye.index("window.__pfRefreshInsights")]
    assert "var BYE_RETRY = [3000, 10000, 20000, 30000];" in bye
    assert "loadBye(attempt + 1)" in bye
    # Embedded first paint: both loaders consume the server-embedded
    # payload (computed in the page build) before any fetch, and each
    # block is consumed at most once so refresh hooks still fetch.
    assert "function pfEmbedded(id)" in js
    assert "_pfConsumed" in js
    assert "pfEmbedded('pf-bye-embedded')" in js
    assert "pfEmbedded('pf-rooting-embedded')" in js
