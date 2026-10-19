"""Cross-league player availability: service logic + API route.

Fake contexts mirror the real adapter shape: roster dicts with
``players`` / ``starters`` / ``reserve`` / ``taxi`` string-id lists, a
``roster_map`` of roster_id -> team name, and a per-request ``viewer``
dict, exactly as build_league_context produces them.
"""
from dashboard_services.availability import (
    availability_in_league,
    build_availability,
    league_sub,
    player_slot,
)


def _roster(rid, players, starters=(), reserve=(), taxi=()):
    return {
        "roster_id": rid,
        "owner_id": f"user-{rid}",
        "players": list(players),
        "starters": list(starters),
        "reserve": list(reserve),
        "taxi": list(taxi),
    }


def _ctx(rosters, viewer_rid=None, **over):
    ctx = {
        "league": {"name": "Blackedraw", "settings": {"type": 0}},
        "rosters": rosters,
        "users": [],
        "roster_map": {1: "Caleb's Casting Couch", 2: "Never Chase"},
        "scoring_settings": {"rec": 1.0},
        "roster_positions": ["QB", "RB", "WR", "FLEX"],
        "total_rosters": 10,
        "viewer": {"viewer_roster_id": viewer_rid},
    }
    ctx.update(over)
    return ctx


# ── player_slot ──────────────────────────────────────────────────────────────

def test_slot_starter_ir_taxi_bench():
    roster = _roster(1, ["a", "b", "c", "d"], starters=["a"], reserve=["b"], taxi=["c"])
    assert player_slot(roster, "a") == "Starter"
    assert player_slot(roster, "b") == "IR"
    assert player_slot(roster, "c") == "Taxi"
    assert player_slot(roster, "d") == "Bench"
    assert player_slot(roster, "zzz") is None


def test_slot_coerces_int_ids():
    roster = _roster(1, [11647], starters=[11647])
    assert player_slot(roster, "11647") == "Starter"


# ── availability_in_league ───────────────────────────────────────────────────

def test_rostered_by_other_with_owner_and_slot():
    ctx = _ctx([_roster(1, ["x"]), _roster(2, ["11647"], starters=["11647"])], viewer_rid=1)
    out = availability_in_league(ctx, "11647")
    assert out == {"status": "rostered", "owner": "Never Chase", "slot": "Starter", "mine": False}


def test_rostered_by_viewer_is_mine():
    ctx = _ctx([_roster(1, ["11560"]), _roster(2, ["x"])], viewer_rid="1")
    out = availability_in_league(ctx, "11560")
    assert out["status"] == "rostered"
    assert out["mine"] is True
    assert out["owner"] == "Caleb's Casting Couch"


def test_available_when_not_on_any_roster():
    ctx = _ctx([_roster(1, ["x"]), _roster(2, ["y"])], viewer_rid=1)
    assert availability_in_league(ctx, "12497")["status"] == "available"


def test_unknown_when_context_has_no_rosters():
    assert availability_in_league({}, "12497")["status"] == "unknown"
    assert availability_in_league({"rosters": []}, "12497")["status"] == "unknown"


def test_owner_name_falls_back_to_roster_label():
    ctx = _ctx([_roster(7, ["p1"])], roster_map={})
    assert availability_in_league(ctx, "p1")["owner"] == "Roster 7"


# ── league_sub ───────────────────────────────────────────────────────────────

def test_sub_full_ppr():
    assert league_sub("sleeper", _ctx([])) == "Sleeper - 10-Team - PPR"


def test_sub_superflex_dynasty_half_ppr():
    ctx = _ctx(
        [],
        league={"name": "Dynasty Degens", "settings": {"type": 2}},
        scoring_settings={"rec": 0.5},
        roster_positions=["QB", "RB", "WR", "SUPER_FLEX"],
        total_rosters=12,
    )
    assert league_sub("sleeper", ctx) == "Sleeper - 12-Team - Half PPR - Superflex - Dynasty"


def test_sub_omits_unknown_pieces():
    ctx = _ctx([], scoring_settings={}, roster_positions=[], total_rosters=0,
               league={"name": "X", "settings": {}})
    # No size signal except roster count (0 rosters here -> nothing), no scoring.
    assert league_sub("espn", ctx) == "ESPN"


# ── build_availability ───────────────────────────────────────────────────────

_LEAGUES = [
    {"platform": "sleeper", "league_id": "L1", "season": 2026, "name": "Blackedraw"},
    {"platform": "espn", "league_id": "L2", "season": 2026, "name": "Sunday Funday"},
    {"platform": "yahoo", "league_id": "L3", "season": 2026, "name": "Gridiron Gurus"},
]


def _getter_factory(ctxs):
    def getter(platform, league_id, season):
        return ctxs.get((platform, league_id, season))
    return getter


def test_build_mixed_statuses_and_summary():
    ctxs = {
        ("sleeper", "L1", 2026): _ctx(
            [_roster(1, ["mine1"]), _roster(2, ["11647"], starters=["11647"])], viewer_rid=1),
        ("espn", "L2", 2026): _ctx([_roster(1, ["other"])], viewer_rid=3,
                                   league={"name": "Sunday Funday", "settings": {}}),
        # L3 context not cached -> unknown.
    }
    out = build_availability(_LEAGUES, ["11647", "12497"], _getter_factory(ctxs))

    vidal = out["11647"]
    assert vidal["summary"] == {"total": 3, "checked": 2, "available": 1, "rostered": 1, "mine": 0}
    by_id = {r["league_id"]: r for r in vidal["leagues"]}
    assert by_id["L1"]["status"] == "rostered"
    assert by_id["L1"]["owner"] == "Never Chase"
    assert by_id["L1"]["href"] == "/sleeper/2026/L1/teams"
    assert by_id["L2"]["status"] == "available"
    assert by_id["L2"]["href"] == "/espn/2026/L2/waivers"
    assert by_id["L3"]["status"] == "unknown"
    assert by_id["L3"]["href"] is None

    horton = out["12497"]
    assert horton["summary"]["available"] == 2
    assert horton["summary"]["checked"] == 2


def test_build_mine_count():
    ctxs = {
        ("sleeper", "L1", 2026): _ctx([_roster(1, ["11560"])], viewer_rid=1),
        ("espn", "L2", 2026): _ctx([_roster(9, ["11560"])], viewer_rid=9,
                                   league={"name": "Sunday Funday", "settings": {}}),
    }
    out = build_availability(_LEAGUES[:2], ["11560"], _getter_factory(ctxs))
    assert out["11560"]["summary"]["mine"] == 2
    assert all(r["mine"] for r in out["11560"]["leagues"])


def test_build_ctx_getter_called_once_per_league_for_batch():
    calls = []

    def getter(platform, league_id, season):
        calls.append((platform, league_id, season))
        return _ctx([_roster(1, ["a"])])

    build_availability(_LEAGUES, ["a", "b", "c"], getter)
    assert len(calls) == 3


def test_build_getter_exception_marks_league_unknown():
    def getter(platform, league_id, season):
        raise RuntimeError("cache down")

    out = build_availability(_LEAGUES[:1], ["a"], getter)
    assert out["a"]["leagues"][0]["status"] == "unknown"
    assert out["a"]["summary"]["checked"] == 0


# ── route ────────────────────────────────────────────────────────────────────

def _route_client(monkeypatch, ctxs, leagues=_LEAGUES, signed_in=True):
    pytest = __import__("pytest")
    pytest.importorskip("flask")
    import sys
    import types

    from flask import Flask

    import dashboard_services.accounts as accounts
    import routes.players_bp as players_bp

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(players_bp.players_bp)
    monkeypatch.setattr(players_bp, "get_nfl_state", lambda: {"season": 2026})
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda *a, **k: (list(leagues), 2026),
    )
    fake_app = types.ModuleType("app")

    def fake_ctx(platform, league_id, season, allow_build=True):
        assert allow_build is False  # availability must never build inline
        return ctxs.get((platform, str(league_id), int(season)))

    fake_app.get_league_ctx_from_cache = fake_ctx

    fake_app._FAKE_REDIS_CTX = {}
    # None = the resolver finds no viewer, so baked ctx viewers stand.
    # Tests set a rid to prove the route's request-thread re-resolution.
    fake_app._FAKE_VIEWER_RID = [None]

    def _redis_load(platform, league_id, season):
        return fake_app._FAKE_REDIS_CTX.get((platform, str(league_id), season))

    def _viewer_for(users, rosters, platform, league_id, season):
        rid = fake_app._FAKE_VIEWER_RID[0]
        return {"viewer_roster_id": rid} if rid is not None else {}

    fake_app._league_ctx_redis_load = _redis_load
    fake_app.get_viewer_session_for_league = _viewer_for
    monkeypatch.setitem(sys.modules, "app", fake_app)
    client = app.test_client()
    if signed_in:
        with client.session_transaction() as sess:
            sess["account_id"] = 42
    return client


def test_route_requires_player_id(monkeypatch):
    client = _route_client(monkeypatch, {})
    assert client.get("/api/player-availability").status_code == 400


def test_route_guest_gets_no_data(monkeypatch):
    client = _route_client(monkeypatch, {}, signed_in=False)
    resp = client.get("/api/player-availability?player_id=11647")
    assert resp.status_code == 200
    assert resp.json == {"ok": True, "signed_in": False, "players": {}}
    assert resp.headers.get("Cache-Control") == "no-store"


def test_route_single_and_batch(monkeypatch):
    ctxs = {
        ("sleeper", "L1", 2026): _ctx(
            [_roster(1, ["x"]), _roster(2, ["11647"], starters=["11647"])], viewer_rid=1),
        ("espn", "L2", 2026): _ctx([_roster(1, ["z"])],
                                   league={"name": "Sunday Funday", "settings": {}}),
    }
    client = _route_client(monkeypatch, ctxs)
    resp = client.get("/api/player-availability?player_id=11647")
    assert resp.status_code == 200
    players = resp.json["players"]
    assert players["11647"]["summary"]["rostered"] == 1
    assert players["11647"]["summary"]["available"] == 1
    assert players["11647"]["summary"]["total"] == 3  # L3 unresolved ctx -> unknown

    resp = client.get("/api/player-availability?player_ids=11647,12497")
    assert set(resp.json["players"]) == {"11647", "12497"}
    assert resp.json["players"]["12497"]["summary"]["available"] == 2


def test_route_batch_capped_at_twelve(monkeypatch):
    client = _route_client(monkeypatch, {}, leagues=[])
    ids = ",".join(str(i) for i in range(20))
    resp = client.get(f"/api/player-availability?player_ids={ids}")
    assert resp.status_code == 200
    assert len(resp.json["players"]) == 12


def test_route_hydrates_from_redis_when_local_cache_misses(monkeypatch):
    # The per-worker cache holds only a few league contexts, so a league
    # can be absent locally while its built ctx sits in shared Redis.
    # The route must answer from Redis instead of reporting unknown.
    import sys

    client = _route_client(monkeypatch, {})
    fake_app = sys.modules["app"]
    fake_app._FAKE_REDIS_CTX[("sleeper", "L1", 2026)] = _ctx(
        [_roster(1, ["x"]), _roster(2, ["11647"], starters=["11647"])], viewer_rid=1)
    resp = client.get("/api/player-availability?player_id=11647")
    assert resp.status_code == 200
    entry = resp.json["players"]["11647"]
    l1 = next(r for r in entry["leagues"] if r["name"] == "Blackedraw")
    assert l1["status"] == "rostered"
    assert entry["summary"]["checked"] == 1


def test_route_mine_resolved_on_request_thread(monkeypatch):
    # Pool-thread ctx fetches carry the empty viewer a background warm
    # baked in. The route re-resolves the viewer from the request session,
    # so "mine" works even when the fetched ctx's viewer is blank.
    import sys

    ctxs = {
        ("sleeper", "L1", 2026): _ctx(
            [_roster(1, ["11560"]), _roster(2, ["y"])], viewer_rid=None),
    }
    client = _route_client(monkeypatch, ctxs)
    sys.modules["app"]._FAKE_VIEWER_RID[0] = 1
    resp = client.get("/api/player-availability?player_id=11560")
    assert resp.status_code == 200
    entry = resp.json["players"]["11560"]
    l1 = next(r for r in entry["leagues"] if r["name"] == "Blackedraw")
    assert l1["status"] == "rostered"
    assert l1["mine"] is True
    assert entry["summary"]["mine"] == 1


# ── frontend pins ────────────────────────────────────────────────────────────

def _read(rel):
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / rel).read_text(encoding="utf-8")


def test_modal_section_markup_and_lazy_loader():
    js = _read("static/player_modal.js")
    assert 'id="pmAvail"' in js
    assert "_pmSetupAvail(playerId, data)" in js
    # Collapsed at start, chevron/hint/secnote header, hidden body.
    assert 'id="pmAvailToggle" aria-expanded="false"' in js
    assert "pm-avail-toggle" in js  # opts out of the global accent button fill
    assert 'id="pmAvailBody" hidden' in js
    # Lazy: fetch fires only from the toggle path's loader.
    assert "if (_pmAvailOpen && !body.dataset.loaded)" in js
    assert "'/api/player-availability?player_id='" in js or \
        '"/api/player-availability?player_id="' in js or \
        "/api/player-availability?player_id=" in js
    # Skeleton + error escape both wired.
    assert "_pmAvailSkeleton()" in js
    assert "brErrorState" in js
    # Guests and draft picks never render the section.
    assert "_pmAvailSignedIn()" in js


def test_modal_unknown_leagues_are_loading_rows_not_values():
    js = _read("static/player_modal.js")
    css = _read("static/dashboard.css")
    # An unknown league renders as a loading row (skeleton), never a status.
    assert "pm-avail-row loading" in js
    assert "pm-avail-status unk" not in js
    assert ".pm-avail-status.unk" not in css
    assert "League not loaded yet" not in js
    # The header summary stays unpopulated while any league is unknown,
    # and the section re-checks on a bounded poll as leagues warm.
    assert "Checking your leagues" in js
    assert "_PM_AVAIL_MAX_POLLS" in js
    assert "_pmAvailRender(entry)" in js


def test_nav_search_bits_lazy_fill():
    js = _read("static/app.js")
    assert "_availMetaSuffix(p)" in js
    assert "nav-search-avail-slot" in js
    assert "'/api/player-availability?player_ids='" in js or \
        "/api/player-availability?player_ids=" in js
    assert "window.__brAvailCache" in js
    assert "_fillAvailBits();" in js
    # Bit copy matches the approved mock.
    assert "Available in " in js and "Rostered in all " in js


def test_nav_search_bit_waits_until_every_league_known():
    js = _read("static/app.js")
    # The completeness gate: no value populates from a partial summary.
    assert "_availComplete" in js
    assert "s.checked === s.total" in js
    # The old partial-claim copy is gone.
    assert "Rostered in ' + s.checked" not in js
    # One delayed retry for warming leagues, then no bit rather than a guess.
    assert "_scheduleAvailRetry" in js


def test_availability_css_present():
    css = _read("static/dashboard.css")
    for sel in (".pm-avail-list", ".pm-avail-row", ".pm-avail-status.avail",
                ".pm-avail-secnote", ".pm-avail-toggle", ".nav-search-avail-bit", ".nav-search-avail-sk"):
        assert sel in css, sel


def test_service_worker_bumped():
    sw = _read("static/sw.js")
    assert "br-fantasy-v66" in sw
