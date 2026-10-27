"""Partial-progress recovery for /api/portfolio-actions (2026-10-27).

Production failure: a viewer with 11 leagues saw "Scanned 0 of 11
leagues / Could not check 11 of 11 leagues yet" on every load,
permanently. Trace of the shipped code:

- A scan pass only counts a league whose context is readable at that
  instant (allow_build=False), and a pass payload was cached ONLY when
  complete (scanned == total); partial results were discarded.
- The process-local DASHBOARD_CACHE holds at most DASHBOARD_CACHE_MAX
  (default 4) league contexts, so for a viewer with more leagues than
  that, background warms rebuild evicted leagues into the same small
  cache and evict each other in a carousel: no single pass can ever
  see every league (absent a Redis hydration hit for every evicted
  league), so no pass ever completes and nothing is ever cached.
- Since #2825 the endpoint never waits for the worker and returns the
  hardcoded deferred payload (scanned 0) on every cache miss, so the
  board's only visible states were "complete cached scan" or zeros.

Fix: per-league fragments persist each scanned league's raw actions +
board meta across passes (_ACTIONS_FRAGMENTS), later passes reuse
fresh fragments, and cache misses serve the latest finished pass
payload (_ACTIONS_LATEST) instead of the zero-count deferred payload.
Completion becomes the accumulation of per-league results, not one
all-or-nothing pass.

The carousel is simulated with a fake context provider whose resident
set holds 2 leagues while the viewer has 6: a cold read registers a
background warm (exactly like app.py's allow_build=False path) and the
test settles warms between polls, evicting the least-recently-used
resident -- the production dynamic at small scale.
"""
import time
from collections import OrderedDict

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_LEAGUE_IDS = ["L1", "L2", "L3", "L4", "L5", "L6"]


def _ctx_for(lid, name):
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1",
            "starters": [], "players": [],
            "reserve": [], "taxi": [], "settings": {},
        }],
        "league": {"name": name, "settings": {}},
        "users": [{"user_id": "owner-1", "display_name": "tester"}],
        "roster_positions": ["QB", "BN"],
        "proj_by_week": {},
        "players_index": {},
        "model_value_table": [],
    }


class _CarouselCtx:
    """Fake get_league_ctx_from_cache with a tiny resident set.

    Cold reads register a warm and return {} (the allow_build=False
    contract); settle() lands pending warms into the resident set,
    evicting least-recently-used residents beyond capacity -- so at
    most `capacity` league contexts are ever readable at once.
    """

    def __init__(self, league_ids, capacity=2):
        self.capacity = capacity
        self.resident = OrderedDict()
        self.pending = []
        self.broken = {}
        self.names = {lid: f"League {lid}" for lid in league_ids}

    def __call__(self, plat, lid, season, allow_build=True):
        if lid in self.broken:
            return self.broken[lid]
        if lid in self.resident:
            self.resident.move_to_end(lid)
            return _ctx_for(lid, self.names[lid])
        if lid not in self.pending:
            self.pending.append(lid)
        return {}

    def settle(self, count=2):
        landed = 0
        while self.pending and landed < count:
            lid = self.pending.pop(0)
            self.resident[lid] = True
            self.resident.move_to_end(lid)
            while len(self.resident) > self.capacity:
                self.resident.popitem(last=False)
            landed += 1


def _leagues(ids):
    return [{"league_id": lid, "platform": "sleeper", "season": 2026,
             "name": f"League {lid}"} for lid in ids]


def _client(monkeypatch, provider, league_ids, user_id="owner-1"):
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

    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", provider)
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached", lambda *a, **k: [])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (_leagues(league_ids), 2026))
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
        sess["viewer_user_id"] = user_id
    return client, bp_mod


def _wait_for_idle(bp_mod, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not bp_mod._ACTIONS_PENDING and not bp_mod._ACTIONS_INFLIGHT:
            return True
        time.sleep(0.05)
    return False


def test_carousel_viewer_eventually_completes(monkeypatch):
    # 6 leagues, only 2 contexts resident at a time: no single pass can
    # see them all, so pre-fix this viewer NEVER completes a scan and
    # every response is the zero-count deferred payload. With fragments
    # the scanned count accumulates pass over pass until it completes
    # and lands in the complete-scan cache.
    provider = _CarouselCtx(_LEAGUE_IDS, capacity=2)
    client, bp_mod = _client(monkeypatch, provider, _LEAGUE_IDS)

    seen = []
    final = None
    for _cycle in range(10):
        data = client.get("/api/portfolio-actions").get_json()
        assert _wait_for_idle(bp_mod)
        seen.append(data["scan"]["scanned"])
        provider.settle()
        if data["scan"]["incomplete"] is False:
            final = data
            break
    assert final is not None, f"scan never completed; scanned counts: {seen}"
    assert final["scan"]["total"] == 6
    assert final["scan"]["scanned"] == 6
    # Progress was visible and monotonic: real intermediate counts, not
    # the permanent zeros of the deferred payload.
    assert seen == sorted(seen), f"scanned counts regressed: {seen}"
    assert any(0 < s < 6 for s in seen), f"no partial progress shown: {seen}"
    # Completion landed in the complete-scan cache like any full scan.
    assert ("viewer:owner-1", 2026, 7) in bp_mod._ACTIONS_CACHE
    again = client.get("/api/portfolio-actions").get_json()
    assert again["scan"]["scanned"] == 6
    assert again["scan"]["incomplete"] is False


def test_failed_pass_never_overwrites_latest_partial(monkeypatch):
    # One league, warm: first pass completes and is stored as latest.
    # A later pass whose compute raises outright (the starters shape
    # that escapes every per-section guard) must leave the latest
    # payload in place: the next response serves the last good scan,
    # not zeros, and the single-flight marks are cleared for recovery.
    provider = _CarouselCtx(["L1"], capacity=2)
    provider.settle()
    provider.resident["L1"] = True
    client, bp_mod = _client(monkeypatch, provider, ["L1"])

    # First load kicks the background pass (its own response may be the
    # deferred payload or, if the pass lands first, the finished one --
    # the dispatch serves a just-landed payload; either is fine here).
    client.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    done = client.get("/api/portfolio-actions").get_json()
    assert done["scan"]["scanned"] == 1
    assert done["scan"]["incomplete"] is False

    # Expire the complete cache + fragments, then break the league's
    # data so the next compute raises outside every guard.
    bp_mod._ACTIONS_CACHE.clear()
    bp_mod._ACTIONS_FRAGMENTS.clear()
    bad = _ctx_for("L1", "League L1")
    bad["rosters"][0]["starters"] = 12345
    bad["rosters"][0]["players"] = ["p1"]
    provider.broken["L1"] = bad

    during = client.get("/api/portfolio-actions").get_json()
    # The last good scan is served while the doomed pass runs.
    assert during["scan"]["scanned"] == 1
    assert _wait_for_idle(bp_mod)
    key = ("viewer:owner-1", 2026, 7)
    assert key in bp_mod._ACTIONS_LATEST
    assert bp_mod._ACTIONS_LATEST[key][1]["scan"]["scanned"] == 1
    assert bp_mod._ACTIONS_CACHE == {}

    # Recovery: fix the data; the next pass completes again.
    provider.broken.clear()
    client.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    healed = client.get("/api/portfolio-actions").get_json()
    assert healed["scan"]["scanned"] == 1
    assert healed["scan"]["incomplete"] is False


def test_no_roster_outcome_is_not_fragmented(monkeypatch):
    # A league whose viewer roster cannot be resolved counts as
    # skipped_no_roster and must re-attempt on the next pass (a
    # transient identity failure must not stick for the fragment TTL).
    import utils.scorezone_user as szu

    real_match = szu.match_viewer_roster
    state = {"fail": True}

    def _flaky(*a, **k):
        if state["fail"]:
            raise RuntimeError("identity row shape changed")
        return real_match(*a, **k)

    monkeypatch.setattr(szu, "match_viewer_roster", _flaky)
    provider = _CarouselCtx(["L1"], capacity=2)
    provider.resident["L1"] = True
    client, bp_mod = _client(monkeypatch, provider, ["L1"])

    client.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    assert bp_mod._ACTIONS_FRAGMENTS == {}
    assert bp_mod._ACTIONS_LATEST == {}
    assert bp_mod._ACTIONS_CACHE == {}

    state["fail"] = False
    client.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    data = client.get("/api/portfolio-actions").get_json()
    assert data["scan"]["scanned"] == 1
    assert data["scan"]["incomplete"] is False


def test_latest_partial_is_viewer_scoped(monkeypatch):
    # Viewer A accumulates a partial scan; viewer B's first response
    # must be the zero-count deferred payload, never A's progress.
    ids = ["L1", "L2", "L3"]
    provider = _CarouselCtx(ids, capacity=2)
    client_a, bp_mod = _client(monkeypatch, provider, ids,
                               user_id="owner-1")
    client_a.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    provider.settle()
    client_a.get("/api/portfolio-actions")
    assert _wait_for_idle(bp_mod)
    a_view = client_a.get("/api/portfolio-actions").get_json()
    assert a_view["scan"]["scanned"] == 2

    client_b, _ = _client(monkeypatch, provider, ids,
                          user_id="owner-2")
    # _client cleared the shared stores; rebuild A's latest under A's
    # key to prove B cannot read it.
    import routes.user_pages_bp as bp_mod2
    bp_mod2._ACTIONS_LATEST[("viewer:owner-1", 2026, 7)] = (
        time.time(), a_view)
    b_first = client_b.get("/api/portfolio-actions").get_json()
    assert b_first["scan"]["scanned"] == 0
    assert b_first["scan"]["total"] == 3
    assert b_first["scan"]["incomplete"] is True
    assert _wait_for_idle(bp_mod2)
