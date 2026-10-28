"""User-scope (My Leagues) ScoreZone must share work like league scope does.

Regression coverage for the duplicate-call audit findings:

  * League slices used to call ``_scorezone_collect`` directly, rebuilding
    the (viewer-independent) collect for every viewer on every 15s poll.
    They now read through ``_rz_cached_collect`` -- the same 60s
    single-flight cache league scope uses.
  * The aggregate (``_scorezone_fetch_user``) had no cache at all and the
    user branch never set an ETag, so user-scope polls could never 304 and
    re-paid the whole portfolio resolution every time. ``_rz_cached_fetch_user``
    adds a bounded per-viewer TTL cache + content ETag, and the user branch
    of ``_scorezone_fetch`` stamps it on ``g`` like the league branch.
  * The per-game player-metadata build rescanned the whole NFL player index
    once per game; the index is now flattened once per collect
    (``_rz_player_index_rows``) and sliced per game
    (``_rz_player_meta_for_teams``) with identical output.
  * The page render stamps the payload's ETag next to ``window.__rz__`` and
    the client seeds its conditional-poll state from it, and the box-score
    sheet caches finals per session (live games mutate per play and keep
    fetch-per-open).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_rz_caches(monkeypatch):
    monkeypatch.setattr(app, "_RZ_COLLECT_CACHE", {})
    monkeypatch.setattr(app, "_RZ_COLLECT_INFLIGHT", {})
    monkeypatch.setattr(app, "_RZ_USER_CACHE", {})
    monkeypatch.setattr(app, "_RZ_USER_INFLIGHT", {})


def _user_payload(n=0):
    return {
        "week": 8, "season": 2026, "platform": "sleeper", "league_id": "L1",
        "scope": "user",
        "matchups": [], "rosters": [], "users": [], "leagues": [],
        "player_info": {}, "scoring": {}, "scoring_by_league": {},
        "pid_league": {}, "pbp_by_game": {}, "games": {},
        "viewer_roster_id": "0:1", "viewer_roster_ids": ["0:1"],
        "updated_at": 1700000000.0 + n,
    }


def _stub_fetch_user(monkeypatch):
    calls = []

    def fake(platform, league_id, season, week):
        calls.append((platform, league_id, season, week))
        return _user_payload(len(calls))

    monkeypatch.setattr(app, "_scorezone_fetch_user", fake)
    return calls


def _ctx(account_id=42, viewer_uid="u1"):
    ctx = app.app.test_request_context("/sleeper/2026/L1/scorezone?scope=user")
    ctx.push()
    from flask import session
    if account_id is not None:
        session["account_id"] = account_id
    if viewer_uid is not None:
        session["viewer_user_id"] = viewer_uid
    return ctx


# ── Per-viewer aggregate cache ──────────────────────────────────────────────

def test_user_aggregate_second_poll_within_ttl_does_not_rebuild(monkeypatch):
    calls = _stub_fetch_user(monkeypatch)
    ctx = _ctx()
    try:
        d1, etag1 = app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
        d2, etag2 = app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
    finally:
        ctx.pop()
    assert calls == [("sleeper", "L1", 2026, 8)], "second poll rebuilt the aggregate"
    assert d1 is d2, "second poll must serve the cached payload object"
    assert etag1 == etag2 and etag1.startswith('"rzu-')


def test_user_aggregate_cache_is_per_viewer(monkeypatch):
    calls = _stub_fetch_user(monkeypatch)
    ctx = _ctx(account_id=42)
    try:
        app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
    finally:
        ctx.pop()
    ctx = _ctx(account_id=43)
    try:
        app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
    finally:
        ctx.pop()
    assert len(calls) == 2, "a different viewer must not be served another viewer's cache entry"


def test_user_aggregate_failures_are_not_cached(monkeypatch):
    calls = []

    def flaky(platform, league_id, season, week):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("user scope requires a signed-in viewer")
        return _user_payload(len(calls))

    monkeypatch.setattr(app, "_scorezone_fetch_user", flaky)
    ctx = _ctx()
    try:
        with pytest.raises(ValueError):
            app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
        d, _etag = app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
        app._rz_cached_fetch_user("sleeper", "L1", 2026, 8)
    finally:
        ctx.pop()
    assert d["scope"] == "user"
    assert len(calls) == 2, "the failure was cached or the success was rebuilt"


def test_user_aggregate_concurrent_polls_single_flight(monkeypatch):
    calls = []

    def slow(platform, league_id, season, week):
        calls.append(1)
        time.sleep(0.4)
        return _user_payload(1)

    monkeypatch.setattr(app, "_scorezone_fetch_user", slow)
    # All threads share one request context (same viewer) -- the single-flight
    # contract is per cache key, not per thread.
    ctx = _ctx()
    results, errors = [], []

    def poll():
        try:
            results.append(app._rz_cached_fetch_user("sleeper", "L1", 2026, 8))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=poll) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        ctx.pop()
    assert not any(t.is_alive() for t in threads), "a poll hung"
    assert not errors
    assert len(calls) == 1, f"expected 1 shared build, got {len(calls)}"
    assert len(results) == 4


# ── _scorezone_fetch user branch: etag + copy semantics ─────────────────────

def test_fetch_user_scope_sets_etag_and_never_mutates_cache(monkeypatch):
    calls = _stub_fetch_user(monkeypatch)
    monkeypatch.setattr(app, "_games_scheduled_today", lambda s, w: True)
    import dashboard_services.api as api_mod
    monkeypatch.setattr(api_mod, "get_nfl_state", lambda: {"season": 2026, "week": 8})
    from flask import g
    ctx = _ctx()
    try:
        d1 = app._scorezone_fetch("sleeper", "L1", 2026, week=8, scope="user")
        etag1 = g.get("rz_etag")
        d2 = app._scorezone_fetch("sleeper", "L1", 2026, week=8, scope="user")
        etag2 = g.get("rz_etag")
    finally:
        ctx.pop()
    assert len(calls) == 1
    assert etag1 and etag1 == etag2 and etag1.startswith('"rzu-')
    assert d1["games_today"] is True and d2["games_today"] is True
    # The cached payload itself must not carry the per-request stamp.
    entry = next(iter(app._RZ_USER_CACHE.values()))
    assert "games_today" not in entry[2]


def test_api_user_scope_answers_304_on_unchanged_poll(monkeypatch):
    calls = _stub_fetch_user(monkeypatch)
    monkeypatch.setattr(app, "_games_scheduled_today", lambda s, w: False)
    import dashboard_services.api as api_mod
    monkeypatch.setattr(api_mod, "get_nfl_state", lambda: {"season": 2026, "week": 8})
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["viewer_user_id"] = "u1"
    url = "/api/sleeper/2026/L1/scorezone-data?scope=user"
    r1 = client.get(url)
    assert r1.status_code == 200
    etag = r1.headers.get("ETag")
    assert etag and etag.startswith('"rzu-')
    r2 = client.get(url, headers={"If-None-Match": etag})
    assert r2.status_code == 304
    assert len(calls) == 1, "a 304 poll must not rebuild the aggregate"
    # A non-matching tag still gets the full payload (from cache, no rebuild).
    r3 = client.get(url, headers={"If-None-Match": '"rzu-stale"'})
    assert r3.status_code == 200
    assert r3.get_json()["scope"] == "user"
    assert len(calls) == 1


# ── League slices share the collect cache ───────────────────────────────────

def _slice_payload():
    return {
        "rosters": [
            {"roster_id": 1, "owner_id": "u1"},
            {"roster_id": 2, "owner_id": "u2"},
        ],
        "matchups": [
            {"roster_id": 1, "matchup_id": 10, "players": ["p1"], "points": 5},
            {"roster_id": 2, "matchup_id": 10, "players": ["p2"], "points": 3},
        ],
        "users": [
            {"user_id": "u1", "display_name": "Me"},
            {"user_id": "u2", "display_name": "Opp"},
        ],
        "player_info": {"p1": {"name": "p1", "pos": "WR"}, "p2": {"name": "p2", "pos": "RB"}},
        "scoring": {"rec": 1.0},
    }


def test_user_slices_share_one_collect_build(monkeypatch):
    calls = []

    def fake_collect(plat, lid, season, week):
        calls.append((plat, lid, season, week))
        return _slice_payload()

    monkeypatch.setattr(app, "_scorezone_collect", fake_collect)
    lg = {"platform": "sleeper", "league_id": "LA", "name": "League A", "season": 2026}
    s1 = app._scorezone_user_league_slice(0, lg, 2026, 8, None, "u1", {})
    s2 = app._scorezone_user_league_slice(1, lg, 2026, 8, None, "u1", {})
    assert s1 is not None and s2 is not None
    assert s1["viewer_roster_id"] == "0:1"
    assert s2["viewer_roster_id"] == "1:1"
    assert calls == [("sleeper", "LA", 2026, 8)], (
        "the second slice must be served by the shared collect cache, not rebuilt"
    )


# ── Player index flattened once per collect ─────────────────────────────────

class _CountingDict(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return super().items()


def _fake_players():
    return _CountingDict({
        "1": {"full_name": "Patrick Mahomes", "team": "KC", "position": "QB",
              "espn_id": "3139477", "yahoo_id": ""},
        "2": {"full_name": "Travis Kelce", "team": "kc", "position": "TE"},
        "3": {"full_name": "Josh Allen", "team": "BUF", "position": "QB",
              "gsis_id": "00-0034857"},
        "4": {"full_name": "San Francisco 49ers", "team": "49ers D/ST", "position": "DEF"},
        "5": {"team": "BUF", "position": "WR"},  # no full_name -> excluded
        "6": {"full_name": "No Team", "team": None, "position": "WR"},
        "7": {"full_name": "Stefon Diggs", "team": "BUF", "position": "WR",
              "espn_id": None, "tank01_id": "diggs"},
    })


def _legacy_meta(nfl_players, game_teams):
    """The pre-fix per-game full-index scan, kept as the parity oracle."""
    out = {}
    for pid, p in nfl_players.items():
        team = app.canon_team(p.get("team")) or ""
        pos = p.get("position", "")
        full_name = p.get("full_name", "")
        if team in game_teams and full_name:
            if pos != "DEF":
                out[pid] = {
                    "name": full_name, "team": team, "position": pos,
                    **{k: p.get(k) for k in app._RZ_META_ID_FIELDS
                       if p.get(k) not in (None, "")},
                }
    return out


def test_player_meta_matches_legacy_scan_for_any_game_count():
    players = _fake_players()
    rows = app._rz_player_index_rows(players)
    assert players.items_calls == 1, "index must be flattened exactly once per collect"
    for teams in ({"KC"}, {"BUF"}, {"KC", "BUF"}, {"SF"}, set()):
        expected = _legacy_meta(players, teams)
        before = players.items_calls
        got = app._rz_player_meta_for_teams(rows, teams)
        assert players.items_calls == before, "slicing a game must not rescan the raw index"
        assert got == expected, f"meta mismatch for teams={teams}"
        assert list(got) == list(expected), "insertion order must match the legacy scan"
    buf = _legacy_meta(players, {"BUF"})
    assert buf["7"]["tank01_id"] == "diggs"
    assert "espn_id" not in buf["7"], "empty/None id fields stay absent, never guessed"


def test_collect_source_scans_index_once():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    block = src[src.index("def _scorezone_collect("):src.index("def _scorezone_fetch(")]
    assert "nfl_players.items()" not in block, "per-game code must not rescan the raw index"
    assert "_rz_player_index_rows(nfl_players)" in block
    assert "_rz_player_meta_for_teams(_rz_index_rows, game_teams)" in block


# ── Client seeds + box-score session cache (source pins) ────────────────────

def test_page_render_stamps_embedded_etag():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    block = src[src.index("def page_scorezone("):src.index("def _scorezone_user_stream_response")]
    assert "window.__rz_etag__=" in block


def test_client_seeds_etag_and_caches_boxscore():
    src = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
    assert "window.__rz_etag__" in src
    assert "_rzEtagByScope[_state.scope || 'league'] = window.__rz_etag__" in src
    # Box-score sheet: per-session cache for finals only (live games mutate
    # per play and keep fetch-per-open).
    assert "_boxCache" in src
    assert "data.status === 'final'" in src
