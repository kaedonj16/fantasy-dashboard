"""Unit tests for Redis read-through persistence of the league context.

A built league context costs 7-28s of provider fan-out, but historically lived
only in each worker's process-local DASHBOARD_CACHE. These tests cover the
Redis layer (dashboard_services/league_ctx_store.py) and the app.py glue with a
dict-backed fake Redis client -- no real Redis required:

- round-trip: build-shaped ctx -> slice (globals/viewer excluded) -> Redis ->
  load with globals reattached from the local worker
- fail-soft: Redis raising / unreachable / unconfigured degrades to no-op
- invalidation: build-generation, bust-marker, and TTL rules mirror the local
  cache-validity checks, so a stale payload is never resurrected
- guards: per-payload size cap, LEAGUE_CTX_REDIS=0 kill-switch, corrupt blobs
"""
import pickle
import time

import pytest

# app.py imports pandas (and flask) at module load. The fast "lint" CI shard
# has flask but not pandas, so guard on pandas first -- otherwise importing app
# here raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")

import pandas as pd  # noqa: E402

import app  # noqa: E402
from dashboard_services import league_ctx_store as store  # noqa: E402

PLATFORM, SEASON, LEAGUE_ID = "sleeper", 2026, "1234567890"


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.ttls[key] = ttl
        self.data[key] = value
        return True


class RaisingRedis:
    def get(self, key):
        raise ConnectionError("redis down")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis down")


@pytest.fixture()
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("LEAGUE_CTX_REDIS", raising=False)
    monkeypatch.delenv("LEAGUE_CTX_REDIS_MAX_BYTES", raising=False)
    monkeypatch.setattr(store, "_redis_client", lambda: fake)
    store._LOGGED_ONCE.clear()
    return fake


def _built_ctx():
    """A build_league_context-shaped ctx, including the excluded globals."""
    return {
        "platform": PLATFORM,
        "league": {"name": "Test League"},
        "league_id": LEAGUE_ID,
        "resolved_league_id": LEAGUE_ID,
        "season": SEASON,
        "rosters": [{"roster_id": 1, "players": ["p1"]}],
        "users": [{"user_id": "u1", "display_name": "Owner"}],
        "traded": [],
        "current_season": 2026,
        "current_week": 4,
        "current_leg": 4,
        "season_type": "regular",
        "season_complete": False,
        "weeks": 4,
        "df_weekly": pd.DataFrame({"week": [1], "points": [101.5]}),
        "team_stats": pd.DataFrame({"team": ["A"], "Wins": [3]}),
        "roster_map": {"1": "Team A"},
        "injury_df": pd.DataFrame({"player": ["p1"]}),
        "activity_df": pd.DataFrame({"kind": ["trade"], "week": [3]}),
        "standings_map": {"1": {"wins": 3}},
        "picks_by_roster": {},
        "draft_capital_available": False,
        "team_game_lookup": {},
        "scoring_settings": {"rec": 1.0},
        "raw_scoring_settings": {"rec": 1.0},
        "roster_positions": ["QB", "RB"],
        "league_settings": {"playoff_week_start": 15},
        "total_rosters": 12,
        "mode": "in_season",
        "offseason_mode": False,
        "drafts": [],
        "latest_draft": None,
        # League-independent globals: excluded from the Redis slice and
        # reattached from the loading worker instead.
        "players": {"p1": {"full_name": "Player One"}},
        "players_map": {"p1": {"name": "Player One"}},
        "players_index": {"p1": {}},
        "teams_index": {"BUF": {}},
        "model_value_table": [{"player_id": "p1", "value": 42}],
        "rookie_rankings": [{"player_id": "r1"}],
        # Per-request / transient: never persisted.
        "viewer": {"viewer_username": "someone-else"},
        "_cache_synced_at": "2026-01-01T00:00:00+00:00",
        "_cache_stale": False,
    }


def _patch_load_side(monkeypatch, generation=1, bust_mtime=0.0):
    """Stub the app-level dependencies of _league_ctx_redis_load."""
    from dashboard_services import league_singleflight

    monkeypatch.setattr(
        league_singleflight, "read_generation",
        lambda *a: {"generation": generation, "completed_at": 0.0},
    )
    monkeypatch.setattr(app, "_league_bust_mtime", lambda *a: bust_mtime)
    monkeypatch.setattr(app, "get_players_global", lambda: {"p1": {"full_name": "Local Player"}})
    monkeypatch.setattr(app, "get_players_map", lambda players: {"p1": {"name": "Local Player"}})
    monkeypatch.setattr(app, "load_players_index", lambda: {"p1": {"local": True}})
    monkeypatch.setattr(app, "load_teams_index", lambda: {"BUF": {"local": True}})
    monkeypatch.setattr(app, "get_model_value_table_cached", lambda: [{"player_id": "p9", "value": 7}])
    monkeypatch.setattr(app, "_load_rookie_rankings_for_ctx", lambda: [{"player_id": "r9"}])
    monkeypatch.setattr(
        app, "get_viewer_session_for_league",
        lambda users, rosters, *a: {"viewer_username": "me"},
    )


@pytest.fixture()
def cache_cleanup():
    key = app._cache_key(PLATFORM, SEASON, LEAGUE_ID)
    app.DASHBOARD_CACHE.pop(key, None)
    yield key
    app.DASHBOARD_CACHE.pop(key, None)


def test_round_trip_reattaches_globals(fake_redis, monkeypatch, cache_cleanup):
    built_at = time.time()
    ctx = _built_ctx()
    app._league_ctx_redis_store(PLATFORM, SEASON, LEAGUE_ID, ctx, built_at, generation=3)

    key = store.ctx_key(PLATFORM, SEASON, LEAGUE_ID)
    assert key in fake_redis.data
    assert fake_redis.ttls[key] == app.CACHE_TTL
    blob = fake_redis.data[key]
    # The whole point: the ~38MB players payload never leaves the worker.
    assert b"Player One" not in blob

    _patch_load_side(monkeypatch, generation=3)
    loaded = app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON)
    assert loaded is not None
    assert loaded["league"] == {"name": "Test League"}
    assert loaded["rosters"] == ctx["rosters"]
    pd.testing.assert_frame_equal(loaded["df_weekly"], ctx["df_weekly"])
    pd.testing.assert_frame_equal(loaded["team_stats"], ctx["team_stats"])
    # Globals come from THIS worker's caches, not from the stored payload.
    assert loaded["players"] == {"p1": {"full_name": "Local Player"}}
    assert loaded["players_map"] == {"p1": {"name": "Local Player"}}
    assert loaded["players_index"] == {"p1": {"local": True}}
    assert loaded["teams_index"] == {"BUF": {"local": True}}
    assert loaded["model_value_table"] == [{"player_id": "p9", "value": 7}]
    assert loaded["rookie_rankings"] == [{"player_id": "r9"}]
    # Viewer is recomputed for the current request, never persisted.
    assert loaded["viewer"] == {"viewer_username": "me"}
    assert loaded["_cache_stale"] is False
    # The local cache is populated with the original build timestamp.
    entry = app.DASHBOARD_CACHE[cache_cleanup]
    assert entry["ctx"] is loaded
    assert entry["ts"] == pytest.approx(built_at)


def test_slice_excludes_globals_and_transients():
    slice_ = app._league_ctx_redis_slice(_built_ctx())
    for key in app._LEAGUE_CTX_REDIS_GLOBAL_KEYS:
        assert key not in slice_
    assert "viewer" not in slice_
    assert "_cache_synced_at" not in slice_
    assert "_cache_stale" not in slice_
    assert slice_["league"] == {"name": "Test League"}
    assert "df_weekly" in slice_


def test_load_miss_returns_none(fake_redis, monkeypatch):
    _patch_load_side(monkeypatch)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None


def test_generation_invalidation(fake_redis, monkeypatch, cache_cleanup):
    """A payload stored before the latest successful build is not served."""
    built_at = time.time()
    blob = store.dump_envelope(2, built_at, {"league": {"name": "Old"}})
    assert store.save(PLATFORM, SEASON, LEAGUE_ID, blob, ttl_seconds=600)
    _patch_load_side(monkeypatch, generation=3)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None
    # At the current generation the same payload is fine.
    _patch_load_side(monkeypatch, generation=2)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is not None


def test_bust_marker_invalidation(fake_redis, monkeypatch):
    """A Refresh / roster-change bust newer than the build invalidates it."""
    built_at = time.time()
    blob = store.dump_envelope(1, built_at, {"league": {}})
    assert store.save(PLATFORM, SEASON, LEAGUE_ID, blob, ttl_seconds=600)
    _patch_load_side(monkeypatch, generation=1, bust_mtime=built_at + 5)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None


def test_ttl_expiry(fake_redis, monkeypatch):
    built_at = time.time() - app.CACHE_TTL - 1
    blob = store.dump_envelope(1, built_at, {"league": {}})
    assert store.save(PLATFORM, SEASON, LEAGUE_ID, blob, ttl_seconds=600)
    _patch_load_side(monkeypatch, generation=1)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None


def test_fail_soft_when_redis_raises(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("LEAGUE_CTX_REDIS", raising=False)
    monkeypatch.setattr(store, "_redis_client", lambda: RaisingRedis())
    store._LOGGED_ONCE.clear()
    blob = store.dump_envelope(1, time.time(), {"league": {}})
    assert store.save(PLATFORM, SEASON, LEAGUE_ID, blob, ttl_seconds=600) is False
    assert store.load(PLATFORM, SEASON, LEAGUE_ID) is None
    # The app-level helpers swallow everything too.
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None
    app._league_ctx_redis_store(PLATFORM, SEASON, LEAGUE_ID, _built_ctx(), time.time(), 1)


def test_fail_soft_when_redis_unconfigured(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("LEAGUE_CTX_REDIS", raising=False)
    assert store.enabled() is False
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None
    app._league_ctx_redis_store(PLATFORM, SEASON, LEAGUE_ID, _built_ctx(), time.time(), 1)


def test_kill_switch(fake_redis, monkeypatch):
    monkeypatch.setenv("LEAGUE_CTX_REDIS", "0")
    assert store.enabled() is False
    blob = store.dump_envelope(1, time.time(), {"league": {}})
    assert store.save(PLATFORM, SEASON, LEAGUE_ID, blob, ttl_seconds=600)
    _patch_load_side(monkeypatch, generation=1)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None
    app._league_ctx_redis_store(PLATFORM, SEASON, LEAGUE_ID, _built_ctx(), time.time(), 1)
    key = store.ctx_key(PLATFORM, SEASON, LEAGUE_ID)
    envelope = store.parse_envelope(fake_redis.data[key])
    # Still the manual save (empty league): the kill-switched store wrote nothing.
    assert envelope["ctx"]["league"] == {}


def test_size_cap_skips_oversized_payload(fake_redis, monkeypatch):
    monkeypatch.setenv("LEAGUE_CTX_REDIS_MAX_BYTES", "1024")
    big_slice = {"league": {}, "blob": "x" * 5000}
    assert store.dump_envelope(1, time.time(), big_slice) is None
    app._league_ctx_redis_store(PLATFORM, SEASON, LEAGUE_ID, {"league": {}, "blob": "x" * 5000}, time.time(), 1)
    assert fake_redis.data == {}


def test_unpicklable_slice_skips_store(fake_redis):
    assert store.dump_envelope(1, time.time(), {"fn": lambda x: x}) is None


def test_corrupt_payload_is_a_miss(fake_redis, monkeypatch):
    key = store.ctx_key(PLATFORM, SEASON, LEAGUE_ID)
    fake_redis.data[key] = b"this is not a pickle"
    assert store.load(PLATFORM, SEASON, LEAGUE_ID) is None
    fake_redis.data[key] = pickle.dumps({"v": 999, "ctx": {}})
    assert store.load(PLATFORM, SEASON, LEAGUE_ID) is None
    fake_redis.data[key] = pickle.dumps(["not", "an", "envelope"])
    assert store.load(PLATFORM, SEASON, LEAGUE_ID) is None
    _patch_load_side(monkeypatch)
    assert app._league_ctx_redis_load(PLATFORM, LEAGUE_ID, SEASON) is None


def test_key_is_league_scoped_and_stable():
    assert store.ctx_key("sleeper", 2026, "a") == store.ctx_key("SLEEPER", 2026, "a")
    assert store.ctx_key("sleeper", 2026, "a") != store.ctx_key("sleeper", 2026, "b")
    assert store.ctx_key("sleeper", 2026, "a") != store.ctx_key("sleeper", 2025, "a")
