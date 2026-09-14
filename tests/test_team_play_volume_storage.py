from datetime import datetime, timedelta, timezone
import json

import dashboard_services.team_play_volume as store


class _Result:
    def __init__(self, rows): self.rows = rows
    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, state): self.state = state
    def cursor(self): return _CM(self)
    def executemany(self, sql, values):
        assert "ON CONFLICT (season, team) DO UPDATE" in sql
        for value in values:
            self.state[(value[0], value[1])] = value
    def execute(self, sql, args):
        rows = []
        for (season, team), value in self.state.items():
            if season == args[0]:
                rows.append({"team": team, "plays_faced_pg": value[2],
                    "plays_faced_l4_pg": value[3], "off_plays_pg": value[4],
                    "games": value[5], "nfl_avg_plays_faced_pg": value[6],
                    "generated_at": datetime.fromisoformat(value[7])})
        return _Result(rows)


class _CM:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self.conn
    def __exit__(self, *args): pass


def _blob(season=2026, value=64.0, generated_at=None):
    return {"season": season, "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
            "nfl_avg_plays_faced_pg": value,
            "teams": {f"T{i:02}": {"plays_faced_pg": value, "plays_faced_l4_pg": value + 1,
                                      "off_plays_pg": value - 1, "games": 8} for i in range(32)}}


def test_persist_32_and_upsert_then_separate_read(monkeypatch):
    # The state represents Postgres: writer and reader have no shared filesystem.
    state = {}
    monkeypatch.setattr(store, "get_conn", lambda: _CM(_Conn(state)))
    assert store.persist_team_play_volume(_blob(value=64)) == 32
    assert len(state) == 32
    assert store.persist_team_play_volume(_blob(value=66)) == 32
    assert len(state) == 32
    store.invalidate_team_play_volume(2026)  # simulate a fresh web process
    loaded = store.load_team_play_volume(2026, allow_local=False)
    assert loaded["play_volume_source"] == "postgres"
    assert loaded["teams"]["T00"]["plays_faced_pg"] == 66


def test_empty_refresh_retains_snapshot(monkeypatch):
    called = False
    def fail_conn():
        nonlocal called
        called = True
        raise AssertionError("empty data must not connect")
    monkeypatch.setattr(store, "get_conn", fail_conn)
    assert store.persist_team_play_volume({"season": 2026, "teams": {}}) == 0
    assert not called


def test_local_json_fallback_and_missing_is_not_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(store, "_read_postgres", lambda season: (_ for _ in ()).throw(RuntimeError("no db")))
    store.invalidate_team_play_volume(2027)
    assert store.load_team_play_volume(2027, allow_local=True) == {}
    (tmp_path / "team_play_volume_s2027.json").write_text(json.dumps(_blob(2027)))
    store.invalidate_team_play_volume(2027)
    loaded = store.load_team_play_volume(2027, allow_local=True)
    assert loaded["play_volume_source"] == "local_json"
    assert loaded["teams"]["T00"]["plays_faced_pg"] == 64.0


def test_cache_keys_are_season_specific(monkeypatch):
    monkeypatch.setattr(store, "_read_postgres", lambda season: {"season": season, "teams": {}})
    store.invalidate_team_play_volume(2025); store.invalidate_team_play_volume(2026)
    assert store.load_team_play_volume(2025, allow_local=False)["season"] == 2025
    assert store.load_team_play_volume(2026, allow_local=False)["season"] == 2026


def test_stale_postgres_snapshot_remains_usable(monkeypatch):
    state = {}
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    monkeypatch.setattr(store, "get_conn", lambda: _CM(_Conn(state)))
    store.persist_team_play_volume(_blob(generated_at=old))
    loaded = store.load_team_play_volume(2026, allow_local=False)
    assert loaded["stale"] is True
    assert len(loaded["teams"]) == 32
