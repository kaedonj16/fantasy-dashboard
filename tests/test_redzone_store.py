"""Server-side RedZone play store: upsert idempotency, watermark TD reads, prune."""
from __future__ import annotations

import json

import pytest

import utils.redzone_store as rs


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeStoreCursor:
    """Stand-in for a psycopg3 cursor.

    Real psycopg3 Connections have no ``executemany`` -- it lives on the
    cursor. The fake mirrors that so a regression to ``conn.executemany``
    fails here exactly as it does in production.
    """

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def executemany(self, sql, rows):
        return self._conn._do_executemany(sql, rows)

    def execute(self, sql, params=None):
        return self._conn.execute(sql, params)

    def fetchall(self):
        raise AssertionError("cursor.fetchall without execute")

    def fetchone(self):
        raise AssertionError("cursor.fetchone without execute")


class _FakeStoreConn:
    """In-memory stand-in for the redzone_plays / app_state tables.

    now is a one-element list holding the fake clock (epoch seconds); tests
    advance it to prove observed_at only moves on real play changes.
    """

    def __init__(self, now):
        self.now = now
        self.plays = {}  # (season, game_id, play_id) -> row dict
        self.state = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def cursor(self):
        return _FakeStoreCursor(self)

    def _do_executemany(self, sql, rows):
        up = " ".join(sql.split()).upper()
        assert up.startswith("INSERT INTO REDZONE_PLAYS"), sql[:60]
        for season, game_id, play_id, seq, is_td, week, payload_json in rows:
            key = (int(season), str(game_id), str(play_id))
            existing = self.plays.get(key)
            if existing is None:
                self.plays[key] = {
                    "season": int(season),
                    "game_id": str(game_id),
                    "play_id": str(play_id),
                    "seq": int(seq),
                    "is_td": bool(is_td),
                    "week": int(week) if week is not None else None,
                    "payload": payload_json,
                    "observed_at": self.now[0],
                }
            elif existing["payload"] != payload_json:
                # Mirrors the ON CONFLICT ... IS DISTINCT FROM bump.
                # Mirrors COALESCE(EXCLUDED.week, redzone_plays.week).
                existing.update(
                    seq=int(seq),
                    is_td=bool(is_td),
                    week=int(week) if week is not None else existing["week"],
                    payload=payload_json,
                    observed_at=self.now[0],
                )
            # identical re-upsert: observed_at untouched
        return _FakeCursor([])

    def execute(self, sql, params=None):
        up = " ".join(sql.split()).upper()
        params = params or ()
        if up.startswith("CREATE TABLE") or up.startswith("CREATE INDEX"):
            return _FakeCursor([])
        if up.startswith("INSERT INTO REDZONE_PLAYS"):
            self._do_executemany(sql, [params])
            return _FakeCursor([])
        if up.startswith("INSERT INTO APP_STATE"):
            self.state[str(params[0])] = str(params[1])
            return _FakeCursor([])
        if up.startswith("SELECT VALUE FROM APP_STATE"):
            val = self.state.get(str(params[0]))
            return _FakeCursor([{"value": val}] if val is not None else [])
        if "FROM REDZONE_PLAYS" in up and "WHERE SEASON = %S AND GAME_ID = ANY" in up:
            season, gids = int(params[0]), {str(g) for g in params[1]}
            rows = [
                {"game_id": r["game_id"], "payload": json.loads(r["payload"])}
                for r in self.plays.values()
                if r["season"] == season and r["game_id"] in gids
            ]
            rows.sort(key=lambda r: (r["game_id"], 0))
            # order by seq within game
            by_game = {}
            for r in rows:
                by_game.setdefault(r["game_id"], []).append(r)
            ordered = []
            for gid in sorted(by_game):
                key_rows = [x for x in self.plays.values()
                            if x["season"] == season and x["game_id"] == gid]
                key_rows.sort(key=lambda x: x["seq"])
                ordered.extend(
                    {"game_id": gid, "payload": json.loads(x["payload"])}
                    for x in key_rows
                )
            return _FakeCursor(ordered)
        if "FROM REDZONE_PLAYS" in up and "IS_TD" in up and "OBSERVED_AT > TO_TIMESTAMP" in up:
            season, since = int(params[0]), float(params[1])
            rows = [
                {"game_id": r["game_id"], "payload": json.loads(r["payload"]),
                 "ts": r["observed_at"]}
                for r in self.plays.values()
                if r["season"] == season and r["is_td"] and r["observed_at"] > since
            ]
            rows.sort(key=lambda r: r["ts"])
            return _FakeCursor(rows)
        if "FROM REDZONE_PLAYS" in up and "PAYLOAD->>'PID' = ANY" in up:
            season, days = int(params[0]), int(params[1])
            pid_set = {str(p) for p in params[2]}
            week = int(params[3]) if len(params) > 3 else None
            cutoff = self.now[0] - days * 86400
            rows = []
            for r in self.plays.values():
                if r["season"] != season or r["observed_at"] < cutoff:
                    continue
                payload = json.loads(r["payload"])
                if str(payload.get("pid") or "") not in pid_set:
                    continue
                # Mirrors the optional AND week = %s clause: week-less rows
                # never match a week-scoped read.
                if week is not None and r["week"] != week:
                    continue
                rows.append({
                    "game_id": r["game_id"],
                    "payload": r["payload"],
                    "ts": r["observed_at"],
                })
            rows.sort(key=lambda r: r["ts"], reverse=True)
            return _FakeCursor(rows)
        if up.startswith("DELETE FROM REDZONE_PLAYS"):
            cutoff = self.now[0] - int(params[0]) * 86400
            doomed = [k for k, r in self.plays.items() if r["observed_at"] < cutoff]
            for k in doomed:
                del self.plays[k]
            return _FakeCursor([{"1": 1}] * len(doomed))
        raise AssertionError("unexpected store SQL: %s" % sql[:80])


@pytest.fixture()
def store_db(monkeypatch):
    import dashboard_services.db as db

    now = [1_700_000_000.0]
    conn = _FakeStoreConn(now)
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: conn)
    return conn, now


def _play(play_id, seq, is_td=False, text="run"):
    return {
        "play_id": play_id, "seq": seq, "game_id": "20260927_KC@BUF",
        "quarter": "1", "clock": "10:00", "down": "1", "distance": "10",
        "play_text": text, "stat_line": {}, "is_td": is_td,
    }


def test_upsert_and_get_plays_ordered(store_db):
    conn, _now = store_db
    plays = [_play("p3", 3), _play("p1", 1), _play("p2", 2)]
    assert rs.upsert_plays(2026, "20260927_KC@BUF", plays) == 3
    got = rs.get_plays(2026, ["20260927_KC@BUF"])
    assert [p["play_id"] for p in got["20260927_KC@BUF"]] == ["p1", "p2", "p3"]


def test_upsert_idempotent_keeps_observed_at(store_db):
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("p1", 1)])
    first = conn.plays[(2026, "g", "p1")]["observed_at"]
    now[0] += 3600  # an hour of polls later
    rs.upsert_plays(2026, "g", [_play("p1", 1)])
    assert conn.plays[(2026, "g", "p1")]["observed_at"] == first


def test_upsert_revision_bumps_observed_at(store_db):
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run")])
    first = conn.plays[(2026, "g", "p1")]["observed_at"]
    now[0] += 3600
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run, scoring changed")])
    assert conn.plays[(2026, "g", "p1")]["observed_at"] > first


def test_td_since_watermark_semantics(store_db):
    _conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("td1", 1, is_td=True)])
    now[0] += 100
    rs.upsert_plays(2026, "g", [_play("td2", 2, is_td=True)])
    now[0] += 100
    rs.upsert_plays(2026, "g", [_play("run1", 3, is_td=False)])

    tds = rs.get_td_plays_since(2026, 1_700_000_000.0 + 50)
    assert [(g, p["play_id"]) for g, p, _ts in tds] == [("g", "td2")]
    assert tds[0][2] == pytest.approx(1_700_000_100.0)


def test_watermark_roundtrip(store_db):
    assert rs.get_watermark() == 0.0
    rs.set_watermark(1234.5)
    assert rs.get_watermark() == pytest.approx(1234.5)


def test_prune_removes_only_old(store_db):
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("old", 1)])
    now[0] += 8 * 86400
    rs.upsert_plays(2026, "g", [_play("new", 2)])
    assert rs.prune_plays(retention_days=7) == 1
    assert set(conn.plays) == {(2026, "g", "new")}


def test_get_plays_unknown_game_absent(store_db):
    assert rs.get_plays(2026, ["nope"]) == {}


def test_poll_once_backfills_recent_unseen_finals(monkeypatch):
    import sys
    import types

    # Tank01 game ids carry the date: backfill window is 3 days.
    monkeypatch.setattr(rs, "discover_live_games", lambda: [
        {"game_id": "20260928_KC@BUF", "live": True, "final": False},
        {"game_id": "20260927_NE@MIA", "live": False, "final": True},
        {"game_id": "20260927_SF@ARI", "live": False, "final": True},
        {"game_id": "20260920_DAL@PHI", "live": False, "final": True},
    ])
    monkeypatch.setattr(rs, "get_plays", lambda season, gids: {"20260927_NE@MIA": []})

    fetched = []

    def fake_pbp(gid, **kw):
        fetched.append((gid, kw.get("live"), kw.get("final")))
        return [{"play_id": "p1", "seq": 1, "is_td": False}]

    def fake_parse(gid):
        date_part = gid.split("_", 1)[0] if "_" in gid else ""
        return (date_part, "KC", "BUF")

    fake_alt = types.ModuleType("utils.redzone_alt_pbp")
    fake_alt.fetch_alt_pbp_plays = fake_pbp
    fake_alt.parse_tank_game_id = fake_parse
    monkeypatch.setitem(sys.modules, "utils.redzone_alt_pbp", fake_alt)

    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {"season": 2026, "week": 4}
    fake_api.get_nfl_players = lambda: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)

    monkeypatch.setattr(rs, "_build_name_maps", lambda *a: ({}, {}))
    upserted = []
    monkeypatch.setattr(rs, "upsert_plays",
                        lambda season, gid, plays, week=None: upserted.append((gid, week)) or len(plays))

    stats = rs.poll_once()
    # live + seen final re-polled; recent unseen final backfilled; old unseen final skipped
    assert stats["games"] == 3
    # the collector stamps the current week from nfl state
    assert {w for _, w in upserted} == {4}
    assert sorted(gid for gid, _w in upserted) == ["20260927_NE@MIA", "20260927_SF@ARI", "20260928_KC@BUF"]
    assert [f[0] for f in fetched] == ["20260928_KC@BUF", "20260927_NE@MIA", "20260927_SF@ARI"]


def test_ensure_table_runs_once_per_process():
    import utils.redzone_store as rs
    from types import SimpleNamespace

    rs._ENSURED_TABLES.clear()
    calls = []

    class _Conn:
        info = None

        def execute(self, sql, params=None):
            calls.append(" ".join(sql.split())[:12].upper())

    rs._ensure_table(_Conn())
    first_run = list(calls)
    assert len(first_run) == 1 + len(rs._INDEX_DDL)
    assert first_run[0] == "CREATE TABLE"

    # Second call on the same database: no DDL.
    rs._ensure_table(_Conn())
    assert calls == first_run

    # A different database still gets its DDL.
    other = _Conn()
    other.info = SimpleNamespace(dbname="otherdb")
    rs._ensure_table(other)
    assert len(calls) == 2 * len(first_run)


def _play_with_pid(play_id, seq, pid, week_text="run"):
    p = _play(play_id, seq, text=week_text)
    p["pid"] = pid
    return p


def test_upsert_stamps_week(store_db):
    conn, _now = store_db
    rs.upsert_plays(2026, "20260927_KC@BUF", [_play("p1", 1)], week=3)
    assert conn.plays[(2026, "20260927_KC@BUF", "p1")]["week"] == 3


def test_upsert_weekless_reupsert_keeps_week(store_db):
    """Mirrors COALESCE(EXCLUDED.week, redzone_plays.week): a re-upsert that
    carries no week must not wipe the week stamped by an earlier upsert."""
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run")], week=3)
    now[0] += 60
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run, revised")])
    assert conn.plays[(2026, "g", "p1")]["week"] == 3


def test_get_plays_for_pids_filters_by_week(store_db):
    """Regression: a week-4 moments request must not return week-3 plays.

    Same player involved in both weeks; only the requested week's plays
    may come back.
    """
    _conn, _now = store_db
    rs.upsert_plays(2026, "20260927_KC@BUF", [_play_with_pid("w3p", 1, "123")], week=3)
    rs.upsert_plays(2026, "20261004_KC@DEN", [_play_with_pid("w4p", 1, "123")], week=4)

    got4 = rs.get_plays_for_pids(2026, ["123"], week=4)
    assert [p["play_id"] for p in got4] == ["w4p"]

    got3 = rs.get_plays_for_pids(2026, ["123"], week=3)
    assert [p["play_id"] for p in got3] == ["w3p"]

    # A week with no collected plays yet returns nothing (the launcher for a
    # future week stays hidden instead of showing last week's plays).
    assert rs.get_plays_for_pids(2026, ["123"], week=5) == []


def test_get_plays_for_pids_without_week_returns_all(store_db):
    """Legacy recency-only behavior is preserved when no week is given."""
    _conn, _now = store_db
    rs.upsert_plays(2026, "20260927_KC@BUF", [_play_with_pid("w3p", 1, "123")], week=3)
    rs.upsert_plays(2026, "20261004_KC@DEN", [_play_with_pid("w4p", 1, "123")], week=4)
    got = rs.get_plays_for_pids(2026, ["123"])
    assert {p["play_id"] for p in got} == {"w3p", "w4p"}
