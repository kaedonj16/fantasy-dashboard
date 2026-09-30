"""Server-side ScoreZone play store: upsert idempotency, watermark TD reads, prune."""
from __future__ import annotations

import json

import pytest

import utils.scorezone_store as rs


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
        self.plays = {}  # (season, game_id, play_id, pid) -> row dict
        self.state = {}
        # Schema state for the pid-key migration: a fresh DB reports the new
        # key (the CREATE TABLE below is the new DDL). The migration test
        # flips these to stage a pre-migration table.
        self.table_exists = False
        self.pk_has_pid = True
        # A 041-shape table also lacks the week column (042 adds it); the
        # no-week migration regression test flips this off.
        self.table_has_week = True
        self._mig_plays = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def rollback(self):
        pass

    def cursor(self):
        return _FakeStoreCursor(self)

    def _do_executemany(self, sql, rows):
        up = " ".join(sql.split()).upper()
        assert up.startswith("INSERT INTO REDZONE_PLAYS"), sql[:60]
        for season, game_id, play_id, pid, seq, is_td, week, payload_json in rows:
            key = (int(season), str(game_id), str(play_id), str(pid))
            existing = self.plays.get(key)
            if existing is None:
                self.plays[key] = {
                    "season": int(season),
                    "game_id": str(game_id),
                    "play_id": str(play_id),
                    "pid": str(pid),
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
        if "PG_CONSTRAINT" in up:
            # Primary-key introspection for the pid-key migration.
            if not self.table_exists:
                return _FakeCursor([])
            cols = "season,game_id,play_id,pid" if self.pk_has_pid else "season,game_id,play_id"
            return _FakeCursor([{"pk_cols": cols}])
        if up.startswith("ALTER TABLE REDZONE_PLAYS ADD COLUMN"):
            # In-place column heal (week from 042, pid from 043): existing
            # rows gain the column with its default (NULL week).
            if "WEEK" in up:
                self.table_has_week = True
            return _FakeCursor([])
        if up.startswith("CREATE TABLE REDZONE_PLAYS_PIDMIG"):
            self._mig_plays = {}
            return _FakeCursor([])
        if up.startswith("INSERT INTO REDZONE_PLAYS_PIDMIG"):
            # The rebuild SELECTs week from the source table; on a
            # week-less (041-shape) source Postgres raises UndefinedColumn
            # (prod 2026-09-30). Mirror that so the regression test fails
            # if the migration ever assumes the column again.
            if not self.table_has_week:
                raise Exception('column "week" does not exist')
            # Rebuild preserving rows; pid comes from each row's payload.
            for row in self.plays.values():
                payload = json.loads(row["payload"])
                pid = str(payload.get("pid") or "")
                new_row = dict(row)
                new_row.setdefault("week", None)
                new_row["pid"] = pid
                key = (row["season"], row["game_id"], row["play_id"], pid)
                self._mig_plays[key] = new_row
            return _FakeCursor([])
        if up.startswith("DROP TABLE REDZONE_PLAYS"):
            return _FakeCursor([])
        if up.startswith("ALTER TABLE REDZONE_PLAYS_PIDMIG RENAME"):
            self.plays = self._mig_plays or {}
            self._mig_plays = None
            self.pk_has_pid = True
            self.table_exists = True
            return _FakeCursor([])
        if up.startswith("CREATE TABLE") or up.startswith("CREATE INDEX"):
            self.table_exists = True
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
            season = int(params[0])
            if "OBSERVED_AT >= NOW()" in up:
                # Legacy recency-only shape: (season, days, pid_list).
                days = int(params[1])
                pid_set = {str(p) for p in params[2]}
                week = None
                cutoff = self.now[0] - days * 86400
            else:
                # Week-scoped shape: (season, pid_list, week). The week stamp
                # is the only time filter -- no recency cutoff.
                pid_set = {str(p) for p in params[1]}
                week = int(params[2])
                cutoff = None
            rows = []
            for r in self.plays.values():
                if r["season"] != season:
                    continue
                if cutoff is not None and r["observed_at"] < cutoff:
                    continue
                payload = json.loads(r["payload"])
                if str(payload.get("pid") or "") not in pid_set:
                    continue
                # Mirrors the week = %s clause: week-less rows never match a
                # week-scoped read.
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
    first = conn.plays[(2026, "g", "p1", "")]["observed_at"]
    now[0] += 3600  # an hour of polls later
    rs.upsert_plays(2026, "g", [_play("p1", 1)])
    assert conn.plays[(2026, "g", "p1", "")]["observed_at"] == first


def test_upsert_revision_bumps_observed_at(store_db):
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run")])
    first = conn.plays[(2026, "g", "p1", "")]["observed_at"]
    now[0] += 3600
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run, scoring changed")])
    assert conn.plays[(2026, "g", "p1", "")]["observed_at"] > first


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
    assert set(conn.plays) == {(2026, "g", "new", "")}


def test_get_plays_unknown_game_absent(store_db):
    assert rs.get_plays(2026, ["nope"]) == {}


def test_discover_live_games_covers_previous_week(monkeypatch):
    """Discovery fetches the current AND previous week scoreboards explicitly.

    Regression: relying on the unparameterized scoreboard alone went blind to
    last week's finals the moment ESPN flipped it forward, so collection gaps
    from the previous week could never be backfilled. Each game is tagged
    with the week it was discovered under.
    """
    import sys
    import types

    requested_weeks = []

    class _Resp:
        status_code = 200

        def __init__(self, week):
            self._week = week

        def json(self):
            return {"sbData": {"events": []}}

    class _Requests:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            requested_weeks.append(params.get("week"))
            return _Resp(params.get("week"))

    monkeypatch.setitem(sys.modules, "requests", _Requests)

    fake_alt = types.ModuleType("utils.scorezone_alt_pbp")
    fake_alt._ESPN_SCOREBOARD = "https://example.invalid/scoreboard"
    fake_alt._UA = "test"

    def fake_lookup(payload):
        # One live game per requested scoreboard week.
        wk = requested_weeks[-1]
        gid = f"2026092{wk}_AAA@BBB"
        return {f"t{wk}": {"gameID": gid, "gameStatusCode": "1"}}

    fake_alt.extract_espn_scoreboard_lookup = fake_lookup
    monkeypatch.setitem(sys.modules, "utils.scorezone_alt_pbp", fake_alt)

    games = rs.discover_live_games(current_week=4)
    assert requested_weeks == ["4", "3"]
    by_gid = {g["game_id"]: g for g in games}
    assert by_gid["20260924_AAA@BBB"]["week"] == 4
    assert by_gid["20260923_AAA@BBB"]["week"] == 3
    assert all(g["live"] for g in games)


def test_discover_live_games_skips_previous_week_one(monkeypatch):
    """Week 1 has no previous week to fetch."""
    import sys
    import types

    requested_weeks = []

    class _Resp:
        status_code = 200

        def json(self):
            return {}

    class _Requests:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            requested_weeks.append(params.get("week"))
            return _Resp()

    monkeypatch.setitem(sys.modules, "requests", _Requests)
    fake_alt = types.ModuleType("utils.scorezone_alt_pbp")
    fake_alt._ESPN_SCOREBOARD = "https://example.invalid/scoreboard"
    fake_alt._UA = "test"
    fake_alt.extract_espn_scoreboard_lookup = lambda payload: {}
    monkeypatch.setitem(sys.modules, "utils.scorezone_alt_pbp", fake_alt)

    assert rs.discover_live_games(current_week=1) == []
    assert requested_weeks == ["1"]


def test_poll_once_backfills_recent_unseen_finals(monkeypatch):
    import sys
    import types

    # Tank01 game ids carry the date: backfill window is 7 days.
    monkeypatch.setattr(rs, "discover_live_games", lambda current_week=None: [
        {"game_id": "20260928_KC@BUF", "live": True, "final": False, "week": 4},
        {"game_id": "20260927_NE@MIA", "live": False, "final": True, "week": 3},
        {"game_id": "20260927_SF@ARI", "live": False, "final": True, "week": 3},
        {"game_id": "20260920_DAL@PHI", "live": False, "final": True, "week": 3},
    ])
    monkeypatch.setattr(rs, "get_plays", lambda season, gids: {"20260927_NE@MIA": []})

    fetched = []

    def fake_pbp(gid, **kw):
        fetched.append((gid, kw.get("live"), kw.get("final")))
        return [{"play_id": "p1", "seq": 1, "is_td": False}]

    def fake_parse(gid):
        date_part = gid.split("_", 1)[0] if "_" in gid else ""
        return (date_part, "KC", "BUF")

    fake_alt = types.ModuleType("utils.scorezone_alt_pbp")
    fake_alt.fetch_alt_pbp_plays = fake_pbp
    fake_alt.parse_tank_game_id = fake_parse
    monkeypatch.setitem(sys.modules, "utils.scorezone_alt_pbp", fake_alt)

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
    # the collector stamps each game's own week (falling back to nfl state)
    assert sorted((gid, w) for gid, w in upserted) == [
        ("20260927_NE@MIA", 3),
        ("20260927_SF@ARI", 3),
        ("20260928_KC@BUF", 4),
    ]
    assert [f[0] for f in fetched] == ["20260928_KC@BUF", "20260927_NE@MIA", "20260927_SF@ARI"]


def test_ensure_table_runs_once_per_process():
    import utils.scorezone_store as rs
    from types import SimpleNamespace

    rs._ENSURED_TABLES.clear()
    calls = []

    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    class _Conn:
        info = None

        def execute(self, sql, params=None):
            calls.append(" ".join(sql.split()).upper())
            if "PG_CONSTRAINT" in calls[-1]:
                # Already on the per-player key: no migration rebuild.
                return _Cursor([{"pk_cols": "season,game_id,play_id,pid"}])
            return _Cursor([])

    rs._ensure_table(_Conn())
    first_run = list(calls)
    # PK introspection first, then the table DDL, then the in-place
    # column heals, then the indexes.
    assert "PG_CONSTRAINT" in first_run[0]
    assert first_run[1].startswith("CREATE TABLE")
    assert first_run[2].startswith("ALTER TABLE REDZONE_PLAYS ADD COLUMN")
    assert "WEEK" in first_run[2]
    assert first_run[3].startswith("ALTER TABLE REDZONE_PLAYS ADD COLUMN")
    assert "PID" in first_run[3]
    assert len(first_run) == 4 + len(rs._INDEX_DDL)

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
    assert conn.plays[(2026, "20260927_KC@BUF", "p1", "")]["week"] == 3


def test_upsert_weekless_reupsert_keeps_week(store_db):
    """Mirrors COALESCE(EXCLUDED.week, redzone_plays.week): a re-upsert that
    carries no week must not wipe the week stamped by an earlier upsert."""
    conn, now = store_db
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run")], week=3)
    now[0] += 60
    rs.upsert_plays(2026, "g", [_play("p1", 1, text="run, revised")])
    assert conn.plays[(2026, "g", "p1", "")]["week"] == 3


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


# ── Per-player key regressions ──────────────────────────────────────────


def _play_pid(play_id, seq, pid, is_td=False, stat_line=None, text="play"):
    p = _play(play_id, seq, is_td=is_td, text=text)
    p["pid"] = pid
    if stat_line is not None:
        p["stat_line"] = stat_line
    return p


def test_upsert_keeps_every_players_row_for_one_play(store_db):
    """Regression: extractors emit one row per involved player under the
    same play_id, but the old (season, game_id, play_id) table key kept
    only one of them -- a TD pass lost either the QB's or the WR's row."""
    conn, _now = store_db
    qb = _play_pid("p1", 1, "111", is_td=True, stat_line={"pass_td": 1, "pass_yds": 25})
    wr = _play_pid("p1", 1, "222", is_td=True, stat_line={"rec_td": 1, "rec_yds": 25})
    assert rs.upsert_plays(2026, "g", [qb, wr], week=4) == 2
    assert (2026, "g", "p1", "111") in conn.plays
    assert (2026, "g", "p1", "222") in conn.plays
    got = rs.get_plays(2026, ["g"])
    assert {p["pid"] for p in got["g"]} == {"111", "222"}
    # Each player's row is independently retrievable by pid.
    assert [p["pid"] for p in rs.get_plays_for_pids(2026, ["111"], week=4)] == ["111"]
    assert [p["pid"] for p in rs.get_plays_for_pids(2026, ["222"], week=4)] == ["222"]


def test_qb_interception_row_retrievable_by_qb_pid(store_db):
    """Regression: an interception produces a QB row and a defender row
    under one play_id. The QB's row must survive the upsert and be found
    by his pid (the moments endpoint classifies it as his turnover)."""
    _conn, _now = store_db
    qb = _play_pid("int1", 7, "111", stat_line={"int": 1}, text="pass intercepted")
    defender = _play_pid("int1", 7, "999", stat_line={"def_int": 1}, text="pass intercepted")
    rs.upsert_plays(2026, "g", [qb, defender], week=4)
    got = rs.get_plays_for_pids(2026, ["111"], week=4)
    assert len(got) == 1
    assert got[0]["play_id"] == "int1"
    assert got[0]["stat_line"]["int"] == 1


def test_td_scan_returns_both_rows_of_a_td_pass(store_db):
    """Both scoring rows must reach the TD notifier: it groups rows by
    canonical play and claims once per (play, owner), picking each owner's
    best contribution -- so per-player rows do not double-fire pushes, but
    dropping one upstream would lose that owner's push."""
    _conn, _now = store_db
    qb = _play_pid("p1", 1, "111", is_td=True, stat_line={"pass_td": 1})
    wr = _play_pid("p1", 1, "222", is_td=True, stat_line={"rec_td": 1})
    rs.upsert_plays(2026, "g", [qb, wr], week=4)
    tds = rs.get_td_plays_since(2026, 0)
    assert {p["pid"] for _g, p, _ts in tds} == {"111", "222"}


def test_week_scoped_read_survives_past_recency_window(store_db):
    """Regression: a week-scoped read made more than ``days`` after the
    plays were observed must still return them -- the old 5-day
    observed_at window silently dropped Sunday plays by the following
    weekend. The week-less legacy read keeps its recency bound."""
    _conn, now = store_db
    rs.upsert_plays(2026, "g", [_play_with_pid("sun", 1, "123")], week=3)
    now[0] += 8 * 86400  # the following Monday
    got = rs.get_plays_for_pids(2026, ["123"], week=3)
    assert [p["play_id"] for p in got] == ["sun"]
    assert rs.get_plays_for_pids(2026, ["123"]) == []


def test_migrate_pid_key_rebuilds_preserving_rows():
    """A pre-pid table (old 3-column key) is rebuilt by _ensure_table:
    rows survive, keyed by the pid from their payloads, and ensuring
    again is a no-op (idempotent)."""
    conn = _FakeStoreConn([1_700_000_000.0])
    conn.table_exists = True
    conn.pk_has_pid = False
    payload = json.dumps({"play_id": "p1", "pid": "111", "seq": 1})
    conn.plays[(2026, "g", "p1")] = {
        "season": 2026, "game_id": "g", "play_id": "p1", "seq": 1,
        "is_td": True, "week": 3, "payload": payload,
        "observed_at": 1_700_000_000.0,
    }
    rs._ENSURED_TABLES.discard("default")
    rs._ensure_table(conn)
    assert conn.pk_has_pid is True
    assert set(conn.plays) == {(2026, "g", "p1", "111")}
    assert conn.plays[(2026, "g", "p1", "111")]["payload"] == payload
    before = dict(conn.plays)
    rs._ENSURED_TABLES.discard("default")
    rs._ensure_table(conn)
    assert conn.plays == before


def test_migrate_pid_key_handles_table_without_week_column():
    """Regression (prod 2026-09-30): a 041-shape table has no week column
    -- 042 adds it, but migrations run in post-deploy after the web
    process starts serving, so the in-code pid-key rebuild fired first
    and its INSERT ... SELECT referenced week, failing with
    UndefinedColumn (DETAIL: week exists only on redzone_plays_pidmig).
    The failed statement also aborted the transaction, taking the rest
    of _ensure_table down with it on every store call. The migration
    must add the column itself; rows survive with week NULL."""
    conn = _FakeStoreConn([1_700_000_000.0])
    conn.table_exists = True
    conn.pk_has_pid = False
    conn.table_has_week = False
    payload = json.dumps({"play_id": "p1", "pid": "111", "seq": 1})
    conn.plays[(2026, "g", "p1")] = {
        "season": 2026, "game_id": "g", "play_id": "p1", "seq": 1,
        "is_td": True, "payload": payload,
        "observed_at": 1_700_000_000.0,
    }
    rs._ENSURED_TABLES.discard("default")
    rs._ensure_table(conn)
    assert conn.table_has_week is True
    assert conn.pk_has_pid is True
    assert set(conn.plays) == {(2026, "g", "p1", "111")}
    assert conn.plays[(2026, "g", "p1", "111")]["payload"] == payload
    assert conn.plays[(2026, "g", "p1", "111")]["week"] is None


# ── Poller loudness ─────────────────────────────────────────────────────


def test_in_nfl_game_window():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    assert rs._in_nfl_game_window(datetime(2026, 9, 27, 14, 0, tzinfo=et))      # Sunday 2pm
    assert not rs._in_nfl_game_window(datetime(2026, 9, 27, 10, 0, tzinfo=et))  # Sunday morning
    assert not rs._in_nfl_game_window(datetime(2026, 9, 29, 14, 0, tzinfo=et))  # Tuesday
    assert rs._in_nfl_game_window(datetime(2026, 10, 1, 21, 0, tzinfo=et))      # Thursday night
    assert rs._in_nfl_game_window(datetime(2026, 9, 28, 21, 0, tzinfo=et))      # Monday night
    assert not rs._in_nfl_game_window(datetime(2026, 9, 26, 15, 0, tzinfo=et))  # September Saturday
    assert rs._in_nfl_game_window(datetime(2027, 1, 9, 15, 0, tzinfo=et))       # January Saturday


def test_poll_once_warns_when_no_games_during_game_window(monkeypatch, caplog):
    """A poll that discovers zero games while NFL games are on means
    discovery is broken; it must be loud, not a silent empty return."""
    import logging
    import sys
    import types

    monkeypatch.setattr(rs, "discover_live_games", lambda current_week=None: [])
    monkeypatch.setattr(rs, "_in_nfl_game_window", lambda now=None: True)
    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {"season": 2026, "week": 4, "season_type": "reg"}
    fake_api.get_nfl_players = lambda: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)

    with caplog.at_level(logging.WARNING, logger="utils.scorezone_store"):
        stats = rs.poll_once()
    assert stats == {"games": 0, "plays": 0}
    assert "0 live/final games" in caplog.text


def test_poll_once_quiet_when_no_games_outside_window(monkeypatch, caplog):
    """Zero games on a Tuesday (or in preseason) is normal -- no warning."""
    import logging
    import sys
    import types

    monkeypatch.setattr(rs, "discover_live_games", lambda current_week=None: [])
    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {"season": 2026, "week": 4, "season_type": "pre"}
    fake_api.get_nfl_players = lambda: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)

    with caplog.at_level(logging.WARNING, logger="utils.scorezone_store"):
        stats = rs.poll_once()
    assert stats == {"games": 0, "plays": 0}
    assert "0 live/final games" not in caplog.text
