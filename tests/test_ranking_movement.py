"""Unit tests for generic team-ranking movement (dashboard_services.ranking_movement).

The DB is faked so the rank-delta logic can be exercised without a live
connection (the pure test env has no psycopg).
"""
from dashboard_services import ranking_movement as R


class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, handler, writes):
        self._handler = handler
        self._writes = writes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "INSERT INTO ranking_movement" in sql:
            self._writes.append(params)
        return _FakeCur(self._handler(sql, params))


def _install(monkeypatch, handler, writes):
    monkeypatch.setattr(R, "_TABLE_READY", True)       # skip CREATE TABLE
    monkeypatch.setattr(R, "_LAST_WRITE", {})          # reset write throttle
    monkeypatch.setattr(R, "get_conn", lambda: _FakeConn(handler, writes))


def test_no_movement_without_prior_snapshot(monkeypatch):
    writes = []

    def handler(sql, params):
        if "MAX(snap_date)" in sql:
            return [{"d": None}]
        return []
    _install(monkeypatch, handler, writes)

    mv = R.record_daily_and_movement("L", 2026, "value", [1, 2, 3])
    assert mv == {}
    assert len(writes) == 3   # today's three ranks still written


def test_movement_is_rank_delta(monkeypatch):
    writes = []

    # Yesterday: order was [1, 2, 3] -> ranks 1,2,3.
    def handler(sql, params):
        if "MAX(snap_date)" in sql:
            return [{"d": "2026-08-06"}]
        if "SELECT roster_id, rank" in sql:
            return [{"roster_id": 1, "rank": 1},
                    {"roster_id": 2, "rank": 2},
                    {"roster_id": 3, "rank": 3}]
        return []
    _install(monkeypatch, handler, writes)

    # Today team 3 jumps to #1 (e.g. after a trade); 1 and 2 slide down one.
    mv = R.record_daily_and_movement("L", 2026, "value", [3, 1, 2])
    assert mv == {"3": 2, "1": -1, "2": -1}   # +2 = climbed two spots


def test_write_throttled_within_window(monkeypatch):
    writes = []

    def handler(sql, params):
        if "MAX(snap_date)" in sql:
            return [{"d": None}]
        return []
    _install(monkeypatch, handler, writes)

    R.record_daily_and_movement("L", 2026, "power", [1, 2])
    n_after_first = len(writes)
    R.record_daily_and_movement("L", 2026, "power", [1, 2])   # immediately again
    assert len(writes) == n_after_first   # second call throttled — no new writes


def test_kinds_are_independent(monkeypatch):
    # The throttle is per (league, season, kind), so a different kind still writes.
    writes = []

    def handler(sql, params):
        if "MAX(snap_date)" in sql:
            return [{"d": None}]
        return []
    _install(monkeypatch, handler, writes)

    R.record_daily_and_movement("L", 2026, "value", [1, 2])
    first = len(writes)
    R.record_daily_and_movement("L", 2026, "power", [1, 2])
    assert len(writes) > first   # different kind is not throttled by "value"
