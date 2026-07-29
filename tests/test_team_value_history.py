"""Unit tests for team value history (dashboard_services.team_value_history).

The DB is faked so the write + series/delta logic runs without a connection.
"""
from dashboard_services import team_value_history as T


class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, handler, writes):
        self._handler = handler
        self._writes = writes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if "INSERT INTO team_value_history" in sql:
            self._writes.append(params)
        return _FakeCur(self._handler(sql, params))


def _install(monkeypatch, handler, writes):
    monkeypatch.setattr(T, "_TABLE_READY", True)
    monkeypatch.setattr(T, "get_conn", lambda: _FakeConn(handler, writes))


def test_skips_before_week_one(monkeypatch):
    writes = []
    _install(monkeypatch, lambda sql, p: [], writes)
    out = T.record_and_series("L", 2026, 0, 1, 8000.0)
    assert out == {"series": [], "current": None, "delta": None}
    assert writes == []


def test_records_and_returns_series_with_delta(monkeypatch):
    writes = []

    def handler(sql, params):
        if "SELECT week, total_value" in sql:
            return [{"week": 5, "total_value": 7800.0},
                    {"week": 6, "total_value": 8240.0}]
        return []
    _install(monkeypatch, handler, writes)

    out = T.record_and_series("L", 2026, 6, 3, 8240.0, write=True)
    assert out["series"] == [{"week": 5, "value": 7800.0}, {"week": 6, "value": 8240.0}]
    assert out["current"] == 8240.0
    assert out["delta"] == 440.0
    assert len(writes) == 1   # snapshotted this week


def test_read_only_when_write_false(monkeypatch):
    writes = []

    def handler(sql, params):
        if "SELECT week, total_value" in sql:
            return [{"week": 6, "total_value": 8240.0}]
        return []
    _install(monkeypatch, handler, writes)

    out = T.record_and_series("L", 2026, 6, 3, 8240.0, write=False)
    assert writes == []             # cache hit → no snapshot
    assert out["current"] == 8240.0
    assert out["delta"] is None     # only one point
