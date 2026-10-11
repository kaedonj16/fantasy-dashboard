"""get_top_movers must tolerate get_latest_snapshot_date returning a str.

Regression test: the real get_latest_snapshot_date() wraps the DATE column
in str(), and _get_top_movers_from_db did `latest_date - timedelta(...)`,
which raised `unsupported operand type(s) for -: 'str' and
'datetime.timedelta'` and killed the value_drops cron notification.
"""
import pytest

pytest.importorskip("numpy")

from datetime import date

import dashboard_services.player_value_history as pvh


class _FakeCursor:
    def __init__(self):
        self._calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._calls += 1

    def fetchone(self):
        # First (and only) fetchone: the MAX(as_of_date) baseline lookup.
        return {"comparison_date": date(2026, 10, 3)}

    def fetchall(self):
        return [
            {
                "player_id": "1",
                "name": "Test Player",
                "position": "RB",
                "team": "DET",
                "old_value": 100.0,
                "new_value": 60.0,
                "delta": -40.0,
            }
        ]


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor()


def _fake_get_conn(*a, **k):
    return _FakeConn()


def test_get_top_movers_handles_string_latest_date(monkeypatch):
    # Mirror the real implementation: ISO string, not a date object.
    monkeypatch.setattr(pvh, "get_latest_snapshot_date", lambda source="model": "2026-10-10")
    monkeypatch.setattr(pvh, "init_value_history_db", lambda: None)
    monkeypatch.setattr(pvh, "get_conn", _fake_get_conn)

    out = pvh.get_top_movers(days=7, limit=15)

    assert out["latest_date"] == "2026-10-10"
    assert out["comparison_date"] == "2026-10-03"
    assert out["used_days"] == 7
    assert out["fallers"] and out["fallers"][0]["delta"] == -40.0
