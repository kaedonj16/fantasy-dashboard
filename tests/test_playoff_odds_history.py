"""Unit tests for playoff-odds movement (dashboard_services.playoff_odds_history).

The DB is faked so the ranking + week-over-week delta logic can be exercised
without a live connection.
"""
from dashboard_services import playoff_odds_history as H


# ── _rank ────────────────────────────────────────────────────────────────────

def test_rank_orders_by_playoff_then_wins():
    rows = [
        {"roster_id": "1", "playoff_probability": 40.0, "avg_final_wins": 7.0},
        {"roster_id": "2", "playoff_probability": 90.0, "avg_final_wins": 9.0},
        {"roster_id": "3", "playoff_probability": 40.0, "avg_final_wins": 8.0},  # ties #1 on prob, more wins
    ]
    assert H._rank(rows) == {"2": 1, "3": 2, "1": 3}


def test_rank_handles_missing_values():
    rows = [{"roster_id": "1"}, {"roster_id": "2", "playoff_probability": 10.0}]
    assert H._rank(rows) == {"2": 1, "1": 2}


# ── record_and_movement (faked DB) ───────────────────────────────────────────

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
        if "INSERT INTO playoff_odds" in sql:
            self._writes.append(params)
        return _FakeCur(self._handler(sql, params))


def _install(monkeypatch, handler, writes):
    monkeypatch.setattr(H, "_TABLE_READY", True)   # skip CREATE TABLE
    monkeypatch.setattr(H, "get_conn", lambda: _FakeConn(handler, writes))


def test_no_movement_before_week_one(monkeypatch):
    writes = []
    _install(monkeypatch, lambda sql, p: [], writes)
    assert H.record_and_movement("L", 2026, 0, [{"roster_id": "1", "playoff_pct": 50}]) == {}


def test_movement_vs_prior_week(monkeypatch):
    writes = []

    # Prior week: team 1 led, team 2 second. This week team 2 overtakes.
    def handler(sql, params):
        if "MAX(week)" in sql:
            return [{"w": 4}]
        if "SELECT roster_id" in sql:
            return [
                {"roster_id": 1, "playoff_probability": 80.0, "avg_final_wins": 9.0},
                {"roster_id": 2, "playoff_probability": 60.0, "avg_final_wins": 8.0},
            ]
        return []
    _install(monkeypatch, handler, writes)

    odds = [
        {"roster_id": 1, "playoff_pct": 55.0, "avg_final_wins": 8.0},
        {"roster_id": 2, "playoff_pct": 85.0, "avg_final_wins": 9.0},
    ]
    mv = H.record_and_movement("L", 2026, 5, odds, write=True)
    # prev ranks: 1->1, 2->2 ; cur ranks: 2->1, 1->2
    assert mv == {"1": -1, "2": 1}
    assert len(writes) == 2   # both teams snapshotted this week


def test_write_skipped_on_cache_hit(monkeypatch):
    writes = []

    def handler(sql, params):
        if "MAX(week)" in sql:
            return [{"w": None}]     # no prior snapshot
        return []
    _install(monkeypatch, handler, writes)

    H.record_and_movement("L", 2026, 5,
                          [{"roster_id": 1, "playoff_pct": 50.0}], write=False)
    assert writes == []   # cache hit → no write
