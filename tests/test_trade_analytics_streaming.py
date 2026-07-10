"""The streaming trade loader must reconstruct trades identically to the old
list loader, so run_analytics' switch to streaming can't change results.

`_iter_trades` replaced a full ``list(trades.values())`` materialization (which
OOMed the trade-recrawl cron as the league pool grew). It reads the exact same
trades×assets join, so for the same rows it must yield exactly what `_load_trades`
returned. We mock the DB cursor and assert byte-for-byte equality, then confirm
the compute passes still run against the streamed source.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

# analytics.py imports `from dashboard_services.db import get_conn`, which pulls in
# psycopg. The DB is mocked in every test here, so stub that module if the real
# driver isn't installed (e.g. a DB-less CI runner) — the stub's get_conn is never
# called because each test monkeypatches analytics.get_conn.
if "dashboard_services.db" not in sys.modules:
    try:
        import psycopg  # noqa: F401
    except Exception:
        _stub = types.ModuleType("dashboard_services.db")
        _stub.get_conn = lambda *a, **k: None
        if "dashboard_services" not in sys.modules:
            # Keep it a real package (with __path__ to the source dir) so other
            # test modules can still import sibling submodules like
            # dashboard_services.archetype_engine — a plain ModuleType here has
            # no __path__ and would shadow the namespace package for the rest of
            # the run, breaking their imports at collection time.
            import os
            _pkg = types.ModuleType("dashboard_services")
            _pkg.__path__ = [os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dashboard_services")]
            sys.modules["dashboard_services"] = _pkg
        sys.modules["dashboard_services.db"] = _stub

from data_building.trade_intel import analytics


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.itersize = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._sql = sql

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self, name=None):
        return _FakeCursor(self._rows)


def _row(tid, txn, created, num_teams, side, atype, pid,
         pick_season=None, pick_round=None, pick_order=None):
    return {
        "id": tid, "transaction_id": txn, "created_at": created,
        "num_teams": num_teams, "side": side, "asset_type": atype,
        "player_id": pid, "pick_season": pick_season,
        "pick_round": pick_round, "pick_order": pick_order,
    }


def _sample_rows():
    c1 = datetime(2025, 9, 1, tzinfo=timezone.utc)
    c2 = datetime(2025, 9, 5, tzinfo=timezone.utc)
    c3 = datetime(2025, 9, 9, tzinfo=timezone.utc)
    # Ordered by trade id, as the SQL guarantees (ORDER BY t.id).
    return [
        # trade 1: 2 players for 1 player + a pick (assets span multiple rows)
        _row(1, "txn1", c1, 12, "a", "player", "p100"),
        _row(1, "txn1", c1, 12, "a", "player", "p101"),
        _row(1, "txn1", c1, 12, "b", "player", "p200"),
        _row(1, "txn1", c1, 12, "b", "pick", None, 2026, 1, "early"),
        # trade 2: a trade row with NO assets (LEFT JOIN -> side is None)
        _row(2, "txn2", c2, 10, None, None, None),
        # trade 3: single player each way
        _row(3, "txn3", c3, 14, "a", "player", "p300"),
        _row(3, "txn3", c3, 14, "b", "player", "p301"),
    ]


def test_iter_trades_matches_load_trades(monkeypatch):
    rows = _sample_rows()
    monkeypatch.setattr(analytics, "get_conn", lambda *a, **k: _FakeConn(rows))

    loaded = analytics._load_trades(2025)
    streamed = list(analytics._iter_trades(2025))

    assert streamed == loaded, "streamed trades diverge from the list loader"
    # Spot-check the structure that downstream passes rely on.
    assert [t["trade_id"] for t in streamed] == [1, 2, 3]
    assert len(streamed[0]["assets"]) == 4          # multi-asset trade assembled
    assert streamed[1]["assets"] == []              # assetless trade preserved
    assert streamed[0]["num_teams"] == 12


def test_compute_passes_run_against_stream(monkeypatch):
    rows = _sample_rows()
    monkeypatch.setattr(analytics, "get_conn", lambda *a, **k: _FakeConn(rows))
    values = {
        "p100": {"value_1qb": 100.0, "value_sf": 120.0},
        "p101": {"value_1qb": 50.0, "value_sf": 60.0},
        "p200": {"value_1qb": 140.0, "value_sf": 170.0},
        "p300": {"value_1qb": 80.0, "value_sf": 90.0},
        "p301": {"value_1qb": 75.0, "value_sf": 85.0},
    }
    make_trades = lambda: analytics._iter_trades(2025)

    # New signatures accept a stream factory and must not error.
    pstats = analytics._compute_player_stats(make_trades, values, 2025)
    kstats = analytics._compute_pick_stats(make_trades, values, 2025)
    pkgs = analytics._compute_packages(make_trades, 2025)

    assert isinstance(pstats, list) and isinstance(kstats, list) and isinstance(pkgs, list)
    # Feeding the identical trades as a plain list yields identical player rows
    # (proves the loop-source swap is behavior-preserving).
    trades_list = list(analytics._iter_trades(2025))
    pstats_list = analytics._compute_player_stats(lambda: iter(trades_list), values, 2025)
    assert pstats == pstats_list
