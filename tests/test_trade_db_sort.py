"""Focused tests for /api/trade-database ?sort= (date/value ordering).

The endpoint lives in app.py (excluded from the refactor), so these tests
extract the real function via AST and run it with a stubbed DB connection
and request args -- no Flask app import, no live database.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _FakeArgs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, args):
        self.args = _FakeArgs(args)


class _FakeCursor:
    def __init__(self, recorder, count_n=5, rows=()):
        self._rec = recorder
        self._count_n = count_n
        self._rows = rows

    def fetchone(self):
        return {"n": self._count_n}

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, recorder, count_n=5, rows=()):
        self._rec = recorder
        self._count_n = count_n
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        self._rec.append((query, list(params)))
        return _FakeCursor(self._rec, self._count_n, self._rows)


def _load_endpoint(monkeypatch):
    """Exec the real api_trade_database from app.py with stubbed deps."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "api_trade_database"
    )
    seg = ast.get_source_segment(src, node)
    assert seg, "could not extract api_trade_database"

    # Stub the modules the function imports from internally.
    # Scoped via monkeypatch so the stubs are removed after each test;
    # leaving them in sys.modules poisons later-collected test modules
    # (e.g. a bare "utils" stub breaks tests importing the real utils).
    db_mod = types.ModuleType("dashboard_services.db")
    calls = []
    db_mod.get_conn = lambda: _FakeConn(calls)
    ds_pkg = types.ModuleType("dashboard_services")
    ds_pkg.db = db_mod
    monkeypatch.setitem(sys.modules, "dashboard_services", ds_pkg)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_mod)

    utils_mod = types.ModuleType("utils.utils")
    utils_mod.load_players_index = lambda: {}
    utils_pkg = types.ModuleType("utils")
    utils_pkg.utils = utils_mod
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_mod)

    lt_mod = types.ModuleType("data_building.trade_intel.league_types")
    lt_mod.league_format_sql_param = lambda fmt: None
    ti_pkg = types.ModuleType("data_building.trade_intel")
    ti_pkg.league_types = lt_mod
    db_build_pkg = types.ModuleType("data_building")
    db_build_pkg.trade_intel = ti_pkg
    monkeypatch.setitem(sys.modules, "data_building", db_build_pkg)
    monkeypatch.setitem(sys.modules, "data_building.trade_intel", ti_pkg)
    monkeypatch.setitem(sys.modules, "data_building.trade_intel.league_types", lt_mod)

    import datetime as _dt
    import logging as _logging

    ns = {
        "request": _FakeRequest({}),
        "datetime": _dt.datetime,
        "logger": _logging.getLogger("test"),
        "jsonify": lambda obj, *a: obj,
        "_TRADE_DB_COUNT_CACHE": {},
        "_TRADE_DB_COUNT_TTL": 300,
    }
    exec(compile(seg, "api_trade_database", "exec"), ns)  # noqa: S102 - test harness
    return ns["api_trade_database"], calls, ns


@pytest.fixture()
def endpoint(monkeypatch):
    fn, calls, ns = _load_endpoint(monkeypatch)
    yield fn, calls, ns


def _run(endpoint, args):
    fn, calls, ns = endpoint
    ns["request"] = _FakeRequest(args)
    ns["_TRADE_DB_COUNT_CACHE"] = {}
    calls.clear()
    return fn(), calls


def _row_query(calls):
    # Second execute call is the row query (first is the COUNT).
    assert len(calls) == 2, f"expected 2 queries, got {len(calls)}"
    return calls[1]


def test_default_sort_is_newest_first(endpoint):
    _, calls = _run(endpoint, {"season": "2026"})
    query, params = _row_query(calls)
    assert "ORDER BY t.created_at DESC NULLS LAST" in query
    assert "trade_value" not in query


def test_date_asc(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "sort": "date_asc"})
    query, params = _row_query(calls)
    assert "ORDER BY t.created_at ASC NULLS LAST" in query
    assert "trade_value" not in query


def test_value_desc_adds_market_value_join(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "sort": "value_desc"})
    query, params = _row_query(calls)
    assert "ORDER BY trade_value DESC" in query
    assert "COALESCE(tv.trade_value, 0) AS trade_value" in query
    assert "trade_intel_player_stats" in query
    assert "weighted_market_value_1qb" in query


def test_value_desc_sf_uses_sf_column(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "sort": "value_desc", "league_type": "sf"})
    query, params = _row_query(calls)
    assert "weighted_market_value_sf" in query


def test_value_asc(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "sort": "value_asc"})
    query, params = _row_query(calls)
    assert "ORDER BY trade_value ASC" in query


def test_bad_sort_falls_back_to_default(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "sort": "drop table"})
    query, params = _row_query(calls)
    assert "ORDER BY t.created_at DESC NULLS LAST" in query
    assert "drop table" not in query.lower()


def test_both_sides_player_filter_param_order(endpoint):
    # Regression: the JOIN placeholders come textually before the WHERE
    # placeholders, so the A/B id lists must lead the param list. The old
    # code put season first and 500'd on every Side A + Side B search.
    _, calls = _run(endpoint, {
        "season": "2026",
        "player_a": "1,2",
        "player_b": "3",
    })
    query, params = _row_query(calls)
    assert params[0] == ["1", "2"], f"A ids must bind the first JOIN placeholder, got {params[0]!r}"
    assert params[1] == ["3"], f"B ids must bind the second JOIN placeholder, got {params[1]!r}"
    assert params[2] == 2026


def test_single_side_param_order_unchanged(endpoint):
    _, calls = _run(endpoint, {"season": "2026", "player_a": "9"})
    query, params = _row_query(calls)
    assert params[0] == 2026
    assert params[-3] == ["9"]  # before LIMIT/OFFSET


def test_value_sort_param_order_with_filters(endpoint):
    _, calls = _run(endpoint, {
        "season": "2026",
        "sort": "value_desc",
        "player_a": "1,2",
        "player_b": "3",
    })
    query, params = _row_query(calls)
    # join params, then the value-subquery season, then WHERE season
    assert params[0] == ["1", "2"]
    assert params[1] == ["3"]
    assert params[2] == 2026  # _vt.season inside the value subquery
    assert params[3] == 2026  # t.season in WHERE
