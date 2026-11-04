"""Focused tests for the /api/trade-database load-speed query shape.

Live diagnosis 2026-11-04: the default request took 11-40s (first paint
timed out at 30s) because both queries aggregated the whole season:
COUNT(DISTINCT t.id) with an unconditional leagues join, and
SELECT DISTINCT over ~545k trades before the LIMIT. t.id is the PK and
the leagues join is 1:1, so dedupe is only needed when the two-sided
asset joins can multiply rows. These tests pin that conditional shape
by exec'ing the real endpoint with stubbed deps (same harness pattern
as test_trade_db_sort.py).
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
    def fetchone(self):
        return {"n": 5}

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, recorder):
        self._rec = recorder

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        self._rec.append((query, list(params)))
        return _FakeCursor()


def _load_endpoint(monkeypatch):
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "api_trade_database"
    )
    seg = ast.get_source_segment(src, node)
    assert seg, "could not extract api_trade_database"

    calls = []
    db_mod = types.ModuleType("dashboard_services.db")
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
    lt_mod.league_format_sql_param = lambda fmt: {"dynasty": 2, "redraft": 0, "keeper": 1}.get(fmt)
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
        "_TRADE_DB_COUNT_TTL": 3600,
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
    fn()
    assert len(calls) == 2, f"expected count+row queries, got {len(calls)}"
    return calls[0], calls[1]


def test_default_count_is_plain_and_joinless(endpoint):
    (count_q, count_p), (row_q, _row_p) = _run(endpoint, {"season": "2026"})
    assert "COUNT(*)" in count_q
    assert "COUNT(DISTINCT" not in count_q
    assert "trade_intel_leagues" not in count_q
    assert count_p == [2026]
    assert "SELECT DISTINCT" not in row_q


def test_sf_filter_keeps_leagues_join_but_plain_count(endpoint):
    (count_q, count_p), _ = _run(endpoint, {"season": "2026", "league_type": "sf"})
    assert "LEFT JOIN trade_intel_leagues" in count_q
    assert "l.is_superflex = %s" in count_q
    assert "COUNT(*)" in count_q
    assert count_p == [2026, True]


def test_format_filter_keeps_leagues_join_but_plain_count(endpoint):
    (count_q, count_p), _ = _run(endpoint, {"season": "2026", "league_format": "dynasty"})
    assert "LEFT JOIN trade_intel_leagues" in count_q
    assert "l.league_type = %s" in count_q
    assert "COUNT(*)" in count_q
    assert count_p == [2026, 2]


def test_two_sided_keeps_distinct_on_both_queries(endpoint):
    (count_q, _), (row_q, _) = _run(endpoint, {
        "season": "2026", "player_a": "9487", "player_b": "12514",
    })
    assert "COUNT(DISTINCT t.id)" in count_q
    assert "SELECT DISTINCT" in row_q


def test_migration_adds_season_created_index():
    sql = (ROOT / "migrations" / "052_trade_db_season_created_index.sql").read_text(encoding="utf-8")
    assert "CREATE INDEX IF NOT EXISTS idx_tit_season_created" in sql
    assert "(season, created_at DESC NULLS LAST, id DESC)" in sql
