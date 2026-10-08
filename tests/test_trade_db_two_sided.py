"""Focused tests for the trade-database two-sided (Side A + Side B) filter.

Regression test for Kaedon's 2026-10-08 report: with Parker Washington in
Side A and Emeka Egbuka in Side B, every result contained only Parker
Washington -- the Side B constraint was silently dropped.

Root cause: `loadTDBPage` used `if (loading) return;`, so adding the Side B
player while the Side A request was still in flight dropped the two-sided
request entirely, leaving stale single-side results on screen with both chips
set. The fix is latest-wins via AbortController: a new filter change aborts
the in-flight request instead of being dropped.

The server-side AND semantics (both players on opposite sides via the
`_ja`/`_jb` JOINs with `_jb.side <> _ja.side`) were already correct; these
tests pin that behavior too so a future refactor cannot OR it or drop a side.

Server-side tests exec the real `api_trade_database` from app.py with a
stubbed DB connection (same harness as test_trade_db_sort.py). Client-side
tests read the page's inline JS from routes/trade_bp.py (same approach as
test_trade_db_card_spacing.py).
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
TDB_SRC = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Server-side: AND semantics for the two-sided search
# ---------------------------------------------------------------------------


class _FakeArgs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, args):
        self.args = _FakeArgs(args)


class _FakeCursor:
    def __init__(self, recorder):
        self._rec = recorder

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
        return _FakeCursor(self._rec)


def _load_endpoint(monkeypatch):
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "api_trade_database"
    )
    seg = ast.get_source_segment(src, node)
    assert seg, "could not extract api_trade_database"

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
    fn()
    return calls


def _queries(calls):
    # (count_query, row_query); each is (sql, params)
    assert len(calls) == 2, f"expected 2 queries, got {len(calls)}"
    return calls[0], calls[1]


def test_both_sides_enforces_opposite_sides(endpoint):
    count_q, row_q = _queries(_run(endpoint, {
        "season": "2026", "player_a": "9487", "player_b": "12514",
        "league_format": "dynasty",
    }))
    for label, (sql, params) in (("count", count_q), ("rows", row_q)):
        assert "JOIN trade_intel_assets _ja" in sql, f"{label}: missing _ja join"
        assert "JOIN trade_intel_assets _jb" in sql, f"{label}: missing _jb join"
        assert "_jb.side <> _ja.side" in sql, f"{label}: missing opposite-side condition"
        assert params[0] == ["9487"], f"{label}: _ja must bind player_a ids, got {params[0]}"
        assert params[1] == ["12514"], f"{label}: _jb must bind player_b ids, got {params[1]}"


def test_both_sides_does_not_use_single_side_subquery(endpoint):
    _, (sql, _) = _queries(_run(endpoint, {
        "season": "2026", "player_a": "9487", "player_b": "12514",
    }))
    # The single-side IN-subqueries must not appear when both sides are given;
    # otherwise Side B would be silently OR'd/dropped.
    assert "_fa.trade_id" not in sql and "_fb.trade_id" not in sql


def test_side_a_only_matches_either_side(endpoint):
    _, (sql, params) = _queries(_run(endpoint, {"season": "2026", "player_a": "9487"}))
    assert "JOIN trade_intel_assets _jb" not in sql
    assert "_fa.trade_id" in sql
    assert params[1] == ["9487"]


def test_side_b_only_matches_either_side(endpoint):
    _, (sql, params) = _queries(_run(endpoint, {"season": "2026", "player_b": "12514"}))
    assert "JOIN trade_intel_assets _ja" not in sql
    assert "_fb.trade_id" in sql
    assert params[1] == ["12514"]


# ---------------------------------------------------------------------------
# Client-side: a new filter change must never be silently dropped
# ---------------------------------------------------------------------------


def test_no_drop_guard_on_load():
    # The old `if (loading) return;` dropped the two-sided request when the
    # Side B chip was added while the Side A request was in flight, leaving
    # stale single-side results on screen with both chips set.
    assert "if (loading) return" not in TDB_SRC


def test_latest_wins_abort_controller():
    assert "tdbAbort.abort()" in TDB_SRC
    assert "new AbortController()" in TDB_SRC
    assert "signal: tdbSignal" in TDB_SRC


def test_aborted_requests_stay_silent():
    assert "err.name === 'AbortError'" in TDB_SRC


def test_both_player_params_still_sent():
    assert "params.set('player_a'" in TDB_SRC
    assert "params.set('player_b'" in TDB_SRC
