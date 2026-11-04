"""Focused tests for /api/players pagination defaults.

The endpoint used to treat a missing limit as "return all" (~5,925
players, 1.1MB live). Every first-party caller passes an explicit limit
(search 8-20, trade-db preload 100), so the default is now a 100-row
page and explicit limits are capped at 500. Explicit limit=0 keeps the
legacy return-all behavior (and its bare-list response shape) as a
deliberate escape. Exec'd from app.py with stubbed deps, same harness
pattern as the trade-db tests.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_N = 600
_PLAYERS = {
    str(i): {"name": f"Player {i}", "pos": "RB", "team": "DET"}
    for i in range(_N)
}
_VALUES = {
    str(i): {"value": float(_N - i), "sf_value": float(_N - i)}
    for i in range(_N)
}


class _FakeArgs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeRequest:
    def __init__(self, args):
        self.args = _FakeArgs(args)


def _load_endpoint(monkeypatch):
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "api_players"
    )
    seg = ast.get_source_segment(src, node)
    assert seg, "could not extract api_players"

    utils_mod = types.ModuleType("utils.utils")
    utils_mod.load_players_index = lambda: dict(_PLAYERS)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.utils = utils_mod
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_mod)

    import logging as _logging
    import math as _math

    ns = {
        "request": _FakeRequest({}),
        "logger": _logging.getLogger("test"),
        "jsonify": lambda obj, *a: obj,
        "math": _math,
        "get_model_value_table_cached": lambda: [],
        "_model_value_sidecar": lambda: {"index": dict(_VALUES)},
        "_api_err": lambda msg, e: {"error": msg},
    }
    exec(compile(seg, "api_players", "exec"), ns)  # noqa: S102 - test harness
    return ns["api_players"], ns


@pytest.fixture()
def endpoint(monkeypatch):
    fn, ns = _load_endpoint(monkeypatch)
    yield fn, ns


def _run(endpoint, args):
    fn, ns = endpoint
    ns["request"] = _FakeRequest(args)
    return fn()


def test_missing_limit_defaults_to_a_100_row_page(endpoint):
    data = _run(endpoint, {})
    assert isinstance(data, dict)
    assert len(data["players"]) == 100
    assert data["total"] == _N
    assert data["page"] == 1
    assert data["pages"] == 6


def test_explicit_limit_is_capped_at_500(endpoint):
    data = _run(endpoint, {"limit": "10000"})
    assert len(data["players"]) == 500
    assert data["total"] == _N


def test_small_explicit_limit_is_honored(endpoint):
    data = _run(endpoint, {"limit": "15"})
    assert len(data["players"]) == 15


def test_explicit_zero_keeps_legacy_return_all(endpoint):
    data = _run(endpoint, {"limit": "0"})
    assert isinstance(data, list)
    assert len(data) == _N
