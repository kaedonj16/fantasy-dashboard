"""Regression tests for _usage_for_player() in app.py.

Covers the depth-chart usage resolution behind the NFL Teams page:
- The daily usage table (fresh) beats stale usage embedded in the
  relevant-players index.
- List-form usage tables return the nested usage dict, not the wrapper row.
- Dict-form usage tables keep working.
- Embedded index usage remains as a fallback when the table has no record.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_usage_for_player():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _usage_for_player(")
    assert start > 0, "_usage_for_player not found in app.py"
    # Function ends at the next top-level def.
    end = src.find("\ndef ", start + 10)
    assert end > start
    ns: dict = {}
    exec(compile(src[start:end], "app.py::_usage_for_player", "exec"), ns)
    return ns["_usage_for_player"]


_usage_for_player = _load_usage_for_player()

FRESH = {"target_share": 0.125, "carry_share": 0.5965, "touch_share": 0.3628,
         "ppr_per_game": 15.4}
STALE = {"target_share": 0.0, "carry_share": 0.1, "touch_share": 0.05,
         "ppr_per_game": 18.6}


def test_table_beats_stale_embedded_usage_dict_form():
    table = {"8138": dict(FRESH)}
    index = {"8138": {"usage": dict(STALE)}}
    got = _usage_for_player("8138", index, table)
    assert got["target_share"] == 0.125
    assert got["carry_share"] == 0.5965
    assert got["ppr_per_game"] == 15.4


def test_table_beats_stale_embedded_usage_list_form():
    table = [{"id": "8138", "sleeper_id": "8138", "usage": dict(FRESH)}]
    index = {"8138": {"usage": dict(STALE)}}
    got = _usage_for_player("8138", index, table)
    assert got == FRESH


def test_list_form_returns_nested_usage_not_wrapper_row():
    table = [{"id": "8138", "name": "James Cook", "usage": dict(FRESH)}]
    got = _usage_for_player("8138", {}, table)
    assert "usage" not in got, "must return the inner usage dict, not the wrapper row"
    assert got["touch_share"] == 0.3628


def test_embedded_usage_is_fallback_when_table_misses():
    table = [{"id": "9999", "usage": dict(FRESH)}]
    index = {"8138": {"usage": dict(STALE)}}
    got = _usage_for_player("8138", index, table)
    assert got == STALE


def test_empty_when_nothing_found():
    assert _usage_for_player("8138", {}, []) == {}
    assert _usage_for_player("8138", {}, {}) == {}
    assert _usage_for_player("", {"8138": {"usage": dict(STALE)}}, {}) == {}


def test_skips_non_dict_rows_in_list_table():
    table = ["junk", None, {"id": "8138", "usage": dict(FRESH)}]
    got = _usage_for_player("8138", {}, table)
    assert got == FRESH
