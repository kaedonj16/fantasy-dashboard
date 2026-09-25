"""Regression tests for 7-day dynasty value rank movement.

Bug (2026-09-25): rank_change_7d compared today's enumerate-style rank
against the 7-day-ago snapshot's pandas min-tie rank over a DIFFERENT pool.
With ~987 players tied at value 0.0 sharing one snapshot rank but spread
across ~987 enumerate slots today, plus pool growth from newly added
players, veterans showed phantom moves like "down 58 spots". Live data at
the time: 1031 players "down" vs 339 "up", changes summing to -67,887
instead of ~0 for a fair re-ranking.

The fix: rank_change_vs_snapshot ranks the INTERSECTION of both pools with
competition ("1224") tie handling on both sides, so spots only move when
players actually pass each other.

The module under test imports utils.utils (which needs Flask, absent in
the pure test env), so it is stubbed here; the helpers under test are pure.
"""
import sys
import types

import pytest

pytest.importorskip("pandas")

_utils_stub = types.ModuleType("utils.utils")
_utils_stub.load_model_value_table = lambda *a, **k: []
sys.modules.setdefault("utils.utils", _utils_stub)
_utils_pkg = types.ModuleType("utils")
_utils_pkg.utils = _utils_stub
sys.modules.setdefault("utils", _utils_pkg)

from data_building.update_player_values_with_rankings import (  # noqa: E402
    _competition_ranks,
    rank_change_vs_snapshot,
)


def test_competition_ranks_ties_share_best_rank():
    assert _competition_ranks({"a": 100.0, "b": 100.0, "c": 50.0}) == {
        "a": 1,
        "b": 1,
        "c": 3,
    }


def test_competition_ranks_all_tied():
    ranks = _competition_ranks({f"p{i}": 0.0 for i in range(500)})
    assert set(ranks.values()) == {1}


def test_no_movement_when_values_unchanged():
    cur = {"a": 300.0, "b": 200.0, "c": 100.0}
    assert rank_change_vs_snapshot(cur, dict(cur)) == {"a": 0, "b": 0, "c": 0}


def test_mass_zero_tie_does_not_manufacture_moves():
    """The exact reported failure: ~987 players tied at 0 in both pools."""
    cur = {f"star{i}": 500.0 - i for i in range(400)}
    cur.update({f"scrub{i}": 0.0 for i in range(987)})
    changes = rank_change_vs_snapshot(cur, dict(cur))
    assert len(changes) == 1387
    assert all(v == 0 for v in changes.values()), "identical pools must show zero movement"


def test_pure_reorder_sums_to_zero():
    cur = {"a": 100.0, "b": 300.0, "c": 200.0}
    hist = {"a": 300.0, "b": 100.0, "c": 200.0}
    changes = rank_change_vs_snapshot(cur, hist)
    assert changes == {"a": -2, "b": 2, "c": 0}
    assert sum(changes.values()) == 0


def test_pool_growth_does_not_move_incumbents():
    """New players entering the pool must not push incumbents down."""
    hist = {"a": 300.0, "b": 200.0, "c": 100.0}
    cur = {
        "rookie1": 400.0,  # inserted at the very top
        "a": 300.0,
        "rookie2": 250.0,  # inserted in the middle
        "b": 200.0,
        "c": 100.0,
        "rookie3": 0.0,  # inserted at the bottom
    }
    changes = rank_change_vs_snapshot(cur, hist)
    assert changes == {"a": 0, "b": 0, "c": 0}


def test_new_player_absent_from_snapshot_gets_no_trend():
    changes = rank_change_vs_snapshot(
        {"a": 300.0, "new": 250.0}, {"a": 300.0}
    )
    assert changes == {"a": 0}
    assert "new" not in changes


def test_departed_player_ignored():
    changes = rank_change_vs_snapshot(
        {"a": 300.0}, {"a": 300.0, "gone": 999.0}
    )
    assert changes == {"a": 0}


def test_empty_snapshot_returns_empty():
    assert rank_change_vs_snapshot({"a": 1.0}, {}) == {}


def test_real_passing_detected_with_ties():
    # b passes a; tied pair below holds still
    cur = {"a": 100.0, "b": 150.0, "c": 0.0, "d": 0.0}
    hist = {"a": 150.0, "b": 100.0, "c": 0.0, "d": 0.0}
    changes = rank_change_vs_snapshot(cur, hist)
    assert changes["b"] == 1
    assert changes["a"] == -1
    assert changes["c"] == 0
    assert changes["d"] == 0


def test_sf_value_map_uses_same_logic():
    """SF ordering is just another value map through the same helper."""
    cur_sf = {"qb": 900.0, "rb": 700.0, "wr": 600.0}
    hist_sf = {"qb": 700.0, "rb": 700.0, "wr": 600.0}
    changes = rank_change_vs_snapshot(cur_sf, hist_sf)
    # qb was tied-1st, now sole 1st: no rank movement; rb tied-1st now 2nd
    assert changes == {"qb": 0, "rb": -1, "wr": 0}
