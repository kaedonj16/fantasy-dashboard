"""Lineup tab This Week view: current starters vs projected-optimal.

Fixture projections only; the builder reuses the matchup slide's projection
lookups and win probability, and utils.lineups.analyze_lineup for the
projected-optimal assignment. No Apply control exists anywhere (Kaedon's
site has no apply functionality; swaps are recommendations only).
"""
from __future__ import annotations

import pytest

# build_optimal_body reaches app.py (which imports pandas at module load)
# through the optimal-lineup helpers. The fast "lint" CI shard has flask
# but not pandas, so guard on pandas first — otherwise the wrapper test
# raises ModuleNotFoundError there. These tests run in the
# full-dependency shard.
pytest.importorskip("pandas")

from dashboard_services.pages.optimal_page import (
    _build_lineup_this_week,
    build_optimal_body,
)


def _ctx():
    players = {
        "q1": {"name": "Josh Allen", "pos": "QB", "team": "BUF"},
        "r1": {"name": "Bijan Robinson", "pos": "RB", "team": "ATL"},
        "r2": {"name": "Weak Back", "pos": "RB", "team": "NYJ"},
        "w1": {"name": "Ja'Marr Chase", "pos": "WR", "team": "CIN"},
        "t1": {"name": "Trey McBride", "pos": "TE", "team": "ARI"},
        "b1": {"name": "Bench Back", "pos": "RB", "team": "DAL"},
        "oq": {"name": "Justin Herbert", "pos": "QB", "team": "LAC"},
        "or1": {"name": "Saquon Barkley", "pos": "RB", "team": "PHI"},
    }
    proj = {"q1": 24.0, "r1": 18.0, "r2": 8.0, "w1": 19.0, "t1": 11.0,
            "b1": 14.5, "oq": 22.0, "or1": 20.0}
    mine = ["q1", "r1", "r2", "w1", "t1"]
    return {
        "viewer": {"viewer_roster_id": "1"},
        "platform": "sleeper",
        "league_id": "L1",
        "season": 2026,
        "current_week": 8,
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
        "rosters": [
            {"roster_id": 1, "players": mine + ["b1"], "starters": mine},
            {"roster_id": 2, "players": ["oq", "or1"], "starters": ["oq", "or1"]},
        ],
        "roster_map": {"1": "You", "2": "Fire Riley"},
        "players_index": players,
        "matchups_by_week": {8: [{
            "left": {"roster_id": "1", "starters": [{"pid": p} for p in mine]},
            "right": {"roster_id": "2",
                      "starters": [{"pid": "oq"}, {"pid": "or1"}]},
        }]},
        "statuses": {8: {"statuses": {}}},
        "proj_by_week": {8: proj},
    }


def test_this_week_computes_the_swap():
    html, swap_count = _build_lineup_this_week(_ctx())
    assert swap_count == 1
    # Metric band: current 80.0, optimal 86.5, +6.5 available.
    assert "80.0" in html and "86.5" in html and "+6.5" in html
    assert "Current proj" in html and "Optimal proj" in html
    assert "Available" in html
    # Win prob moves with the optimal lineup (carousel function).
    assert "Win prob" in html and "→" in html
    # The swap itself: Weak Back out, Bench Back in, with the honest reason.
    assert "Weak Back" in html and "Bench Back" in html
    assert "Bench Back projects 6.5 more than Weak Back." in html
    # Full table with the swap row highlighted and pills.
    assert "lw-table" in html
    assert "lw-sw" in html
    assert "Swap +6.5" in html
    assert "Keep" in html
    assert "Week 8 vs Fire Riley" in html
    assert "Apply" not in html


def test_this_week_no_projections_is_empty():
    ctx = _ctx()
    ctx["proj_by_week"] = {8: {}}
    html, swap_count = _build_lineup_this_week(ctx)
    assert html == "" and swap_count is None


def test_wrapper_seg_defaults_to_this_week():
    ctx = _ctx()
    html = build_optimal_body(ctx)
    assert "data-opt-seg='thisweek'" in html
    assert "data-opt-seg='history'" in html
    assert "data-opt-panel='thisweek'" in html
    assert "data-opt-panel='history' hidden" in html
    # History is the unchanged completed-weeks view (empty state here).
    assert "No verified completed weeks yet." in html
    # The hub reads the swap count off ctx for the Lineup tab chip.
    assert ctx.get("lineup_swap_count") == 1
    assert "Apply" not in html
