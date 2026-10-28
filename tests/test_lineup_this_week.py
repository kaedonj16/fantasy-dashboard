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


# ── Kaedon's Week 9 vs Fire Riley scenario (regressions, 2026-10-28) ───────
# Her live Lineup tab showed five phantom swap pills (Walker 20.8 ->
# Gordon 12.9 "Swap +7.9"), a "No swaps needed" banner over a +1.9
# Available metric, and Carnell Tate -- a taxi-squad player -- as the
# suggested add. Root causes: the optimizer pool included taxi players,
# and the solver's raw assignment rotates continuing starters across
# slots depending on roster iteration order, which the old slot-by-slot
# diff read as swaps. The roster orders below are orders that make the
# raw solver rotate (verified against utils.lineups.analyze_lineup).

_HER_STARTERS = ["wil", "wal", "war", "lam", "lon", "mcb", "egb", "gor",
                 "mcp", "sea"]
_HER_PROJ = {"wil": 15.1, "wal": 20.8, "war": 14.8, "lam": 22.3, "lon": 16.8,
             "mcb": 19.3, "egb": 11.7, "gor": 12.9, "mcp": 7.7, "sea": 7.3,
             "tate": 13.6, "b1": 5.0, "oq": 22.0, "or1": 20.0}


def _her_ctx(roster_order, taxi=()):
    players = {
        "wil": {"name": "Caleb Williams", "pos": "QB", "team": "CHI"},
        "wal": {"name": "Kenneth Walker III", "pos": "RB", "team": "KC"},
        "war": {"name": "Jaylen Warren", "pos": "RB", "team": "PIT"},
        "lam": {"name": "CeeDee Lamb", "pos": "WR", "team": "DAL"},
        "lon": {"name": "Drake London", "pos": "WR", "team": "ATL"},
        "mcb": {"name": "Trey McBride", "pos": "TE", "team": "ARI"},
        "egb": {"name": "Emeka Egbuka", "pos": "WR", "team": "TB"},
        "gor": {"name": "Ollie Gordon II", "pos": "RB", "team": "MIA"},
        "mcp": {"name": "Evan McPherson", "pos": "K", "team": "CIN"},
        "sea": {"name": "Seattle Seahawks", "pos": "DEF", "team": "SEA"},
        "tate": {"name": "Carnell Tate", "pos": "WR", "team": ""},
        "b1": {"name": "Bench Back", "pos": "RB", "team": "NYJ"},
        "oq": {"name": "Justin Herbert", "pos": "QB", "team": "LAC"},
        "or1": {"name": "Saquon Barkley", "pos": "RB", "team": "PHI"},
    }
    return {
        "viewer": {"viewer_roster_id": "1"},
        "platform": "sleeper",
        "league_id": "L1",
        "season": 2026,
        "current_week": 9,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "FLEX", "K", "DEF"] + ["BN"] * 10,
        "rosters": [
            {"roster_id": 1, "players": roster_order,
             "starters": list(_HER_STARTERS), "taxi": list(taxi)},
            {"roster_id": 2, "players": ["oq", "or1"],
             "starters": ["oq", "or1"]},
        ],
        "roster_map": {"1": "Caleb's Casting Couch", "2": "Fire Riley"},
        "players_index": players,
        "matchups_by_week": {9: [{
            "left": {"roster_id": "1",
                     "starters": [{"pid": p} for p in _HER_STARTERS]},
            "right": {"roster_id": "2",
                      "starters": [{"pid": "oq"}, {"pid": "or1"}]},
        }]},
        "statuses": {9: {"statuses": {}}},
        "proj_by_week": {9: dict(_HER_PROJ)},
    }


def test_this_week_taxi_player_is_never_suggested():
    # Roster order that makes the raw solver rotate AND reach for Tate.
    order = ["gor", "b1", "lam", "tate", "mcp", "lon", "sea", "wal", "wil",
             "egb", "war", "mcb"]
    ctx = _her_ctx(order, taxi=["tate"])
    html, swap_count = _build_lineup_this_week(ctx)
    assert swap_count == 0
    # Tate is on taxi: he is never named, and his +1.9 phantom gain is gone.
    assert "Carnell Tate" not in html
    assert "150.6" not in html and "+1.9" not in html
    # Current == optimal, banner agrees, no swap rows or pills anywhere.
    assert html.count("148.7") >= 2
    assert "No swaps needed" in html
    assert "lw-sw" not in html and "Swap +" not in html
    assert "Week 9 vs Fire Riley" in html


def test_this_week_pure_rotation_is_all_keep():
    # Same starter set, but this roster order makes the raw solver rotate
    # Walker/Warren and London/Egbuka across slots. Display must not care.
    order = ["wil", "lam", "war", "mcb", "wal", "egb", "lon", "sea", "b1",
             "mcp", "gor"]
    ctx = _her_ctx(order)
    del ctx["rosters"][0]["taxi"]
    ctx["rosters"][0]["players"] = [p for p in order if p != "tate"]
    html, swap_count = _build_lineup_this_week(ctx)
    assert swap_count == 0
    assert "No swaps needed" in html
    assert "lw-sw" not in html and "Swap +" not in html
    assert html.count("148.7") >= 2


def test_this_week_real_bench_upgrade_survives_rotation():
    # Tate as a normal bench player (no taxi): exactly one honest swap,
    # Tate in for Egbuka at +1.9, despite the rotating roster order.
    order = ["gor", "b1", "lam", "tate", "mcp", "lon", "sea", "wal", "wil",
             "egb", "war", "mcb"]
    ctx = _her_ctx(order)
    html, swap_count = _build_lineup_this_week(ctx)
    assert swap_count == 1
    assert "No swaps needed" not in html
    assert "148.7" in html and "150.6" in html and "+1.9" in html
    assert "Carnell Tate projects 1.9 more than Emeka Egbuka." in html
    # Exactly one swap row/pill, and no phantom rotation pills.
    assert html.count("lw-swap") == 1
    assert html.count("Swap +") == 1 and "Swap +1.9" in html
    assert "Swap +-" not in html and "Swap +7.9" not in html
    assert "Swap +8.7" not in html


def test_this_week_week_label_matches_hub_display_week():
    # Completed season: the hub's slides sit on the last finalized week,
    # so the Lineup label must too (the raw resolve used current_week).
    import pandas as pd

    ctx = _her_ctx(list(_HER_STARTERS) + ["b1"])
    ctx["season_complete"] = True
    ctx["weeks"] = 17
    ctx["current_week"] = 9
    ctx["matchups_by_week"][8] = ctx["matchups_by_week"][9]
    ctx["proj_by_week"][8] = ctx["proj_by_week"][9]
    ctx["statuses"][8] = ctx["statuses"][9]
    ctx["df_weekly"] = pd.DataFrame(
        [{"week": 8, "finalized": True}, {"week": 9, "finalized": False}])
    html, swap_count = _build_lineup_this_week(ctx)
    assert "Week 8 vs Fire Riley" in html
    assert "Week 9 vs" not in html


def _metric_tiles(html):
    return (html.count("<div class='lw-metric'>")
            + html.count("<div class='lw-metric acc'>"))


def test_this_week_no_swap_collapses_to_starters_list():
    # Same no-swap landing as the taxi regression: the view must collapse
    # to the banner + two honest metric tiles + a compact starters list.
    # A Current-vs-Optimal table here would print the same lineup twice.
    order = ["gor", "b1", "lam", "tate", "mcp", "lon", "sea", "wal", "wil",
             "egb", "war", "mcb"]
    ctx = _her_ctx(order, taxi=["tate"])
    html, swap_count = _build_lineup_this_week(ctx)
    assert swap_count == 0
    # Exactly two metric tiles (Current proj, Win prob): no Optimal tile,
    # no +0.0 Available tile repeating the same numbers.
    assert _metric_tiles(html) == 2
    assert "lw-metrics-solo" in html
    assert "Optimal proj" not in html and "Available" not in html
    assert "Current proj" in html and "Win prob" in html
    # Starters list instead of the comparison table: no table, no Keep
    # column, every starter listed once with his projection.
    assert "lw-stlist" in html and "lw-strow" in html
    assert "lw-table" not in html and "Keep" not in html
    for name in ("Caleb Williams", "Kenneth Walker III", "Jaylen Warren",
                 "CeeDee Lamb", "Drake London", "Trey McBride",
                 "Emeka Egbuka", "Ollie Gordon II", "Evan McPherson",
                 "Seattle Seahawks"):
        assert name in html
    # Total footer equals the current projection.
    assert "lw-strow-total" in html and "Total" in html


def test_this_week_swap_state_keeps_table_and_full_band():
    html, swap_count = _build_lineup_this_week(_ctx())
    assert swap_count == 1
    # Full band: Current, Optimal, Available, Win prob.
    assert _metric_tiles(html) == 4
    assert "lw-metrics-solo" not in html
    # Comparison table (not the collapsed starters list).
    assert "lw-table" in html and "lw-stlist" not in html


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
