"""Regression: a player Sleeper zeroes for a week must stay zeroed.

Sleeper's weekly feed drops the stat line of a doubtful/out player (the
row carries only ADP fields, which the fetch filters out), so the player
is simply absent from that week's cached projection file -- exactly how
Caleb Williams (Doubtful, hamstring) and Breece Hall showed 0.0 on both
Sleeper and ESPN for 2026 Week 4 while still carrying other weeks' lines.

build_projections_by_week used to refill any absent player with the
median of their other weeks, resurrecting a full ~15-20pt projection for
the zeroed week (and inventing points in bye weeks). These tests pin the
contract: the bundle for a week is Sleeper's line for that week, nothing
else. Absent stays absent; an explicit 0 stays 0.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")


def _bundles(monkeypatch, weeks_map):
    import app
    app._PROJ_BY_WEEK_MEMO.clear()
    monkeypatch.setattr(app, "load_week_projection",
                        lambda season, w, **kw: weeks_map.get(w, {}))
    monkeypatch.setattr(app, "load_players_index", lambda: {
        "11560": {"name": "Caleb Williams", "pos": "QB"},
        "8155": {"name": "Breece Hall", "pos": "RB"},
        "4984": {"name": "Josh Allen", "pos": "QB"},
    })
    return app.build_projections_by_week(2026, 5, {})


def test_absent_week_is_not_refilled_from_other_weeks(monkeypatch):
    # Caleb has full lines in weeks 1-3 and 5, but no Week 4 line at all
    # (Sleeper zeroed him). Week 4 must not inherit the median (~20).
    weeks = {
        1: {"11560": 19.0, "4984": 23.0},
        2: {"11560": 21.0, "4984": 22.0},
        3: {"11560": 20.0, "4984": 24.0},
        4: {"4984": 23.12},                      # Caleb absent = Sleeper 0.0
        5: {"11560": 20.39, "4984": 25.45},
    }
    bundles = _bundles(monkeypatch, weeks)
    w4 = bundles[4]["projections"]
    assert "11560" not in w4
    assert w4["4984"] == 23.12
    # His real Week 5 line is untouched.
    assert bundles[5]["projections"]["11560"] == 20.39
    assert bundles["_available"] is True


def test_explicit_zero_line_stays_zero(monkeypatch):
    weeks = {
        1: {"8155": 15.0},
        2: {"8155": 14.0},
        3: {"8155": 16.0},
        4: {"8155": 0.0},                        # explicit Sleeper 0
        5: {"8155": 14.93},
    }
    bundles = _bundles(monkeypatch, weeks)
    assert bundles[4]["projections"]["8155"] == 0.0


def test_bye_week_gets_no_invented_points(monkeypatch):
    # Same mechanism as the injury zero: a bye-week absence is not a hole
    # to fill with the season median.
    weeks = {
        1: {"4984": 23.0},
        2: {"4984": 22.0},
        3: {"4984": 24.0},
        4: {"4984": 23.0},
        5: {},                                   # bye week: no line
    }
    bundles = _bundles(monkeypatch, weeks)
    assert "4984" not in bundles[5]["projections"]
