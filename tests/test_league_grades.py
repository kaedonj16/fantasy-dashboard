"""Regression guard for dashboard_services/ai/league_grades.py.

The module once imported ``count_roster_positions`` / ``get_roster_positions``
from ``utils.lineup_slots``, where neither name exists. The ImportError was
swallowed by the graceful-degradation try/except, so the dashboard hero
silently showed no grade while the Teams page showed one. This test fails
loudly if those imports ever break again.
"""
import pytest

from dashboard_services.ai.league_grades import viewer_league_context_grade

POS = ["QB", "RB", "WR", "TE"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                    "K", "DEF", "BN", "BN", "BN", "BN", "BN", "BN"]


def _player(pid, name, pos, value, age):
    return {
        "id": pid,
        "player_id": pid,
        "name": name,
        "search_name": name.strip().lower(),
        "position": pos,
        "value": value,
        "sf_value": value,
        "age": age,
    }


def _ctx():
    """Four-team dynasty league; roster 1 is stacked at every position."""
    vals, rosters = [], []
    for rid in (1, 2, 3, 4):
        pids = []
        scale = 1.0 if rid == 1 else 0.5
        for i, pos in enumerate(POS * 3):
            pid = f"p{rid}_{i}"
            vals.append(_player(pid, f"Player {pid}", pos,
                                round(400 * scale - i * 5, 1), 24 + (i % 8)))
            pids.append(pid)
        rosters.append({"roster_id": rid, "players": pids})
    return {
        "platform": "sleeper",
        "rosters": rosters,
        "model_value_table": vals,
        "roster_positions": list(ROSTER_POSITIONS),
        "picks_by_roster": {},
    }


def test_viewer_grade_returns_grade_for_stacked_team():
    grade = viewer_league_context_grade(_ctx(), 1)
    assert grade, "expected a grade dict, got empty (import regression?)"
    assert grade.get("grade"), grade
    assert grade.get("win_window"), grade
    assert grade.get("formula_id") == "dynasty_ctx", grade


def test_viewer_grade_weak_team_grades_lower():
    strong = viewer_league_context_grade(_ctx(), 1)
    weak = viewer_league_context_grade(_ctx(), 4)
    assert strong and weak
    assert strong["score"] > weak["score"]


def test_viewer_grade_degrades_gracefully():
    assert viewer_league_context_grade({}, 1) == {}
    assert viewer_league_context_grade(_ctx(), "not-a-roster") == {}


def test_fixed_imports_resolve():
    # Direct regression pin: these names must exist at their real homes.
    from utils.live_stats import count_roster_positions
    from utils.lineup_slots import is_superflex_lineup
    from dashboard_services.api import get_roster_positions

    counts = count_roster_positions(ROSTER_POSITIONS)
    assert counts["QB"] == 1 and counts["RB"] == 2
    assert not is_superflex_lineup(ROSTER_POSITIONS)
