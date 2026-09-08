"""Teams Schedule tab remaining-schedule ranking uses standings SOS Future."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pd = pytest.importorskip("pandas")

from dashboard_services.service import (
    build_team_strength,
    compute_sos_by_team,
    remaining_schedule_strength,
)

ROOT = Path(__file__).resolve().parents[1]


def _rosters(*rids: str) -> list[dict]:
    return [{"roster_id": rid} for rid in rids]


def _week_pair(mid: int, a: str, a_pts: float, b: str, b_pts: float) -> list[dict]:
    return [
        {"matchup_id": mid, "roster_id": a, "points": a_pts},
        {"matchup_id": mid, "roster_id": b, "points": b_pts},
    ]


def test_schedule_strength_source_uses_sos_not_roster_value():
    src = (ROOT / "routes" / "schedule_api_bp.py").read_text(encoding="utf-8")
    start = src.find("def api_schedule_strength")
    assert start > 0
    body = src[start:start + 4000]
    assert "remaining_schedule_strength" in body
    assert "regular_season_length" in body
    assert "roster_val / 50.0" not in body
    assert "preseason_opponent_strength" not in body
    assert "load_pick_value_table" not in body
    helper = (ROOT / "dashboard_services" / "service.py").read_text(encoding="utf-8")
    fn = helper[helper.find("def remaining_schedule_strength"):]
    fn = fn[: fn.find("\nfrom collections import defaultdict")]
    assert "build_team_strength" in fn
    assert "compute_sos_by_team" in fn
    assert "0.65" not in fn  # blend lives in build_team_strength, not a local copy


def test_remaining_schedule_matches_standings_sos_future():
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 118, "2", 120) + _week_pair(2, "3", 90, "4", 110),
        2: _week_pair(1, "3", 0, "1", 0) + _week_pair(2, "4", 0, "2", 0),
    }
    rows, no_games = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=2,
        roster_names={"1": "A", "2": "B", "3": "C", "4": "D"},
    )
    assert no_games is False
    by_id = {r["roster_id"]: r for r in rows}
    assert by_id["3"]["games_remaining"] == 1
    assert by_id["4"]["games_remaining"] == 1

    stats = pd.DataFrame([
        {"owner": "1", "AVG": 118.0, "Win%": 0.0},
        {"owner": "2", "AVG": 120.0, "Win%": 1.0},
        {"owner": "3", "AVG": 90.0, "Win%": 0.0},
        {"owner": "4", "AVG": 110.0, "Win%": 1.0},
    ])
    # Team 1 scored 118 but lost; team 2 scored 120 and won.
    expected = compute_sos_by_team(
        matchups,
        build_team_strength(stats),
        weeks_past=1,
        users=[],
        regular_season_weeks=2,
    )
    assert by_id["3"]["avg_opp_points"] == round(float(expected["3"]["ros_sos"]), 2)
    assert by_id["4"]["avg_opp_points"] == round(float(expected["4"]["ros_sos"]), 2)


def test_remaining_schedule_ranks_by_sos_blend_not_ppg_only():
    """High win% / slightly lower PPG is a tougher SOS opponent than PPG-only."""
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 118, "3", 90) + _week_pair(2, "2", 120, "4", 130),
        2: _week_pair(1, "3", 0, "1", 0) + _week_pair(2, "4", 0, "2", 0),
    }
    rows, _no_games = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=2,
    )
    by_id = {r["roster_id"]: r for r in rows}
    # PPG-only: 3 faces 118, 4 faces 120 → 4 harder. SOS: 1 won at 118, 2 lost at 120.
    assert by_id["3"]["avg_opp_points"] > by_id["4"]["avg_opp_points"]
    order = [r["roster_id"] for r in rows]
    assert order.index("3") < order.index("4")


def test_remaining_schedule_ignores_playoff_weeks():
    rosters = _rosters("1", "2")
    matchups = {
        1: _week_pair(1, "1", 100, "2", 90),
        2: _week_pair(1, "1", 0, "2", 0),
    }
    rows, _ = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=1,
    )
    assert all(r["games_remaining"] == 0 for r in rows)
    assert all(r["avg_opp_points"] == 0.0 for r in rows)


def test_remaining_schedule_preseason_is_even():
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 0, "2", 0) + _week_pair(2, "3", 0, "4", 0),
    }
    rows, no_games = remaining_schedule_strength(
        rosters, matchups, current_week=0, regular_season_weeks=1,
    )
    assert no_games is True
    values = {r["avg_opp_points"] for r in rows}
    assert values == {100.0}
    assert all(r["games_remaining"] == 1 for r in rows)
