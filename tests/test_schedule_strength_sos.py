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
    assert "_projected_starter_avgs" in body
    assert "using_projections" in body
    assert "roster_val / 50.0" not in body
    assert "preseason_opponent_strength" not in body
    assert "load_pick_value_table" not in body
    helper = src[src.find("def _projected_starter_avgs"):src.find("def api_schedule_strength")]
    assert "build_ppg_map" in helper
    assert "_position_aware_lineup" in helper
    assert "resolve_projected_ppg_many" in helper
    helper_fn = (ROOT / "dashboard_services" / "service.py").read_text(encoding="utf-8")
    fn = helper_fn[helper_fn.find("def remaining_schedule_strength"):]
    fn = fn[: fn.find("\nfrom collections import defaultdict")]
    assert "build_team_strength" in fn
    assert "compute_sos_by_team" in fn
    assert "projected_avg_by_rid" in fn
    assert "0.65" not in fn  # blend lives in build_team_strength, not a local copy


def test_remaining_schedule_matches_standings_sos_future():
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 118, "2", 120) + _week_pair(2, "3", 90, "4", 110),
        2: _week_pair(1, "3", 0, "1", 0) + _week_pair(2, "4", 0, "2", 0),
    }
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=2,
        roster_names={"1": "A", "2": "B", "3": "C", "4": "D"},
    )
    assert source == "actual"
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
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=2,
    )
    assert source == "actual"
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
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=1,
    )
    assert source == "actual"
    assert all(r["games_remaining"] == 0 for r in rows)
    assert all(r["avg_opp_points"] == 0.0 for r in rows)


def test_projected_starter_avgs_uses_playoff_odds_lineup(monkeypatch):
    pytest.importorskip("flask")
    from routes.schedule_api_bp import _projected_starter_avgs

    def fake_ppg_map(_ctx):
        return (
            {"a": {"ppg": 12.0, "pos": "RB"}, "b": {"ppg": 18.0, "pos": "WR"}},
            {"a": "RB", "b": "WR"},
        )

    def fake_lineup(pids, ppg_map, _pos_map, _slots):
        total = sum(float((ppg_map.get(str(p)) or {}).get("ppg") or 0.0) for p in pids)
        return total, []

    monkeypatch.setattr("data_building.simulate_playoff_odds.build_ppg_map", fake_ppg_map)
    monkeypatch.setattr(
        "data_building.simulate_playoff_odds._position_aware_lineup", fake_lineup
    )
    rosters = [
        {"roster_id": "1", "players": ["a", "b"]},
        {"roster_id": "2", "players": ["a"]},
    ]
    out = _projected_starter_avgs(rosters, {"roster_positions": ["RB", "WR", "BN"]}, 2026, 0)
    assert out["1"] == 30.0
    assert out["2"] == 12.0


def test_remaining_schedule_preseason_is_even_without_projections():
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 0, "2", 0) + _week_pair(2, "3", 0, "4", 0),
    }
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=0, regular_season_weeks=1,
    )
    assert source == "even"
    values = {r["avg_opp_points"] for r in rows}
    assert values == {100.0}
    assert all(r["games_remaining"] == 1 for r in rows)


def test_remaining_schedule_preseason_uses_projected_scoring():
    """Before week 1, remaining SOS ranks by projected starter AVG (win rate even)."""
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "3", 0, "1", 0) + _week_pair(2, "4", 0, "2", 0),
    }
    # Week 1 remaining: 3 faces high-proj 1, 4 faces low-proj 2.
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=0, regular_season_weeks=1,
        projected_avg_by_rid={"1": 140.0, "2": 90.0, "3": 110.0, "4": 100.0},
    )
    assert source == "projected"
    by_id = {r["roster_id"]: r for r in rows}
    assert by_id["3"]["avg_opp_points"] > by_id["4"]["avg_opp_points"]
    assert by_id["3"]["my_avg_points"] == 110.0
    order = [r["roster_id"] for r in rows]
    assert order.index("3") < order.index("4")
    # Not the even-100 placeholder.
    assert {r["avg_opp_points"] for r in rows} != {100.0}


def test_remaining_schedule_ignores_projections_once_games_are_played():
    rosters = _rosters("1", "2", "3", "4")
    matchups = {
        1: _week_pair(1, "1", 118, "2", 120) + _week_pair(2, "3", 90, "4", 110),
        2: _week_pair(1, "3", 0, "1", 0) + _week_pair(2, "4", 0, "2", 0),
    }
    rows, source = remaining_schedule_strength(
        rosters, matchups, current_week=1, regular_season_weeks=2,
        projected_avg_by_rid={"1": 200.0, "2": 10.0, "3": 10.0, "4": 200.0},
    )
    assert source == "actual"
    by_id = {r["roster_id"]: r for r in rows}
    assert by_id["1"]["my_avg_points"] == 118.0
