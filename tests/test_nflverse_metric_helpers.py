"""Unit tests for nflverse metric helpers (no Flask / nfl_data_py)."""
from data_building.external_data.nflverse_metrics import (
    NGS_FLOOR,
    _apply_created_separation,
    _created_separation,
    _flag,
    _rate_pct,
    build_ngs_passing_for_season,
    build_ngs_receiving_for_season,
    build_ngs_rushing_for_season,
    _aggregate_ngs_weekly_records,
)


def test_rate_pct():
    assert _rate_pct(3, 10) == 30.0
    assert _rate_pct(0, 0) is None
    assert _rate_pct(1, 3, digits=1) == 33.3


def test_flag():
    assert _flag(True) == 1.0
    assert _flag(False) == 0.0
    assert _flag(None) == 0.0
    assert _flag(1) == 1.0
    assert _flag(0) == 0.0


def test_created_separation():
    assert _created_separation(3.2, 5.8) == round(3.2 - 5.8, 2)
    assert _created_separation(None, 5.0) is None
    assert _created_separation(3.0, None) is None


def test_apply_created_separation_mutates_row():
    row = {"ngs_avg_separation": 2.5, "ngs_avg_cushion": 6.0}
    _apply_created_separation(row)
    assert row["ngs_created_separation"] == round(2.5 - 6.0, 2)
    empty = {"ngs_avg_separation": 2.5}
    _apply_created_separation(empty)
    assert "ngs_created_separation" not in empty


def test_ngs_builders_skip_below_floor():
    assert NGS_FLOOR == 2016
    assert build_ngs_passing_for_season(2015) == {}
    assert build_ngs_receiving_for_season(2015) == {}
    assert build_ngs_rushing_for_season(2015) == {}


def test_ngs_weekly_fallback_uses_volume_weights_and_max():
    weekly = [
        {"player_gsis_id": "p1", "season_type": "REG", "week": 1,
         "attempts": 10, "avg_time_to_throw": 2.0,
         "max_completed_air_distance": 20},
        {"player_gsis_id": "p1", "season_type": "REG", "week": 2,
         "attempts": 30, "avg_time_to_throw": 4.0,
         "max_completed_air_distance": 50},
    ]
    row = _aggregate_ngs_weekly_records(weekly, "passing")[0]
    assert row["avg_time_to_throw"] == 3.5
    assert row["max_completed_air_distance"] == 50
