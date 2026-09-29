"""Contracts for the unrealized/intended air-yards metrics on Advanced Metrics."""

from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    PRO_METRICS,
    PREMIUM_METRICS,
    _V_TARGETS,
)


def test_air_yards_metrics_registered():
    for key, label in (
        ("unrealized_air_yards", "Unrealized Air Yards"),
        ("unrealized_air_yards_per_game", "Unrealized AY/G"),
        ("intended_air_yards_per_game", "Intended AY/G"),
        ("intended_air_yards_share", "Intended AY Share"),
    ):
        spec = LEADERBOARD_METRICS[key]
        assert spec["label"] == label
        assert spec["category"] == "Receiving"
        assert spec["positions"] == ["WR", "TE"]
        assert spec["min_vol"] is _V_TARGETS


def test_air_yards_metrics_are_free():
    for key in (
        "unrealized_air_yards",
        "unrealized_air_yards_per_game",
        "intended_air_yards_per_game",
        "intended_air_yards_share",
    ):
        assert key not in PRO_METRICS
        assert key not in PREMIUM_METRICS


def test_unrealized_sql_references_ngs_columns():
    sql = LEADERBOARD_METRICS["unrealized_air_yards"]["computed_sql"]
    assert "ngs_avg_intended_air_yards" in sql
    assert "total_targets" in sql
    assert "yards_per_reception" in sql
    assert "total_receptions" in sql
    assert LEADERBOARD_METRICS["unrealized_air_yards"]["computed_null"]


def test_unrealized_per_game_sql_divides_by_games():
    sql = LEADERBOARD_METRICS["unrealized_air_yards_per_game"]["computed_sql"]
    assert "NULLIF(m.games, 0)" in sql
    assert "ngs_avg_intended_air_yards" in sql


def test_intended_per_game_sql():
    sql = LEADERBOARD_METRICS["intended_air_yards_per_game"]["computed_sql"]
    assert "ngs_avg_intended_air_yards" in sql
    assert "total_targets" in sql
    assert "NULLIF(m.games, 0)" in sql


def test_intended_share_maps_to_ngs_column():
    # The DB column is ngs_pct_share_intended_air_yards; the metric key must
    # resolve to it via computed_sql (the leaderboard uses the key as a column
    # name when computed_sql is absent).
    spec = LEADERBOARD_METRICS["intended_air_yards_share"]
    assert spec["computed_sql"] == "m.ngs_pct_share_intended_air_yards"
    assert spec["computed_null"] == "m.ngs_pct_share_intended_air_yards IS NOT NULL"
    assert spec["pct"] is True


def _unrealized_py(row):
    """Python mirror of the unrealized_air_yards computed_sql."""
    intended = (row.get("ngs_avg_intended_air_yards") or 0) * (row.get("total_targets") or 0)
    earned = (row.get("yards_per_reception") or 0) * (row.get("total_receptions") or 0)
    return intended - earned


def test_unrealized_formula_math():
    # 10.0 intended AY/target x 100 targets = 1000 intended; 12.5 YPR x 60 rec = 750.
    row = {
        "ngs_avg_intended_air_yards": 10.0,
        "total_targets": 100,
        "yards_per_reception": 12.5,
        "total_receptions": 60,
    }
    assert _unrealized_py(row) == 250.0
    # Missing NGS data -> None (computed_null keeps the row off the board),
    # and the COALESCE form never goes negative from NULLs.
    assert _unrealized_py({}) == 0.0
