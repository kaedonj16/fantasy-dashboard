"""Guards the public (non-PFF) nflverse advanced-metric catalog wiring."""
import pytest

pytest.importorskip("flask")

from data_building.advanced_metrics import (  # noqa: E402
    LEADERBOARD_METRICS,
    PREMIUM_METRICS,
    WEEKLY_ADV_METRIC_COLS,
    WEEKLY_ADV_WEIGHT_COLS,
    _ADV_WEEKLY_WEIGHTED_METRICS,
    _ADV_WEEKLY_TOTAL_METRICS,
)


PUBLIC_NEW_METRICS = (
    "ngs_avg_time_to_throw",
    "ngs_aggressiveness",
    "ngs_avg_completed_air_yards",
    "ngs_avg_air_yards_differential",
    "ngs_avg_air_yards_to_sticks",
    "ngs_cpoe",
    "ngs_max_completed_air_distance",
    "ngs_avg_time_to_los",
    "ngs_percent_attempts_gte_eight_defenders",
    "ngs_created_separation",
    "play_action_rate",
    "play_action_epa",
    "out_of_pocket_rate",
    "blitz_rate_faced",
    "epa_vs_blitz",
    "epa_vs_stacked_box",
    "rushing_success_rate",
    "receiving_success_rate",
    "rushing_epa_per_att",
    "receiving_epa_per_target",
    "qb_hit_rate",
    "explosive_pass_rate",
    "pacr",
    "racr",
)


def test_new_metrics_are_in_leaderboard_and_public():
    missing = [m for m in PUBLIC_NEW_METRICS if m not in LEADERBOARD_METRICS]
    assert missing == [], f"missing leaderboard specs: {missing}"
    leaked = [m for m in PUBLIC_NEW_METRICS if m in PREMIUM_METRICS]
    assert leaked == [], f"public metrics gated as premium: {leaked}"
    for key in PUBLIC_NEW_METRICS:
        spec = LEADERBOARD_METRICS[key]
        assert spec.get("label"), key
        assert spec.get("category") in {"Passing", "Rushing", "Receiving"}, key
        assert spec.get("positions"), key
        assert spec.get("desc"), key


def test_new_metrics_are_week_filterable():
    missing_cols = [m for m in PUBLIC_NEW_METRICS if m not in WEEKLY_ADV_METRIC_COLS]
    assert missing_cols == [], missing_cols
    missing_weights = [
        m for m in PUBLIC_NEW_METRICS
        if m not in _ADV_WEEKLY_TOTAL_METRICS and m not in _ADV_WEEKLY_WEIGHTED_METRICS
    ]
    assert missing_weights == [], missing_weights
    for col in ("w_pass_air_yards", "w_rec_air_yards"):
        assert col in WEEKLY_ADV_WEIGHT_COLS


def test_pacr_and_racr_weight_by_air_yards():
    assert _ADV_WEEKLY_WEIGHTED_METRICS["pacr"] == "w_pass_air_yards"
    assert _ADV_WEEKLY_WEIGHTED_METRICS["racr"] == "w_rec_air_yards"
