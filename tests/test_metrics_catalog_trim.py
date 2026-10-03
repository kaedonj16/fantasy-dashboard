"""Guards the 2026-09-30 Advanced Metrics catalog trim.

26 redundant metrics were removed from the user-facing catalog
(LEADERBOARD_METRICS and everything derived from it): hard duplicates of a
surviving keeper, season totals whose per-game twin survives, red-zone
per-game counts whose share metrics survive, and trivia/composite entries.
The underlying DB columns, writers, and internal computations intentionally
remain; only the catalog entries are gone. These tests pin the removal so a
removed key cannot quietly come back, and pin that every preset still
points at metrics that exist.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services.pages.advanced_metrics_page import (  # imports after the flask guard
    ADVANCED_METRIC_PRESETS,
)
from data_building.advanced_metrics import (  # imports after the flask guard
    LEADERBOARD_METRICS,
    PRO_METRICS,
)

REMOVED_KEYS = (
    # Hard duplicates (keeper in parentheses):
    # ngs_cpoe (cpoe), catchable_tgt_pct (drop_rate),
    # passing_epa_per_att (epa_per_play), explosive_runs_pg
    # (explosive_run_rate), xfp_stddev (fp_cv)
    "ngs_cpoe",
    "catchable_tgt_pct",
    "passing_epa_per_att",
    "explosive_runs_pg",
    "xfp_stddev",
    # Season totals; the per-game twin survives.
    "total_pass_tds",
    "total_rush_tds",
    "total_rec_tds",
    "total_carries",
    "total_targets",
    "total_receptions",
    "total_touches",
    "total_tds",
    "total_rec_yards",
    "total_rush_yards",
    "explosive_runs_10_plus",
    "avoided_tackles",
    # Red-zone per-game counts; the share metrics survive.
    "rz_targets_pg",
    "rz_carries_pg",
    # Trivia / composite entries.
    "ngs_max_completed_air_distance",
    "ngs_avg_time_to_los",
    "out_of_pocket_rate",
    "pacr",
    "racr",
    "ngs_avg_air_yards_differential",
    "red_zone_usage",
)

# The designated keeper each removed duplicate defers to.
KEEPERS = (
    "cpoe",
    "drop_rate",
    "epa_per_play",
    "explosive_run_rate",
    "fp_cv",
    "pass_tds_per_game",
    "rush_tds_per_game",
    "rec_tds_per_game",
    "carries_per_game",
    "targets_per_game",
    "receptions_per_game",
    "touches_per_game",
    "total_tds_per_game",
    "rec_yards_per_game",
    "rush_yards_per_game",
    "avoided_tackles_per_carry",
    "rz_target_share",
    "rz_opp_share",
)


def test_removed_keys_are_out_of_the_catalog():
    present = [k for k in REMOVED_KEYS if k in LEADERBOARD_METRICS]
    assert present == [], f"trimmed metrics back in catalog: {present}"


def test_removed_keys_are_out_of_pro_metrics():
    present = [k for k in REMOVED_KEYS if k in PRO_METRICS]
    assert present == [], f"trimmed metrics still PRO-gated: {present}"


def test_keepers_survive():
    missing = [k for k in KEEPERS if k not in LEADERBOARD_METRICS]
    assert missing == [], f"designated keepers missing: {missing}"


def test_presets_reference_only_catalog_metrics():
    for name, preset in ADVANCED_METRIC_PRESETS.items():
        assert preset["primary"] in LEADERBOARD_METRICS, (
            f"preset {name} primary {preset['primary']} not in catalog")
        missing = [k for k in preset["metrics"] if k not in LEADERBOARD_METRICS]
        assert missing == [], f"preset {name} references non-catalog: {missing}"
