"""Focused contracts for Advanced Metrics preset and aggregation metadata."""
from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    _ADV_WEEKLY_DERIVED_METRICS,
    _adv_weekly_agg_sql,
)

EXPECTED = {
    "rushing": ("rushing_epa_per_att", None),
    "receiving": ("target_share", None),
    "passing": ("epa_per_play", "QB"),
    "rb": ("opportunity_share", "RB"),
    "wr": ("target_share", "WR"),
    "te": ("target_share", "TE"),
    "qb": ("epa_per_play", "QB"),
    "general": ("opportunity_share", None),
    "expected": ("expected_ppr_per_game", None),
    "receiving_profile": ("target_share", None),
}


def test_presets_have_valid_ordered_metrics_primary_position_and_sort():
    assert set(ADVANCED_METRIC_PRESETS) == set(EXPECTED)
    for preset_id, (primary, position) in EXPECTED.items():
        preset = ADVANCED_METRIC_PRESETS[preset_id]
        assert preset["primary"] == preset["metrics"][0] == primary
        assert preset["position"] == position
        assert preset["sort"] == "desc"
        assert len(preset["metrics"]) == 7
        assert len(set(preset["metrics"])) == 7
        assert all(key in LEADERBOARD_METRICS for key in preset["metrics"])


def test_wr_and_receiving_rank_target_share_not_yards_per_target():
    assert ADVANCED_METRIC_PRESETS["wr"]["primary"] == "target_share"
    assert ADVANCED_METRIC_PRESETS["receiving"]["primary"] == "target_share"


def test_no_route_metrics_in_presets():
    forbidden = {"total_routes", "routes_per_game", "route_participation", "yprr"}
    selected = {key for preset in ADVANCED_METRIC_PRESETS.values() for key in preset["metrics"]}
    assert selected.isdisjoint(forbidden)
    assert not any("route" in key for key in selected)


def test_efficiency_epa_presets_use_rate_not_cumulative_metric():
    assert ADVANCED_METRIC_PRESETS["rushing"]["primary"] == "rushing_epa_per_att"
    assert "rushing_epa" not in ADVANCED_METRIC_PRESETS["rushing"]["metrics"]
    assert "receiving_epa_per_target" in ADVANCED_METRIC_PRESETS["receiving"]["metrics"]
    assert "receiving_epa" not in ADVANCED_METRIC_PRESETS["receiving"]["metrics"]
    assert LEADERBOARD_METRICS["epa_per_play"]["label"] == "Passing EPA / Dropback"


def test_weighted_epa_sql_uses_only_covered_denominator():
    rush_sql, _ = _adv_weekly_agg_sql("rushing_epa_per_att")
    rec_sql, _ = _adv_weekly_agg_sql("receiving_epa_per_target")
    assert "SUM(rushing_epa_per_att * w_carries)" in rush_sql
    assert "CASE WHEN rushing_epa_per_att IS NOT NULL THEN w_carries END" in rush_sql
    assert "SUM(receiving_epa_per_target * w_targets)" in rec_sql
    assert "CASE WHEN receiving_epa_per_target IS NOT NULL THEN w_targets END" in rec_sql


def test_expected_actual_and_fpoe_share_matched_components():
    actual_sql = _ADV_WEEKLY_DERIVED_METRICS["actual_ppr_per_game"][0]
    expected_sql = _ADV_WEEKLY_DERIVED_METRICS["expected_ppr_per_game"][0]
    fpoe_sql = _ADV_WEEKLY_DERIVED_METRICS["ppr_over_expected_per_game"][0]
    assert "expected_ppr + ppr_over_expected" in actual_sql
    assert "expected_ppr" in expected_sql
    assert "ppr_over_expected" in fpoe_sql
