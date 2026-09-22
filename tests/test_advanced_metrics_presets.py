"""Focused contracts for Advanced Metrics preset and aggregation metadata."""
from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
from dashboard_services.pages.advanced_metrics_page import _AM_JS
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


def test_preset_context_counts_are_ordered_and_complete():
    assert ADVANCED_METRIC_PRESETS["rushing"]["samples"] == ["games", "carries"]
    assert ADVANCED_METRIC_PRESETS["receiving"]["samples"] == ["games", "targets", "receptions"]
    assert ADVANCED_METRIC_PRESETS["wr"]["samples"] == ["games", "targets", "receptions"]
    assert ADVANCED_METRIC_PRESETS["te"]["samples"] == ["games", "targets", "receptions"]
    assert ADVANCED_METRIC_PRESETS["rb"]["samples"] == ["games", "carries", "targets", "receptions"]
    assert ADVANCED_METRIC_PRESETS["passing"]["samples"] == ["games", "attempts", "dropbacks"]
    assert ADVANCED_METRIC_PRESETS["qb"]["samples"] == ["games", "attempts", "dropbacks", "carries"]


def test_table_uses_one_semantic_schema_for_headers_rows_skeletons_and_exports():
    assert "function tableColumnSchema()" in _AM_JS
    assert "tableColumnSchema().forEach" in _AM_JS
    assert "data-column-id=\"' + c.id" in _AM_JS
    assert "schemaSkeletonRows()" in _AM_JS
    assert "const sampleCols = contextColsFor();" in _AM_JS
    # Regression: never relabel the Games header from the primary metric's
    # volume field. That was the extra CAR/Att header which shifted perception.
    assert "VOL_LABELS[state.volCol]" not in _AM_JS


def test_distinct_fixture_values_bind_to_semantic_metric_keys():
    # Primary and comparison values are looked up from the schema's metric key,
    # never from a neighboring sample cell or a parallel positional array.
    assert "function schemaValue(column, row)" in _AM_JS
    assert "if (column.metricKey === state.metric) return row.value" in _AM_JS
    assert "state.extraData[column.metricKey]" in _AM_JS
    assert "data-column-id=\"metric:' + state.metric" in _AM_JS
    assert "data-column-id=\"metric:' + key" in _AM_JS


def test_context_values_preserve_zero_and_do_not_alias_attempts_to_dropbacks():
    assert "if (row[key] != null) return row[key]" in _AM_JS
    assert "attempts:{key:'att'" in _AM_JS
    assert "dropbacks:{key:'db'" in _AM_JS
    assert "r.att || r.db" not in _AM_JS
    assert "r.db || r.att" not in _AM_JS


def test_stale_schema_responses_are_discarded():
    assert "requestToken: 0" in _AM_JS
    assert "requestToken !== state.requestToken" in _AM_JS


def test_season_context_counts_resolve_across_provider_snapshot_dates():
    import inspect
    from data_building.advanced_metrics import get_metric_leaderboard

    source = inspect.getsource(get_metric_leaderboard)
    assert "SELECT MAX(cx.{col})" in source
    assert "cx.player_id=m.player_id AND cx.season=m.season" in source
