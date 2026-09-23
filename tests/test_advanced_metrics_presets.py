"""Focused contracts for Advanced Metrics preset and aggregation metadata."""
import pytest

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

# Decision presets: (primary metric, position filter, sort direction).
DECISION_EXPECTED = {
    "key_metrics": ("expected_ppr_per_game", None, "desc"),
    "start_sit": ("expected_ppr_per_game", None, "desc"),
    "buy_low_sell_high": ("ppr_over_expected_per_game", None, "asc"),
    "waiver_wire": ("opportunity_trend", None, "desc"),
    "breakout_check": ("xfp_trend", None, "desc"),
    "ceiling_dfs": ("boom_rate", None, "desc"),
}


def test_presets_have_valid_ordered_metrics_primary_position_and_sort():
    assert set(ADVANCED_METRIC_PRESETS) == set(EXPECTED) | set(DECISION_EXPECTED)
    for preset_id, (primary, position) in EXPECTED.items():
        preset = ADVANCED_METRIC_PRESETS[preset_id]
        assert preset["primary"] == preset["metrics"][0] == primary
        assert preset["position"] == position
        assert preset["sort"] == "desc"
        assert preset.get("kind") != "decision"
        assert len(preset["metrics"]) == 7
        assert len(set(preset["metrics"])) == 7
        assert all(key in LEADERBOARD_METRICS for key in preset["metrics"])


def test_decision_presets_have_valid_metrics_taglines_and_sort():
    decision_ids = {k for k, p in ADVANCED_METRIC_PRESETS.items() if p.get("kind") == "decision"}
    assert decision_ids == set(DECISION_EXPECTED)
    for preset_id, (primary, position, sort) in DECISION_EXPECTED.items():
        preset = ADVANCED_METRIC_PRESETS[preset_id]
        assert preset["primary"] == preset["metrics"][0] == primary
        assert preset["position"] == position
        assert preset["sort"] == sort
        assert preset.get("tagline"), preset_id
        assert 5 <= len(preset["metrics"]) == len(set(preset["metrics"])) <= 10
        assert all(key in LEADERBOARD_METRICS for key in preset["metrics"])


def test_buy_low_preset_sorts_ascending_for_most_negative_first():
    # The whole point of the preset: most negative FPOE (buy low) on top.
    assert ADVANCED_METRIC_PRESETS["buy_low_sell_high"]["sort"] == "asc"
    assert ADVANCED_METRIC_PRESETS["buy_low_sell_high"]["primary"] == "ppr_over_expected_per_game"


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


def test_decision_pills_tagline_and_movers_are_wired_in_js():
    # Pills render from cfg.presets and drive amLoadPreset / amClearDecision.
    assert "amDecisionPills" in _AM_JS
    assert "_updateDecisionUI" in _AM_JS
    assert "amClearDecision" in _AM_JS
    assert "amPresetTagline" in _AM_JS
    # Last-used decision view is remembered across visits.
    assert "localStorage.getItem('amLastPreset')" in _AM_JS
    assert "localStorage.setItem('amLastPreset'" in _AM_JS
    # Movers strip fetches the dedicated endpoint and renders three groups.
    assert "_loadMovers" in _AM_JS
    assert "/api/advanced-metrics/movers" in _AM_JS
    assert "Heating up" in _AM_JS and "Cooling off" in _AM_JS and "Efficiency outliers" in _AM_JS


def test_key_metrics_is_the_default_landing_view():
    # Fresh loads (no ?preset=) land on the last-used decision view, else Key Metrics.
    assert "'key_metrics'" in _AM_JS
    assert "amLastPreset" in _AM_JS
    assert "_PRESETS[_landing]" in _AM_JS


def test_preset_choice_is_shareable_via_url():
    assert "p.set('preset', _activePresetId)" in _AM_JS
    assert "_initParams.get('preset')" in _AM_JS


def test_schedule_ease_is_displayable_for_start_sit():
    # The Start/Sit matchup column needs Schedule Ease visible to the table.
    from data_building.advanced_metrics import LEADERBOARD_METRICS
    assert not LEADERBOARD_METRICS["schedule_ease"].get("hidden")
    assert "schedule_ease" in ADVANCED_METRIC_PRESETS["start_sit"]["metrics"]


def test_movers_endpoint_shape():
    import inspect
    from routes.advanced_metrics_bp import api_advanced_metrics_movers
    source = inspect.getsource(api_advanced_metrics_movers)
    assert '"heating"' in source and '"cooling"' in source and '"outliers"' in source
    assert "opportunity_trend" in source
    assert "ppr_over_expected_per_game" in source


def test_pro_metrics_are_valid_leaderboard_keys():
    from data_building.advanced_metrics import PRO_METRICS
    assert isinstance(PRO_METRICS, frozenset) and len(PRO_METRICS) >= 5
    assert all(key in LEADERBOARD_METRICS for key in PRO_METRICS)


def test_pro_presets_are_the_advanced_three():
    # Product scope: "the data is free, the answers are PRO." Key Metrics,
    # Start / Sit, and Ceiling / DFS stay free (their primaries —
    # expected_ppr_per_game and boom_rate — are free; boom/bust are already
    # visible to everyone on the Start / Sit compare table). The three
    # remaining advanced decision presets are PRO-locked.
    from data_building.advanced_metrics import PRO_METRICS
    decision = {k: p for k, p in ADVANCED_METRIC_PRESETS.items() if p.get("kind") == "decision"}
    assert len(decision) == 6
    locked = {pid for pid, p in decision.items() if p["primary"] in PRO_METRICS}
    assert locked == {"buy_low_sell_high", "waiver_wire", "breakout_check"}
    assert "expected_ppr_per_game" not in PRO_METRICS
    assert "boom_rate" not in PRO_METRICS
    assert "bust_rate" not in PRO_METRICS


def test_free_page_locks_pro_presets_and_strips_pro_metrics():
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    # The three advanced presets are locked; Key Metrics, Start / Sit, and
    # Ceiling / DFS are not (boom_rate is free — it's already on Start / Sit).
    for pid in ("buy_low_sell_high", "waiver_wire", "breakout_check"):
        assert 'data-preset="%s" data-locked="1"' % pid in html, pid
    assert 'data-preset="key_metrics" data-locked="0"' in html
    assert 'data-preset="start_sit" data-locked="0"' in html
    assert 'data-preset="ceiling_dfs" data-locked="0"' in html
    assert "🔒" in html
    # PRO metrics are not offered in the free picker; expected_ppr_per_game
    # (the free default view's anchor) and boom/bust rates (already visible
    # on the Start / Sit compare table) are.
    assert 'value="expected_ppr_per_game"' in html
    assert 'value="boom_rate"' in html
    assert 'value="bust_rate"' in html
    assert 'value="ppr_over_expected_per_game"' not in html
    assert 'value="opportunity_trend"' not in html
    # Raw metrics stay free.
    assert 'value="opportunity_share"' in html
    assert 'value="target_share"' in html


def test_premium_page_has_no_locks_and_full_picker():
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    html = build_advanced_metrics_body(True, LEADERBOARD_METRICS)
    assert 'data-locked="1"' not in html
    assert 'value="expected_ppr_per_game"' in html


def test_pro_gating_is_wired_in_js():
    assert "cfg.proPresets" in _AM_JS
    assert "_isProPreset" in _AM_JS
    assert "showPaywall('advanced-metrics-'" in _AM_JS
    # Movers strip hidden for free users.
    assert "host.style.display = 'none'; return;" in _AM_JS
    # 403 pro_only opens the paywall and falls back to a free metric.
    assert "_proOnly" in _AM_JS
    assert "state.metric = 'opportunity_share'" in _AM_JS


def test_config_marks_pro_metrics(offline_client):
    resp = offline_client.get("/api/advanced-metrics/config")
    assert resp.status_code == 200
    metrics = resp.get_json()["metrics"]
    # expected_ppr_per_game anchors the free Key Metrics / Start-Sit views.
    # boom_rate/bust_rate are free: the Start / Sit compare table already
    # shows them to everyone.
    assert metrics["expected_ppr_per_game"]["pro"] is False
    assert metrics["boom_rate"]["pro"] is False
    assert metrics["bust_rate"]["pro"] is False
    assert metrics["ppr_over_expected_per_game"]["pro"] is True
    assert metrics["opportunity_trend"]["pro"] is True
    assert metrics["opportunity_share"]["pro"] is False


def test_leaderboard_pro_metric_403_for_non_premium(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: False)
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=ppr_over_expected_per_game&season=2026")
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "pro_only"


def test_leaderboard_expected_ppr_is_free_for_non_premium(offline_client, monkeypatch):
    # The flagship metric stays free: it anchors Key Metrics / Start-Sit.
    import routes.advanced_metrics_bp as bp
    import data_building.advanced_metrics as am
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: False)
    monkeypatch.setattr(am, "get_metric_leaderboard", lambda *a, **k: [])
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=expected_ppr_per_game&season=2026")
    assert resp.status_code == 200, resp.get_data(as_text=True)


def test_leaderboard_free_metric_ok_for_non_premium(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    import data_building.advanced_metrics as am
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: False)
    monkeypatch.setattr(am, "get_metric_leaderboard", lambda *a, **k: [])
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=opportunity_share&season=2026")
    assert resp.status_code == 200, resp.get_data(as_text=True)


def test_leaderboard_pro_metric_ok_for_premium(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    import data_building.advanced_metrics as am
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: True)
    monkeypatch.setattr(am, "get_metric_leaderboard", lambda *a, **k: [])
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=ppr_over_expected_per_game&season=2026")
    assert resp.status_code == 200, resp.get_data(as_text=True)


def test_movers_endpoint_is_pro_only(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: False)
    resp = offline_client.get("/api/advanced-metrics/movers?season=2026")
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "pro_only"


# --- Degenerate trend windows: weeks 1-3 must not render a false 0.0 flat ---

def test_recent_vs_season_delta_is_none_for_degenerate_windows():
    from data_building.weekly_metrics import _recent_vs_season_delta
    assert _recent_vs_season_delta([]) is None
    assert _recent_vs_season_delta([4.0]) is None
    assert _recent_vs_season_delta([4.0, 7.0]) is None
    assert _recent_vs_season_delta([4.0, 7.0, 5.0]) is None


def test_recent_vs_season_delta_computes_from_week_four():
    from data_building.weekly_metrics import _recent_vs_season_delta
    # vals [2,2,2,8]: season avg 3.5, last-3 avg 4.0 -> +0.5
    assert _recent_vs_season_delta([2.0, 2.0, 2.0, 8.0]) == 0.5
    # declining usage: season avg 6.5, last-3 avg 6.0 -> -0.5
    assert _recent_vs_season_delta([8.0, 8.0, 8.0, 2.0]) == -0.5


def test_recent_vs_season_ratio_is_none_for_degenerate_windows():
    from data_building.advanced_metrics import _recent_vs_season_ratio
    assert _recent_vs_season_ratio([]) is None
    assert _recent_vs_season_ratio([1.0, 2.0]) is None
    assert _recent_vs_season_ratio([1.0, 2.0, 3.0]) is None
    # 4+ samples compute normally: last-3 avg 3.0 vs season avg 2.25 -> +1/3
    assert _recent_vs_season_ratio([0.0, 2.0, 3.0, 4.0]) == pytest.approx(1 / 3)


# --- PRO-locked extra columns render a lock, not blank cells ---

def test_pro_locked_column_js_wiring():
    # 403 on an extra-metric fetch marks the column proLocked; the column
    # renders lock cells that open the paywall on tap.
    assert "proLocked" in _AM_JS
    assert "data-pro-metric" in _AM_JS
    assert "showPaywall('advanced-metrics-metric-' +" in _AM_JS


def test_page_metrics_config_marks_pro_flag():
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    # The premium page includes every metric, each carrying its pro flag.
    html = build_advanced_metrics_body(True, LEADERBOARD_METRICS)
    assert '"pro": true' in html
    assert '"pro": false' in html
    # The free page strips PRO metrics from the picker config entirely.
    free_html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    assert '"pro": true' not in free_html
