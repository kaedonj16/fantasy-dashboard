"""Focused contracts for Targets Per Snap surfacing.

The metric itself (season leaderboard, config, modal rank cache) already
existed; this covers the week-range volume-gate fix and the preset /
player-compare surfacing added with it.

Regression background: the leaderboard handler derives its default
min_vol from the SEASON spec (targets_per_snap min_vol is _V_SNAPS,
col total_snaps, first opt 100). The weekly spec used to gate on
SUM(targets) / SUM(touches), so a default week-range query demanded
50-100 targets inside the range and always returned an empty board.
The weekly gate must be in snaps, matching the season units.
"""
import contextlib
from pathlib import Path

from dashboard_services.pages.advanced_metrics_page import ADVANCED_METRIC_PRESETS
from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    PRO_METRICS,
    _WEEKLY_METRICS,
)

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_weekly_per_snap_specs_gate_on_snaps():
    for key in ("targets_per_snap", "touches_per_snap"):
        spec = _WEEKLY_METRICS[key]
        assert spec["min_col"] == "SUM(snaps)", key
        assert spec["min_label"] == "Min Snaps", key
        assert spec["min_opts"] == [50, 100, 200, 400], key


def test_weekly_per_snap_sql_is_unchanged():
    assert _WEEKLY_METRICS["targets_per_snap"]["sql"] == (
        "SUM(targets)::float / NULLIF(SUM(snaps), 0)")
    assert _WEEKLY_METRICS["touches_per_snap"]["sql"] == (
        "SUM(touches)::float / NULLIF(SUM(snaps), 0)")


def test_season_registry_entry_is_intact_and_free():
    spec = LEADERBOARD_METRICS["targets_per_snap"]
    assert spec["label"] == "Targets / Snap"
    assert spec["category"] == "Receiving"
    assert spec["positions"] == ["WR", "TE"]
    assert spec["computed_sql"] == (
        "m.total_targets::float / NULLIF(m.total_snaps, 0)")
    assert "targets_per_snap" not in PRO_METRICS


def test_presets_surface_targets_per_snap_for_receiving_wr_te_only():
    for preset_id in ("receiving", "wr", "te"):
        metrics = ADVANCED_METRIC_PRESETS[preset_id]["metrics"]
        assert "targets_per_snap" in metrics, preset_id
        # Placed directly after target_share, per the surfacing spec.
        assert metrics.index("targets_per_snap") == (
            metrics.index("target_share") + 1), preset_id
    assert "targets_per_snap" not in ADVANCED_METRIC_PRESETS["rb"]["metrics"]


def test_compare_fallback_entries_in_app_js():
    # wrTeMetrics fallback list carries the metric.
    wr_te = APP_JS[APP_JS.index("const wrTeMetrics = ["):]
    wr_te = wr_te[: wr_te.index("];")]
    assert "'targets_per_snap'" in wr_te
    # Fallback label and a realistic compare-bar range (was the default
    # 100, which rendered the bars near-empty for a 0-0.25 rate).
    assert "targets_per_snap:'Targets/Snap'" in APP_JS
    assert "'targets_per_snap': 0.25" in APP_JS


def test_weekly_leaderboard_gates_on_snaps_not_targets(monkeypatch):
    """A default-scaled gate (50) must be applied to in-range snaps.

    Player in range: 40 targets / 280 snaps. Under the old SUM(targets)
    gate a min_vol of 50 would have excluded them; under the snaps gate
    their _wvol is 280 and they pass. The fake conn captures the SQL the
    builder emits and returns the aggregated row a real DB would return
    for that gate.
    """
    import data_building.advanced_metrics as metrics

    captured = {}

    class Conn:
        def execute(self, sql, params=()):
            captured["sql"] = sql
            captured["params"] = params
            return self

        def fetchall(self):
            return [{
                "player_id": "p1", "position": "WR", "weeks_played": 8,
                "value": 40 / 280, "ctx_receptions": 30, "ctx_targets": 40,
                "ctx_carries": 0, "_wvol": 280,
            }]

    @contextlib.contextmanager
    def conn():
        yield Conn()

    monkeypatch.setattr(metrics, "get_conn", conn)
    rows = metrics.get_weekly_range_leaderboard(
        "targets_per_snap", position="WR", season=2026,
        week_start=1, week_end=8, min_vol=50)

    assert "SUM(snaps) AS _wvol" in captured["sql"]
    assert "SUM(targets) AS _wvol" not in captured["sql"]
    assert 50 in captured["params"]
    assert len(rows) == 1
    assert rows[0]["vol"] == 280
    assert rows[0]["value"] == 40 / 280
