"""Compare Usage tab: ONE overlaid head-to-head panel (2-player compare).

Replaces the two side-by-side weekly columns (#compareWeekly1/2) with a
single #compareUsagePanel rendered by cmpRenderTrends() in app.js, using
pmSparklineDuo + buildCompareWeeklyTrendRows / buildCompareSeasonTrendRows
from player_modal.js. Both players share one scale per metric; the series
align on the union of week numbers, so a bye/DNP week (no API row) becomes
a gap in that player's line, never a zero point.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
PM_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

_node_required = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available")


def _extract(js, start_marker, end_marker):
    start = js.index(start_marker)
    end = js.index(end_marker, start)
    return js[start:end]


def _compare_src():
    """The real compare-overlay functions from player_modal.js, for node."""
    parts = [
        _extract(PM_JS, "function pmStFmt", "function pmSeasonTrendChartHTML"),
        _extract(PM_JS, "function pmSparklineDuo",
                 "// Weekly datapoint hover tooltip"),
        _extract(PM_JS, "// ── Compare usage overlay: weekly metric specs",
                 "// ── Player modal: TRENDING hot/cold section"),
    ]
    prelude = (
        "var _pmEsc = function(s) { return String(s)"
        ".replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); };\n"
    )
    return prelude + "\n".join(parts)


def _run_node(driver):
    res = subprocess.run(["node", "-e", _compare_src() + driver],
                         capture_output=True, text=True, timeout=20,
                         check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _wk(week, **kw):
    base = {"week": week, "opponent": "BUF", "snap_pct": 0, "targets": 0,
            "target_share": 0, "touches": 0, "carry_share": 0, "carries": 0,
            "rush_yards": 0, "receptions": 0, "rec_yards": 0, "rz_targets": 0,
            "rz_carries": 0, "ppr_pts": 0}
    base.update(kw)
    return base


# Player A: 4 full weeks. Player B: misses week 3 (bye -- no row at all).
GIBBS = [
    _wk(1, snap_pct=70, targets=5, target_share=20, touches=20, carry_share=60,
        carries=12, rush_yards=60, receptions=4, rec_yards=30, rz_targets=1,
        rz_carries=2, ppr_pts=18),
    _wk(2, snap_pct=80, targets=7, target_share=28, touches=24, carry_share=70,
        carries=16, rush_yards=80, receptions=6, rec_yards=50, rz_targets=2,
        rz_carries=2, ppr_pts=24),
    _wk(3, snap_pct=75, targets=6, target_share=24, touches=22, carry_share=65,
        carries=14, rush_yards=70, receptions=5, rec_yards=40, rz_targets=1,
        rz_carries=1, ppr_pts=21),
    _wk(4, snap_pct=85, targets=8, target_share=32, touches=26, carry_share=75,
        carries=18, rush_yards=90, receptions=7, rec_yards=60, rz_targets=3,
        rz_carries=2, ppr_pts=27),
]
WALKER = [
    _wk(1, snap_pct=60, targets=3, target_share=12, touches=18, carry_share=55,
        carries=14, rush_yards=70, receptions=2, rec_yards=10, rz_targets=0,
        rz_carries=2, ppr_pts=15),
    _wk(2, snap_pct=62, targets=4, target_share=14, touches=19, carry_share=58,
        carries=15, rush_yards=75, receptions=3, rec_yards=15, rz_targets=1,
        rz_carries=2, ppr_pts=16.5),
    _wk(4, snap_pct=58, targets=2, target_share=10, touches=17, carry_share=52,
        carries=13, rush_yards=65, receptions=1, rec_yards=5, rz_targets=0,
        rz_carries=1, ppr_pts=13.5),
]


@_node_required
def test_duo_shared_scale_and_gap_splits_line():
    out = _run_node(
        "\nconst html = pmSparklineDuo([10, 20, null, 40, 50],"
        " [10, 20, 30, 40, 50], '#111111', '#222222',"
        " ['a1','a2',null,'a4','a5'], ['b1','b2','b3','b4','b5']);\n"
        "process.stdout.write(JSON.stringify({\n"
        "  polylines: (html.match(/<polyline/g) || []).length,\n"
        "  dashes: (html.match(/stroke-dasharray/g) || []).length,\n"
        "  dots: (html.match(/class=\"wk-dot\"/g) || []).length,\n"
        "  html: html }));\n"
    )
    # Gapped series A splits into 2 segments; contiguous B is 1 line.
    assert out["polylines"] == 3
    # One dashed average line per series.
    assert out["dashes"] == 2
    # Hover dots only on present points: A has 4, B has 5.
    assert out["dots"] == 9
    # No area fill (the single-series pmSparkline gradient is absent).
    assert "linearGradient" not in out["html"]


@_node_required
def test_compare_weekly_rows_align_bye_week_and_math():
    driver = (
        "\nconst html = buildCompareWeeklyTrendRows("
        + json.dumps(GIBBS) + ", " + json.dumps(WALKER)
        + ", 'RB', 'RB', 'Jahmyr Gibbs', 'Kenneth Walker III',"
        " '#0076b6', '#12284b');\n"
        "process.stdout.write(JSON.stringify({ html: html }));\n"
    )
    html = _run_node(driver)["html"]

    # Target % is a row for an RB pair, with averages from target_share:
    # Gibbs (20+28+24+32)/4 = 26.0, Walker (12+14+10)/3 = 12.0.
    assert "Target %" in html
    assert "26.0%" in html and "12.0%" in html
    # Snap % averages and edge: 77.5 vs 60.0 -> Gibbs +17.5.
    assert "77.5%" in html and "60.0%" in html
    assert "Gibbs +17.5" in html
    # Carries edge: 15.0 vs 14.0.
    assert "Gibbs +1.0" in html
    # Yds/Carry is 5.0 on both sides -> Even, no lead counted for it.
    assert "Even" in html
    # Summary: Gibbs leads every decided category (13 of 14 rows).
    assert "leads 13 of 14" in html
    # Walker has NO week-3 point (bye), and no zero is plotted for him:
    # his W4 snap tip is present, a W3 tip is not.
    assert "Walker · Wk 4 · 58%" in html
    assert "Walker · Wk 3" not in html
    # Gibbs W3 exists with opponent.
    assert "Gibbs · Wk 3 · 75% vs BUF" in html
    # LG = last present week: Gibbs snap LG 85 (W4), Walker snap LG 58 (W4,
    # his last present week -- not a W3 zero).
    assert "LG: 85.0%" in html and "LG: 58.0%" in html
    # Total hover dots: 14 rows x (4 Gibbs weeks + 3 Walker weeks).
    assert html.count('class="wk-dot"') == 14 * 7
    # Footer carries the range + gap explanation.
    assert "Wks 1&ndash;4" in html and "gaps = bye / DNP" in html


@_node_required
def test_compare_weekly_row_dropped_when_both_sides_zero():
    # Nobody rushes: zero out rushing fields on both sides -> the Rush Yds
    # and Yds/Carry rows must not render at all.
    a = [dict(w, rush_yards=0) for w in GIBBS]
    b = [dict(w, rush_yards=0) for w in WALKER]
    driver = (
        "\nconst html = buildCompareWeeklyTrendRows("
        + json.dumps(a) + ", " + json.dumps(b)
        + ", 'RB', 'RB', 'Jahmyr Gibbs', 'Kenneth Walker III',"
        " '#0076b6', '#12284b');\n"
        "process.stdout.write(JSON.stringify({ html: html }));\n"
    )
    html = _run_node(driver)["html"]
    assert "Rush Yds" not in html
    assert "Yds/Carry" not in html
    # Other rows survive.
    assert "Snap %" in html and "PPR Pts" in html


@_node_required
def test_compare_weekly_missing_side_names_that_side():
    driver = (
        "\nconst html = buildCompareWeeklyTrendRows([], "
        + json.dumps(WALKER)
        + ", 'RB', 'RB', 'Jahmyr Gibbs', 'Kenneth Walker III',"
        " '#0076b6', '#12284b');\n"
        "process.stdout.write(JSON.stringify({ html: html }));\n"
    )
    html = _run_node(driver)["html"]
    assert "Not enough weekly data for Jahmyr Gibbs this season." in html


@_node_required
def test_compare_season_rows_merge_and_latest_edge():
    opts = [{"key": "ppg", "label": "PPR PPG"},
            {"key": "snap", "label": "Snap %", "pct": True}]
    s1 = {"ppg": [{"season": 2023, "value": 15.0}, {"season": 2024, "value": 19.0}],
          "snap": [{"season": 2024, "value": 70.0}]}
    s2 = {"ppg": [{"season": 2023, "value": 12.0}, {"season": 2024, "value": 14.0}],
          "snap": [{"season": 2023, "value": 55.0}, {"season": 2024, "value": 60.0}]}
    driver = (
        "\nconst html = buildCompareSeasonTrendRows("
        + json.dumps(opts) + ", " + json.dumps(s1) + ", [], " + json.dumps(s2)
        + ", 'Jahmyr Gibbs', 'Kenneth Walker III', '#0076b6', '#12284b');\n"
        "process.stdout.write(JSON.stringify({ html: html }));\n"
    )
    html = _run_node(driver)["html"]
    # Both metrics render (snap qualifies via player 2's two seasons, and
    # player 1's single season still shows its latest value + dot).
    assert "PPR PPG" in html and "Snap %" in html
    # Latest-value edge on PPG: 19.0 vs 14.0.
    assert "Gibbs +5" in html
    assert "2023&ndash;2024" in html


def test_app_js_single_overlay_panel_contract():
    assert 'id="compareUsagePanel"' in APP_JS
    assert "compareWeekly1" not in APP_JS
    assert "compareWeekly2" not in APP_JS
    # One render entry point, no per-side calls left.
    assert "cmpRenderTrends(1)" not in APP_JS
    assert "cmpRenderTrends(2)" not in APP_JS
    metrics = APP_JS[APP_JS.index("function cmpRenderMetrics()"):
                     APP_JS.index("function cmpRenderWeekly(which)")]
    assert "cmpRenderTrends()" in metrics
    # Wrappers kept for the slicing tests / legacy retry closures.
    assert "async function cmpRenderWeekly(which)" in APP_JS
    assert "async function cmpRenderSeason(which)" in APP_JS
    # Tier averages get the shared note, teams are stashed for line colors.
    assert "Usage trends aren\\'t shown for tier averages." in APP_JS
    assert "_cmpSides[1].team = p1.team || '';" in APP_JS
    assert "_cmpSides[2].team = p2.team || '';" in APP_JS
    assert "function _cmpUsageColor(side, team, otherColor)" in APP_JS


def test_compare_usage_css_grid_and_restack():
    assert ".cmp-u-row" in CSS and ".cmp-u-summary" in CSS
    assert "grid-template-columns: 104px 86px 1fr 86px 96px;" in CSS
    assert '"metric metric edge" "val1 chart val2"' in CSS
    assert "grid-template-columns: 64px 1fr 64px;" in CSS


def test_builders_live_in_player_modal_js():
    assert "function pmSparklineDuo(seriesA, seriesB, colorA, colorB, tipsA, tipsB)" in PM_JS
    assert "function buildCompareWeeklyTrendRows(" in PM_JS
    assert "function buildCompareSeasonTrendRows(" in PM_JS
    # The single-player builders are untouched entry points.
    assert "function buildWeeklyTrendRows(weeks, position)" in PM_JS
    assert "function buildSeasonTrendRows(options, series, position)" in PM_JS
