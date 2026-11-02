"""Wave 2 of the chart redesign: team modal Weekly Scoring + z-bars, the
uniform team palette, League Graphs payloads, the History single renderer,
and the Teams page dead-renderer deletion.

Source pins follow the repo's JS-test conventions; payload tests construct
DataFrames by hand (auto-marked integration via importorskip("pandas")).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "static" / "app.js").read_text()
GRAPHS_PY = (ROOT / "dashboard_services" / "pages" / "graphs_page.py").read_text()
HISTORY_PY = (ROOT / "dashboard_services" / "pages" / "history_page.py").read_text()
TEAMS_PY = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text()

pd = pytest.importorskip("pandas")


# ── Palette: one color per team, viewer in the navy slot ────────────────────

def test_palette_slots_match_client_constant():
    from dashboard_services.pages.graphs_page import TEAM_PALETTE

    m = re.search(r"var SLOTS = \[([^\]]+)\]", APP)
    assert m, "BRTeamPalette SLOTS not found in app.js"
    client = [s.strip().strip("'\"") for s in m.group(1).split(",")]
    assert client == TEAM_PALETTE
    assert TEAM_PALETTE[0] == "#122d4b"  # navy viewer slot


def test_palette_assignment_is_stable_and_viewer_first():
    from dashboard_services.pages.graphs_page import owner_color_map, owner_slot_map

    owners = ["Alpha", "Bravo", "Charlie"]
    slots = owner_slot_map(owners, viewer="Bravo")
    assert slots["Bravo"] == 0
    # Same inputs -> same assignment, every chart, every call.
    assert owner_slot_map(owners, viewer="Bravo") == slots
    assert owner_color_map(owners, viewer="Bravo")["Bravo"] == "#122d4b"
    # A different chart asking in a different owner order still gives the
    # viewer slot 0 and never double-books a slot.
    slots2 = owner_slot_map(["Charlie", "Bravo", "Alpha"], viewer="Bravo")
    assert slots2["Bravo"] == 0
    assert len(set(slots2.values())) == 3


def test_old_rainbow_cycle_retired():
    # The old 8-color COLOR_CYCLE is gone (its unique hexes survive nowhere
    # in this module; positional POS_COLORS are a separate, untouched
    # non-Plotly palette).
    assert "#eb6834" not in GRAPHS_PY
    assert "COLOR_CYCLE = TEAM_PALETTE" in GRAPHS_PY
    assert "go.Figure" not in GRAPHS_PY and "go.Scatter" not in GRAPHS_PY
    assert "apply_brand_layout" not in GRAPHS_PY


# ── Team modal conversions ──────────────────────────────────────────────────

def test_team_modal_charts_have_no_plotly_calls():
    assert "Plotly.newPlot('teamWeeklyChart'" not in APP
    assert "Plotly.newPlot('teamRadarChart'" not in APP
    assert "Plotly.newPlot('historyChartPlotly'" not in APP
    assert "scatterpolar" not in APP


def test_weekly_scoring_renders_through_brvaluechart():
    start = APP.index("var weeklyEl = document.getElementById('teamWeeklyChart');")
    block = APP[start:start + 2600]
    assert "window.BRValueChart.render(weeklyEl" in block
    assert "hero: true, width: 3" in block
    assert "name: 'Lg Avg', width: 1.5, dash: '8 5'" in block
    assert "shape: 'spline'" not in block and "spline" not in block


def test_team_breakdown_is_zbars_not_radar():
    start = APP.index("var radarEl = document.getElementById('teamRadarChart');")
    block = APP[start:start + 1600]
    assert "window.BRCharts.zbars(radarEl" in block
    assert "z_scores" in block and "raw_stats" in block


def test_eff_chart_uses_monotone_path_and_end_labels():
    start = APP.index("function _tmBuildEffChart(weeks) {")
    block = APP[start:start + 4200]
    assert "window.BRValueChart.monotonePath" in block
    assert "pathOf('actual')" in block and "pathOf('optimal')" in block
    assert "endLabels" in block
    assert "<polyline points=\"${pts('actual')}" not in block


def test_zbar_sign_color_rule_present():
    start = APP.index("function zbars(container, opts) {")
    block = APP[start:start + 2600]
    # Positive z: navy hero. Negative z: loss red at 60%.
    assert "z >= 0 ? heroCol : th.loss" in block
    assert 'fill-opacity="0.6"' in block or "fill-opacity" in block
    assert "League avg" in block


# ── Graphs page payloads ────────────────────────────────────────────────────

def _weekly(rows):
    return pd.DataFrame(rows, columns=[
        "week", "matchup_id", "roster_id", "owner", "points",
        "points_against", "avatar", "finalized"])


def _paired(games):
    rows = []
    rids = {}
    for wk, mid, o1, p1, o2, p2 in games:
        rids.setdefault(o1, len(rids) + 1)
        rids.setdefault(o2, len(rids) + 1)
        rows.append({"week": wk, "matchup_id": mid, "roster_id": rids[o1],
                     "owner": o1, "points": p1, "points_against": p2,
                     "avatar": None, "finalized": True})
        rows.append({"week": wk, "matchup_id": mid, "roster_id": rids[o2],
                     "owner": o2, "points": p2, "points_against": p1,
                     "avatar": None, "finalized": True})
    return _weekly(rows)


def test_bump_payload_is_straight_reversed_with_end_label_data():
    from dashboard_services.pages.graphs_page import _bump_card, owner_color_map

    df = _paired([(1, 1, "A", 120.0, "B", 100.0), (2, 1, "A", 90.0, "B", 110.0)])
    figs = {}
    html = _bump_card(df, owner_color_map(["A", "B"], "A"), "A", figs)
    assert "Standings Bump Chart" in html
    payload = json.loads(figs["chart-bump"])
    assert payload["kind"] == "line"
    assert payload["straight"] is True  # ranks are integers: no curves
    assert payload["yReverse"] is True
    assert payload["dots"] == "all"
    assert len(payload["series"]) == 2
    assert "refLines" not in payload  # no settings -> no invented cutoff


def test_bump_payload_playoff_line_only_when_cutoff_known():
    from dashboard_services.pages.graphs_page import _bump_card, owner_color_map

    df = _paired([(1, 1, "A", 120.0, "B", 100.0), (2, 1, "A", 90.0, "B", 110.0)])
    figs = {}
    _bump_card(df, owner_color_map(["A", "B"], "A"), "A", figs, playoff_cutoff=1)
    payload = json.loads(figs["chart-bump"])
    assert payload["refLines"] == [{"value": 1.5, "label": "Playoff line"}]


def test_heatmap_payload_marks_missing_week_as_null():
    from dashboard_services.pages.graphs_page import _heatmap_card, owner_color_map

    # B has no week-2 game: the cell must be null (a bye), never a score.
    df = _paired([(1, 1, "A", 120.0, "B", 100.0), (2, 1, "A", 130.0, "C", 90.0),
                  (1, 2, "C", 95.0, "A", 120.0)])
    figs = {}
    _heatmap_card(df, owner_color_map(["A", "B", "C"]), figs)
    payload = json.loads(figs["chart-heatmap"])
    assert payload["kind"] == "heatmap"
    row_b = [r for r in payload["rows"] if r["label"] == "B"][0]
    w2 = payload["weeks"].index(2)
    assert row_b["cells"][w2] is None


def test_career_winpct_payload_is_pct_with_record_notes():
    from dashboard_services.pages.graphs_page import _career_winpct_card, owner_color_map

    rec = pd.DataFrame({
        "season": [2024, 2025], "owner": ["A", "A"],
        "wins": [9, 10], "losses": [5, 4], "ties": [0, 0],
    })
    figs = {}
    html = _career_winpct_card(rec, "A", owner_color_map(["A"], "A"), figs)
    assert "Career Win % by Season" in html
    payload = json.loads(figs["chart-career-wpct"])
    assert payload["kind"] == "line" and payload["fmt"] == "pct"
    s = payload["series"][0]
    assert s["slot"] == 0  # navy, not an owner-cycle color
    assert s["endName"] == "2025"
    assert payload["pointNotes"] == ["9-5", "10-4"]
    assert payload["yMin"] >= 0 and payload["yMax"] <= 100


def test_radar_comparison_is_paired_zbars_with_raw_map():
    from dashboard_services.pages.graphs_page import _radar_card, owner_color_map

    ts = pd.DataFrame({
        "owner": ["A", "B"],
        "PF": [1000.0, 900.0], "PA": [900.0, 1000.0], "MAX": [150.0, 140.0],
        "MIN": [80.0, 70.0], "AVG": [120.0, 110.0], "STD": [15.0, 18.0],
    })
    figs = {}
    html = _radar_card(ts, owner_color_map(["A", "B"], "A"), figs)
    assert "Radar Comparison" in html
    assert "zbarPairs" in html
    assert "var RAWMAP=" in html
    assert "scatterpolar" not in html
    assert "ensurePlotly" not in html


def test_sos_fallback_payload_uses_palette_slots():
    from dashboard_services.pages.graphs_page import _sos_fig, owner_color_map

    df = _paired([(1, 1, "A", 120.0, "B", 100.0), (2, 1, "A", 90.0, "B", 110.0)])
    payload = _sos_fig(df, owner_color_map(["A", "B"], "A"))
    assert payload["kind"] == "barsH"
    slots = {r["label"]: r["slot"] for r in payload["rows"]}
    assert slots["A"] == 0  # viewer navy
    assert payload["refLabel"].startswith("League avg")


# ── History + Teams ─────────────────────────────────────────────────────────

def test_history_chart_is_single_svg_renderer_no_plotly():
    assert "plotly_white" not in HISTORY_PY
    assert "import plotly" not in HISTORY_PY
    assert "go.Figure" not in HISTORY_PY
    assert "br-history-chart" in HISTORY_PY
    assert "BRGraphsRender" in APP and "BRRenderHistoryCharts" in APP
    assert "historyChartSvg" in APP


def test_history_chart_payload_slots_stable_by_sorted_name():
    from dashboard_services.pages.history_page import _history_chart

    df = pd.DataFrame({
        "week": [1, 1, 2, 2],
        "owner": ["Zulu", "Alpha", "Zulu", "Alpha"],
        "points": [100.0, 120.0, 110.0, 90.0],
        "finalized": [True] * 4,
    })
    html = _history_chart(df)
    assert "br-history-chart" in html
    m = re.search(r"data-chart='([^']+)'", html)
    assert m, "payload attribute missing"
    payload = json.loads(m.group(1).replace("&quot;", '"'))
    slots = {s["name"]: s["slot"] for s in payload["series"]}
    # Sorted-name assignment: Alpha first -> slot 1 (no viewer on this
    # surface), Zulu second -> slot 2; identical in every season.
    assert slots == {"Alpha": 1, "Zulu": 2}


def test_teams_page_dead_lazy_plotly_renderer_deleted():
    assert "team-chart-lazy" not in TEAMS_PY
    assert "Lazy-render Plotly charts" not in TEAMS_PY
    # The Value by Position mix bar itself is untouched (Kaedon rejected
    # its redesign): its markup builder is still present.
    assert "pos-seg" in TEAMS_PY or "Value by position" in TEAMS_PY


# ── Node behavioral: palette + z-bar sign rule + bump straight/reversed ─────

_NODE_DRIVER = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const start = src.indexOf('window.BRValueChart = (function () {');
const end = src.indexOf('\nfunction _fixOnePlotly', start);
if (start < 0 || end < 0) { console.error('EXTRACT FAIL'); process.exit(1); }
function fakeEl() {
  return { style: {}, offsetWidth: 140, isConnected: true,
    setAttribute() {}, addEventListener() {},
    querySelector() { return fakeEl(); }, querySelectorAll() { return []; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 600 }; } };
}
function makeContainer(w) {
  const c = fakeEl();
  c.clientWidth = w; c.parentNode = { clientWidth: w };
  let html = '';
  Object.defineProperty(c, 'innerHTML', { get() { return html; }, set(v) { html = v; } });
  c._html = () => html;
  return c;
}
const sandbox = { window: {}, console,
  document: { documentElement: { getAttribute: () => sandbox._theme || null },
    querySelectorAll() { return []; }, addEventListener() {}, readyState: 'complete' },
  getComputedStyle: () => ({ getPropertyValue: () => '', position: 'static' }) };
sandbox.window.addEventListener = () => {};
vm.createContext(sandbox);
vm.runInContext(src.slice(start, end), sandbox);
const W = sandbox.window;
if (!W.BRTeamPalette || !W.BRCharts || !W.BRValueChart) { console.error('MISSING MODULES'); process.exit(1); }

// Palette: viewer slot 0; stability; dark navy swap.
const map = W.BRTeamPalette.assign(['Alpha', 'Bravo', 'Charlie'], 'Bravo');
if (map.Bravo !== 0 || map.Alpha !== 1 || map.Charlie !== 2) { console.error('PALETTE ASSIGN'); process.exit(1); }
if (W.BRTeamPalette.color(0) !== '#122d4b') { console.error('PALETTE NAVY'); process.exit(1); }
sandbox._theme = 'dark';
if (W.BRTeamPalette.color(0) !== '#60a5fa') { console.error('PALETTE DARK NAVY'); process.exit(1); }
if (W.BRTeamPalette.color(1) !== '#38bdf8') { console.error('PALETTE SKY'); process.exit(1); }
sandbox._theme = null;
console.log('PALETTE_OK');

// Z-bars: sign -> color rule, zero line, end text with raw + z.
const cz = makeContainer(560);
W.BRCharts.zbars(cz, { rows: [
  { label: 'PF', z: 1.2, raw: '128.4' },
  { label: 'PA', z: -0.8, raw: '110.0' } ], height: 120 });
const hz = cz._html();
if (!hz.includes('#122d4b')) { console.error('ZBAR NAVY'); process.exit(1); }
if (!hz.includes('#ef4444') || !hz.includes('fill-opacity="0.6"')) { console.error('ZBAR LOSS'); process.exit(1); }
if (!hz.includes('128.4 · +1.2') || !hz.includes('110.0 · -0.8')) { console.error('ZBAR END TEXT'); process.exit(1); }
if (!hz.includes('League avg')) { console.error('ZBAR ZERO LABEL'); process.exit(1); }
console.log('ZBARS_OK');

// Bump: straight path (no C curves) and reversed axis (rank 1 above rank 12).
const cb = makeContainer(760);
W.BRValueChart.render(cb, { height: 300, labels: ['W1', 'W2', 'W3'], straight: true,
  yReverse: true, yStep: 1, yMin: 0.5, yMax: 12.5, dots: 'all',
  series: [{ name: 'A', values: [1, 5, 12], color: '#122d4b', width: 3 }] });
const hb = cb._html();
const d = (hb.match(/<path d="([^"]+)"/) || [])[1] || '';
if (!d || d.includes('C')) { console.error('BUMP NOT STRAIGHT: ' + d); process.exit(1); }
const ys = d.match(/[\d.]+,([\d.]+)/g).map(s => parseFloat(s.split(',')[1]));
if (!(ys[0] < ys[1] && ys[1] < ys[2])) { console.error('BUMP NOT REVERSED: ' + ys); process.exit(1); }
console.log('BUMP_OK');

// Heatmap: null cell renders the distinct bye treatment, never a pale score.
const ch = makeContainer(620);
W.BRCharts.heatmap(ch, { weeks: [1, 2], rows: [
  { label: 'A', avg: 115, cells: [120, null] },
  { label: 'B', avg: 100, cells: [100, 95] } ] });
const hh = ch._html();
if (!hh.includes('br-hm-bye')) { console.error('HEATMAP BYE'); process.exit(1); }
if (!hh.includes('Bye')) { console.error('HEATMAP BYE TIP'); process.exit(1); }
console.log('HEATMAP_OK');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_wave2_client_behaviors_in_node(tmp_path):
    driver = tmp_path / "wave2_driver.js"
    driver.write_text(_NODE_DRIVER, encoding="utf-8")
    res = subprocess.run(
        ["node", str(driver), str(ROOT / "static" / "app.js")],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    for marker in ("PALETTE_OK", "ZBARS_OK", "BUMP_OK", "HEATMAP_OK"):
        assert marker in res.stdout
