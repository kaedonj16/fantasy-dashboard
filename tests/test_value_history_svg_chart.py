"""Value-history charts: shared BRValueChart SVG renderer (wave 1).

The player modal chart, the compare 2-player / 3-player charts and the
public player page chart used to be three separate Plotly copies. They now
render through ONE hand-built SVG helper, window.BRValueChart, defined in
static/app.js core (above the @public-js:core-end marker) so every shell
that draws a value chart can reach it: the modal and compare ship in the
full app bundle, and the public player page loads the lite bundle built
from the same core.

Locked design (approved mock): navy hero line (#122d4b light / #60a5fa
dark) with a 12% gradient to the padded axis baseline, dashed sky
(#38bdf8) secondary, sky band fill between the two series, stacked end
labels (name over value, no pills), flat dots with no rings, green Peak /
red Low on the modal primary series, and the stored values plotted EXACTLY
(no data smoothing) through a monotone cubic path that never overshoots.

Other Plotly charts (team modal weekly/radar, Teams page, graphs page,
history page) are later waves and must stay untouched here.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
MODAL = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
PPAGE = (ROOT / "static" / "player_page.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _block(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start:end]


HELPER = _block(APP, "window.BRValueChart = (function () {", "function _fixOnePlotly")


# ── Shared renderer placement ────────────────────────────────────────────

def test_helper_lives_in_app_core_so_all_shells_reach_it():
    # Above the core-end marker = shipped in BOTH the full bundle and the
    # lite public bundle the player page loads.
    assert APP.index("window.BRValueChart = (function () {") < APP.index(
        "// @public-js:core-end  (everything below is app/feature code"
    )
    # The monotone path is exported so tests (and only tests) can exercise
    # the geometry directly.
    assert "monotonePath: _monoPath" in HELPER
    assert "var HERO = { light: '#122d4b', dark: '#60a5fa' };" in HELPER
    assert "var SKY = '#38bdf8';" in HELPER


# ── No Plotly on the value-history paths; other charts untouched ─────────

def test_no_plotly_on_the_three_value_history_paths():
    assert "Plotly" not in MODAL
    assert "ensurePlotly" not in MODAL
    assert "Plotly" not in PPAGE
    assert "ensurePlotly" not in PPAGE
    cmp2 = _block(APP, "const _drawCmpChart = () => {", "_drawCmpChart();")
    assert "Plotly" not in cmp2
    assert "window.BRValueChart.render(chartDiv, {" in cmp2
    cmp3 = _block(APP, "function _renderTripleValueChart(players)", "function openComparisonView(p1, p2)")
    assert "Plotly" not in cmp3
    assert "window.BRValueChart.render(chartDiv, {" in cmp3


def test_other_plotly_charts_untouched():
    # Wave 1 scope guard, updated for wave 2: the three surfaces wave 1
    # left on Plotly (team weekly, team radar, history) have since been
    # converted by wave 2, so none of their Plotly calls remain either.
    assert "Plotly.newPlot('teamWeeklyChart'" not in APP
    assert "Plotly.newPlot('teamRadarChart'" not in APP
    assert "Plotly.newPlot('historyChartPlotly'" not in APP


# ── Modal chart spec ─────────────────────────────────────────────────────

def test_modal_chart_uses_locked_series_treatment():
    region = _block(MODAL, "function renderChart(history) {", "// Time-range filter")
    # Hero = navy role color, secondary = sky dashed, band between them,
    # peak/low on the primary series.
    assert "color: window.BRValueChart.heroColor, width: 3, hero: true" in region
    assert "color: window.BRValueChart.sky, width: 1.5, dash: '8 5'" in region
    assert "band: hasDualSeries" in region
    assert "peakLow: true" in region
    assert "name: hasDualSeries ? '1QB' : 'Value'" in region
    # SF gaps stay gaps: no fallback substitution onto the 1QB line.
    assert "d.value_sf != null ? Number(d.value_sf) : null" in region


def test_modal_summary_peak_floor_take_win_loss_colors():
    assert "pm-vh-ext pm-vh-ext-peak" in MODAL
    assert "pm-vh-ext pm-vh-ext-floor" in MODAL
    assert ".pm-vh-ext-peak .pm-vh-ext-k" in CSS
    assert ".pm-vh-ext-floor .pm-vh-ext-k" in CSS
    peak_css = _block(CSS, ".pm-vh-ext-peak .pm-vh-ext-k,", "@media (max-width: 768px)")
    assert "color: var(--win)" in peak_css
    assert "color: var(--loss)" in peak_css


def test_redraft_toggle_still_swaps_series_and_label():
    # The renderer draws the ACTIVE format's series: redraft reads the
    # redraft payload, the section label restamps per format, and the
    # thin-history note stays honest (never dynasty data under it).
    assert "? data.value_history_redraft : data.value_history;" in MODAL
    assert "fmt === 'redraft' ? 'Redraft Value History' : 'Dynasty Value History'" in MODAL
    assert "_vhLbl.textContent = _pmVhLabel(_vhFmt);" in MODAL
    assert "Redraft history is still building." in MODAL


# ── Compare + player page spec ───────────────────────────────────────────

def test_compare_keeps_bands_ref_lines_and_third_color():
    cmp2 = _block(APP, "const _drawCmpChart = () => {", "_drawCmpChart();")
    # Dynasty-only shaded value bands + dashed tier-average reference line.
    assert "_bands.push({ min: band[0], max: band[1], fill:" in cmp2
    assert "_refLines.push({ value: Number(v), color: _cmpColors[i], name: p.name });" in cmp2
    assert "bands: _bands, refLines: _refLines" in cmp2
    # Hide-block-when-no-active-format-series behavior preserved.
    assert "_cmpToggleChartBlock(chartDiv, _anyActive);" in cmp2
    cmp3 = _block(APP, "function _renderTripleValueChart(players)", "function openComparisonView(p1, p2)")
    assert "'#22c55e'" in cmp3
    # Both compare charts share the point builder (active-format series).
    assert "points: _cmpHistPoints(p)" in cmp2
    assert "points: _cmpHistPoints(p)" in cmp3


def test_player_page_keeps_history_messages_and_uses_renderer():
    assert 'showChartMessage("Not enough value history yet for a chart");' in PPAGE
    assert 'showChartMessage("Could not load the chart library");' in PPAGE
    assert "window.BRValueChart.render(div, {" in PPAGE
    assert "band: hasDual," in PPAGE
    assert "peakLow: true," in PPAGE


# ── Behavioral: monotone geometry + rendered SVG (node) ──────────────────

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
    getBoundingClientRect() { return { left: 0, top: 0 }; } };
}
function makeContainer(w) {
  const c = fakeEl();
  c.clientWidth = w; c.parentNode = { clientWidth: w };
  const kids = {};
  c.querySelector = (sel) => (kids[sel] || (kids[sel] = fakeEl()));
  let html = '';
  Object.defineProperty(c, 'innerHTML', { get() { return html; }, set(v) { html = v; } });
  c._html = () => html;
  return c;
}
const sandbox = { window: {}, console,
  document: { documentElement: { getAttribute: () => null } },
  getComputedStyle: () => ({ getPropertyValue: () => '', position: 'static' }) };
sandbox.window.addEventListener = () => {};
vm.createContext(sandbox);
vm.runInContext(src.slice(start, end), sandbox);
const B = sandbox.window.BRValueChart;
if (!B) { console.error('NO HELPER'); process.exit(1); }

// Monotone path must never overshoot a segment's endpoint range.
function cubic(p0, p1, p2, p3, t) {
  const u = 1 - t;
  return u*u*u*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t*t*t*p3;
}
function checkPath(pts) {
  const nums = B.monotonePath(pts).match(/-?\d+\.?\d*/g).map(Number);
  let idx = 2, prev = { x: nums[0], y: nums[1] }, maxOver = 0;
  while (idx < nums.length) {
    const endY = nums[idx + 5];
    const lo = Math.min(prev.y, endY), hi = Math.max(prev.y, endY);
    for (let s = 0; s <= 100; s++) {
      const y = cubic(prev.y, nums[idx + 1], nums[idx + 3], endY, s / 100);
      maxOver = Math.max(maxOver, lo - y, y - hi);
    }
    prev = { x: nums[idx + 4], y: endY }; idx += 6;
  }
  return maxOver;
}
const jag = [7900, 7880, 8100, 7950, 8610, 8300, 8420, 7000, 7050, 7050, 6900, 7200]
  .map((v, i) => ({ x: i * 50, y: 300 - (v - 6800) / 10 }));
if (checkPath(jag) > 1e-6) { console.error('OVERSHOOT'); process.exit(1); }
console.log('MONOTONE_OK');

// Dual-series modal-shape render: stacked end labels, band, peak/low.
const labels = [], qb = [], sf = [];
for (let i = 0; i < 52; i++) {
  labels.push('W' + i);
  qb.push(8100 + Math.round(Math.sin(i / 5) * 60));
  sf.push(qb[i] + 1490);
}
qb[22] = 8610; qb[2] = 7880; qb[51] = 8420; sf[51] = 9910; sf[10] = null;
const c1 = makeContainer(760);
B.render(c1, { height: 250, labels, band: true, peakLow: true, series: [
  { name: '1QB', values: qb, color: B.heroColor, width: 3, hero: true },
  { name: 'SF', values: sf, color: B.sky, width: 1.5, dash: '8 5' } ] });
const h1 = c1._html();
for (const n of ['<svg class="br-vh-svg"', 'Peak 8,610', 'Low 7,880', '>1QB<', '8,420',
    '9,910', 'fill-opacity="0.07"', '#122d4b', '#38bdf8', '#16a34a', '#ef4444']) {
  if (!h1.includes(n)) { console.error('MISSING ' + n); process.exit(1); }
}
if (/stroke="#ffffff"|stroke="#fff"/.test(h1)) { console.error('RING'); process.exit(1); }
// X-axis ticks must land at real positions with real labels: the index-mode
// axis entries are {u, label} objects, never undefined-u strings (a string
// entry put the tick at x="NaN" with empty text and killed hover snapping).
if (h1.includes('x="NaN"')) { console.error('NAN TICK'); process.exit(1); }
if (!/<text x="[0-9.]+" y="242"[^>]*>[^<]+<\/text>/.test(h1)) { console.error('NO X TICKS'); process.exit(1); }
console.log('RENDER_OK');

// Short redraft-shape series: discrete point dots + merged peak suffix.
const c2 = makeContainer(380);
const red = [6200, 6400, 6300, 6700, 6600, 6900, 6800, 7000, 6950, 7050];
B.render(c2, { height: 200, labels: red.map((_, i) => 'Oct ' + (i + 1)), peakLow: true,
  series: [{ name: '1QB', values: red, color: B.heroColor, width: 3, hero: true }] });
const h2 = c2._html();
if (!h2.includes('· Peak')) { console.error('NO MERGED PEAK'); process.exit(1); }
if ((h2.match(/r="3"/g) || []).length < 10) { console.error('NO DISCRETE DOTS'); process.exit(1); }
console.log('SHORT_OK');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_monotone_path_never_overshoots_and_render_matches_spec(tmp_path):
    driver = tmp_path / "vh_driver.js"
    driver.write_text(_NODE_DRIVER, encoding="utf-8")
    res = subprocess.run(
        ["node", str(driver), str(ROOT / "static" / "app.js")],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "MONOTONE_OK" in res.stdout
    assert "RENDER_OK" in res.stdout
    assert "SHORT_OK" in res.stdout


# ── Behavioral: hover events (regression for the dead modal hover) ───────
# The modal chart was not hoverable on the live site: index mode passed
# plain string labels where the hover snap expected {u, label} entries, so
# the snap stayed undefined, every series lookup missed, and onMove hid the
# tooltip on every mousemove. The render-only tests above could not see it
# (their shim's addEventListener was a no-op). This driver records the real
# listeners BRValueChart binds and dispatches real events at them, with an
# innerHTML setter that models the browser replacing the chart's nodes on
# every render (a re-render must rebind hover on the fresh overlay).

_HOVER_DRIVER = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const start = src.indexOf('window.BRValueChart = (function () {');
const end = src.indexOf('\nfunction _fixOnePlotly', start);
if (start < 0 || end < 0) { console.error('EXTRACT FAIL'); process.exit(1); }
function fakeEl() {
  const el = { style: {}, offsetWidth: 140, isConnected: true, attrs: {}, _handlers: {},
    setAttribute(k, v) { el.attrs[k] = v; },
    addEventListener(t, fn) { (el._handlers[t] = el._handlers[t] || []).push(fn); },
    dispatch(t, evt) { (el._handlers[t] || []).forEach(fn => fn(evt)); },
    getBoundingClientRect() { return { left: 0, top: 0 }; } };
  return el;
}
function makeContainer(w) {
  const c = fakeEl();
  c.clientWidth = w; c.parentNode = { clientWidth: w };
  let kids = {}, html = '';
  c.querySelector = (sel) => (kids[sel] || (kids[sel] = fakeEl()));
  Object.defineProperty(c, 'innerHTML', {
    get() { return html; },
    set(v) { html = v; kids = {}; } });  // node replacement, like a browser
  c._html = () => html;
  return c;
}
const sandbox = { window: {}, console,
  document: { documentElement: { getAttribute: () => null } },
  getComputedStyle: () => ({ getPropertyValue: () => '', position: 'static' }) };
sandbox.window.addEventListener = () => {};
vm.createContext(sandbox);
vm.runInContext(src.slice(start, end), sandbox);
const B = sandbox.window.BRValueChart;
if (!B) { console.error('NO HELPER'); process.exit(1); }

const labels = [], qb = [], sf = [];
for (let i = 0; i < 30; i++) { labels.push('Oct ' + (i + 1)); qb.push(8000 + i * 10); sf.push(qb[i] + 1400); }
qb[29] = 8420; sf[29] = 9910;
const c = makeContainer(760);
function draw(vals, labs) {
  B.render(c, { height: 250, labels: labs, band: true, peakLow: true, series: [
    { name: '1QB', values: vals[0], color: B.heroColor, width: 3, hero: true },
    { name: 'SF', values: vals[1], color: B.sky, width: 1.5, dash: '8 5' } ] });
}
draw([qb, sf], labels);

// Sweep the pointer across the plot: the tooltip must track point to
// point (many distinct dates), and the point at index 14 must show its
// exact date + both series' true values with the crosshair on a real x.
let overlay = c.querySelector('.br-vh-overlay');
let tip = c.querySelector('.br-vh-tip');
let hoverG = c.querySelector('.br-vh-hover');
let cross = c.querySelector('.br-vh-cross');
let hitX = null, seenDates = {};
for (let mx = 55; mx <= 700; mx += 3) {
  overlay.dispatch('mousemove', { clientX: mx });
  if (tip.style.display !== '') { console.error('TIP NOT SHOWN at ' + mx); process.exit(1); }
  const m = tip.innerHTML.match(/Oct \d+/);
  if (m) seenDates[m[0]] = 1;
  if (tip.innerHTML.includes('Oct 15') && tip.innerHTML.includes('8,140')) hitX = mx;
}
if (Object.keys(seenDates).length < 20) { console.error('HOVER DOES NOT TRACK: ' + Object.keys(seenDates).length); process.exit(1); }
if (hitX == null) { console.error('POINT 14 TOOLTIP NEVER SHOWN'); process.exit(1); }
overlay.dispatch('mousemove', { clientX: hitX });
if (hoverG.style.display !== '') { console.error('HOVER GROUP NOT SHOWN'); process.exit(1); }
for (const want of ['Oct 15', '1QB', '8,140', 'SF', '9,540']) {
  if (!tip.innerHTML.includes(want)) { console.error('TIP MISSING ' + want + ': ' + tip.innerHTML); process.exit(1); }
}
const cx1 = Number(cross.attrs.x1);
if (!isFinite(cx1) || cx1 < 40 || cx1 > 720) { console.error('CROSS BAD: ' + cross.attrs.x1); process.exit(1); }
console.log('HOVER_OK');
overlay.dispatch('mouseleave', {});
if (tip.style.display !== 'none' || hoverG.style.display !== 'none') { console.error('NOT HIDDEN'); process.exit(1); }
console.log('LEAVE_OK');
// Touch goes through the same handler.
overlay.dispatch('touchmove', { touches: [{ clientX: hitX }] });
if (tip.style.display !== '' || !tip.innerHTML.includes('Oct 15')) { console.error('TOUCH FAIL'); process.exit(1); }
console.log('TOUCH_OK');
// A re-render (range pill / format toggle path) replaces the nodes; hover
// must work on the fresh overlay with the new window's values.
draw([qb.slice(20), sf.slice(20)], labels.slice(20));
overlay = c.querySelector('.br-vh-overlay');
tip = c.querySelector('.br-vh-tip');
let rrHit = false;
for (let mx = 55; mx <= 720 && !rrHit; mx += 3) {
  overlay.dispatch('mousemove', { clientX: mx });
  if (tip.style.display === '' && tip.innerHTML.includes('Oct 30')
      && tip.innerHTML.includes('8,420') && tip.innerHTML.includes('9,910')) rrHit = true;
}
if (!rrHit) { console.error('RERENDER HOVER FAIL: ' + tip.innerHTML); process.exit(1); }
console.log('RERENDER_HOVER_OK');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_hover_events_drive_tooltip_and_crosshair(tmp_path):
    driver = tmp_path / "vh_hover_driver.js"
    driver.write_text(_HOVER_DRIVER, encoding="utf-8")
    res = subprocess.run(
        ["node", str(driver), str(ROOT / "static" / "app.js")],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "HOVER_OK" in res.stdout
    assert "LEAVE_OK" in res.stdout
    assert "TOUCH_OK" in res.stdout
    assert "RERENDER_HOVER_OK" in res.stdout
