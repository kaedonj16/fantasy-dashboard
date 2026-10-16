"""Graph Metrics modal: controls layout, Labels count, and zoom.

Kaedon's report (2026-10-13, screenshot of Created Sep vs Uncatchable Tgt %):
the filter/control rows were broken, the Labels pick did not stick (it kept
printing only the original 12), and the graph had no zoom.

Root causes pinned here:

1. The position control <div> was never closed, so the actions block was
   nested INSIDE it (plus an extra stray </div>): the TE pill clipped at
   the card edge and the action buttons were squeezed into the position
   row. And the global .am-metric-btn min-width:180px overflowed its flex
   cell while the label wrapped to two lines, growing the Y picker taller
   than its row ("Uncatchable Tgt" / "%").
2. Labels: _amSyncGraphModeCtrls overwrote #amGraphLabels with 8 (Simple)
   or 12 (Detailed) on every modal open / mode switch, and the scatter
   builder capped the count at 8 in Simple mode regardless of the pick.
   Non-star labels that collided were also dropped outright.
3. Zoom: none existed. The viewBox zoom (buttons / wheel / drag-pan /
   pinch) is wired per render and exports stay full-frame.
"""
import json
import re
import subprocess
from html.parser import HTMLParser

import pytest

from dashboard_services.pages.advanced_metrics_page import _AM_JS, build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


# ---------------------------------------------------------------------------
# 1. Controls markup + CSS
# ---------------------------------------------------------------------------

class _Tree(HTMLParser):
    """Minimal element tree so nesting (not just substrings) is asserted."""

    VOID = {"br", "img", "input", "hr", "meta", "link", "circle", "path", "rect", "line", "image"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = ("__root__", {}, [])
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = (tag, dict(attrs), [])
        self.stack[-1][2].append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1][2].append((tag, dict(attrs), []))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return


def _find(node, pred):
    if pred(node):
        return node
    for child in node[2]:
        hit = _find(child, pred)
        if hit is not None:
            return hit
    return None


def _has_class(node, cls):
    return cls in (node[1].get("class") or "").split()


def test_actions_block_is_not_nested_inside_position_control():
    parser = _Tree()
    parser.feed(_html())
    controls = _find(parser.root, lambda n: _has_class(n, "am-graph-controls"))
    assert controls is not None
    pos_ctrl = _find(controls, lambda n: _has_class(n, "am-graph-pos-ctrl"))
    actions = _find(controls, lambda n: _has_class(n, "am-graph-actions"))
    assert pos_ctrl is not None and actions is not None
    # The bug: actions lived inside the position control (unclosed div).
    assert _find(pos_ctrl, lambda n: _has_class(n, "am-graph-actions")) is None
    # Both are direct children of the controls container.
    assert pos_ctrl in controls[2] and actions in controls[2]
    # The position bar keeps all five pills, unclipped by nesting.
    bar = _find(pos_ctrl, lambda n: n[1].get("id") == "amGraphPosBar")
    assert bar is not None and len(bar[2]) == 5


def test_graph_controls_css_fixes():
    css = _style(_html())
    # Picker buttons fill their cell on one line instead of the global
    # 180px min-width overflow + two-line wrap.
    assert ".am-graph-controls .am-graph-picker .am-metric-btn" in css
    btn_rule = css.split(".am-graph-controls .am-graph-picker .am-metric-btn {", 1)[1].split("}", 1)[0]
    assert "min-width:0" in btn_rule and "white-space:nowrap" in btn_rule
    assert "text-overflow:ellipsis" in css
    # Actions wrap on their own row; the card is wide enough for the rows.
    assert ".am-graph-actions-row" in css and "flex-wrap:wrap" in css
    assert ".am-graph-card { max-width:1140px;" in css
    # Selects and picker buttons share one height so rows align.
    assert "height:32px" in css


# ---------------------------------------------------------------------------
# 1b. Desktop right rail (2026-10-15): controls move out of the plot's way
# ---------------------------------------------------------------------------

def _desktop_block(css):
    start = css.index("@media (min-width:601px)")
    end = css.index(".am-graph-svg { width:100%; height:auto;", start)
    return css[start:end]


def _mobile_block(css):
    # The graph modal's own mobile block follows the ctrl-hidden rules;
    # other max-width:600px blocks exist elsewhere in the page CSS.
    anchor = css.index(".am-graph-pos-bar.am-ctrl-hidden")
    start = css.index("@media (max-width:600px)", anchor)
    end = css.index("@media (min-width:601px)", start)
    return css[start:end]


def test_controls_live_in_a_right_rail_beside_the_plot():
    parser = _Tree()
    parser.feed(_html())
    body = _find(parser.root, lambda n: _has_class(n, "am-graph-body"))
    assert body is not None
    rail = _find(body, lambda n: _has_class(n, "am-graph-rail"))
    plot = _find(body, lambda n: _has_class(n, "am-graph-plot-wrap"))
    assert rail is not None and plot is not None
    # Both are direct children of the body; the rail carries the controls,
    # the plot wrap carries the plot, and neither nests inside the other.
    assert rail in body[2] and plot in body[2]
    assert _find(rail, lambda n: _has_class(n, "am-graph-controls")) is not None
    assert _find(plot, lambda n: _has_class(n, "am-graph-controls")) is None
    assert _find(rail, lambda n: n[1].get("id") == "amGraphPlot") is None
    assert _find(plot, lambda n: n[1].get("id") == "amGraphPlot") is not None
    # Download + Copy link sit in their own file row inside the actions block.
    actions = _find(rail, lambda n: _has_class(n, "am-graph-actions"))
    file_row = _find(actions, lambda n: _has_class(n, "am-graph-file-row"))
    view_row = _find(actions, lambda n: _has_class(n, "am-graph-view-row"))
    assert file_row is not None and view_row is not None
    assert _find(file_row, lambda n: n[1].get("id") == "amGraphDownloadBtn") is not None
    assert _find(file_row, lambda n: n[1].get("id") == "amGraphCopyBtn") is not None
    assert _find(view_row, lambda n: n[1].get("id") == "amGraphThemeBtn") is not None
    assert _find(view_row, lambda n: n[1].get("id") == "amGraphModeDetailed") is not None


def test_desktop_rail_css():
    css = _style(_html())
    desk = _desktop_block(css)
    # Rail on the RIGHT: body becomes a row, rail is ordered after the plot
    # and pinned at 238px with a left border; it scrolls on its own.
    assert ".am-graph-body { flex-direction:row; }" in desk
    rail_rule = desk.split(".am-graph-rail {", 1)[1].split("}", 1)[0]
    assert "order:2" in rail_rule and "width:238px" in rail_rule
    assert "border-left:1px solid var(--border)" in rail_rule
    plot_rule = desk.split(".am-graph-plot-wrap {", 1)[1].split("}", 1)[0]
    assert "order:1" in plot_rule and "overflow:hidden" in plot_rule
    # No plot scrolling on desktop: the SVG is height-constrained.
    svg_rule = desk.split(".am-graph-svg {", 1)[1].split("}", 1)[0]
    assert "height:100%" in svg_rule and "max-height:100%" in svg_rule
    # Controls stack vertically inside the rail and scroll independently.
    ctrl_rule = desk.split(".am-graph-controls {", 1)[1].split("}", 1)[0]
    assert "flex-direction:column" in ctrl_rule and "overflow-y:auto" in ctrl_rule
    # Download / Copy link pinned at the rail bottom.
    file_rule = desk.split(".am-graph-file-row {", 1)[1].split("}", 1)[0]
    assert "margin-top:auto" in file_rule


def test_mobile_keeps_top_controls_layout():
    html = _html()
    css = _style(html)
    mob = _mobile_block(css)
    # The Settings toggle still shows on mobile and hides the controls.
    assert ".am-graph-ctrl-toggle { display:flex; }" in mob
    assert 'id="amGraphCtrlToggle"' in html
    assert "onclick=\"amToggleGraphControls()\"" in html
    # The View/file split dissolves on mobile so the action buttons share
    # one wrapping row exactly as before the rail. The actions container
    # must out-specify the .am-gctrl column rule or the buttons stack.
    assert ".am-graph-actions-row { display:contents; }" in mob
    assert ".am-graph-controls .am-graph-actions { display:flex; flex-direction:row;" in mob
    # No rail border/width leaks into the mobile layout.
    assert ".am-graph-rail" not in mob


def test_og_render_hides_the_rail():
    css = _style(_html())
    assert "html.og-render .am-graph-rail" in css
    og = css[css.index("html.og-render .am-graph-body"):]
    assert "display:block !important" in og.split("}", 1)[0]
    # The 1200x630 export sizing still wins for the plot itself.
    assert "html.og-render .am-graph-svg { width:auto !important; height:594px !important;" in css


def test_graph_picker_panels_pin_fixed_so_rail_overflow_cannot_clip():
    # amInitSearchPicker.positionPanel pins the open panel position:fixed
    # relative to its button; the graph X/Y/Z pickers share that builder,
    # so the rail's overflow-y:auto never clips the metric dropdowns.
    assert "panel.style.position = 'fixed'" in _AM_JS
    assert "function amInitGraphPickers()" in _AM_JS
    assert _AM_JS.count("amInitSearchPicker({") >= 4  # primary + X + Y + Z


# ---------------------------------------------------------------------------
# 2. Labels: the user's count is authoritative
# ---------------------------------------------------------------------------

def _extract_fn(js, name):
    start = js.index(f"function {name}(")
    brace = js.index("{", start)
    depth = 0
    for i in range(brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start:i + 1]
    raise AssertionError("unbalanced braces")


def test_mode_sync_never_overwrites_labels_choice():
    sync = _extract_fn(_AM_JS, "_amSyncGraphModeCtrls")
    assert "amGraphLabels" not in sync, "mode sync must not reset the Labels select"
    assert "Math.min(_lblCount, 8)" not in _AM_JS, "Simple mode must not cap the label count"
    assert "const showCut = Math.min(pts.length, _lblCount);" in _AM_JS
    # Collided non-star labels keep the least-crowded anchor instead of
    # being dropped, so a bigger pick actually prints more labels.
    assert "if (!isStar || !best) return;" not in _AM_JS


_NODE_PREAMBLE = """
let __lblValue = '12';
const document = { getElementById: function(id) {
  if (id === 'amGraphLabels') return { value: __lblValue };
  if (id === 'amGraphCtxNote') return { textContent: '' };
  if (id === 'amGraphZoomPct') return { textContent: '' };
  return null;
}, querySelector: function() { return null; } };
const window = { innerWidth: 1280 };
const cfg = { metrics: {
  xk: { label: 'Created Sep' }, yk: { label: 'Uncatchable Tgt %' },
  zk: { label: 'Targets' } } };
let _amGraphSimple = false, _amGraphQuadrants = false, _amGraphLabelAxis = 'x';
function _amGraphPalette() { return { bg: '#ffffff', text: '#1e293b', muted: '#64748b',
  grid: 'rgba(100,116,139,.16)', axis: 'rgba(100,116,139,.40)', border: 'rgba(100,116,139,.20)' }; }
function fmtVal(v) { return String(Math.round(v * 100) / 100); }
function posColor() { return '#f59e0b'; }
"""


def _scatter_js():
    parts = [
        _extract_fn(_AM_JS, "_amEsc"),
        _extract_fn(_AM_JS, "_amLastName"),
        _extract_fn(_AM_JS, "_amBuildScatter"),
    ]
    return "\n".join(parts)


def _count_labels(lbl_value, simple):
    pts = []
    for i in range(75):
        # Deterministic spread resembling the reported chart's cloud.
        x = -6 + (i % 15) * 0.36 + ((i * 7) % 5) * 0.07
        y = 8 + ((i * 37) % 62) + (i % 3)
        pts.append({"pid": str(i), "name": f"Player{i} Last{i}",
                    "position": "WR", "headshot": "", "x": x, "y": y, "z": None})
    pts.sort(key=lambda p: -p["x"])
    script = (
        _NODE_PREAMBLE
        + f"\n__lblValue = '{lbl_value}'; _amGraphSimple = {str(simple).lower()};\n"
        + _scatter_js()
        + "\nconst pts = " + json.dumps(pts) + ";"
        + "\nconst svg = _amBuildScatter(pts, 'xk', 'yk', '', '', {m: -1, b: 40, r2: 0.1}, 'none');"
        + "\nconsole.log(JSON.stringify({ labels: (svg.match(/class=\"am-graph-lbl\"/g) || []).length }));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["labels"]


def test_labels_pick_is_honored_in_both_modes():
    n12 = _count_labels("12", simple=False)
    n25_detailed = _count_labels("25", simple=False)
    n25_simple = _count_labels("25", simple=True)
    assert n12 >= 11, f"12 picked, only {n12} printed"
    assert n25_detailed >= 20, f"25 picked, only {n25_detailed} printed"
    assert n25_simple == n25_detailed, "Simple mode must not cap the picked count"
    assert n25_detailed > n12


# ---------------------------------------------------------------------------
# 3. Zoom
# ---------------------------------------------------------------------------

def test_zoom_controls_and_wiring_exist():
    html = _html()
    for el_id, handler in (("amGraphZoomIn", "amZoomGraph(1.45)"),
                           ("amGraphZoomOut", "amZoomGraph(0.7)"),
                           ("amGraphZoomReset", "amResetGraphZoom()")):
        m = re.search(r'<[^>]*\bid="%s"[^>]*>' % el_id, html)
        assert m, f"#{el_id} missing"
        assert handler in m.group(0)
    assert 'id="amGraphZoomPct"' in html
    # Wired per render; wheel + pointer (pan/pinch) handlers; base viewBox
    # stamped on the svg and used for full-frame PNG export.
    assert "_amWireGraphZoom(plot.querySelector('svg.am-graph-svg'))" in _AM_JS
    assert "addEventListener('wheel'" in _AM_JS
    assert "addEventListener('pointerdown'" in _AM_JS
    assert "data-base-vb=" in _AM_JS
    assert "svg.getAttribute('data-base-vb')" in _AM_JS
    # og social renders hide the zoom chrome.
    assert "html.og-render .am-graph-zoom" in _style(html)


def _zoom_js():
    start = _AM_JS.index("let _amZoomK")
    end = _AM_JS.index("window.amZoomGraph")
    return _AM_JS[start:end]


def test_zoom_math_scales_and_clamps():
    fake_svg = """
const svg = { attrs: { 'data-base-vb': '0 0 720 600', viewBox: '0 0 720 600' },
  classList: { toggle: function() {} },
  setAttribute: function(k, v) { this.attrs[k] = v; },
  getAttribute: function(k) { return this.attrs[k]; } };
"""
    script = (
        _NODE_PREAMBLE + fake_svg + _zoom_js()
        + "\n_amApplyGraphZoom(svg); const full = svg.attrs.viewBox;"
        + "\n_amZoomAt(svg, 2, 0.5, 0.5); const ctr = svg.attrs.viewBox;"
        + "\n_amZoomAt(svg, 99, 0.5, 0.5); const maxed = svg.attrs.viewBox;"
        + "\n_amZoomAt(svg, 0.01, 0.5, 0.5); const back = svg.attrs.viewBox;"
        + "\nconsole.log(JSON.stringify({ full: full, ctr: ctr, maxed: maxed, back: back, k: _amZoomK }));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["full"] == "0.0 0.0 720.0 600.0"
    assert res["ctr"] == "180.0 150.0 360.0 300.0", "2x zoom centers on the focal point"
    # Clamped at 5x, and zooming back out restores the full frame.
    assert res["maxed"] == "288.0 240.0 144.0 120.0"
    assert res["back"] == "0.0 0.0 720.0 600.0" and res["k"] == 1
