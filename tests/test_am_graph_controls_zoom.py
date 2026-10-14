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
    assert ".am-graph-card { max-width:880px;" in css
    # Selects and picker buttons share one height so rows align.
    assert "height:32px" in css


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
