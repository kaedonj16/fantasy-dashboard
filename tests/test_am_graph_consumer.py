"""Consumer-appeal contracts for the Advanced Metrics graph modal.

Covers the three Kaedon-approved upgrades: (1) an auto-generated headline
takeaway naming the top-right standout in plain words, (2) entrance animations
for dots/labels/trend/quadrants/headline with reduced-motion and og-render
guards, (3) a sticky Simple/Detailed mode (Simple default: bigger dots, fewer
labels, no quadrant dividers, no R^2 chip).
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _js(html):
    m = re.search(r"<script>\n\(function\(\)\{\n(.*?)\n\}\)\(\);\n</script>", html, re.S)
    assert m, "embedded graph script missing"
    return m.group(1)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def test_mode_toggle_exists_and_simple_default():
    html = _html()
    for el_id in ("amGraphModeSimple", "amGraphModeDetailed"):
        m = re.search(r'<[^>]*\bid="%s"[^>]*>' % el_id, html)
        assert m, f"#{el_id} missing"
    simple = re.search(r'<[^>]*\bid="amGraphModeSimple"[^>]*>', html).group(0)
    assert "amSetGraphMode('simple')" in simple
    assert "am-active" in simple, "Simple must be the default-active mode"
    js = _js(html)
    assert "window.amSetGraphMode" in js
    assert "localStorage.getItem('amGraphMode')" in js
    assert "localStorage.setItem('amGraphMode'" in js


def test_headline_takeaway_wired():
    js = _js(html := _html())
    assert "am-graph-headline" in js
    assert "_hlLines" in js
    # Standout = top-right quadrant by combined z-score; skipped when tiny.
    assert "pts.length < 4" in js
    assert "_qdHi(yk)" in js and "_qdHi(xk)" in js


def test_entrance_animations_with_safety_guards():
    html = _html()
    css = _style(html)
    assert "@keyframes amDotIn" in css
    assert "@keyframes amFadeUp" in css
    assert "prefers-reduced-motion" in css
    # og=1 social screenshots must capture the finished frame.
    assert "html.og-render .am-graph-svg .am-graph-dot" in css
    js = _js(html)
    assert "am-graph-trend" in js
    assert "am-graph-quad" in js
    assert "animation-delay" in js


def test_simple_mode_simplifies_and_detailed_restores():
    js = _js(html := _html())
    # Bigger dots + tighter label cap in simple mode.
    assert "_dotBoost" in js
    assert "Math.min(_lblCount, 8)" in js
    # R^2 chip hidden in simple mode (trend line stays).
    assert "if (!_amGraphSimple)" in js
    # Simple presets quadrants off; Detailed restores her standing default.
    assert "_amGraphQuadrants = !_amGraphSimple" in js
