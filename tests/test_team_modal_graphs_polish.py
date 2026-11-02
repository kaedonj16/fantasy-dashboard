"""Team modal Graphs tab polish: source contracts for the weekly chart layout
and the mobile header close button.

Locks in the fixes for the visual defects Kaedon reported 2026-09-29:
fractional week ticks (1, 1.5, 2, ...) on the Weekly Scoring chart, the
legend overlapping the plot area, the cut-off "Week" axis title, and the
close button wrapping onto its own line between the meta rows and the stat
tiles on narrow screens.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _app_js() -> str:
    return (REPO / "static" / "app.js").read_text()


def _weekly_block() -> str:
    """The Weekly Scoring render block (BRValueChart since wave 2)."""
    src = _app_js()
    start = src.index("var weeklyEl = document.getElementById('teamWeeklyChart');")
    end = src.index("window.BRValueChart.render(weeklyEl, {", start)
    return src[start:end] + src[end:src.index("});", end)]


def test_weekly_chart_uses_whole_week_ticks():
    block = _weekly_block()
    # X labels are built per data point as W1, W2, W3 (matching the Actual
    # vs Optimal chart): whole weeks only, never fractional auto ticks.
    assert "ws.map(function (d) { return 'W' + d.week; })" in block
    assert "labels: wkLabels" in block


def test_weekly_chart_has_no_detached_legend():
    # The old floating Plotly legend is gone entirely: series identity is
    # carried by the stacked end labels, so nothing can overlap the data.
    block = _weekly_block()
    assert "legend" not in block
    assert "Plotly" not in block
    assert "showlegend" not in _app_js()


def test_weekly_chart_has_no_redundant_axis_titles():
    block = _weekly_block()
    assert "title: 'Week'" not in block
    assert "title: 'Points'" not in block


def _inside_768_media(css: str, rule_start: int) -> bool:
    """True when the rule at rule_start sits inside a 768px media query."""
    media_before = css.rfind("@media (max-width: 768px)", 0, rule_start)
    if media_before == -1:
        return False
    segment = css[media_before:rule_start]
    return segment.count("{") > segment.count("}")


def test_eff_chart_svg_scales_to_container_width():
    # The Actual vs Optimal SVG must not pin a fixed pixel height: with
    # width="100%" and the default preserveAspectRatio="meet", a fixed
    # height letterboxes the 340-unit drawing instead of stretching it to
    # the container width (the Weekly Scoring Plotly chart fills it).
    src = _app_js()
    start = src.index('aria-label="Weekly actual versus optimal points"')
    tag_start = src.rindex("<svg", 0, start)
    tag = src[tag_start:start]
    assert 'width="100%"' in tag
    assert 'height="${H}"' not in tag
    assert "height:auto" in tag


def test_mobile_close_button_pinned_top_right():
    css = (REPO / "static" / "dashboard.css").read_text()
    m = re.search(r"\.team-modal-close\s*\{[^}]*position:\s*absolute[^}]*\}", css)
    assert m, "no absolutely-positioned .team-modal-close rule"
    assert _inside_768_media(css, m.start()), "close rule is not inside the 768px media query"
    decls = m.group(0)
    assert "top:" in decls and "right:" in decls
    # The header must anchor the absolutely positioned button.
    mh = re.search(r"\.team-modal-header\s*\{[^}]*position:\s*relative[^}]*\}", css)
    assert mh, "no position:relative .team-modal-header rule"
    assert _inside_768_media(css, mh.start()), \
        "header anchor rule is not inside the 768px media query"
    # The actions menu must clear the pinned button.
    m2 = re.search(r"\.tm-menu-wrap\s*\{[^}]*margin-right:[^}]*\}", css)
    assert m2, "no .tm-menu-wrap margin-right rule"
    assert _inside_768_media(css, m2.start()), "menu rule is not inside the 768px media query"
