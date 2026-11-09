"""Team modal Graphs tab, Layout A (approved by Kaedon 2026-11-09).

Layout only: every chart becomes the same tm3 card, paired into rows
that share a height (Weekly wide beside Breakdown, Score vs Opponent
beside Points Race, Seed beside Weekly Finish, the two league scatters
paired, Actual vs Optimal full width at the bottom). No chart, data or
formula changes, and no new colors: cards use the tm3 surface/border
tokens and the renderers keep their site-token colors.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _app_js() -> str:
    return (REPO / "static" / "app.js").read_text()


def _css() -> str:
    return (REPO / "static" / "dashboard.css").read_text()


def _graphs_block() -> str:
    src = _app_js()
    start = src.index("// Scoring section (Layout A")
    end = src.index("// Populate tab panels", start)
    return src[start:end]


def test_every_chart_is_a_unified_tm3_card():
    block = _graphs_block()
    for cls in ("tm-chart-weekly", "tm-chart-radar", "tm-chart-svg", "tm-chart-eff"):
        assert "tm3-card " + cls in block or ("tm3-card " + cls) in _app_js(), cls
    # The old un-carded section treatment is gone from the whole tab build,
    # including the efficiency chart builder.
    assert "team-modal-section tm-chart" not in _app_js()
    assert "team-modal-section tm-chart-eff" not in _app_js()


def test_layout_a_pairings_and_order():
    block = _graphs_block()
    assert "_tmGraphRow([weeklyCard, radarCard], true)" in block
    assert "_tmGraphRow([vsOppCard, raceCard], false)" in block
    assert "_tmGraphRow([luckCard, valueAgeCard], false)" in block
    # Weekly/Breakdown row renders before the VsOpp/Race row.
    assert block.index("_tmGraphRow([weeklyCard, radarCard], true)") < block.index(
        "_tmGraphRow([vsOppCard, raceCard], false)"
    )


def test_eff_chart_is_a_tm3_card():
    src = _app_js()
    assert '<div class="tm3-card tm-chart-eff">' in src


def test_panel_is_column_flow_with_14px_gaps():
    css = _css()
    m = re.search(r"#tm-panel-charts\.active\s*\{([^}]*)\}", css)
    assert m, "no #tm-panel-charts.active rule"
    body = m.group(1)
    assert "flex-direction: column" in body
    assert "gap: 14px" in body


def test_graph_rows_are_grids_with_shared_heights():
    css = _css()
    m = re.search(r"\.tm-graph-row\s*\{([^}]*)\}", css)
    assert m, "no .tm-graph-row rule"
    body = m.group(1)
    assert "display: grid" in body
    assert "grid-template-columns: 1fr 1fr" in body
    assert "align-items: stretch" in body
    wide = re.search(r"\.tm-graph-row-wide\s*\{([^}]*)\}", css)
    assert wide and "2fr 1fr" in wide.group(1)
    # Lone card in a row spans the full row.
    assert ".tm-graph-row > .tm3-card:only-child" in css


def test_paired_chart_containers_share_one_height():
    css = _css()
    m = re.search(r"#tm-panel-charts \.team-chart-container\s*\{([^}]*)\}", css)
    assert m, "no panel chart-container rule"
    assert "height: 250px" in m.group(1)
    radar = re.search(r"#tm-panel-charts #teamRadarChart\s*\{([^}]*)\}", css)
    assert radar and "height: 250px" in radar.group(1)


def test_rows_collapse_to_one_column_when_narrow():
    css = _css()
    # Container twin (narrow modal) and viewport twin both collapse rows.
    container = re.search(
        r"@container \(max-width: 700px\)\s*\{\s*\.tm-graph-row", css
    )
    assert container, "no @container collapse for .tm-graph-row"
    assert css.count("grid-template-columns: 1fr;") >= 2


def test_no_new_colors_or_banned_radius_in_layout_css():
    css = _css()
    start = css.index("/* Layout A (approved 2026-11-09)")
    end = css.index("@container (max-width: 700px)", start)
    block = css[start:end]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block), "Layout A CSS must use tokens only"
    assert "999px" not in block
    assert "border-radius" not in block  # cards keep the tm3 12px radius
