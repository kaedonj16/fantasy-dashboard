"""Mobile visual-rework contracts for the Advanced Metrics toolbar.

Kaedon approved the mobile mockup (header + compact icon-only actions, no
shouting control labels, search+season on one row, solid + Metric button,
fading Decide rail, wrapping chips with a "Clear all" text link, collapsible
field averages). The sticky player column now applies on all viewports (base
table CSS); the rework block only adds the mobile edge fade.

The command-bar rework (mock A) superseded the header and the search+season
row: the strip now holds search / filter / overflow icon buttons, the search
row toggles open, and the old toolbar rows became the filter sheet.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def _rework_css(html):
    """Only the mobile rework block appended by this change."""
    marker = "mobile visual rework"
    style = _style(html)
    start = style.index(marker)
    return style[start:]


def _tag(html, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), html)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_header_uses_compact_icon_buttons():
    # Command-bar rework: the slim sticky strip holds the title plus search /
    # filter / overflow icon buttons; the three old actions moved into the
    # overflow menu with their label spans kept for the menu rows.
    html = _html()
    assert 'id="amCmdBar"' in html
    for el_id in ("amSearchToggle", "amFilterToggle", "amMoreToggle"):
        tag = _tag(html, el_id)
        assert 'title="' in tag, f"#{el_id} needs a title for the icon-button state"
    for el_id in ("amGraphBtn", "amLegendBtn", "amExportBtn"):
        tag = _tag(html, el_id)
        assert 'class="am-legend-btn"' in tag, f"#{el_id} must stay a legend button"
    assert html.count('class="am-legend-btn-label"') == 3
    css = _rework_css(html)
    assert ".am-cmdbar {" in css


def test_control_labels_removed_from_markup_with_aria_names():
    # Desktop rework: the all-caps section labels are gone from the markup
    # entirely; the controls carry accessible names instead.
    html = _html()
    body = html.split("<style>")[0]  # labels would live in the markup, not CSS/JS
    assert ">Primary Metric<" not in body
    assert ">Seasons<" not in body
    assert ">Search<" not in body
    metric_btn = _tag(html, "amMetricBtn")
    assert 'aria-label="Primary metric"' in metric_btn
    search = _tag(html, "amSearch")
    assert 'aria-label="Search players"' in search
    season_btn = _tag(html, "amSeasonBtn")
    assert 'aria-label="Select seasons"' in season_btn


def test_search_and_season_share_one_mobile_row():
    # Command-bar rework: the search+season row became a search row toggled
    # from the strip; the season picker moved into the filter sheet.
    html = _html()
    body = html.split("<style>")[0]
    search_row = _tag(html, "amSearchRow")
    assert "hidden" in search_row
    search = _tag(html, "amSearch")
    assert 'aria-label="Search players"' in search
    assert 'id="amSearchToggle"' in body
    css = _rework_css(html)
    assert "#amSearchRow { margin:8px 0 0; }" in css
    assert "#amSearchRow .am-search { font-size:14px; }" in css
    # Season controls live in the sheet, not the top row.
    assert body.index('id="amSeasonMulti"') > body.index('id="amFilterSheet"')


def test_add_metric_is_a_solid_button_not_a_ghost():
    html = _html()
    css = _rework_css(html)
    assert "#amAddStatBtn {" in css
    assert "border:1px solid var(--border); border-radius:8px;" in css
    assert "background:var(--card);" in css
    # Boxy per --radius-pill: no new stadium pills introduced by the rework.
    assert "border-radius:999px" not in css


def test_decide_presets_get_edge_fade_and_no_label():
    html = _html()
    css = _rework_css(html)
    # The "Decide:" label was removed from the markup by the desktop rework.
    assert "am-decisions-label" not in html.split("<style>")[0]
    assert "-webkit-mask-image:linear-gradient(to right,#000 88%,transparent 100%);" in css
    assert "mask-image:linear-gradient(to right,#000 88%,transparent 100%);" in css
    # Accessible name is preserved on the pill group itself.
    assert 'aria-label="Decision views"' in html


def test_compare_chips_ride_one_flat_line_with_clear_all_link():
    html = _html()
    css = _rework_css(html)
    # Same treatment as the presets rail: single line, sideways scroll,
    # edge fade, chips never shrink.
    assert ".am-compare-bar { flex-wrap:nowrap; }" in css
    assert "flex-wrap:nowrap; overflow-x:auto; -webkit-overflow-scrolling:touch;" in css
    assert ".am-compare-chips .am-chip { flex-shrink:0; }" in css
    assert ".am-compare-chips { flex-wrap:wrap" not in css
    # The ghost pill hides on mobile; the text link takes over.
    assert "#amClearExtrasBtn { display:none !important; }" in css
    assert ".am-clear-link.am-clear-link-show { display:inline-block; }" in css
    tag = _tag(html, "amClearExtrasLink")
    assert 'class="am-clear-link"' in tag
    assert 'onclick="amClearExtras()"' in tag
    assert ">Clear all</button>" in html
    # JS toggles the link visibility alongside the desktop button.
    assert "am-clear-link-show" in html


def test_field_averages_collapse_to_one_liner():
    html = _html()
    css = _rework_css(html)
    # The stray "|" (am-avg-swatch) is gone.
    assert "am-avg-swatch" not in html
    # Collapsible markup.
    toggle = _tag(html, "amAvgToggle")
    assert 'aria-expanded="false"' in toggle
    assert 'aria-controls="amAvgGrid"' in toggle
    assert 'id="amAvgToggleText"' in html
    assert 'id="amAvgGrid"' in html
    # Mobile CSS: paragraph hides, toggle + grid take over.
    assert ".am-avg-full { display:none; }" in css
    assert ".am-avg-note.am-avg-open .am-avg-grid {" in css
    # JS builds the one-liner ("Field avg · <Metric> <value>") and the grid,
    # and persists the open state across re-renders.
    js = html[html.index("<script>"):html.index("</script>")]
    assert "amAvgToggleText" in js
    assert "am-avg-open" in js
    assert "aria-expanded" in js


def test_results_table_has_fade_and_sticky_player_column():
    html = _html()
    css = _rework_css(html)
    assert ".am-table-wrap {" in css
    assert "mask-image:linear-gradient(to right,#000 92%,transparent 100%);" in css
    # Sticky player column lives in the base table CSS so it applies on all
    # viewports (promoted from the old mobile-only rule).
    base = _style(html)
    sticky = ".am-table thead th.am-player, .am-table tbody td.am-player {"
    assert sticky in base
    block = base[base.index(sticky):base.index(sticky) + 260]
    assert "position:sticky" in block
    assert "left:0" in block
    assert "background:var(--card)" in block
    # The thead cell paints above horizontally scrolling body cells.
    assert ".am-table thead th.am-player { z-index:3; }" in base
    # Row washes (hover / owned / pinned) stay opaque on the sticky cells.
    assert ".am-table tbody tr.am-row:hover td.am-player" in base
    assert ".am-table tbody tr.am-row.am-owned td.am-player" in base
    assert ".am-table tbody tr.am-row.am-owned:hover td.am-player" in base
    assert ".am-table tbody tr.am-row.am-pinned td.am-player" in base
    assert ".am-table tbody tr.am-row.am-pinned.am-owned td.am-player" in base


def test_add_metric_stays_visible_on_mobile():
    # Command-bar rework: + Metric moved into the sheet's Metrics section
    # (it is a secondary control; the strip keeps the metric picker visible).
    html = _html()
    body = html.split("<style>")[0]
    _tag(html, "amAddStatBtn")  # still present with its id and handlers
    assert body.index('id="amAddStatBtn"') > body.index('id="amFilterSheet"')
