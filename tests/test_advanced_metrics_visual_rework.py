"""Mobile visual-rework contracts for the Advanced Metrics toolbar.

Kaedon approved the mobile mockup (header + compact icon-only actions, no
shouting control labels, search+season on one row, solid + Metric button,
fading Decide rail, wrapping chips with a "Clear all" text link, collapsible
field averages, sticky player column). Desktop rules are untouched by the
rework block.
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
    html = _html()
    for el_id in ("amGraphBtn", "amLegendBtn", "amExportBtn"):
        tag = _tag(html, el_id)
        assert 'class="am-legend-btn"' in tag, f"#{el_id} must stay a legend button"
        assert 'title="' in tag, f"#{el_id} needs a title for the icon-only state"
    # Labels live in their own spans so mobile can hide them; the description
    # gets a hook so it can be dropped on phones.
    assert html.count('class="am-legend-btn-label"') == 3
    assert 'class="am-head-desc"' in html
    assert 'class="am-head-actions"' in html
    css = _rework_css(html)
    assert ".am-head .am-head-desc { display:none; }" in css
    assert ".am-head .am-legend-btn-label { display:none; }" in css


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
    html = _html()
    css = _rework_css(html)
    assert "#amSeasonCtrl { flex:0 1 128px;" in css
    assert ".am-ctrl-search { flex:1 1 0;" in css
    # Search grows, season stays compact; visual order matches the approved
    # mockup (search left, season right) despite the DOM order.
    assert "#amSeasonCtrl { flex:0 1 128px; min-width:0; order:2; }" in css
    assert ".am-ctrl-search { flex:1 1 0; min-width:0; order:1; }" in css


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


def test_compare_chips_wrap_with_clear_all_link():
    html = _html()
    css = _rework_css(html)
    assert ".am-compare-chips { flex-wrap:wrap; overflow-x:visible;" in css
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
    sticky = ".am-table th.am-player, .am-table td.am-player {"
    assert sticky in css
    block = css[css.index(sticky):css.index(sticky) + 220]
    assert "position:sticky" in block
    assert "left:0" in block
    assert "background:var(--card)" in block


def test_add_metric_stays_visible_on_mobile():
    # No regression of the declutter contract: + Metric is a primary control.
    tag = _tag(_html(), "amAddStatBtn")
    assert "am-mobile-filter" not in tag
