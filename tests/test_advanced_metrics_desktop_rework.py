"""Desktop visual-rework contracts for the Advanced Metrics toolbar.

Kaedon approved the desktop mockup (icon-only header actions, no all-caps
control labels, compact control rows, solid + Metric / + Filter buttons,
Usage-trends / My-roster toggle chips instead of checkboxes, unlabeled
preset pills, single-row week range with a slimmer slider, Clear-all text
link, collapsible field averages). The data table and all data logic are
untouched, and the approved mobile rework keeps its behavior: desktop-only
overrides stay inside min-width:601px.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def _desktop_css(html):
    """Only the desktop rework block appended by this change."""
    style = _style(html)
    start = style.index("desktop visual rework")
    return style[start:]


def _desktop_min_width_block(css):
    """The min-width:601px section of the desktop block (desktop-only rules)."""
    start = css.index("@media (min-width:601px)")
    return css[start:]


def _tag(html, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), html)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_header_actions_are_icon_only_on_desktop():
    html = _html()
    css = _desktop_css(html)
    # Unguarded: mobile already hides these, so sharing the rule is safe.
    assert ".am-head .am-legend-btn-label { display:none; }" in css
    assert ".am-head .am-legend-btn { padding:8px; }" in css
    for el_id in ("amGraphBtn", "amLegendBtn", "amExportBtn"):
        assert 'title="' in _tag(html, el_id), f"#{el_id} needs a title for the icon-only state"


def test_control_labels_removed_with_aria_names():
    html = _html()
    body = html.split("<style>")[0]
    assert ">Primary Metric<" not in body
    assert ">Seasons<" not in body
    assert ">Search<" not in body
    assert 'aria-label="Primary metric"' in _tag(html, "amMetricBtn")
    assert 'aria-label="Search players"' in _tag(html, "amSearch")
    assert 'aria-label="Select seasons"' in _tag(html, "amSeasonBtn")


def test_row1_is_one_clean_row_on_desktop():
    css = _desktop_min_width_block(_desktop_css(_html()))
    assert "#amControls { align-items:center; }" in css
    assert "#amControls > .am-ctrl:first-child { flex:0 1 250px; min-width:0; }" in css
    assert ".am-ctrl-search { flex:1 1 200px; min-width:160px; }" in css


def test_row2_has_solid_buttons_divider_and_no_custom_echo():
    html = _html()
    css = _desktop_min_width_block(_desktop_css(html))
    for el_id in ("amAddStatBtn", "amAddFilterBtn", "amSaveSetBtn", "amDeleteSetBtn"):
        _tag(html, el_id)  # still present
    assert "#amAddStatBtn, #amAddFilterBtn, #amSaveSetBtn, #amDeleteSetBtn {" in css
    assert "border:1px solid var(--border); border-radius:var(--radius-pill,8px);" in css
    # Boxy per --radius-pill: no new stadium pills introduced by the rework.
    assert "border-radius:999px" not in _desktop_css(html)
    # Thin divider before the toggles; the redundant "Custom" echo hides.
    assert 'class="am-ctl-divider"' in html
    assert ".am-ctl-divider { display:none; }" in _desktop_css(html)
    assert ".am-ctl-divider { display:block; width:1px; height:24px;" in css
    assert "#amActiveSet { display:none; }" in css


def test_view_options_are_toggle_chips_with_working_checkboxes():
    html = _html()
    css = _desktop_css(html)
    for wrap_id, chk_id in (("amTrendToggleWrap", "amTrendToggle"),
                            ("amRosterToggleWrap", "amRosterToggle")):
        wrap = _tag(html, wrap_id)
        assert "am-toggle-chip" in wrap, f"#{wrap_id} must be a toggle chip"
        assert "am-mobile-filter" in wrap, f"#{wrap_id} must keep collapsing on mobile"
        chk = _tag(html, chk_id)
        assert 'type="checkbox"' in chk, f"#{chk_id} checkbox must survive for existing JS"
        assert "am-toggle-input" in chk
    assert 'class="am-toggle-box"' in html
    assert 'class="am-toggle-text"' in html
    # Checked state paints the box; mobile falls back to the native checkbox.
    assert ".am-toggle-chip .am-toggle-input:checked + .am-toggle-box {" in css
    assert "background:var(--accent,#2563eb);" in css
    assert ".am-toggle-chip .am-toggle-box { display:none; }" in css


def test_decide_presets_have_no_label_but_keep_tagline():
    html = _html()
    assert "am-decisions-label" not in html.split("<style>")[0]
    assert 'aria-label="Decision views"' in html
    tagline = _tag(html, "amPresetTagline")
    assert 'aria-live="polite"' in tagline


def test_week_range_is_one_compact_row_with_slim_slider():
    html = _html()
    css = _desktop_min_width_block(_desktop_css(html))
    assert "#amWeekCtrl { display:flex; flex-direction:row; align-items:center; gap:14px; }" in css
    assert "#amWeekCtrl .am-weekbar-head .am-ctrl-label { display:none; }" in css
    assert "#amWkBarHost { flex:1 1 auto; min-width:0; }" in css
    # Slimmer slider, scoped to the AM host: the shared wk-bar component
    # (player/compare modals) keeps its size.
    assert "#amWkBarHost .wk-bar-track { height:24px; }" in css
    assert "#amWkBarHost .wk-bar-grip { height:22px; width:20px; }" in css
    assert ".wk-bar-track { height:24px; }" not in css.replace("#amWkBarHost .wk-bar-track", "")


def test_clear_all_link_replaces_ghost_button_on_desktop():
    html = _html()
    css = _desktop_css(html)
    assert "#amClearExtrasBtn { display:none !important; }" in css
    assert ".am-clear-link.am-clear-link-show { display:inline-block; }" in css
    tag = _tag(html, "amClearExtrasLink")
    assert 'onclick="amClearExtras()"' in tag
    assert ">Clear all</button>" in html


def test_field_averages_collapse_on_desktop():
    html = _html()
    css = _desktop_min_width_block(_desktop_css(html))
    toggle = _tag(html, "amAvgToggle")
    assert 'aria-expanded="false"' in toggle
    assert 'aria-controls="amAvgGrid"' in toggle
    assert ".am-avg-toggle {" in css
    assert "display:flex;" in css[css.index(".am-avg-toggle {"):css.index(".am-avg-toggle {") + 200]
    assert ".am-avg-full { display:none; }" in css
    assert ".am-avg-note.am-avg-open .am-avg-grid {" in css
    assert "grid-template-columns:repeat(auto-fill,minmax(220px,1fr));" in css


def test_what_changed_strip_stays_but_quieter():
    html = _html()
    css = _desktop_min_width_block(_desktop_css(html))
    _tag(html, "amMovers")
    _tag(html, "amMoversHead")
    assert ".am-movers { padding:8px 14px; }" in css


def test_results_table_is_untouched():
    html = _html()
    body = html.split("<style>")[0]
    thead = body[body.index("<thead>"):body.index("</thead>")]
    for cls in ("am-rank", "am-player", "am-season-col", "am-games", "am-weeks", "am-barcell"):
        assert cls in thead, f"table header lost .{cls}"
    assert 'id="amTableBody"' in body
    assert 'id="amPagination"' in body
    # No table CSS in the desktop block at all.
    assert "am-table" not in _desktop_css(html)
