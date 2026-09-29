"""Command-bar collapsing contracts for the Advanced Metrics top.

The old design hid secondary controls behind the Filters toggle on phones
(`am-mobile-filter`). The command-bar rework replaces that mechanism: every
secondary control now lives in the filter sheet (#amFilterSheet, toggled by
#amFilterToggle with a count badge), and the primary controls (metric picker,
positions, decision presets, context line) stay visible. Nothing is dropped:
each moved control keeps its ID and JS behavior.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS

# Secondary controls: grouped into the sheet instead of the page top.
SHEET_IDS = (
    "amSavedSet",
    "amActiveSet",
    "amSaveSetBtn",
    "amDeleteSetBtn",
    "amTrendToggleWrap",
    "amRosterToggleWrap",
    "amSeasonMulti",
    "amCombineToggle",
    "amQuickRanges",
    "amWkBarHost",
    "amTeamFilter",
    "amGamesCtrl",
    "amAgeWrap",
    "amAddFilterBtn",
    "amFilterChips",
    "amSortBtn",
)

# Primary controls: always visible, never inside the sheet.
VISIBLE_IDS = (
    "amMetricBtn",
    "amSearchToggle",
    "amFilterToggle",
    "amMoreToggle",
    "amPositions",
    "amDecisionPills",
    "amContextLine",
)


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _body(html):
    return html.split("<style>")[0]


def _tag(body, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), body)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_secondary_controls_live_in_the_filter_sheet():
    body = _body(_html())
    sheet_start, sheet_end = _sheet_span(body)
    for el_id in SHEET_IDS:
        idx = body.index('id="%s"' % el_id)
        assert sheet_start < idx < sheet_end, \
            f"#{el_id} must live inside the filter sheet"


def _sheet_span(body):
    start = body.index('id="amFilterSheet"')
    # The sheet's opening tag starts at the <div before the id.
    open_idx = body.rindex("<div", 0, start)
    depth = 0
    i = open_idx
    while True:
        nxt_open = body.find("<div", i + 1)
        nxt_close = body.find("</div>", i + 1)
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            i = nxt_open
        else:
            if depth == 0:
                return open_idx, nxt_close + len("</div>")
            depth -= 1
            i = nxt_close


def test_primary_controls_stay_visible_outside_the_sheet():
    body = _body(_html())
    sheet_start, sheet_end = _sheet_span(body)
    for el_id in VISIBLE_IDS:
        idx = body.index('id="%s"' % el_id)
        assert not (sheet_start < idx < sheet_end), \
            f"#{el_id} must stay visible outside the sheet"


def test_old_filters_toggle_is_gone():
    body = _body(_html())
    assert 'id="amFiltersBtn"' not in body
    # One entry point replaces it: the strip's filter icon with a count badge.
    _tag(body, "amFilterToggle")
    _tag(body, "amFilterBadge")
    js = _html()[_html().index("<script>"):_html().index("</script>")]
    assert "function amToggleSheet(" in js


def test_sheet_is_a_bottom_sheet_on_mobile_and_panel_on_desktop():
    html = _html()
    style = html[html.index("<style>"):html.index("</style>")]
    css = style[style.index("command bar rework"):]
    assert "@media (max-width:760px)" in css
    assert "position:fixed" in css
    assert "@media (min-width:761px)" in css
    assert "position:absolute" in css


def test_sheet_has_a_positioned_ancestor_for_the_desktop_panel():
    # The desktop panel is absolutely positioned under the strip; the shell
    # wrapping strip + sheet must be the positioned ancestor.
    body = _body(_html())
    shell_idx = body.index('class="am-cmd-shell"')
    strip_idx = body.index('id="amCmdBar"')
    sheet_idx = body.index('id="amFilterSheet"')
    legend_idx = body.index('id="amLegendModal"')
    assert shell_idx < strip_idx < sheet_idx < legend_idx, \
        "shell must wrap the strip and the sheet"
    style = _html()
    css = style[style.index("<style>"):style.index("</style>")]
    assert ".am-cmd-shell { position:relative; }" in css
