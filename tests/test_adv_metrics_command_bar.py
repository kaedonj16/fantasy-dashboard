"""Command-bar rework contracts for the Advanced Metrics top (mock A).

Kaedon picked the Command bar direction: a slim sticky strip up top (title +
search / filter / overflow actions), the primary metric picker, a position
segmented control, and decision presets always visible. Everything else
(time, players, filters, sets, view) lives in one filter sheet behind the
filter icon with a count badge. The + Metric wrap and the added-metric
compare bar sit with the position row instead of in the sheet, and Graph
Metrics is a first-class strip action rather than an overflow-menu row. The old scattered toolbar
(#amToolbar, #amControls, .am-subcontrols, #amFilterBar, #amFiltersBtn) is
gone; every control keeps its ID and JS behavior.
"""
import os
import re
import shutil
import subprocess

import pytest

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS

# Every control must survive the rework with exactly one element carrying it.
KEY_IDS = (
    "amFilterSheet",
    "amSheetBackdrop",
    "amContextLine",
    "amFilterBadge",
    "amSearch",
    "amSearchToggle",
    "amSearchRow",
    "amMoreToggle",
    "amMoreMenu",
    "amMetricBtn",
    "amSeasonMulti",
    "amQuickRanges",
    "amWkBarHost",
    "amTeamFilter",
    "amMinGames",
    "amTrendToggle",
    "amRosterToggle",
    "amAddStatBtn",
    "amAddFilterBtn",
    "amSavedSet",
    "amActiveSet",
    "amSaveSetBtn",
    "amDeleteSetBtn",
    "amSortBtn",
    "amDecisionPills",
    "amPresetTagline",
    "amMovers",
    "amMoversSummary",
    "amCompareBar",
    "amCompareChips",
    "amFilterChips",
    "amPosRow",
    "amAddStatHost",
    "amCompareHost",
    "amSheetClose",
)

# getElementById targets that are intentionally not in the static markup:
# dynamic prefixes (used with string concatenation) or elements the JS itself
# injects later (stat-picker search, pagination, retry). Same set as main.
DYNAMIC_IDS = {
    "amCmpWk_",
    "amExtraHeader_",
    "amFilterColHdr_",
    "amLoading",
    "amMetricInfo",
    "amPageNext",
    "amPagePrev",
    "amRetryBtn",
    "amSpSearch",
}


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html.split("<style>")[1].split("</style>")[0]


def test_mobile_sheet_clears_the_bottom_dock():
    """The phone bottom sheet must sit above the fixed bottom dock
    (var(--dock-safe-bottom): 56px + safe-area), never behind it."""
    css = _style(_html())
    m = re.search(r"@media\s*\(max-width:\s*760px\)\s*\{(.*?)\n      \}\n    </style>", css, re.S)
    mobile = m.group(1) if m else css
    assert "var(--dock-safe-bottom" in mobile, "sheet must anchor above the dock"
    assert re.search(r"\.am-filter-sheet\s*\{[^}]*bottom:\s*calc\(var\(--dock-safe-bottom", mobile), \
        "sheet bottom edge must clear the dock"


def _body(html):
    return html.split("<style>")[0]


def _tag(body, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), body)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_every_control_survives_exactly_once():
    body = _body(_html())
    for el_id in KEY_IDS:
        count = len(re.findall(r'\bid="%s"' % re.escape(el_id), body))
        assert count == 1, f"#{el_id} must appear exactly once, found {count}"


def test_old_toolbar_layout_is_gone():
    html = _html()
    body = _body(html)
    for el_id in ("amFiltersBtn", "amToolbar", "amControls", "amFilterBar",
                  "amAddFilterBtnM", "amSeasonCtrl"):
        assert f'id="{el_id}"' not in body, f"#{el_id} should be removed by the rework"
    assert "am-subcontrols" not in body
    assert "am-ctl-divider" not in body


def test_command_strip_structure():
    body = _body(_html())
    assert 'id="amCmdBar"' in body
    for el_id in ("amSearchToggle", "amFilterToggle", "amMoreToggle"):
        tag = _tag(body, el_id)
        assert 'aria-expanded="false"' in tag, f"#{el_id} needs aria-expanded"
    # The overflow menu holds the glossary and CSV actions with IDs/handlers
    # intact; Graph Metrics moved out of it into the strip actions.
    menu = _tag(body, "amMoreMenu")
    assert 'hidden' in menu
    assert 'onclick="amOpenGraph()"' in _tag(body, "amGraphBtn")
    actions = body[body.index('class="am-cmd-actions"'):body.index('id="amMoreMenu"')]
    assert 'id="amGraphBtn"' in actions, "#amGraphBtn must live in .am-cmd-actions"
    menu_span = body[body.index('id="amMoreMenu"'):body.index('id="amSearchRow"')]
    assert 'id="amGraphBtn"' not in menu_span, "#amGraphBtn must leave the overflow menu"
    for el_id in ("amLegendBtn", "amExportBtn"):
        assert f'id="{el_id}"' in menu_span, f"#{el_id} stays in the overflow menu"
    legend = _tag(body, "amLegendBtn")
    assert "amLegendModal" in legend
    _tag(body, "amExportBtn")  # CSV keeps its id; JS wires the click listener
    # Sheet + backdrop start hidden; the search row starts hidden too.
    assert "hidden" in _tag(body, "amFilterSheet")
    assert "hidden" in _tag(body, "amSheetBackdrop")
    assert "hidden" in _tag(body, "amSearchRow")


def test_positions_are_a_segmented_control():
    body = _body(_html())
    tag = _tag(body, "amPositions")
    assert "am-segmented" in tag, "#amPositions must read as one segmented unit"
    for pos in ("ALL", "QB", "RB", "WR", "TE"):
        assert f'data-pos="{pos}"' in body, f"position {pos} missing"
    # The existing active-class toggling contract is untouched.
    assert 'class="otc-day-filter am-pos active" data-pos="ALL"' in body


def test_sheet_groups_every_secondary_control():
    body = _body(_html())
    # The sheet element only: everything up to the card body that follows it.
    tail = body[body.index('id="amFilterSheet"'):body.index('class="card-body"')]
    # Grouped, labeled sections.
    for section in ("Time", "Players", "Filters", "Sets", "View"):
        assert f">{section}</h4>" in tail, f"sheet section {section} missing"
    for el_id in ("amSeasonMulti", "amCombineToggle", "amQuickRanges", "amWkBarHost",
                  "amTeamFilter", "amGamesCtrl", "amMinGames", "amAgeWrap",
                  "amTrendToggleWrap", "amRosterToggleWrap", "amAddFilterBtn",
                  "amFilterChips", "amFilterForm", "amSavedSet", "amSaveSetBtn",
                  "amDeleteSetBtn", "amSortBtn"):
        assert f'id="{el_id}"' in tail, f"#{el_id} must live in the filter sheet"
    # The metric controls moved out of the sheet entirely: there is no
    # Metrics section anymore, and neither control renders inside the sheet.
    assert ">Metrics</h4>" not in tail, "the sheet's Metrics section is gone"
    for el_id in ("amMetricsSec", "amCompareBar", "amCompareChips", "amAddStatWrap",
                  "amAddStatBtn", "amStatPicker"):
        assert f'id="{el_id}"' not in tail, f"#{el_id} must NOT live in the filter sheet"


def test_metric_controls_live_beside_positions_at_every_width():
    """The + Metric wrap and the added-metric compare bar live beside /
    directly under the position filters in the server markup itself, at
    every width: the button sits inside the positions row and the chips
    land in the host right under it. There is no runtime relocation: the
    nodes render in their only home, so desktop and mobile agree."""
    body = _body(_html())
    # Hosts: the add-stat host is inside the positions row, after the
    # segmented control, and holds the wrap; the compare host is the next
    # block under the row, before the decision pills, and holds the bar.
    row_idx = body.index('id="amPosRow"')
    pos_idx = body.index('id="amPositions"')
    host_idx = body.index('id="amAddStatHost"')
    wrap_idx = body.index('id="amAddStatWrap"')
    bar_host_idx = body.index('id="amCompareHost"')
    bar_idx = body.index('id="amCompareBar"')
    pills_idx = body.index('id="amDecisionPills"')
    assert row_idx < pos_idx < host_idx < wrap_idx < bar_host_idx < bar_idx < pills_idx
    # The relocation machinery is gone: no mover function, no layout
    # matchMedia listener, no sheet Metrics section to hide while empty.
    html = _html()
    js = html[html.index("<script>"):html.index("</script>")]
    assert "amRelocateMetricControls" not in js
    assert "_amLayoutMq" not in js
    assert 'id="amMetricsSec"' not in body
    assert "amMetricsSec" not in js
    # The stat picker stays viewport-fixed off the button's rect, so it is
    # safe from the wrap's permanent home.
    assert "picker.style.position = 'fixed'" in js
    assert "getBoundingClientRect" in js


def test_sheet_close_button_is_pinned_shorter():
    """The sheet close button at the shared 36px command size read too tall;
    it carries its own shorter, fully pinned box model."""
    css = _style(_html())
    m = re.search(r"#amSheetClose\s*\{([^}]*)\}", css)
    assert m, "#amSheetClose needs its own sizing rule"
    rule = m.group(1)
    assert "height:30px" in rule
    assert "width:30px" in rule
    assert "padding:0" in rule
    assert "box-sizing:border-box" in rule


def test_movers_default_to_slim_banner():
    body = _body(_html())
    head = _tag(body, "amMoversHead")
    assert 'aria-expanded="false"' in head, "movers start collapsed"
    _tag(body, "amMoversSummary")
    js = _html()[_html().index("<script>"):_html().index("</script>")]
    # Collapsed by default; an explicit stored choice still wins.
    assert "stored === null ? true : stored === '1'" in js
    # Banner counts derive from the fetched movers data.
    assert "efficiency outlier" in js
    assert "heating up" in js


def test_badge_and_context_line_wiring_exists():
    html = _html()
    js = html[html.index("<script>"):html.index("</script>")]
    assert "function amRefreshFilterBadge()" in js
    assert "function amUpdateContextLine()" in js
    assert "function amToggleSheet(" in js
    # Badge counts non-default states: team, min games, age, combo filters,
    # usage trends, roster-only, week range, seasons.
    for needle in ("state.team", "state.minVol", "state.ageMin", "state.comboFilters",
                   "state.showTrends", "state.rosterOnly", "state.weekRange",
                   "amSelectedSeasons()"):
        assert needle in js[js.index("function amRefreshFilterBadge()"):js.index("function amRefreshFilterBadge()") + 1200], \
            f"badge must consider {needle}"
    # Context line reads seasons, week range, team, sort.
    ctx = js[js.index("function amUpdateContextLine()"):js.index("function amUpdateContextLine()") + 900]
    assert "amSeasonLabel" in ctx
    assert "state.team" in ctx
    assert "state.sortDir" in ctx


def test_no_em_dashes_in_new_copy():
    # Standing repo rule. Scoped to the markup this rework added (the command
    # strip + search row, the movers banner, and the filter sheet). The metric
    # glossary's pre-existing descriptions are out of scope.
    body = _body(_html())
    strip = body[body.index('id="amCmdBar"'):body.index('id="amLegendModal"')]
    top = body[body.index('id="amMovers"'):body.index('id="amCompareModal"')]
    assert "\u2014" not in strip + top, "no em dashes in the new UI copy"


def test_getelementbyid_targets_exist_in_markup():
    # Static guard against JS-selector breakage from moving elements around:
    # every getElementById target in the page script must exist in the markup,
    # except the known dynamic/injected set.
    html = _html()
    body = _body(html)
    js = html[html.index("<script>"):html.index("</script>")]
    used = set(re.findall(r"getElementById\(\s*['\"]([^'\"]+)['\"]", js))
    defined = set(re.findall(r'id="([^"]+)"', body))
    missing = sorted(i for i in used if i not in defined and i not in DYNAMIC_IDS)
    assert not missing, f"JS references IDs missing from markup: {missing}"


def test_graph_button_is_a_strip_icon_button():
    # Graph Metrics joined the strip as an icon button in the same style as
    # search / filters / overflow; its id, handler and accessible name are
    # unchanged so nothing else (deep links, tests, JS) has to move.
    body = _body(_html())
    tag = _tag(body, "amGraphBtn")
    assert 'class="am-cmd-btn"' in tag, "#amGraphBtn must use the strip icon-button style"
    assert 'aria-label="Graph Metrics"' in tag
    assert 'title="Graph Metrics"' in tag
    assert 'onclick="amOpenGraph()"' in tag
    assert "am-legend-btn" not in tag, "the graph button is no longer a menu row"


def test_compare_bar_visibility_drives_host_and_desktop_grid():
    """The compare bar appears only when extras (or pinned compare) exist.
    updateCompareBar is the single place that decides, and it must drive
    the host's display class and the desktop grid's compare row from the
    same state, so nothing below the bar jumps when it toggles."""
    html = _html()
    css = _style(html)
    js = html[html.index("<script>"):html.index("</script>")]
    # The host takes space only while the bar is shown; the add-stat host
    # is always laid out (its row is part of the positions row).
    assert ".am-add-stat-host { display:flex; }" in css
    assert ".am-compare-host { display:none; }" in css
    assert ".am-compare-host.am-compare-on { display:block; }" in css
    # Desktop grid: the base template is untouched (no gap when nothing is
    # picked); the on-state template inserts one full-width compare row
    # between the positions row and the preset pills.
    assert css.count('"compare compare"') == 1
    assert ".am-cmd-controls.am-has-compare" in css
    assert ".am-cmd-controls.am-has-compare .am-compare-host { grid-area:compare; }" in css
    # The one decision point toggles both hooks with the bar's display.
    fn = js[js.index("function updateCompareBar()"):js.index("function buildStatPicker()")]
    assert "const showBar = state.extraMetrics.length > 0 || hasPinned;" in fn
    assert "bar.style.display = showBar ? 'flex' : 'none';" in fn
    assert "classList.toggle('am-compare-on', showBar)" in fn
    assert "classList.toggle('am-has-compare', showBar)" in fn


def test_sheet_and_badge_do_not_depend_on_a_metrics_section():
    """The sheet lost its Metrics section; open/close and the filter-count
    badge must not reference it. The badge never counted compare extras,
    so retiring the section cannot change the count."""
    html = _html()
    body = _body(html)
    js = html[html.index("<script>"):html.index("</script>")]
    assert 'id="amMetricsSec"' not in body
    assert "amMetricsSec" not in js
    assert "function amToggleSheet(" in js
    badge = _extract_fn(js, "amRefreshFilterBadge")
    assert "extraMetrics" not in badge


def test_graph_open_applies_the_compare_selection_preset():
    html = _html()
    js = html[html.index("<script>"):html.index("</script>")]
    assert "function _amGraphPresetFromSelection(applic)" in js
    open_fn = js[js.index("window.amOpenGraph = function()"):js.index("window.amToggleGraphControls")]
    # The preset is computed from the graphable list for the graph-local
    # position and applied before the axis selects are built, so the modal
    # opens already preset and the normal render path draws it.
    assert "_amGraphPresetFromSelection(applic)" in open_fn
    preset_at = open_fn.index("_amGraphPresetFromSelection(applic)")
    selects_at = open_fn.index("xSel.innerHTML = _amGraphMetricOptions(curX)")
    assert preset_at < selects_at


# --- Node behavior tests ---------------------------------------------------
# The page ships its logic as an inline script string; these extract single
# functions from the built page and run them under node with hand-rolled
# stubs, in the style of tests/test_refresh_freshness.py.


def _page_js():
    html = _html()
    return html[html.index("<script>"):html.index("</script>")]


def _extract_fn(js, name):
    m = re.search(r"function %s\([^)]*\) \{.*?\n  \}" % re.escape(name), js, re.S)
    assert m, f"{name} not found in page script"
    return m.group(0)


def _run_node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_graph_preset_maps_the_compare_selection():
    fn = _extract_fn(_page_js(), "_amGraphPresetFromSelection")
    script = (
        "var state = { metric: 'primary_metric', extraMetrics: [] };\n"
        + fn + "\n"
        "function eq(a, b) { return JSON.stringify(a) === JSON.stringify(b); }\n"
        "function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }\n"
        "var applic = ['primary_metric', 'a', 'b', 'c', 'd'];\n"
        "state.extraMetrics = [];\n"
        "check('zero extras keeps the defaults', _amGraphPresetFromSelection(applic) === null);\n"
        "state.extraMetrics = ['a'];\n"
        "check('one extra: primary on X, extra on Y, no bubble',\n"
        "  eq(_amGraphPresetFromSelection(applic), {x: 'primary_metric', y: 'a', z: ''}));\n"
        "state.extraMetrics = ['a', 'b'];\n"
        "check('two extras: X and Y, no bubble',\n"
        "  eq(_amGraphPresetFromSelection(applic), {x: 'a', y: 'b', z: ''}));\n"
        "state.extraMetrics = ['a', 'b', 'c'];\n"
        "check('three extras: X, Y and bubble',\n"
        "  eq(_amGraphPresetFromSelection(applic), {x: 'a', y: 'b', z: 'c'}));\n"
        "state.extraMetrics = ['a', 'b', 'c', 'd'];\n"
        "check('a fourth extra is ignored',\n"
        "  eq(_amGraphPresetFromSelection(applic), {x: 'a', y: 'b', z: 'c'}));\n"
        "state.extraMetrics = ['not_graphable', 'a', 'b'];\n"
        "check('extras the position filter excludes are skipped first',\n"
        "  eq(_amGraphPresetFromSelection(applic), {x: 'a', y: 'b', z: ''}));\n"
        "state.extraMetrics = ['not_graphable'];\n"
        "check('only excluded extras keeps the defaults', _amGraphPresetFromSelection(applic) === null);\n"
    )
    _run_node(script)


@pytest.mark.skipif(os.environ.get("SKIP_NODE") == "1", reason="node tests disabled")
def test_compare_bar_toggles_host_and_grid_classes():
    fn = _extract_fn(_page_js(), "updateCompareBar")
    script = (
        "var state = { extraMetrics: [] };\n"
        "var MAX_COMPARE = 4;\n"
        "function _mLabel(k) { return k; }\n"
        "function makeEl() {\n"
        "  var el = { innerHTML: '', disabled: false, style: {}, toggles: {} };\n"
        "  el.classList = { toggle: function(c, on) { el.toggles[c] = on; } };\n"
        "  return el;\n"
        "}\n"
        "var els = {};\n"
        "['amCompareBar', 'amCompareChips', 'amAddStatBtn', 'amComparePinnedBtn',\n"
        " 'amClearExtrasBtn', 'amClearExtrasLink', 'amCompareHost'].forEach(function(id) { els[id] = makeEl(); });\n"
        "els.amComparePinnedBtn.style.display = 'none';\n"
        "var controlsEl = makeEl();\n"
        "var document = {\n"
        "  getElementById: function(id) { return els[id] || null; },\n"
        "  querySelector: function(sel) { return sel === '.am-cmd-controls' ? controlsEl : null; }\n"
        "};\n"
        + fn + "\n"
        "function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }\n"
        "state.extraMetrics = ['a', 'b'];\n"
        "updateCompareBar();\n"
        "check('bar shows with extras', els.amCompareBar.style.display === 'flex');\n"
        "check('host takes space', els.amCompareHost.toggles['am-compare-on'] === true);\n"
        "check('desktop grid opens its compare row', controlsEl.toggles['am-has-compare'] === true);\n"
        "check('chips render for the extras', els.amCompareChips.innerHTML.indexOf('amRemoveExtra') !== -1);\n"
        "state.extraMetrics = [];\n"
        "updateCompareBar();\n"
        "check('bar hides when cleared', els.amCompareBar.style.display === 'none');\n"
        "check('host collapses', els.amCompareHost.toggles['am-compare-on'] === false);\n"
        "check('grid row closes', controlsEl.toggles['am-has-compare'] === false);\n"
    )
    _run_node(script)
