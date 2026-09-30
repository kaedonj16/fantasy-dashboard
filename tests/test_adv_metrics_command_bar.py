"""Command-bar rework contracts for the Advanced Metrics top (mock A).

Kaedon picked the Command bar direction: a slim sticky strip up top (title +
search / filter / overflow actions), the primary metric picker, a position
segmented control, and decision presets always visible. Everything else
(time, players, metrics, filters, sets, view) lives in one filter sheet
behind the filter icon with a count badge. The old scattered toolbar
(#amToolbar, #amControls, .am-subcontrols, #amFilterBar, #amFiltersBtn) is
gone; every control keeps its ID and JS behavior.
"""
import re

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
    "amMetricsSec",
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
    # The overflow menu holds the three existing actions with IDs/handlers intact.
    menu = _tag(body, "amMoreMenu")
    assert 'hidden' in menu
    assert 'onclick="amOpenGraph()"' in _tag(body, "amGraphBtn")
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
    tail = body[body.index('id="amFilterSheet"'):]
    # Grouped, labeled sections.
    for section in ("Time", "Players", "Metrics", "Filters", "Sets", "View"):
        assert f">{section}</h4>" in tail, f"sheet section {section} missing"
    for el_id in ("amSeasonMulti", "amCombineToggle", "amQuickRanges", "amWkBarHost",
                  "amTeamFilter", "amGamesCtrl", "amMinGames", "amAgeWrap",
                  "amTrendToggleWrap", "amRosterToggleWrap", "amCompareBar",
                  "amAddStatWrap", "amAddStatBtn", "amStatPicker", "amAddFilterBtn",
                  "amFilterChips", "amFilterForm", "amSavedSet", "amSaveSetBtn",
                  "amDeleteSetBtn", "amSortBtn"):
        assert f'id="{el_id}"' in tail, f"#{el_id} must live in the filter sheet"


def test_mobile_metric_controls_relocate_beside_positions():
    """On phones the + Metric wrap and the added-metric compare bar move out
    of the filter sheet: the button sits beside the position filters and the
    chips land directly under that row. The nodes themselves move (one
    instance of each ID); desktop keeps them in the sheet."""
    body = _body(_html())
    # Hosts: the add-stat host is inside the positions row, after the
    # segmented control; the compare host is the next block under the row,
    # before the decision pills.
    row_idx = body.index('id="amPosRow"')
    pos_idx = body.index('id="amPositions"')
    host_idx = body.index('id="amAddStatHost"')
    bar_host_idx = body.index('id="amCompareHost"')
    pills_idx = body.index('id="amDecisionPills"')
    assert row_idx < pos_idx < host_idx < bar_host_idx < pills_idx
    # Hosts start empty; JS fills them on mobile only.
    assert '<div id="amAddStatHost" class="am-add-stat-host"></div>' in body
    assert '<div id="amCompareHost" class="am-compare-host"></div>' in body
    # Server markup still homes both controls in the sheet's Metrics section
    # (the desktop layout); relocation is a runtime move, not a duplicate.
    sec_idx = body.index('id="amMetricsSec"')
    assert sec_idx < body.index('id="amAddStatWrap"') < body.index('id="amCompareBar"')
    js = _html()[_html().index("<script>"):_html().index("</script>")]
    assert "function amRelocateMetricControls(" in js
    assert "matchMedia('(max-width: 760px)')" in js
    assert "getElementById('amAddStatHost')" in js
    assert "getElementById('amCompareHost')" in js
    # The stat picker stays viewport-fixed off the button's rect, so it is
    # safe no matter which home the wrap currently sits in.
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
