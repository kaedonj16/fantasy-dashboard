"""Advanced Metrics player search: presentation vs behaviour contracts.

The AM page used to reveal a full-width #amSearch input in the page flow;
PR #2736 floated it in a popover under the command bar. Kaedon's final
correction (2026-10-19): the search must extend out to the left from its
icon like the top nav search does. So the control now lives IN the
command bar as the shared .nav-search-* box - collapsed to a 36px icon
in .am-cmd-actions, widening under :focus-within while the other bar
buttons stay pinned right. The behaviour stays this page's own plain
name filter, exactly as the old in-flow input: typing sets state.search
and render() substring-filters the leaderboard rows by name, live. No
popover, no results dropdown, no pick-a-result step.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _body(html):
    return html.split("<style>")[0]


def _style(html):
    return html.split("<style>")[1].split("</style>")[0]


def _js(html):
    return html[html.index("<script>"):html.index("</script>")]


def _tag(body, el_id):
    m = re.search(r'<[^>]*\bid="%s"[^>]*>' % re.escape(el_id), body)
    assert m, f"#{el_id} missing from page"
    return m.group(0)


def test_search_markup_is_the_expanding_bar_box_not_a_popover():
    body = _body(_html())
    # The popover row and its toggle button are gone entirely.
    for gone in ("amSearchRow", "amSearchToggle", "amPlayerSearch", "am-search-row"):
        assert gone not in body, f"{gone} should be gone from the AM search"
    # The input rides the shared nav-search classes inside the expanding
    # wrapper, with the clear button - same chrome as the top nav search.
    _tag(body, "amSearchExpand")
    search = _tag(body, "amSearch")
    assert 'class="nav-search-input am-search-input"' in search
    assert 'aria-label="Search players"' in search
    _tag(body, "amSearchClear")
    # No results dropdown element exists at all.
    assert 'id="amSearchDropdown"' not in body
    assert "nav-search-dropdown" not in body


def test_search_box_sits_in_the_command_bar_actions():
    body = _body(_html())
    # The expander is the first control in .am-cmd-actions, so its growth
    # extends leftward out of the icon while graph / filter / more stay
    # pinned to the bar's right edge.
    actions = body[body.index('class="am-cmd-actions"'):body.index('id="amMoreMenu"')]
    assert 'id="amSearchExpand"' in actions
    assert actions.index('id="amSearchExpand"') < actions.index('id="amGraphBtn"')
    assert body.index('id="amCmdBar"') < body.index('id="amSearchExpand"') < body.index('id="amFilterSheet"')


def test_search_css_mirrors_the_nav_expansion():
    css = _style(_html())
    # No floating panel geometry remains for this control.
    assert ".am-search-row" not in css
    assert "#amSearchRow" not in css
    # The page only aligns the collapsed box with the .am-cmd-btn
    # geometry; the :focus-within width transition itself is the shared
    # dashboard.css .nav-search-* pattern, not re-declared here.
    inner = re.search(r"\.am-search-expand \.nav-search-inner\s*\{([^}]*)\}", css)
    assert inner, ".am-search-expand .nav-search-inner rule missing"
    assert "width:36px" in inner.group(1)
    assert "height:36px" in inner.group(1)
    assert "nav-search-dropdown" not in css
    # Mobile keeps the slightly larger input text (anti iOS focus zoom).
    assert "#amSearchExpand .am-search-input { font-size:15px; }" in css


def test_icon_click_expands_and_focuses_the_input():
    js = _js(_html())
    inner = js[js.index("searchInner.addEventListener('click'"):]
    inner = inner[:inner.index("});")]
    assert "if (document.activeElement !== searchEl) searchEl.focus();" in inner


def test_typing_filters_names_live_with_no_pick_a_result_model():
    js = _js(_html())
    # The exact pre-#2736 semantics: the input listener sets state.search
    # from the trimmed value, resets paging, and re-renders, immediately
    # (no debounce, no intermediate results model).
    assert "state.search = searchEl.value.trim(); state.page = 0; _amSyncSearchClear(); render();" in js
    # render() applies that as a name filter over display rows, through the
    # shared brNameMatch matcher (normalized / compact / initials /
    # token-prefix / typo tiers), substring fallback if it is unavailable.
    assert "const q = state.search.toLowerCase();" in js
    assert "displayRows = displayRows.filter(r => (typeof brScorePlayerName === 'function') ? brScorePlayerName(r, q) > 0 : (r.name || '').toLowerCase().includes(q));" in js
    # The whole nav-search selection model stays gone: no payload fetch,
    # no result rows, no group label, no matcher, no apply-on-select
    # path, no arrow-key selection.
    for gone in (
        "fetch('/api/league-players?view=search')",
        "amSearchDropdown",
        "amApplySearchPlayer",
        "_amSearchMatchesFor",
        "_amLoadSearchPlayers",
        "_amRenderSearchResults",
        "nav-search-result",
        "nav-search-group-label",
        "e.key === 'ArrowDown'",
        "e.key === 'ArrowUp'",
    ):
        assert gone not in js, f"{gone} should be gone from the AM search"


def test_clear_button_restores_the_full_list():
    js = _js(_html())
    clear = js[js.index("searchClear.addEventListener('click'"):]
    clear = clear[:clear.index("});")]
    assert "searchEl.value = '';" in clear
    assert "state.search = '';" in clear
    assert "state.page = 0;" in clear
    assert "render();" in clear


def test_escape_collapses_without_clearing_the_filter():
    js = _js(_html())
    # Escape just blurs: losing :focus-within collapses the box back to
    # the icon, exactly like the nav search. The popover closer and its
    # outside-tap document listener are gone.
    assert "if (e.key === 'Escape') searchEl.blur();" in js
    assert "amCloseSearchPopover" not in js
    assert "row.hidden" not in js
    # Collapsing never touches the filter: the typed name keeps filtering
    # the board while the box is collapsed, as the old in-flow input's
    # text did.


def test_mover_chips_and_deep_link_drive_the_same_filter():
    js = _js(_html())
    # What-changed mover chips write their name into the input and the
    # same state.search, so the chip name shows in the search box.
    assert "searchEl.value = m.name; state.search = m.name; state.page = 0;" in js
    # The ?search= deep link seeds both the input and the filter.
    assert "searchEl.value = _searchInit;" in js
    assert "state.search = _searchInit;" in js
