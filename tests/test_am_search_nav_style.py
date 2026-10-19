"""Advanced Metrics player search: presentation vs behaviour contracts.

The AM page used to reveal a full-width #amSearch input in the page flow.
PR #2736 kept the input but floated it in a popover under the command bar
with the nav search's pill styling. Kaedon's correction (2026-10-19): the
control must NOT work like the nav search's modal picker - no results
dropdown, no pick-a-result step. It just filters the names in the list,
exactly as the old in-flow input did: typing sets state.search and
render() substring-filters the leaderboard rows by name, live.
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


def test_search_markup_is_a_pill_popover_not_a_full_width_row():
    body = _body(_html())
    # The bespoke full-width input is gone; the input rides the shared
    # nav-search pill classes with AM-scoped ids, plus a clear button.
    assert 'class="am-search"' not in body
    _tag(body, "amPlayerSearch")
    search = _tag(body, "amSearch")
    assert 'class="nav-search-input am-player-search-input"' in search
    assert 'aria-label="Search players"' in search
    _tag(body, "amSearchClear")
    # No results dropdown element exists at all.
    assert 'id="amSearchDropdown"' not in body
    assert "nav-search-dropdown" not in body
    # Anchored inside the sticky command bar so the popover tracks the strip.
    assert body.index('id="amCmdBar"') < body.index('id="amSearchRow"') < body.index('id="amFilterSheet"')
    assert "hidden" in _tag(body, "amSearchRow")


def test_search_popover_floats_above_the_page():
    css = _style(_html())
    row = re.search(r"\.am-search-row\s*\{([^}]*)\}", css)
    assert row, ".am-search-row rule missing"
    assert "position:absolute" in row.group(1)
    assert "top:calc(100% + 6px)" in row.group(1)
    assert "width:min(430px, calc(100vw - 32px))" in row.group(1)
    # The page CSS styles no dropdown for this control any more.
    assert "nav-search-dropdown" not in css


def test_typing_filters_names_live_with_no_pick_a_result_model():
    js = _js(_html())
    # The exact pre-#2736 semantics: the input listener sets state.search
    # from the trimmed value, resets paging, and re-renders, immediately
    # (no debounce, no intermediate results model).
    assert "state.search = searchEl.value.trim(); state.page = 0; _amSyncSearchClear(); render();" in js
    # render() applies that as a name substring filter over display rows.
    assert "const q = state.search.toLowerCase();" in js
    assert "displayRows = displayRows.filter(r => (r.name || '').toLowerCase().includes(q));" in js
    # The whole nav-search selection model is gone: no payload fetch, no
    # result rows, no group label, no matcher, no apply-on-select path,
    # no arrow-key selection.
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


def test_popover_closes_on_escape_and_outside_tap_without_clearing_the_filter():
    js = _js(_html())
    closer = js[js.index("function amCloseSearchPopover"):js.index("if (searchEl) {")]
    assert "row.hidden = true;" in closer
    # Closing the popover never touches the filter: the typed name keeps
    # filtering the board while the popover is closed, as the old
    # in-flow input's text did.
    assert "state.search" not in closer
    assert "e.key === 'Escape'" in js
    assert "amCloseSearchPopover();" in js  # outside-tap document listener


def test_mover_chips_and_deep_link_drive_the_same_filter():
    js = _js(_html())
    # What-changed mover chips write their name into the input and the
    # same state.search, so the chip name shows in the search box.
    assert "searchEl.value = m.name; state.search = m.name; state.page = 0;" in js
    # The ?search= deep link seeds both the input and the filter.
    assert "searchEl.value = _searchInit;" in js
    assert "state.search = _searchInit;" in js
