"""Enter-to-select contract for player pickers (Class E audit).

The nav search and the standalone compare page pickers act on Enter:
pressing Enter picks the top/focused match instead of leaving the input
filled with nowhere to go. The trade calculator side searches, the Trade
Hub suggestions search, the in-modal compare search, and the Trade
Database searches only wired mouse clicks, and their dropdown items are
not focusable, so a keyboard user could not select a player at all.
These tests lock the Enter handler on each of those surfaces, plus the
waiver start/sit guest picker's keyboard path (button activation fires
click, not the mousedown the pick was bound to).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
TRADE_BP = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
WAIVERS = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")


def _region(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


def test_sibling_contract_nav_and_compare_page_act_on_enter():
    # The surfaces the fixed pickers are being aligned to.
    nav = _region(APP_JS, "function selectCurrent()", "dropdown.addEventListener('click'")
    assert "openPlayerModal" in nav
    compare = _region(APP_JS, "function _bindPicker(slot)", "_bindPicker(1);")
    assert "e.key === 'Enter'" in compare and "_select(items[" in compare


def test_trade_calculator_side_search_enter_selects_top_match():
    region = _region(APP_JS, "function setupSearch(side)", "function bindPickButtons()")
    assert 'addEventListener("keydown"' in region
    assert 'e.key !== "Enter"' in region
    assert 'dropdown.querySelector(".otc-dropdown-item")' in region
    assert "first.click()" in region


def test_trade_hub_suggestions_search_enter_selects_top_match():
    region = _region(APP_JS, 'playerInput.addEventListener("input"',
                     "playerDropdown.addEventListener(\"click\"")
    assert 'playerInput.addEventListener("keydown"' in region
    assert 'playerDropdown.querySelector(".otc-sugg-dropdown-item")' in region


def test_compare_modal_search_enter_selects_top_result():
    region = _region(APP_JS, "function openCompareSearch(player1Data)",
                     "function _computeSeasonStats(p)")
    assert "addEventListener('keydown'" in region
    assert "resultsBox.querySelector('.compare-search-result-item')" in region


def test_trade_database_search_enter_selects_top_match():
    region = _region(TRADE_BP, "function bindTDBSearch(side)", "bindTDBSearch('A');")
    assert "addEventListener('keydown'" in region
    assert "drop.querySelector('.tdb-dropdown-item')" in region


def test_waiver_guest_picker_keyboard_path():
    # Button activation (Enter/Space while focused) fires click, not
    # mousedown: the pick must be bound to both. wvAddGuest dedupes by
    # player_id, so a mouse press that somehow fires both stays one add.
    assert 'onmousedown="wvGuestPick(this)" onclick="wvGuestPick(this)"' in WAIVERS
    # The input blur must not hide the results while focus is inside them.
    assert "box.contains(document.activeElement)" in WAIVERS
    guest = _region(WAIVERS, "function wvAddGuest(p)", "function wvRemoveGuest")
    assert "wvGuestById(p.player_id)" in guest


def test_no_em_dash_in_new_handlers():
    for region in (
        _region(APP_JS, "function setupSearch(side)", "function bindPickButtons()"),
        _region(TRADE_BP, "function bindTDBSearch(side)", "bindTDBSearch('A');"),
    ):
        assert "\u2014" not in region
