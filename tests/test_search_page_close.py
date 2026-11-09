"""Focused tests: tapping a page/tool result in mobile search must dismiss
the search overlay before navigating.

Bug (reported 2026-11-09): on mobile, searching for a page and tapping the
page result left the full-screen search open on top of the destination --
the user had to tap Cancel to see the page they had navigated to.

Cause: player/team results (.nav-search-result) were dismissed by the
search-screen and More-sheet click handlers, but page/tool results
(.nav-search-cmd) were not covered anywhere. initNavSearch only closed its
own dropdown (closeDropdown), never the containing screen/sheet, and the
container handlers only matched .nav-search-result.

These are file-text checks in the style of
tests/test_search_suggestions_no_zoom.py; they do not import app.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_public_close_search_exposed():
    # initNavSearch lives in the lazy features bundle and can only reach the
    # core closeSearch() through a public handle (same pattern as
    # window.brCloseMoreSheet / window.brOpenSearch).
    assert "window.brCloseSearch = closeSearch;" in APP_JS


def test_search_screen_closes_on_page_result_tap():
    assert (
        "if (e.target.closest('.nav-search-result, .nav-search-cmd')) closeSearch();"
        in APP_JS
    )


def test_more_sheet_closes_on_page_result_tap():
    # Inline search inside the More sheet: a page result tap closes both the
    # inline search and the sheet itself.
    assert (
        "if (e.target.closest('.nav-search-result, .nav-search-cmd')) "
        "{ closeInlineSearch(); setOpen(false); return; }" in APP_JS
    )
    assert (
        "if (e.target.closest('.br-sheet-link, .nav-search-result, .nav-search-cmd')) "
        "setOpen(false);" in APP_JS
    )


def _cmd_click_block() -> str:
    m = re.search(
        r"const cmd = e\.target\.closest\('\.nav-search-cmd'\);.*?closeDropdown\(\);",
        APP_JS,
        re.DOTALL,
    )
    assert m, "dropdown page-result click handler not found"
    return m.group(0)


def test_dropdown_page_click_closes_overlays_before_navigating():
    block = _cmd_click_block()
    close_at = block.index("window.brCloseSearch()")
    sheet_at = block.index("window.brCloseMoreSheet()")
    nav_at = block.index("window.location.href")
    assert close_at < nav_at
    assert sheet_at < nav_at


def test_select_current_page_branch_closes_overlays_before_navigating():
    m = re.search(
        r"el\.dataset\.nsKind === 'cmd'.*?window\.location\.href = el\.getAttribute\('href'\);",
        APP_JS,
        re.DOTALL,
    )
    assert m, "selectCurrent() page-result branch not found"
    block = m.group(0)
    assert block.index("window.brCloseSearch()") < block.index("window.location.href")
    assert block.index("window.brCloseMoreSheet()") < block.index("window.location.href")
