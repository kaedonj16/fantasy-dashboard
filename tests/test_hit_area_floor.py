"""Contracts for the 44px hit-area floor (2026-11-14 audit).

On coarse pointers, standalone tap targets get at least a 44px box via
min-width/min-height overrides (which beat the smaller fixed sizes in
the cascade), so glyphs and desktop geometry stay untouched. Page-local
style blocks carry the same floor for the selectors they own.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def _coarse_block(text, marker):
    """Return the @media (pointer: coarse) block that follows `marker`."""
    start = text.index(marker)
    media = text.index("@media (pointer: coarse)", start)
    brace = text.index("{", media)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[media:i + 1]
    raise AssertionError("unbalanced coarse block")


def test_dashboard_coarse_block_floors_chrome_and_closes():
    css = _read("static/dashboard.css")
    block = _coarse_block(css, "Hit-area floor (44px rule")
    for sel in (".br-top-search", ".br-top-notif", "#darkModeToggle",
                ".utility-icon-btn", ".player-modal-close",
                ".team-modal-close", ".for-modal-close", ".otc-popup-close",
                ".br-asst-bubble-dismiss", ".otc-info-btn"):
        assert sel in block, sel
    # Icon buttons are floored on both axes inside the block.
    assert block.count("min-width: 44px") >= 3
    assert block.count("min-height: 44px") >= 3


def test_dashboard_coarse_block_floors_short_controls():
    css = _read("static/dashboard.css")
    block = _coarse_block(css, "Hit-area floor (44px rule")
    for sel in (".btn", ".csd-trigger", ".otc-main-tab", ".otc-day-filter",
                ".pagination-btn", ".pagination-page", ".fob-skip",
                ".fob-next", ".br-asst-chip", ".tdb-search",
                ".compare-pick-input", ".ddb-search"):
        assert sel in block, sel
    # The settings-row custom select carries a fixed 30px height in a
    # higher-specificity rule; the floor must release it explicitly.
    assert ".otc-settings-row .csd-trigger" in block
    assert "height: auto" in block


def test_floor_is_coarse_only_no_global_44px_minimums():
    css = _read("static/dashboard.css")
    marker = css.index("Hit-area floor (44px rule")
    # Nothing in the hit-area section leaks outside its media block:
    # the section header sits immediately above the @media opener.
    between = css[marker:css.index("@media (pointer: coarse)", marker)]
    assert "min-height: 44px" not in between


def test_metrics_page_block_floors_chips_and_pills():
    text = _read("dashboard_services/pages/advanced_metrics_page.py")
    block = _coarse_block(text, "Hit-area floor (44px rule")
    for sel in (".am-pill", ".am-cmd-btn", ".am-chip", ".am-chip-x",
                ".am-add-stat-btn", ".am-select", ".am-context-line",
                ".am-clear-link"):
        assert sel in block, sel


def test_players_page_block_floors_segmented_and_sort():
    text = _read("dashboard_services/pages/players_page.py")
    block = _coarse_block(text, "Hit-area floor (44px rule")
    for sel in (".pr-segbtn", ".pr-sort-toggle button", "#prCountRow button"):
        assert sel in block, sel


def test_draft_db_block_floors_tabs_and_buttons():
    text = _read("routes/draft_db_bp.py")
    block = _coarse_block(text, "Hit-area floor (44px rule")
    for sel in (".ddb-tab", ".ddb-board-btn", ".ddb-back", ".ddb-search"):
        assert sel in block, sel


def test_trade_calc_block_floors_info_button():
    text = _read("dashboard_services/pages/trade_calculator_page.py")
    block = _coarse_block(text, "Hit-area floor (44px rule")
    assert ".pi-info-btn" in block
    assert "min-width:44px" in block.replace(" ", "") or "min-width: 44px" in block


def test_paywall_close_floored_on_coarse():
    css = _read("static/paywall.css")
    block = _coarse_block(css, "Hit-area floor (44px rule")
    assert ".paywall-close" in block
    assert "min-height: 44px" in block


def test_no_banned_transition_all_in_new_blocks():
    for rel in ("static/dashboard.css", "static/paywall.css",
                "dashboard_services/pages/advanced_metrics_page.py",
                "dashboard_services/pages/players_page.py",
                "dashboard_services/pages/trade_calculator_page.py",
                "routes/draft_db_bp.py"):
        block = _coarse_block(_read(rel), "Hit-area floor (44px rule")
        assert not re.search(r"transition:\s*all\b", block), rel
