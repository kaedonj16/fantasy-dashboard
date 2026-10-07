"""Guards for the Mock 4 teams-page rework: compact card chrome, drawer, ranked lists.

The compact team-strength cards ship HTML classes (tsc-avatar-wrap, tsc-you,
tsc-mix-legend, ...) that must have matching CSS, the cards must carry the
data attributes the sort bar and the drawer depend on, and the grid must not
stretch cards to the tallest row. The drawer shell and the ranked-list
(selectors used by the Value/Schedule tabs) are locked too.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")

# Class names the card HTML emits that must have matching CSS.
_CARD_SELECTORS = (
    ".tsc-avatar-wrap",
    ".tsc-avatar",
    ".tsc-avatar-mono",
    ".tsc-name",
    ".tsc-you",
    ".tsc-dot",
    ".tsc-grade",
    ".tsc-status",
    ".tsc-pi",
    ".tsc-pi-track",
    ".tsc-mix-bar",
    ".tsc-mix-legend",
    ".tsc-mix-dot",
    ".tsc-details",
    ".td-drawer",
    ".td-scrim",
    ".rl-row",
    ".rl-cols",
    ".vbar",
)

# Data attributes the client-side sort bar and drawer JS depend on.
_SORT_ATTRS = (
    "data-sort-grade=",
    "data-sort-posindex=",
    "data-sort-archetype=",
    "data-roster-id=",
    "data-original-index=",
)


def test_compact_card_html_emits_avatar_name_status_and_details():
    assert "tsc-avatar-wrap" in TEAMS_PAGE
    assert "tsc-name-text" not in TEAMS_PAGE  # old chrome is gone
    assert "tsc-you" in TEAMS_PAGE
    assert "tsc-dot" in TEAMS_PAGE
    assert "tsc-grade" in TEAMS_PAGE
    assert "tsc-mix-dot" in TEAMS_PAGE
    assert "View details" in TEAMS_PAGE
    # The expandable in-card position table is gone; the drawer is the detail surface.
    assert "pos-strength-table" not in TEAMS_PAGE
    assert "pos-table-wrap" not in TEAMS_PAGE
    assert "team-card-toggle" not in TEAMS_PAGE
    for attr in _SORT_ATTRS:
        assert attr in TEAMS_PAGE, f"card lost sort/drawer attribute {attr}"


def test_compact_card_css_restores_layout():
    assert "TEAM STRENGTH CARD" in CSS
    for sel in _CARD_SELECTORS:
        assert sel in CSS, f"missing teams CSS for {sel}"


def test_teams_grid_does_not_stretch_cards():
    grid = re.search(r"\.teams-page\s+\.teams-grid\s*\{([^}]+)\}", CSS)
    assert grid, "missing .teams-page .teams-grid rule"
    assert "align-items: start" in grid.group(1)
    # The old height:100% stretch must be undone for the compact cards; the
    # later (winning) .teams-grid .team-strength-card rule sets height:auto.
    cards = re.findall(r"\.teams-grid\s+\.team-strength-card\s*\{([^}]+)\}", CSS)
    assert cards, "missing .teams-grid .team-strength-card rule"
    assert any("height: auto" in body for body in cards)


def test_teams_grid_collapses_three_two_one():
    css = CSS
    assert re.search(
        r"@media\s*\(max-width:\s*1100px\)\s*\{\s*\.teams-page\s+\.teams-grid\s*\{[^}]*repeat\(2,\s*1fr\)",
        css,
    ), "grid must go 3 -> 2 columns at <=1100px"
    assert re.search(
        r"@media\s*\(max-width:\s*640px\)\s*\{\s*\.teams-page\s+\.teams-grid\s*\{[^}]*1fr",
        css,
    ), "grid must go 2 -> 1 column at <=640px"


def test_team_card_avatar_is_clamped():
    wrap = re.search(r"\.tsc-avatar-wrap\s*\{([^}]+)\}", CSS)
    assert wrap, "missing .tsc-avatar-wrap rule"
    body = wrap.group(1)
    assert "38px" in body
    assert "overflow: hidden" in body
    assert "border-radius: 50%" in body


def test_team_card_mix_legend_has_flex_gap():
    legend = re.search(r"\.tsc-mix-legend\s*\{([^}]+)\}", CSS)
    assert legend, "missing .tsc-mix-legend rule"
    body = legend.group(1)
    assert "display: flex" in body
    assert "gap:" in body


def test_team_card_name_ellipsizes_instead_of_wrapping():
    """Long names must ellipsis on one line so compact cards stay compact."""
    name = re.search(r"\.tsc-name\s*\{([^}]+)\}", CSS)
    assert name, "missing .tsc-name rule"
    body = name.group(1)
    assert "text-overflow: ellipsis" in body
    assert "white-space: nowrap" in body
    assert "overflow: hidden" in body


def test_drawer_is_fixed_right_slide_over():
    drawer = re.search(r"\.td-drawer\s*\{([^}]+)\}", CSS)
    assert drawer, "missing .td-drawer rule"
    body = drawer.group(1)
    assert "position: fixed" in body
    assert "right: 0" in body
    assert "translateX(105%)" in body
    assert ".td-drawer.open" in CSS


def test_drawer_goes_full_width_on_mobile():
    assert re.search(
        r"@media\s*\(max-width:\s*560px\)\s*\{[^}]*\.td-drawer\s*\{[^}]*width:\s*100%",
        CSS,
    ), "drawer must be full-width on small phones"


def test_no_stadium_pills_in_teams_css():
    assert "border-radius: 999px" not in CSS
