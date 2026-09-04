"""Guards for teams-grid card chrome.

The reworked team-strength cards ship HTML classes (tc-avatar, tc-you,
tc-mix-legend, …) that a later dashboard.css merge once dropped, leaving
unconstrained banner images, a smashed YOU suffix, and run-on mix stats.
These tests lock the selectors in both the page builder and the stylesheet.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")

# Class names the card HTML emits that must have matching CSS.
_CARD_SELECTORS = (
    ".tc-avatar-wrap",
    ".tc-avatar",
    ".tc-avatar-mono",
    ".tc-name-text",
    ".tc-you",
    ".tc-window",
    ".tc-window-dot",
    ".tc-mix-legend",
    ".tc-mix-leg",
    ".tc-mix-dot",
    ".tc-index",
    ".tc-pos-chip",
    ".tc-head",
    ".tc-strength-track",
)


def test_team_card_html_emits_clamped_avatar_and_you_pill():
    assert "tc-avatar-wrap" in TEAMS_PAGE
    assert "tc-name-text" in TEAMS_PAGE
    assert "tc-you" in TEAMS_PAGE
    assert "tc-mix-dot" in TEAMS_PAGE
    assert "tc-window-dot" in TEAMS_PAGE
    # Name and YOU are siblings so the pill cannot concatenate onto the title.
    assert "<span class='tc-name-text'>{name}</span>{_you_pill}" in TEAMS_PAGE


def test_team_card_css_restores_reworked_layout():
    assert "TEAM STRENGTH CARD" in CSS
    for sel in _CARD_SELECTORS:
        assert sel in CSS, f"missing team-card CSS for {sel}"


def test_team_card_avatar_is_clamped():
    wrap = re.search(r"\.team-strength-card\s+\.tc-avatar-wrap\s*\{([^}]+)\}", CSS)
    assert wrap, "missing .tc-avatar-wrap rule"
    body = wrap.group(1)
    assert "42px" in body
    assert "min-width" in body
    assert "max-width" in body
    assert "overflow: hidden" in body


def test_team_card_mix_legend_has_flex_gap():
    legend = re.search(r"\.team-strength-card\s+\.tc-mix-legend\s*\{([^}]+)\}", CSS)
    assert legend, "missing .tc-mix-legend rule"
    body = legend.group(1)
    assert "display: flex" in body
    assert "column-gap" in body or "gap:" in body


def test_team_card_name_wraps_instead_of_ellipsis():
    """Viewer rows with a YOU pill must not single-line-ellipsis the title.

    A nowrap + text-overflow clamp turned mid-length names into
    "Move the …" once the non-shrinking YOU badge took the last ~40px.
    """
    name = re.search(
        r"\.team-strength-card\s+\.tc-head\s+h2\.tc-name\s*\{([^}]+)\}", CSS
    )
    assert name, "missing h2.tc-name rule"
    name_body = name.group(1)
    assert "white-space: normal" in name_body
    assert "overflow: hidden" not in name_body
    assert "text-overflow" not in name_body

    text = re.search(r"\.team-strength-card\s+\.tc-name-text\s*\{([^}]+)\}", CSS)
    assert text, "missing .tc-name-text rule"
    text_body = text.group(1)
    assert "white-space: normal" in text_body
    assert "overflow-wrap" in text_body
    assert "text-overflow" not in text_body
    assert "ellipsis" not in text_body

    you = re.search(r"\.team-strength-card\s+\.tc-you\s*\{([^}]+)\}", CSS)
    assert you, "missing .tc-you rule"
    you_body = you.group(1)
    assert "flex: 0 0 auto" in you_body or "flex-shrink: 0" in you_body
