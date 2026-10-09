"""Render tests for the Season Hub League Awards section.

Locks in the Hub-style restructure of render_awards_section: os-card shell
with the standard section head and collapse toggle (same pattern as the
Standings card above it), one tile per award with label / winner / value /
context, win-accent tiles for honors and loss-accent tiles for shame
awards, and the Highest Player name keeping its player-clickable behavior.
Data, award set, and the empty-dict -> "" contract are unchanged.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("pandas")

from dashboard_services.awards import render_awards_section  # noqa: E402

AWARDS = {
    "highest_single_week": ("Team Alpha", 3, 152.36),
    "lowest_single_week": ("Team Beta", 5, 61.24),
    "longest_win_streak": (["Team Alpha", "Team Gamma"], 7),
    "longest_loss_streak": (["Team Delta"], 6),
    "most_consistent": ("Team Epsilon", 9.123, 8),
    "highest_player": (4, 41.25, "Star Player", "WR", "CIN", "Team Alpha", "999"),
}


def _tiles(out: str):
    """[(accent, label)] per tile, in render order."""
    return re.findall(
        r'class="award-item (award-\w+)">\s*<div class="award-name">([^<]+)</div>', out
    )


def test_empty_awards_render_nothing():
    assert render_awards_section({}) == ""
    assert render_awards_section(None) == ""


def test_section_uses_hub_card_chrome_with_collapse():
    out = render_awards_section(AWARDS)
    # Redesign review: no outer card chrome on the sidebar; plain section.
    assert '<section class="os-side-plain awards-card" data-section="awards">' in out
    assert 'os-card awards-card' not in out
    assert '<div class="os-section-head">' in out
    assert '<h2 class="os-section-title">' in out
    assert 'class="fa-solid fa-trophy"' in out
    assert "League Awards" in out
    assert '<div class="os-section-subtitle">Season superlatives so far</div>' in out
    # Same collapse pattern as the Standings card: toggle + collapsible body.
    assert 'class="card-collapse-toggle"' in out
    assert 'data-target="dash-awards-body"' in out
    assert 'aria-expanded="true"' in out
    assert '<div class="card-collapsible-body" id="dash-awards-body">' in out
    # Old generic card chrome is gone.
    assert 'class="card awards-card"' not in out
    assert "awards-title" not in out
    assert "award-body" not in out


def test_tiles_have_label_winner_value_context():
    out = render_awards_section(AWARDS)
    assert out.count('class="award-item award-') == 6
    assert '<div class="award-winner">Team Alpha</div>' in out
    # Value splits into a strong number element and a smaller muted unit.
    assert (
        '<div class="award-value"><span class="award-value-num">152.4</span>'
        ' <span class="award-value-unit">points</span></div>'
    ) in out
    assert '<div class="award-context">Week 3</div>' in out
    assert '<span class="award-value-num">61.2</span>' in out
    assert '<div class="award-context">Week 5</div>' in out
    # Tied streak winners stack one per line, never a comma run-on.
    assert (
        '<div class="award-winner"><div class="award-winner-line">Team Alpha</div>'
        '<div class="award-winner-line">Team Gamma</div></div>'
    ) in out
    assert "Team Alpha, Team Gamma" not in out
    assert '<span class="award-value-num">7</span>' in out
    assert '<span class="award-value-unit">games</span>' in out
    assert '<span class="award-value-num">6</span>' in out
    # Consistency: sigma is the headline value (no unit), games are context.
    assert (
        '<div class="award-value"><span class="award-value-num">σ 9.12</span></div>'
    ) in out
    assert '<div class="award-context">over 8 games</div>' in out
    assert '<span class="award-value-num">41.25</span>' in out
    assert '<div class="award-context">Week 4</div>' in out


def test_honor_and_shame_accents_on_the_right_awards():
    out = render_awards_section(AWARDS)
    assert _tiles(out) == [
        ("award-honor", "Highest Single Week"),
        ("award-shame", "Lowest Single Week"),
        ("award-honor", "Longest Win Streak"),
        ("award-shame", "Longest Losing Streak"),
        ("award-honor", "Most Consistent"),
        ("award-honor", "Highest Points By a Player"),
    ]


def test_highest_player_name_stays_clickable():
    out = render_awards_section(AWARDS)
    assert (
        '<div class="award-winner"><span class=\'player-clickable\' '
        "style='cursor:pointer;' data-player-id='999' "
        "data-player-name='Star Player'>Star Player</span></div>"
    ) in out


def test_highest_player_without_id_is_plain():
    awards = {"highest_player": (4, 41.25, "Star Player", "WR", "CIN", "Team Alpha", "")}
    out = render_awards_section(awards)
    assert '<div class="award-winner"><span>Star Player</span></div>' in out
    assert "player-clickable" not in out


def test_partial_awards_render_only_present_tiles():
    out = render_awards_section({"longest_loss_streak": (["Team Delta"], 6)})
    assert out.count('class="award-item award-') == 1
    assert _tiles(out) == [("award-shame", "Longest Losing Streak")]
    # A single streak winner stays plain text on the winner line.
    assert '<div class="award-winner">Team Delta</div>' in out
    assert "Highest Single Week" not in out


def test_awards_css_has_tints_not_accent_bar_and_spans_orphan_tile():
    from pathlib import Path

    css = (
        Path(__file__).resolve().parent.parent / "static" / "dashboard.css"
    ).read_text(encoding="utf-8")
    base = re.search(r"^\.award-item \{(.*?)\}", css, re.DOTALL | re.MULTILINE).group(1)
    # The thick side accent bar is gone from the tile chrome.
    assert "border-left" not in base
    honor = re.search(
        r"^\.award-item\.award-honor \{(.*?)\}", css, re.DOTALL | re.MULTILINE
    ).group(1)
    shame = re.search(
        r"^\.award-item\.award-shame \{(.*?)\}", css, re.DOTALL | re.MULTILINE
    ).group(1)
    # Honor/shame distinction is a soft whole-tile tint instead.
    assert "background: color-mix(in srgb, var(--win)" in honor
    assert "background: color-mix(in srgb, var(--loss)" in shame
    assert "border-left" not in honor
    assert "border-left" not in shame
    # Dashboard awards rail: clean cards, no tint; values keep green/red.
    dash = re.search(
        r"\.awards-card \.award-item\.award-honor,\s*\.awards-card \.award-item\.award-shame \{(.*?)\}",
        css,
        re.DOTALL,
    ).group(1)
    assert "background: var(--card)" in dash
    # Redesign review: awards are always a single column; the 2-column
    # orphan-tile spanning rules are gone.
    grid = re.search(
        r"^\.awards-grid \{(.*?)\}", css, re.DOTALL | re.MULTILINE
    ).group(1)
    assert "grid-template-columns: 1fr" in grid
    assert "repeat(2, 1fr)" not in css.split(".awards-grid")[1].split("}")[0]
    assert ".award-item:last-child:nth-child(odd)" not in css
