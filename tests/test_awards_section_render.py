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
    assert '<section class="os-card awards-card" data-section="awards">' in out
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
    assert '<div class="award-value">152.4 points</div>' in out
    assert '<div class="award-context">Week 3</div>' in out
    assert '<div class="award-value">61.2 points</div>' in out
    assert '<div class="award-context">Week 5</div>' in out
    # Tied streak winners join on the winner line.
    assert '<div class="award-winner">Team Alpha, Team Gamma</div>' in out
    assert '<div class="award-value">7 games</div>' in out
    assert '<div class="award-value">6 games</div>' in out
    # Consistency: sigma is the headline value, game count is the context.
    assert '<div class="award-value">σ 9.12</div>' in out
    assert '<div class="award-context">over 8 games</div>' in out
    assert '<div class="award-value">41.25 points</div>' in out
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
    assert "Highest Single Week" not in out
