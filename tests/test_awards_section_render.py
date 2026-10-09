"""Render tests for the Season Hub League Awards section.

Locks in the character-card layout of render_awards_section: dark gradient
header (icon + title + subtitle + collapse toggle), one row per award with
label / team / detail left and a 24px value right (green for honors, red for
shame awards, dark for neutral), and the Highest Player name keeping its
player-clickable behavior. Data, award set, and the empty-dict -> ""
contract are unchanged.
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


def _rows(out: str):
    """[label] per row, in render order."""
    return re.findall(
        r'<div class="cc-award-label">([^<]+)</div>', out
    )


def test_empty_awards_render_nothing():
    assert render_awards_section({}) == ""
    assert render_awards_section(None) == ""


def test_section_uses_character_chrome_with_collapse():
    out = render_awards_section(AWARDS)
    assert '<section class="os-side-plain cc-aw-card" data-section="awards">' in out
    assert '<div class="cc-aw-head">' in out
    assert '<div class="cc-aw-head-icon">' in out
    assert 'class="fa-solid fa-trophy"' in out
    assert '<h3 class="cc-aw-title">League Awards</h3>' in out
    assert '<div class="cc-aw-sub">Season superlatives so far</div>' in out
    assert 'class="card-collapse-toggle cc-head-toggle"' in out
    assert 'data-target="dash-awards-body"' in out
    assert 'aria-expanded="true"' in out
    assert '<div class="card-collapsible-body" id="dash-awards-body">' in out
    # Old chrome is gone.
    assert "award-item" not in out
    assert "awards-grid" not in out
    assert "os-section-head" not in out
    assert "cc-rw cc-award-row" not in out


def test_rows_have_label_team_detail_value():
    out = render_awards_section(AWARDS)
    assert out.count('class="cc-award"') == 6
    # Label, team, detail on the left, value on the right.
    assert '<div class="cc-award-label">Highest single week</div>' in out
    assert '<div class="cc-award-team">Team Alpha</div>' in out
    assert '<div class="cc-award-detail">Week 3</div>' in out
    assert '<div class="cc-award-val cc-up">152.4</div>' in out
    assert '<div class="cc-award-label">Lowest single week</div>' in out
    assert '<div class="cc-award-team">Team Beta</div>' in out
    assert '<div class="cc-award-detail">Week 5</div>' in out
    assert '<div class="cc-award-val cc-dn">61.2</div>' in out
    # Tied streak winners stack with <br>, never a comma run-on.
    assert "Team Alpha<br>Team Gamma" in out
    assert "Team Alpha, Team Gamma" not in out
    assert '<div class="cc-award-val cc-up">7</div>' in out
    assert '<div class="cc-award-val cc-dn">6</div>' in out
    # Consistency: sigma value, "over N games" detail.
    assert '<div class="cc-award-val">9.12</div>' in out
    assert '<div class="cc-award-detail">over 8 games</div>' in out
    assert "Team Epsilon" in out
    assert '<div class="cc-award-val cc-up">41.2</div>' in out


def test_row_order_and_labels():
    out = render_awards_section(AWARDS)
    labels = _rows(out)
    assert labels == [
        "Highest single week",
        "Lowest single week",
        "Longest win streak",
        "Longest losing streak",
        "Most consistent",
        "Top player week",
    ]


def test_highest_player_name_stays_clickable():
    out = render_awards_section(AWARDS)
    assert (
        "<span class='player-clickable' "
        "style='cursor:pointer;' data-player-id='999' "
        "data-player-name='Star Player'>Star Player</span>" in out
    )


def test_highest_player_without_id_is_plain():
    awards = {"highest_player": (4, 41.25, "Star Player", "WR", "CIN", "Team Alpha", "")}
    out = render_awards_section(awards)
    assert "<span>Star Player</span>" in out
    assert "player-clickable" not in out


def test_partial_awards_render_only_present_rows():
    out = render_awards_section({"longest_loss_streak": (["Team Delta"], 6)})
    assert out.count('class="cc-award"') == 1
    assert _rows(out) == ["Longest losing streak"]
    assert "Highest single week" not in out


def test_team_names_are_escaped():
    out = render_awards_section(
        {"highest_single_week": ("<b>Team</b>", 3, 100.0)}
    )
    assert "<b>Team</b>" not in out
    assert "&lt;b&gt;Team&lt;/b&gt;" in out


def test_awards_css_row_rules():
    from pathlib import Path

    css = (
        Path(__file__).resolve().parent.parent / "static" / "dashboard.css"
    ).read_text(encoding="utf-8")
    # Award rows get generous vertical padding.
    m = re.search(r"\.cc-award \{(.*?)\}", css, re.DOTALL)
    assert m and "padding: 14px 4px" in m.group(1)
    # Values are 24px extra-bold tabular numerals.
    m = re.search(r"\.cc-award-val \{(.*?)\}", css, re.DOTALL)
    assert m and "font-size: 24px" in m.group(1)
    assert m and "font-weight: 800" in m.group(1)
    # Dark header card chrome exists.
    assert re.search(r"\.cc-aw-card \{(.*?)\}", css, re.DOTALL)
    assert re.search(r"\.cc-aw-head \{(.*?)\}", css, re.DOTALL)
    # Green/red accents exist.
    assert re.search(r"\.cc-up \{(.*?)\}", css, re.DOTALL)
    assert re.search(r"\.cc-dn \{(.*?)\}", css, re.DOTALL)
