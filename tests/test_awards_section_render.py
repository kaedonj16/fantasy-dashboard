"""Render tests for the Season Hub League Awards section.

Locks in the command-center row layout of render_awards_section: os-side-plain
shell with the standard section head and collapse toggle, one row per award
with label / detail left and a 16px value right (green for honors, red for
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
    """[(value-class, label)] per row, in render order."""
    return re.findall(
        r'<div class="cc-rw cc-award-row"><span>([^<]+)<br>', out
    )


def test_empty_awards_render_nothing():
    assert render_awards_section({}) == ""
    assert render_awards_section(None) == ""


def test_section_uses_hub_chrome_with_collapse():
    out = render_awards_section(AWARDS)
    assert '<section class="os-side-plain awards-card" data-section="awards">' in out
    assert '<div class="os-section-head">' in out
    assert '<h2 class="os-section-title">' in out
    assert 'class="fa-solid fa-trophy"' in out
    assert "League Awards" in out
    assert '<div class="os-section-subtitle">Season superlatives so far</div>' in out
    assert 'class="card-collapse-toggle"' in out
    assert 'data-target="dash-awards-body"' in out
    assert 'aria-expanded="true"' in out
    assert '<div class="card-collapsible-body" id="dash-awards-body">' in out
    # Old tile chrome is gone.
    assert "award-item" not in out
    assert "awards-grid" not in out


def test_rows_have_label_detail_value():
    out = render_awards_section(AWARDS)
    assert out.count('class="cc-rw cc-award-row"') == 6
    # Label + detail on the left, value on the right.
    assert "<span>Highest week<br>" in out
    assert '<span class="cc-muted">Team Alpha &middot; Wk 3</span>' in out
    assert '<span class="cc-award-val cc-up">152.4</span>' in out
    assert "<span>Lowest week<br>" in out
    assert '<span class="cc-muted">Team Beta &middot; Wk 5</span>' in out
    assert '<span class="cc-award-val cc-dn">61.2</span>' in out
    # Tied streak winners stack with <br>, never a comma run-on.
    assert "Team Alpha<br>Team Gamma" in out
    assert "Team Alpha, Team Gamma" not in out
    assert '<span class="cc-award-val cc-up">7</span>' in out
    assert '<span class="cc-award-val cc-dn">6</span>' in out
    # Consistency: sigma value, team in the detail line.
    assert '<span class="cc-award-val">9.12</span>' in out
    assert "Team Epsilon" in out
    assert '<span class="cc-award-val cc-up">41.2</span>' in out


def test_row_order_and_labels():
    out = render_awards_section(AWARDS)
    labels = _rows(out)
    assert labels == [
        "Highest week",
        "Lowest week",
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
    assert out.count('class="cc-rw cc-award-row"') == 1
    assert _rows(out) == ["Longest losing streak"]
    assert "Highest week" not in out


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
    # Award rows get extra vertical padding vs plain rows.
    m = re.search(r"\.cc-award-row \{(.*?)\}", css, re.DOTALL)
    assert m and "padding: 12px 0" in m.group(1)
    # Values are 16px bold tabular numerals.
    m = re.search(r"\.cc-award-val \{(.*?)\}", css, re.DOTALL)
    assert m and "font-size: 16px" in m.group(1)
    # Green/red accents exist.
    assert re.search(r"\.cc-up \{(.*?)\}", css, re.DOTALL)
    assert re.search(r"\.cc-dn \{(.*?)\}", css, re.DOTALL)
