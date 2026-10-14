"""Player modal follow-up layouts (feat/player-modal-fixes).

- Breakout tab: weekly snap-share / targets charts with amber trend lines
  from the payload's weekly_series, hidden under 2 points.
- Overview: two-column body (main + right rail) with the mock's phone
  ordering; start/sit strip, scoring toggles and hero stay full width.
- Team tab: Room / Usage-by-week card pair at equal height from the
  payload's focus_weekly series.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Breakout weekly charts
# ---------------------------------------------------------------------------

def test_breakout_weekly_charts_render_from_series():
    assert "_pmBkWeeklyChart" in JS
    assert "data.weekly_series" in JS
    assert "'Snap share by week'" in JS
    assert "'Targets by week'" in JS
    # Amber trend line + dots through the bar tops.
    assert "pm-bk-line" in JS and "pm-bk-dot" in JS
    assert ".pm-bk-line { fill: none; stroke: #f59e0b;" in CSS
    # A chart needs 2+ points; anything less renders nothing.
    assert "if (pts.length < 2) return '';" in JS
    # Blue bars with the low bar muted, per the mock.
    assert ".pm-bk-bar { fill: var(--accent); }" in CSS
    assert ".pm-bk-bar-low { fill: #46586f; }" in CSS


# ---------------------------------------------------------------------------
# Overview two-column layout
# ---------------------------------------------------------------------------

def test_overview_two_column_structure():
    assert 'class="pm-ov-cols"' in JS
    assert 'class="pm-ov-main"' in JS
    assert 'class="pm-ov-rail"' in JS
    # Sections keep their ids and gain column/order marker classes.
    assert 'id="pmRecentForm" class="pm-recent pm-ov-recent"' in JS
    assert 'id="pmInLeague" class="pm-inleague pm-ov-inleague"' in JS
    assert 'class="pm-playoff-card pm-ov-playoff"' in JS
    assert 'class="pm-adp-block pm-ov-adp"' in JS
    assert 'class="pm-news-section pm-ov-news"' in JS
    assert 'class="pm-ov-value"' in JS
    # Rail Role card exists and is filled from the game-logs fetch.
    assert 'id="pmOvRole" class="pm-ov-role" hidden' in JS
    assert "getElementById('pmOvRole')" in JS
    # Full-width pieces stay above the columns in the template.
    ov = JS[JS.find('let overviewHTML'):]
    assert ov.find('pmStartSitStrip') < ov.find('pm-ov-cols')
    assert ov.find('${scoringToggles}') < ov.find('pm-ov-cols')
    assert ov.find('pm-hero-row') < ov.find('pm-ov-cols')
    # Desktop grid: main column + 300px rail, mock geometry.
    assert "grid-template-columns: minmax(0, 1fr) 300px" in CSS


def test_overview_phone_order():
    block = CSS[CSS.find(".pm-ov-playoff { order: 1;") - 200:]
    for cls, n in [("pm-ov-playoff", 1), ("pm-ov-recent", 2), ("pm-ov-value", 3),
                   ("pm-ov-role", 4), ("pm-ov-inleague", 5), ("pm-ov-adp", 6),
                   ("pm-ov-news", 7)]:
        assert f".{cls} {{ order: {n}; }}" in block
    # Columns flatten via display:contents so the order interleaves.
    assert ".pm-ov-main, .pm-ov-rail { display: contents; }" in CSS


# ---------------------------------------------------------------------------
# Team tab card pair
# ---------------------------------------------------------------------------

def test_team_pair_structure_and_usage_card():
    assert "pm-team-pair" in JS and "pm-pair-room" in JS
    assert "pm-pair-usage" in JS
    assert "Usage by week · targets" in JS
    assert "data.focus_weekly" in JS
    # Pair rows from the mock: dotted snap list, last-3 per-game, season high.
    assert "Snap share by week" in JS
    assert "Targets, last " in JS
    assert "Season high" in JS
    # Equal height: stretched grid row of card pair.
    assert ".pm-team-pair { display: grid; grid-template-columns: 1fr 1fr;" in CSS
    assert "align-items: stretch" in CSS
    # No usage data -> room card goes solo full width.
    assert "pm-team-pair-solo" in JS and "pm-team-pair-solo" in CSS
    # Phone stacks the pair.
    pair_css = CSS[CSS.find(".pm-team-pair { display: grid;"):]
    assert "@media (max-width: 768px)" in pair_css


def test_no_banned_radius_in_new_css():
    assert "border-radius: 999px" not in CSS
