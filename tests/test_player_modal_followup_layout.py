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


def test_overview_rail_panels_share_one_treatment():
    # Declutter (2026-11-03): every rail section is the same quiet panel --
    # row surface, hairline border, shared padding -- and the sections'
    # own top dividers drop inside the box.
    for sel in (".pm-ov-rail .pm-playoff-card", ".pm-ov-rail .pm-ov-role",
                ".pm-ov-rail .pm-avail", ".pm-ov-rail .pm-inleague",
                ".pm-ov-rail .pm-news-section"):
        assert sel in CSS, sel
    assert "background: var(--row); border: 1px solid var(--border);" in CSS
    assert ".pm-ov-rail .pm-section-divider { display: none; }" in CSS
    # The old per-card fills are gone (no double-boxing).
    assert ".pm-playoff-card { background: var(--bg-alt)" not in CSS
    assert ".pm-ov-role { background: var(--bg-alt)" not in CSS
    # Availability's inner list no longer draws its own border box.
    assert ".pm-avail-list { border: 1px solid var(--grid)" not in CSS


def test_overview_phone_order():
    block = CSS[CSS.find(".pm-ov-recent { order: 1;") - 200:]
    for cls, n in [("pm-ov-recent", 1), ("pm-ov-value", 2), ("pm-ov-playoff", 3),
                   ("pm-ov-role", 4), ("pm-ov-adp", 5), ("pm-ov-inleague", 6),
                   ("pm-avail", 7), ("pm-ov-news", 8)]:
        assert f".{cls} {{ order: {n}; }}" in block
    # Columns flatten via display:contents so the order interleaves.
    assert ".pm-ov-main, .pm-ov-rail { display: contents; }" in CSS


def test_overview_hero_is_one_line_strip():
    # Hero-strip spec (2026-11-04): the row itself is the card and the
    # cells sit inside it split by hairlines -- no per-cell boxes, and no
    # phone 2-column wrap, so 2/3/4 stats stay on one line at any width.
    row = CSS[CSS.find(".pm-hero-row {"):]
    row = row[:row.find("}")]
    assert "background: var(--bg-alt);" in row
    assert "border: 1px solid var(--border);" in row
    assert "overflow: hidden;" in row
    assert "gap: 0;" in row
    assert ".pm-hero-row .pm-hero-stat + .pm-hero-stat {" in CSS
    assert "border-left: 1px solid var(--border);" in CSS
    assert ".pm-hero-row {\n        grid-template-columns: 1fr 1fr !important;" not in CSS
    # Strip type runs a step smaller than the shared tile sizes.
    assert ".pm-hero-row .pm-hero-val {\n    font-size: 22px;\n}" in CSS
    assert ".pm-hero-row .pm-hero-val {\n        font-size: 19px;\n    }" in CSS
    assert ".pm-hero-row .pm-hero-stat {\n        padding: 6px 8px;\n    }" in CSS


def test_overview_avail_below_in_league_in_rail():
    # Availability moved below In this league in the desktop rail DOM
    # (hero-strip spec); its phone order 7 (above) lands it in the same
    # spot once the columns flatten.
    ov = JS[JS.find('let overviewHTML'):]
    assert ov.find('id="pmOvRole"') < ov.find('id="pmInLeague"')
    assert ov.find('id="pmInLeague"') < ov.find('id="pmAvail"')
    assert ov.find('id="pmAvail"') < ov.find('${newsBlock}')


# ---------------------------------------------------------------------------
# Team tab card pair
# ---------------------------------------------------------------------------

def test_team_pair_structure_and_usage_card():
    assert "pm-team-pair" in JS and "pm-pair-room" in JS
    assert "pm-pair-usage" in JS
    # The usage metric is position-dependent (RB touches, QB pass attempts,
    # WR/TE targets), so the header and last-3 label render the picked noun.
    assert "Usage by week · ${_metric.low}" in JS
    assert "data.focus_weekly" in JS
    # Pair rows from the mock: last-3 per-game and season high stay; the
    # dotted snap list became the per-week grid inside the usage card.
    assert "<span>Snap share by week</span>" not in JS
    assert "pm-wk-grid" in JS
    assert "${_metric.noun}, last " in JS
    assert "Season high" in JS
    # The pair no longer stretches to equal height (that left a large
    # blank box under the usage card): the right side is a stack of cards.
    assert ".pm-team-pair { display: grid; grid-template-columns: 1fr 1fr;" in CSS
    pair_rule = CSS[CSS.find(".pm-team-pair { display: grid;"):]
    pair_rule = pair_rule[:pair_rule.find("}")]
    assert "align-items: start" in pair_rule
    assert "pm-team-stack" in JS and ".pm-team-stack {" in CSS
    # No stack content at all -> room card goes solo full width.
    assert "pm-team-pair-solo" in JS and "pm-team-pair-solo" in CSS
    # Phone stacks the pair.
    pair_css = CSS[CSS.find(".pm-team-pair { display: grid;"):]
    assert "@media (max-width: 768px)" in pair_css


def test_no_banned_radius_in_new_css():
    assert "border-radius: 999px" not in CSS
