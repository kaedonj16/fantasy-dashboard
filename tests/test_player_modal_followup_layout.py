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
    assert ".pm-hero-row .pm-hero-stat {\n        padding: 6px 2px;\n    }" in CSS


def test_overview_hero_no_truncation_on_phones():
    # Truncation fix (2026-11-04, live Gibbs modal): "TOTAL PT…" and
    # "POS : 1 · OV…" ellipsized on phones. The subs drop the colons
    # (compare hero in app.js keeps its own format on purpose), the
    # Total label drops "Pts", and the phone strip type/tracking/padding
    # shrink a step so every label and sub fits its cell.
    assert "`POS ${posR} · OVR ${ovrR" in JS
    assert "`POS : ${" not in JS
    assert "'POS : '" not in JS
    assert 'pm-hero-label">Total${seasonLabel}' in JS
    assert "Total Pts${seasonLabel}" not in JS
    assert ".pm-hero-row .pm-hero-label {\n        font-size: 9.5px;" in CSS
    assert "letter-spacing: 0.01em;" in CSS
    assert ".pm-hero-row .pm-hero-sub {" in CSS
    assert "font-size: 8px;\n        letter-spacing: -0.03em;" in CSS


def test_overview_vh_summary_is_one_compact_cluster():
    # Compact fix (2026-11-04): the summary's pieces were spread across
    # the full width (empty right side on the value row, dead middle in
    # the Peak/Floor + filter row), leaving big white gaps on phone and
    # desktop. Now the summary is one cluster: value, delta with the
    # range text inline, and the range bar re-homed as the summary's
    # last flex child, uniform gaps, no spread. Peak/Floor are NOT in
    # the summary at all (Kaedon's call): the chart's own Peak / Low
    # callouts carry them, with dates. Because updateSummary() rebuilds
    # the summary's innerHTML on every render, the bar element is held
    # in a closure and re-appended after each rebuild, and parked back
    # before the chart when the summary hides.
    assert "let _vhBarEl = null;" in JS
    assert "_vhBarEl = bar;" in JS
    assert "if (_vhBarEl) el.appendChild(_vhBarEl);" in JS
    assert "el.contains(_vhBarEl)" in JS
    assert "pm-vh-subrow" not in JS
    assert "pm-vh-ext" not in JS
    # Range text sits inside the move row, after the delta.
    move_row = JS[JS.find('pm-vh-move-row'):JS.find('if (_vhBarEl) el.appendChild')]
    assert "pm-vh-delta" in move_row and "pm-vh-range" in move_row
    # CSS scoped to the modal summary so Compare / player page keep
    # their own .pvc-range-bar contexts, and the Peak/Floor chip CSS
    # is gone.
    assert ".pm-vh-subrow" not in CSS
    assert "pm-vh-ext" not in CSS
    # Tasteful pass: one line at every width; the bar is a quiet
    # segmented control anchored to the right edge (auto left margin),
    # active segment a light accent tint, not the solid fill; the
    # delta is plain colored text, no pill background.
    assert ".pm-vh-summary {\n    display: flex;\n    align-items: flex-end;\n    gap: 14px;\n    flex-wrap: nowrap;" in CSS
    assert ".pm-vh-summary .pvc-range-bar {\n    margin: 0 0 0 auto;" in CSS
    assert ".pm-vh-summary .pvc-range-btn + .pvc-range-btn {" in CSS
    assert ".pm-vh-summary .pvc-range-btn.is-active {" in CSS
    assert ".pm-vh-delta.down { color: var(--loss); }" in CSS
    assert ".pm-vh-summary .pvc-range-btn { padding: 5px 7px; font-size: 11px; }" in CSS


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
