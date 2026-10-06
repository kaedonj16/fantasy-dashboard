"""Player modal Advanced tab TRENDING hot/cold section.

The TRENDING section surfaces only metrics with a significant recent trend at
the top of the Advanced tab, reusing the weekly series and sparkline renderer.
Quiet metrics stay as plain bars below.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_trending_placeholder_at_top_of_advanced_metrics():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Placeholder div is rendered before the metric bars (prepended to rankNote).
    assert 'id="pmTrendingSection"' in js
    assert "'<div id=\"pmTrendingSection\"></div>' + rankNote" in js


def test_trending_uses_weekly_series_and_sparkline():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function pmComputeTrending(weeks, position)" in js
    assert "function pmRenderTrendingSection(playerId, season, position)" in js
    # Reuses the existing sparkline renderer and weekly-metrics endpoint.
    assert "pmSparkline(t.series," in js
    assert "/api/player-weekly-metrics/" in js


def test_trending_is_position_aware():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function _pmTrendMetricSets(pos)" in js
    assert "p === 'RB'" in js
    assert "p === 'WR' || p === 'TE'" in js
    assert "p === 'QB'" in js


def test_trending_threshold_and_limit():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Significant = |3-wk avg vs season avg| >= 15%, top 4 by magnitude.
    assert "_PM_TREND_PCT = 0.15" in js
    assert "return out.slice(0, 4)" in js
    # Needs at least 4 weeks of data (recent-3 vs prior weeks).
    assert "_PM_TREND_MIN_WEEKS = 4" in js


def test_trending_hot_cold_badges():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "HEATING UP" in js
    assert "COOLING OFF" in js
    assert "pm-trend-badge" in js
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".pm-trend-row.hot" in css
    assert ".pm-trend-row.cold" in css


def test_trending_kicked_after_metrics_render():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function _pmKickTrending(" in js
    # Only single-season views get a trending section.
    assert "if (isCareer || isMultiSeason || !activeSeason) return;" in js
    assert "_pmKickTrending(playerId, metricsData, activeSeason, isCareer, isMultiSeason)" in js


def test_trending_shares_weekly_data_with_trends_panel():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Prefetch stores the series on the trends wrap so opening Trends later
    # does not refetch.
    assert "wrap._weeklyData = d.weeks;" in js
    assert "wrap._weeklyDataSeason" in js
    # Season change clears the cached series.
    assert "wtWrap._weeklyData = null;" in js


def test_trending_css_desktop_grid():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".pm-trending-grid" in css
    # Desktop gets a 2x2 grid.
    assert "@media (min-width: 641px)" in css
    assert "grid-template-columns: 1fr 1fr;" in css


def test_trending_copy_has_no_em_dashes():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Callouts use middots, never em dashes. Scope to the trending block.
    start = js.index("Player modal: TRENDING hot/cold section")
    end = js.index("Collapse/expand a section in the player compare view", start)
    block = js[start:end]
    assert "\u2014" not in block


def test_season_week_selectors_have_no_card():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    # The .adv-time-ctl container must not render as a card (no background,
    # border, or padding wrapping the season/week selectors).
    import re
    m = re.search(r"\.adv-time-ctl\s*\{([^}]*)\}", css)
    assert m, ".adv-time-ctl rule missing"
    body = m.group(1)
    assert "background" not in body
    assert "border" not in body or "border-" in body  # allow border-* subprops? no: strict
    assert "border:" not in body
    assert "padding" not in body


def test_season_pills_single_line_rail():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert "flex-wrap: nowrap" in css
    assert "overflow-x: auto" in css
    assert "-webkit-overflow-scrolling: touch" in css
