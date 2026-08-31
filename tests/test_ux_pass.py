"""Tests for the focused UX implementation pass."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DASH = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
OS_DASH = (ROOT / "dashboard_services" / "pages" / "offseason_dashboard_page.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
CHEAT_JS = (ROOT / "static" / "cheat_sheet.js").read_text(encoding="utf-8")
PLAYER_MODAL = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
WAIVERS = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


def test_dashboard_action_first_hierarchy():
    assert "os-action-queue" in DASH
    assert 'data-jump="os-jump-actions"' in DASH
    assert "lineup_alert_html" in DASH
    assert "matchup_html" in DASH
    assert 'data-jump="os-jump-matchup"' in DASH
    # Center column stays action-first: digest, then next steps, then matchup.
    assert DASH.index("sinceLastVisitCard") < DASH.index("os-jump-matchup")
    assert DASH.index("os-jump-matchup") > DASH.index("do_next_waiver_html")
    assert DASH.index("matchup_html") > DASH.index("do_next_waiver_html")
    # Front Office Report fills the left rail instead of stacking under matchup.
    assert DASH.index("{gm_card_html}") < DASH.index('class="os-main-col"')
    assert DASH.index("{matchup_html}") > DASH.index('class="os-main-col"')
    assert "Waiver Wire Targets" not in DASH
    assert "_render_do_next_waiver_card" in DASH


def test_offseason_dashboard_action_queue():
    assert "os-action-queue" in OS_DASH
    assert "_render_do_next_waiver_card" in OS_DASH
    assert "draft_prep_hrefs" in OS_DASH
    assert "Draft Room" in OS_DASH
    assert "Waiver Wire Targets" not in OS_DASH
    assert "Best waiver available" not in OS_DASH


def test_cheat_sheet_terminology_tooltips():
    assert "VALUE: Value over replacement" in CHEAT_JS
    assert "MARKET: Where market signals" in CHEAT_JS
    assert "cs-advanced-col" not in CHEAT_JS
    assert "csShowAdvanced" not in CHEAT_JS


def test_rankings_table_unchanged_from_main():
    rankings = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
    players_page = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
    assert "prShowAdvanced" not in rankings
    assert "prAdvancedToggle" not in players_page


def test_cross_feature_links():
    assert "pmInjectContextActions" in PLAYER_MODAL
    assert "Trade For" in PLAYER_MODAL
    assert "wv-ctx-link" in WAIVERS
    assert "Compare to roster" in WAIVERS
    assert "View schedule" in WAIVERS


def test_command_palette_feature_navigation():
    assert "NAV_COMMANDS" in APP_JS
    assert "nav-search-cmd" in APP_JS
    assert "nav-search-group-label" in APP_JS
    assert "Trade Calculator" in APP_JS
    assert "brNavLeagueUrl" in APP_JS
    assert "Search players or jump to" in APP_PY


def test_player_search_still_works():
    assert "nav-search-result" in APP_JS
    assert "openPlayerModal" in APP_JS
    assert "/api/league-players" in APP_JS


def test_keyboard_command_palette_behavior():
    assert "ArrowDown" in APP_JS
    assert "ArrowUp" in APP_JS
    assert "Escape" in APP_JS
    assert 'e.key === "Enter"' in APP_JS or "e.key === 'Enter'" in APP_JS
    assert "ctrlKey" in APP_JS and "metaKey" in APP_JS


def test_pro_preview_does_not_expose_gated_details():
    assert "brProPreview" in PAYWALL_JS
    assert "preview_count" not in PAYWALL_JS.lower() or "opts.count" in PAYWALL_JS
    assert "br-pro-preview-msg" in PAYWALL_JS
    assert "brProPreview" in APP_JS


def test_analytics_terminology_module():
    from utils.analytics_terminology import LABELS, tooltip, label, title_attr

    assert label("vor") == "VOR"
    assert "Dynasty trade value" in tooltip("br_value")
    assert "VALUE:" in title_attr("br_value")
    assert LABELS["market_vs_adp"]["category"] == "MARKET"
    assert LABELS["historical_hit_rate"]["category"] == "HISTORY"


def test_do_next_waiver_card_helper():
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _render_do_next_waiver_card" in app_src
    assert "Next steps" in app_src
    assert "Do this next" not in app_src
    assert "os-do-next-collapsed" in app_src
    assert "os-do-next-draft" in app_src
    assert "Draft prep" in app_src
    assert "startup_draft_pending" in app_src
    assert "Get ready for your draft" in app_src


def test_hub_column_height_sync_helper():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "initHubColumnHeightSync" in app_js
    assert "os-hub-cols-synced" in app_js
    assert "ResizeObserver" in app_js


def test_dashboard_css_action_and_palette_styles():
    assert ".os-action-queue" in DASH_CSS
    assert ".os-do-next-card" in DASH_CSS
    assert ".os-do-next-draft" in DASH_CSS
    assert ".nav-search-group-label" in DASH_CSS
    assert ".br-pro-preview" in DASH_CSS
    assert ".pr-advanced-col" not in DASH_CSS
    assert ".cs-advanced-col" not in DASH_CSS
