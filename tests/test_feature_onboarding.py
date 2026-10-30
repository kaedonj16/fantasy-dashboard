"""Regression tests for the tour audit cleanup.

The old tour tests only asserted strings exist in app.js, which is how dead
selectors shipped: the per-page coach marks (FeatureOnboarding) targeted
classes no page renders anymore, had no CSS at all, and burned users'
one-time completion keys without showing a single step. These tests pin the
fixed behavior and, where the offline client can render the page, check the
tour targets against real rendered markup.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()

FOB = APP_JS.split("Feature onboarding coach marks")[1]


def test_fob_has_css_for_every_class_it_builds():
    for cls in ["fob-overlay", "fob-spotlight", "fob-tooltip", "fob-step",
                "fob-msg", "fob-row", "fob-skip", "fob-next", "fob-dots"]:
        assert f".{cls}" in CSS, f"missing CSS for .{cls}"
    # The spotlight cut-out comes from its box-shadow dim.
    assert "box-shadow: 0 0 0 9999px" in CSS


def test_fob_selectors_match_current_markup():
    # Advanced Metrics graph dots are circle.am-graph-dot, not .gdot/.dot.
    assert "#amGraphModal circle.am-graph-dot" in FOB
    assert ".gdot" not in FOB
    assert "circle.dot" not in FOB
    # Start/Sit board targets: verdict badges and the quick-compare button.
    assert ".wv-cx-badge" in FOB
    assert ".wv-cx-quickcmp" in FOB
    assert ".ss-verdict" not in FOB
    assert "ss-compare-hint" not in FOB
    assert "data-ss-compare" not in FOB
    # Trade Hub steps target the always-rendered main tabs.
    assert '.otc-main-tab[data-tab="suggestions"]' in FOB
    assert '.otc-main-tab[data-tab="calculator"]' in FOB


def test_fob_copy_is_not_the_stale_copy():
    assert "The number is confidence" not in FOB
    assert "Every row is a metric" not in FOB
    assert "projected points" in FOB


def test_fob_mechanics_fixed():
    # Versioned keys: users burned by the broken v1 tour get the fixed one.
    assert "onboarded-trade-hub-v2" in FOB
    assert "onboarded-start-sit-v2" in FOB
    # A tour that never showed a step must not mark itself done.
    assert "tour.shown > 0" in FOB
    # Retry budget is per step, not shared across the tour.
    assert "tour.retryIdx" in FOB
    # Instant scroll before measuring (the PR #2283 site-tour lesson).
    fob_position = FOB.split("function positionCoach")[1].split("function renderStep")[0]
    assert "behavior: 'auto'" in fob_position
    assert "behavior: 'smooth'" not in fob_position
    # Soft-nav arrivals re-check via initPageRoot.
    assert "window.FeatureOnboarding.check()" in APP_JS


def test_guides_registered_and_clickable():
    for gid in ["trade-hub", "advanced-metrics", "start-sit"]:
        assert f"brFeatureGuides['{gid}']" in APP_JS
    # Desktop dialog handles guide buttons and offers a real PRO tour replay.
    assert "data-help-pro-tour" in APP_JS
    assert "window.FeatureOnboarding.start(gid)" in APP_JS
    # Mobile sheet populator + action are wired.
    assert "brPopulateSheetGuides" in APP_JS
    assert "name === 'feature-guide'" in APP_JS


def test_retired_guest_tour_css_removed_but_live_rules_kept():
    """The guest product tour was retired by the A+ landing; its CSS
    block (bento previews, hp-rz mockups, hm deck, tour CTA, duplicate
    wv-cx / wrapped / otc rules that stomped the live ones) is gone.
    The live hero board (.hp-mb) and the Start/Sit matchup-tier chips
    (.wv-cx-chip.mt1-4, whose only definitions lived in that block)
    survive."""
    css = CSS
    for dead in (".home-bento", ".home-preview", ".tour-cta",
                 ".hm-deck", ".hm-slide", ".wrapped-deck", ".trust-strip",
                 ".quote-card", ".otc-grade"):
        assert dead not in css, f"retired guest-tour CSS still present: {dead}"
    # The .hp-rz game-strip pill rules stay for now: PR #2903's test pins
    # them as the homepage-preview mirror of the ScoreZone pills, even
    # though no current markup uses the scope.
    assert ".hp-rz .rz-gp-versus" in css
    assert ".hp-mb" in css, "live hero board styles must stay"
    for tier in ("mt1", "mt2", "mt3", "mt4"):
        assert f".wv-cx-chip.{tier}" in css, f"live chip tier style lost: {tier}"


def test_site_tour_copy_fixed():
    site = APP_JS.split("New User Site Tour")[1].split("Contextual guides")[0]
    assert "breakout score" not in site
    assert "open Trade Intel" not in site
    assert ".breakouts-root" not in site
    # PRO Market Intel steps actually open the panel they describe.
    assert "_openMarketIntel" in site
    assert "#otcMarketIntelPanel" in site
    # Replay promises point at Help & Tours, where the buttons now live.
    assert "Help &amp; Tours" in site or "Help & Tours" in site


# ── Rendered-page target checks ─────────────────────────────────────────────

def test_trade_page_renders_fob_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/trade?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert 'data-tab="suggestions"' in html
    assert 'data-tab="calculator"' in html
    assert "otc-sugg-subtab-toggle" in html
    assert 'id="otcMarketIntelPanel"' in html
    assert 'id="otcSubtabMarket"' in html


def test_metrics_page_renders_fob_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/metrics?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert "am-cmdbar" in html
    assert 'id="amGraphBtn"' in html
    assert 'id="amTable"' in html or "am-table" in html


def test_waivers_page_renders_fob_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/waivers?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert 'id="wvSectionStartSit"' in html
    assert 'id="wvTabStartSit"' in html
    # The start/sit row markup (badges + quick-compare) ships with the page.
    assert "wv-cx-quickcmp" in html
    assert "wv-cx-badge" in html


# ── The six tours added 2026-10-30 (trade calc + the five recommended) ─────

def test_new_tours_registered_with_real_targets():
    fob = FOB
    for feature, key in (
        ("trade-calc", "onboarded-trade-calc-v1"),
        ("waivers", "onboarded-waivers-v1"),
        ("portfolio", "onboarded-portfolio-v1"),
        ("teams", "onboarded-teams-v1"),
        ("rankings", "onboarded-rankings-v1"),
        ("player-modal", "onboarded-player-modal-v1"),
    ):
        assert key in fob, f"missing versioned key for {feature}"
        assert f"brFeatureGuides['{feature}']" in fob, f"guide not registered: {feature}"
    # Trade page splits by premium: Hub tour for PRO, Calculator tour else.
    assert "isPremiumUser() ? 'trade-hub' : 'trade-calc'" in fob
    # Calc tour targets (all rendered by trade_calculator_page.py).
    for sel in (".otc-search-input", "#tradeVerdict", ".otc-side-title-link"):
        assert sel in fob, f"calc tour target missing: {sel}"
    # Portfolio / Teams / Rankings / Waivers targets.
    for sel in (".pf-lg-card[data-href]", ".pf-lg-fav", ".pf-done-btn",
                ".teams-mode-toggle", ".team-strength-card", ".teams-ptabs",
                "#prSettingsBtn", ".pos-pill[data-pos=\"QB\"]", "#prSortHeader",
                "#wvSectionWaivers .wv-cx-row", ".wv-horizon-btn[data-h=\"four_week\"]"):
        assert sel in fob, f"tour target missing: {sel}"
    # Player modal tour: stable anchors + first-render trigger, because the
    # modal does not exist at page load.
    assert "#pmContextActions .pm-actions-trigger" in fob
    assert "#pmTabBar" in fob
    assert "MutationObserver" in fob and "pmPlayerId" in fob


def test_trade_page_renders_calc_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/trade?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert "otc-search-input" in html
    assert 'id="tradeVerdict"' in html


def test_waivers_page_renders_waivers_tour_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/waivers?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert 'id="wvSectionWaivers"' in html
    assert 'id="wvTabWaivers"' in html
    assert 'data-h="four_week"' in html


def test_teams_page_renders_fob_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/teams?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert "teams-mode-toggle" in html
    assert "team-strength-card" in html
    assert "teams-ptabs" in html


def test_players_page_renders_rankings_targets(offline_client):
    r = offline_client.get("/sleeper/2026/tourdemo/players?tour=1")
    assert r.status_code == 200, r.status_code
    html = r.get_data(as_text=True)
    assert 'id="prSettingsBtn"' in html
    assert "pos-pill" in html
    assert 'id="prSortHeader"' in html


def test_portfolio_and_modal_targets_exist_in_sources():
    app_py = (ROOT / "app.py").read_text()
    portfolio = app_py.split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert "pf-lg-card" in portfolio and "data-href" in portfolio
    assert "pf-lg-fav" in portfolio and "pf-lg-menu-btn" in portfolio
    # Front Office Done buttons are rendered client-side into the moves card.
    assert "pf-done-btn" in APP_JS
    modal_js = (ROOT / "static" / "player_modal.js").read_text()
    assert "pm-actions-trigger" in modal_js
    assert 'id="pmTabBar"' in modal_js or "pmTabBar" in modal_js
