from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_signed_in_home_separates_saved_leagues_from_connect_flow():
    source = (ROOT / "app.py").read_text()
    assert "Your leagues" in source
    assert 'id="connectLeagueFlow"{% if session.get(\'account_id\') %} hidden{% endif %}' in source
    assert 'aria-controls="connectLeagueFlow"' in source


def test_saved_leagues_are_links_with_an_explicit_open_action():
    source = (ROOT / "static" / "app.js").read_text()
    assert 'class="signed-home-league" href="${url}"' in source
    assert 'signed-home-league-open' in source
    assert 'setHomeCardState("connect")' in source


def test_connect_another_league_replaces_and_restores_card_state():
    markup = (ROOT / "app.py").read_text()
    script = (ROOT / "static" / "app.js").read_text()
    assert 'id="homeCardTitle"' in markup
    assert 'id="homeConnectBack"' in markup
    assert 'if (signedInHome) signedInHome.hidden = !connected' in script
    assert 'if (connectLeagueFlow) connectLeagueFlow.hidden = connected || returning' in script
    assert 'homeConnectBack?.addEventListener("click"' in script
    assert 'setHomeCardState(window._hasAccount ? "connected" : "returning")' in script


def test_signed_in_league_save_refreshes_existing_memberships_in_place():
    script = (ROOT / "static" / "app.js").read_text()
    link_route = (ROOT / "routes" / "link_bp.py").read_text()
    assert 'fetch("/api/link/add"' in script
    assert 'await window.refreshHomeLeagues?.()' in script
    assert 'setHomeCardState("connected")' in script
    assert "link_platform_identity(account_id, \"sleeper\"" in link_route
    assert "get_sleeper_user_leagues(platform_user_id, season)" in link_route


def test_returning_users_can_fully_reset_local_identity():
    markup = (ROOT / "app.py").read_text()
    script = (ROOT / "static" / "app.js").read_text()
    auth = (ROOT / "routes" / "auth_bp.py").read_text()
    assert 'class="home-reset-user" href="/reset-user">Not me?</a>' in markup
    assert 'class="saved-viewer-reset" href="/reset-user">Not me?</a>' in script
    assert '@auth_bp.route("/reset-user")' in auth
    assert "localStorage.removeItem('saved_viewer')" in auth
    assert "localStorage.removeItem('saved_account')" in auth
    assert "sessionStorage.clear()" in auth


def test_saved_league_list_is_client_paginated_and_card_width_is_stable():
    script = (ROOT / "static" / "app.js").read_text()
    css = (ROOT / "static" / "dashboard.css").read_text()
    assert "const pageSize = 3" in script
    assert "leagues.slice(leaguePage * pageSize" in script
    assert "Page ${leaguePage + 1} of ${pageCount}" in script
    card_rule = css[css.index(".home-card {"):css.index("/* \"Continue as X\"")]
    assert "width: 100%" in card_rule
    assert "max-width: 380px" in card_rule
    assert "box-sizing: border-box" in card_rule


def test_home_copy_touched_by_state_machine_has_no_em_dash():
    markup = (ROOT / "app.py").read_text()
    home_form = markup[markup.index('FORM_BODY = """'):markup.index('<div class="home-content-wrapper">')]
    assert "—" not in home_form
    assert "&mdash;" not in home_form


def test_lite_css_swap_uses_landing_lite_for_guest_home():
    """Guest home uses landing_lite.css; seo_lite stays free of home-* rules.

    Homepage layout (hero, onboarding card, feature grid, ticker) is not in
    seo_lite.css — guests get landing_lite.css instead. Signed-in home keeps
    full dashboard.css via the non-lite branch.
    """
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '_page_css_file = "landing_lite.css"' in app_py
    assert 'active == "home"' in app_py
    assert "_LANDING_LITE_CSS_V" in app_py
    seo_css = (ROOT / "static" / "seo_lite.css").read_text(encoding="utf-8")
    assert ".home-hero" not in seo_css
    assert ".home-card" not in seo_css
    landing = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
    assert ".home-hero" in landing
    assert ".home-card" in landing
    assert ".home-ticker-band" in landing
    assert ".home-pro" in landing
    home_rules = len([ln for ln in landing.splitlines() if ln.startswith(".home-")])
    assert home_rules >= 100
    # Guest SEO pages still emit dropdowns + the More sheet. Without these
    # hides, every nav link spills in-flow on logged-out pages.
    assert ".nav-pill-dropdown-menu" in seo_css
    assert ".skip-link" in seo_css
    assert (
        ".br-tabbar,\n.br-sheet-scrim,\n.br-sheet,\n.br-search-screen {\n    display: none;\n}"
        in seo_css
    )
    assert ".otc-day-filter" in seo_css
    assert ".csd-wrap" in seo_css
    assert ".csd-list" in seo_css
    assert ".compare-pick-results" in seo_css
    assert "--card-bg:" in seo_css
    assert "--radius-pill:" in seo_css
