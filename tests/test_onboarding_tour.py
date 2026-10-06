"""Contract + route tests for onboarding / welcome tour improvements."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
LEAGUE_PAGES = (ROOT / "routes" / "league_pages_bp.py").read_text()
UI_PREFS = (ROOT / "routes" / "user_pages_bp.py").read_text()


def test_tour_resume_uses_tour_step_not_mock_param():
    assert "tour_step=" in APP_JS
    assert "params.get('tour_step')" in APP_JS
    # Live resume must not use the mock-preview ?tour= param.
    assert " + '?tour=' +" not in APP_JS
    assert "request.args.get(\"tour\") and not request.args.get(\"tour_step\")" in LEAGUE_PAGES


def test_site_tour_is_shortened_with_mobile_path():
    assert "DESKTOP_STEPS" in APP_JS
    assert "MOBILE_STEPS" in APP_JS
    assert "Remind me later" in APP_JS
    assert "show again" in APP_JS
    assert "interactive: true" in APP_JS
    # Mobile path targets the bottom dock, not the desktop hamburger.
    assert "#brMoreTab" in APP_JS
    assert ".br-tabbar" in APP_JS
    assert "#navToggle" not in APP_JS.split("MOBILE_STEPS")[1].split("function tourSteps")[0]
    # Player-card steps invite a tap; they must not auto-open the modal.
    mobile_block = APP_JS.split("MOBILE_STEPS")[1].split("function tourSteps")[0]
    assert "action: 'openPlayerModal'" not in mobile_block
    assert "tour-dismiss-row" in APP_JS
    assert "white-space: nowrap" in CSS or "tour-dismiss-row" in CSS


def test_premium_welcome_restyle_and_replay():
    assert "showSubWelcome" in APP_JS
    assert "sub-welcome-overlay" in APP_JS
    assert "Welcome to PRO" in APP_JS
    # Trade Hub workstream restyled the welcome feature row to point at /trade?tab=suggestions.
    assert "{ label: 'Trade Hub'" in APP_JS
    assert "/trade?tab=suggestions" in APP_JS
    assert "Playoff Impact" in APP_JS
    assert "settingsWelcomeBtn" in APP_PY
    assert "PRO Welcome" in APP_PY
    assert ".sub-welcome-card" in CSS
    assert "tour-hole-shield" in CSS


def test_welcome_is_plan_aware_and_names_pro():
    assert "variant !== 'league'" in APP_JS
    assert "variant !== 'claim'" in APP_JS
    assert "PRO_DESKTOP_STEPS" in APP_JS
    assert "mode: 'pro'" in APP_JS
    assert "br_skip_league_pro_banner" in APP_JS
    invite = (Path(__file__).resolve().parents[1] / "utils" / "league.py").read_text()
    assert "welcome=claim" in invite
    paywall = (Path(__file__).resolve().parents[1] / "static" / "paywall.js").read_text()
    assert "welcome=${_welcome}" in paywall or "welcome=" in paywall
    assert "br_skip_league_pro_banner" in paywall


def test_welcome_gates_site_tour_auto_start():
    assert "__brWelcomePending" in APP_JS
    assert "__brWelcomeActive" in APP_JS
    assert "window.__brWelcomeActive || window.__brWelcomePending" in APP_JS


def test_ui_prefs_and_events_endpoints_exist():
    assert '/api/ui-prefs' in UI_PREFS
    assert '/api/events' in UI_PREFS
    assert "site_tour_done" in UI_PREFS
    assert "sub_welcome_done" in UI_PREFS
    assert "register_blueprint(user_pages_bp)" in APP_PY
    assert "window.brTrack" in APP_JS
    assert "window.brUiPrefs" in APP_JS


def test_help_tours_reuses_site_tour_and_filters_empty_guides():
    assert "window.brOpenHelpTours" in APP_JS
    assert "window.startSiteTour()" in APP_JS
    assert "window.brFeatureGuides" in APP_JS
    assert "guide.steps.length > 0" in APP_JS
    assert "guide.available" in APP_JS
    # The guide registry does not introduce another completion preference.
    assert "feature_guide_done" not in APP_JS


def test_home_onboarding_account_nudge_and_espn_guidance():
    # The bottom "Create Account" nudge was removed; saving now happens only at
    # step 3 via the inline #googleContinueBtn prompt. Keep the checks for the
    # inline save behavior + ESPN guidance copy.
    assert "Success = your league dashboard loads" in APP_PY
    assert "home_league_selected" in APP_JS
    assert "home-google-ready" in APP_JS


def test_tour_dismissal_is_global_not_per_league():
    # "Don't show again" must dismiss the tour for all leagues, not just the
    # current one (Kaedon has multiple leagues and kept getting re-toured).
    assert "br_site_tour_done" in APP_JS
    assert "br_site_tour_later" in APP_JS
    # Global key is checked first in isTourDone.
    assert "lsGet(TOUR_GLOBAL_KEY) === '1'" in APP_JS
    # Legacy per-league keys migrate to the global key.
    assert "migrateLegacyTourKeys" in APP_JS
    # "Remind me later" syncs account-level.
    assert "site_tour_later: now" in APP_JS
    assert '"site_tour_later"' in UI_PREFS or "'site_tour_later'" in UI_PREFS or "site_tour_later" in UI_PREFS


def test_tour_positioning_uses_instant_scroll():
    # positionOverlays must not smooth-scroll then synchronously read the rect
    # (that strands the tooltip thousands of px off-screen).
    assert "behavior: 'auto', block: 'center'" in APP_JS
    # The site tour's positionOverlays specifically must use instant scroll.
    pos_fn = APP_JS.split("function positionOverlays(target, interactive)")[1].split("function renderTooltip")[0]
    assert "behavior: 'smooth'" not in pos_fn


def test_tour_step_param_honors_dismissal():
    # ?tour_step= must not bypass dismissal; only an actively-in-progress tour
    # (session flag, set on cross-page step navigation and manual replay)
    # resumes without the dismissal check.
    assert "br_tour_session" in APP_JS
    assert "tourSessionActive()" in APP_JS
    assert "setTourSession(true)" in APP_JS
    assert "setTourSession(false)" in APP_JS
