"""Contract + route tests for onboarding / welcome tour improvements."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()
APP_PY = (ROOT / "app.py").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
LEAGUE_PAGES = (ROOT / "routes" / "league_pages_bp.py").read_text()
UI_PREFS = (ROOT / "routes" / "ui_prefs_bp.py").read_text()


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
    assert "Open Trade Suggestions" in APP_JS
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
    invite = (Path(__file__).resolve().parents[1] / "utils" / "league_invite.py").read_text()
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
    assert "register_blueprint(ui_prefs_bp)" in APP_PY
    assert "window.brTrack" in APP_JS
    assert "window.brUiPrefs" in APP_JS


def test_home_onboarding_account_nudge_and_espn_guidance():
    assert 'id="homeLeagueReadyNudge"' in APP_PY
    assert "Success = your league dashboard loads" in APP_PY
    assert "home_league_selected" in APP_JS
    assert "home-google-ready" in APP_JS
    assert "Save this league to your account" in APP_JS
