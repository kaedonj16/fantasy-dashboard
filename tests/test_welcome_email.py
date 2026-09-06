"""Signup / PRO welcome emails and onboarding opt-out preferences."""
from __future__ import annotations

from unittest import mock

from utils.email_preferences import (
    ONBOARDING,
    WEEKLY_DIGEST,
    is_enabled,
    unsubscribe_onboarding,
)
from utils import welcome_email as we


def test_onboarding_defaults_on_without_row():
    class _Conn:
        def execute(self, *a, **k):
            class R:
                def fetchone(self):
                    return None
            return R()

    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = _Conn()
        ctx.__exit__.return_value = False
        gc.return_value = ctx
        assert is_enabled(1, ONBOARDING, email_opt_out=True) is True
        assert is_enabled(1, "product_updates", email_opt_out=False) is False


def test_unsubscribe_onboarding_sets_preference_false():
    with mock.patch("utils.email_preferences.set_enabled", return_value=True) as se:
        assert unsubscribe_onboarding(9) is True
        se.assert_called_once_with(9, False, ONBOARDING)


def test_signup_welcome_html_has_logos_and_depth(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasyfootball.com")
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-test-secret")
    out = we.build_signup_welcome(first_name="Pat", dash_url="https://brfantasyfootball.com/")
    html = out["html"]
    assert out["subject"].startswith("Welcome to BR Fantasy")
    assert "BR_Logo_dark.png" in html
    assert "BR_Mark_dark.png" in html
    assert "sleeper-logo.png" in html
    assert "espn-logo.png" in html
    assert "Trade Calculator" in html
    assert "Weekly email digest" in html
    assert "welcome and onboarding emails" in html
    assert "{UNSUB}" in html or "unsubscribe" in html.lower()


def test_pro_welcome_html_covers_toolkit(monkeypatch):
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasyfootball.com")
    out = we.build_pro_welcome(
        first_name="Sam",
        plan="combo",
        platform="sleeper",
        season=2026,
        league_id="12345",
    )
    html = out["html"]
    assert "League + Personal PRO" in out["subject"]
    assert "Trade Suggestions" in html
    assert "Breakout Engine" in html
    assert "Front Office Report" in html
    assert "BR_Logo_dark.png" in html
    assert "/sleeper/2026/12345/trade?tab=suggestions" in html
    assert "welcome and onboarding emails" in html


def test_send_signup_respects_opt_out(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasyfootball.com")
    with mock.patch("utils.welcome_email._should_send", return_value=(False, "opted_out")), \
         mock.patch("utils.welcome_email._claim_once") as claim:
        assert we.send_signup_welcome(3, email="a@b.com", first_name="A") is False
        claim.assert_not_called()


def test_send_pro_dedupes_via_claim(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("SITE_BASE_URL", "https://brfantasyfootball.com")
    with mock.patch("utils.welcome_email._account_email_row", return_value={
             "id": 5, "email": "pro@ex.com", "first_name": "Pro",
         }), \
         mock.patch("utils.welcome_email._should_send", return_value=(True, "ok")), \
         mock.patch("utils.welcome_email._claim_once", return_value=False) as claim, \
         mock.patch("utils.welcome_email._deliver") as deliver:
        assert we.send_pro_welcome(5, plan="user") is False
        claim.assert_called_once()
        deliver.assert_not_called()


def test_weekly_digest_still_defaults_on():
    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("utils.email_preferences._legacy_opt_out", return_value=False), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        gc.side_effect = RuntimeError("no db")
        assert is_enabled(1, WEEKLY_DIGEST, email_opt_out=False) is True
