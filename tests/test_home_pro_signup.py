from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing


ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
GOOGLE = (ROOT / "routes" / "google_auth_bp.py").read_text(encoding="utf-8")
YAHOO = (ROOT / "routes" / "yahoo_auth_bp.py").read_text(encoding="utf-8")
BILLING = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
LANDING_CSS = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")


def _home_form() -> str:
    return APP_PY[APP_PY.index('FORM_BODY = """'):APP_PY.index('<div class="home-content-wrapper">')]


def test_home_proof_is_replaced_by_pro_signup():
    home = _home_form()
    assert 'class="home-proof"' not in home
    assert 'id="homeProSignup"' in home
    assert 'data-home-pro-open' in home
    assert "Unlock PRO" in home
    assert "Trade Intelligence" in home
    assert "Breakout Engine" in home
    assert "Playoff Impact" in home
    assert "data-plan=" not in home
    assert "—" not in home
    assert "&mdash;" not in home


def test_home_pro_styles_live_in_dashboard_and_landing_css():
    for css in (DASH_CSS, LANDING_CSS):
        assert ".home-pro {" in css
        assert ".home-pro-benefits {" in css
        assert ".home-pro-hero-cta" in css
        assert ".home-pro-open-btn" in css
        assert ".home-proof {" not in css
    paywall_css = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
    assert ".paywall-modal .home-pro-fields[hidden]" in paywall_css


def test_home_pro_js_opens_modal_then_stages_google():
    source = PAYWALL_JS[PAYWALL_JS.index("function openHomeProModal"):]
    assert "className = 'paywall-modal'" in source
    assert 'data-plan="single_league"' in source
    assert 'data-plan="combo"' in source
    assert 'id="homeProGoogle" class="google-continue-btn"' in source
    assert "fetch('/api/pro-signup/pending'" in source
    assert "/auth/google?intent=onboarding&next=/pro/resume-checkout" in source
    assert "/api/sleeper-user-leagues?username=" in source
    assert "_initiatePurchaseWithLeague" in source
    assert "checkoutBtn.hidden = !window._hasAccount" in source
    assert "fetch('/api/identify'" not in source
    assert "data-home-pro-open" in PAYWALL_JS[PAYWALL_JS.index("function initHomeProSignup"):]


def test_paywall_subscribe_requires_google_account():
    purchase = PAYWALL_JS[PAYWALL_JS.index("async function initiatePurchase"):]
    purchase = purchase[: PAYWALL_JS.index("function addPremiumBadge") - PAYWALL_JS.index("async function initiatePurchase")]
    assert "_hasGoogleAccount()" in purchase
    assert "brOpenSignin" not in purchase
    assert "signinModal" not in purchase
    ident = PAYWALL_JS[PAYWALL_JS.index("function _showIdentifyModal"):]
    ident = ident[: PAYWALL_JS.index("async function _initiatePurchaseWithLeague") - PAYWALL_JS.index("function _showIdentifyModal")]
    assert "A Google account is required to subscribe" in ident
    assert "goGoogleWithLeague" in ident
    assert "_startGoogleSubscribe" in ident
    checkout = BILLING[BILLING.index("def create_checkout_session"):]
    checkout = checkout[: checkout.index("if plan not in _STRIPE_PRICES")]
    assert "_require_google_to_subscribe" in checkout
    assert 'session.get("account_id")' in BILLING[BILLING.index("def resume_pro_checkout"):]


def test_google_and_yahoo_resume_pending_checkout():
    assert "def _redirect_after_google" in GOOGLE
    assert "pending_checkout_resume_path" in GOOGLE
    assert "pending_checkout_resume_path" in YAHOO
    assert 'return _redirect_after_google(destination or next_url)' in GOOGLE


def test_pro_signup_pending_requires_plan_and_league(offline_client):
    missing = offline_client.post("/api/pro-signup/pending", json={})
    assert missing.status_code == 400
    assert missing.json["ok"] is False

    league_missing = offline_client.post("/api/pro-signup/pending", json={"plan": "league"})
    assert league_missing.status_code == 400

    user_ok = offline_client.post("/api/pro-signup/pending", json={"plan": "user"})
    assert user_ok.status_code == 200
    assert user_ok.json["ok"] is True

    bad_plan = offline_client.post("/api/pro-signup/pending", json={
        "plan": "lifetime", "league_id": "123", "platform": "sleeper",
    })
    assert bad_plan.status_code == 400

    ok = offline_client.post("/api/pro-signup/pending", json={
        "plan": "single_league",
        "league_id": "999",
        "platform": "espn",
        "season": 2026,
        "username": "Ryan",
    })
    assert ok.status_code == 200
    assert ok.json["ok"] is True
    assert ok.json["auth_url"] == "/auth/google?intent=onboarding&next=/pro/resume-checkout"

    with offline_client.session_transaction() as sess:
        assert sess["pending_checkout"]["plan"] == "single_league"
        assert sess["pending_checkout"]["league_id"] == "999"
        assert sess["pending_link"]["platform"] == "espn"
        assert sess["pending_link"]["username"] == "Ryan"


def test_resume_checkout_sends_guests_to_google(offline_client):
    with offline_client.session_transaction() as sess:
        sess["pending_checkout"] = {
            "plan": "user", "league_id": "123", "platform": "sleeper", "season": 2026,
        }
    response = offline_client.get("/pro/resume-checkout")
    assert response.status_code == 302
    assert "/auth/google" in response.headers["Location"]


def test_resume_checkout_requires_google_not_sleeper_only(offline_client):
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "Ryan"
        sess["viewer_user_id"] = "u1"
        sess["pending_checkout"] = {
            "plan": "user", "league_id": "123", "platform": "sleeper", "season": 2026,
        }
    response = offline_client.get("/pro/resume-checkout")
    assert response.status_code == 302
    assert "/auth/google" in response.headers["Location"]


def test_resume_checkout_opens_stripe_for_signed_in_users(offline_client, monkeypatch):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/home-pro")

    fake_stripe = SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession))
    monkeypatch.setattr(billing, "_stripe", lambda: fake_stripe)
    monkeypatch.setattr(
        "dashboard_services.subscriptions.viewer_is_league_member",
        lambda *args, **kwargs: True,
    )

    with offline_client.session_transaction() as sess:
        sess["account_id"] = 44
        sess["viewer_user_id"] = "u44"
        sess["pending_checkout"] = {
            "plan": "single_league",
            "league_id": "555",
            "platform": "sleeper",
            "season": 2026,
        }

    response = offline_client.get("/pro/resume-checkout")
    assert response.status_code == 302
    assert response.headers["Location"] == "https://checkout.stripe.test/home-pro"
    assert captured["metadata"]["plan"] == "single_league"
    assert captured["metadata"]["league_id"] == "555"
    assert "/sleeper/2026/555/dashboard" in unquote(captured["success_url"])

    with offline_client.session_transaction() as sess:
        assert "pending_checkout" not in sess


def test_guest_home_renders_pro_signup(offline_client):
    html = offline_client.get("/").get_data(as_text=True)
    assert "homeProSignup" in html
    assert "home-proof" not in html
    assert "Unlock PRO" in html
    assert "data-home-pro-open" in html
    assert "Continue with Google" in html
    assert "A Google account is required to subscribe" in html


class _JsonResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_google_callback_resumes_home_pro_checkout(monkeypatch):
    import flask
    from routes.google_auth_bp import google_auth_bp

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost/auth/google/callback")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _JsonResp({"id_token": "tok"}))
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *args, **kwargs: {
            "sub": "g-sub", "email": "user@example.com",
            "nonce": "nonce", "given_name": "Pat",
        },
    )
    monkeypatch.setattr("dashboard_services.accounts.upsert_google_account", lambda *a, **k: 77)
    monkeypatch.setattr("dashboard_services.accounts.link_platform_identity", lambda *a, **k: "ok")
    monkeypatch.setattr("dashboard_services.accounts.add_user_league", lambda *a, **k: None)
    monkeypatch.setattr("dashboard_services.accounts.get_post_login_destination", lambda *a, **k: "/")

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(google_auth_bp)

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["google_oauth_state"] = "state"
            sess["google_oauth_nonce"] = "nonce"
            sess["google_pkce_verifier"] = "verifier"
            sess["google_oauth_next"] = "/"
            sess["pending_link"] = {
                "platform": "sleeper", "league_id": "555",
                "season": 2026, "username": "Ryan",
            }
            sess["pending_checkout"] = {
                "plan": "single_league", "league_id": "555",
                "platform": "sleeper", "season": 2026,
            }
        response = client.get("/auth/google/callback?code=code&state=state")

    assert response.status_code == 302
    assert response.headers["Location"] == "/pro/resume-checkout"
