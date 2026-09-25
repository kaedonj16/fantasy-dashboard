"""Stripe Customer Portal: /api/create-portal-session + pricing-page manage banner."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

# The lightweight CI job intentionally installs pytest without the Flask stack;
# app integration tests must skip during collection there, just like the other
# offline_client tests. The full-stack CI job installs Flask and runs these.
pytest.importorskip("flask")

import routes.billing_bp as billing


def _sub_info(**kw):
    base = {
        "has_premium": False,
        "subscription_type": None,
        "has_league_subscription": False,
        "has_user_subscription": False,
        "has_single_league_subscription": False,
        "expires_at": None,
        "subscriber_user_id": None,
        "stripe_customer_id": None,
    }
    base.update(kw)
    return base


def _mock_subscription(monkeypatch, handler):
    monkeypatch.setattr(
        "dashboard_services.subscriptions.get_subscription_info", handler
    )


def _mock_stripe_portal(monkeypatch, captured):
    def _create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(url="https://portal.stripe.test/session_abc")

    fake = SimpleNamespace(
        billing_portal=SimpleNamespace(Session=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)


def test_portal_endpoint_success_returns_url(offline_client, monkeypatch):
    _mock_subscription(
        monkeypatch,
        lambda user_id, league_id, platform="sleeper": _sub_info(
            stripe_customer_id="cus_personal",
            has_user_subscription=True,
            subscription_type="user",
        ),
    )
    captured = {}
    _mock_stripe_portal(monkeypatch, captured)

    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "user1"
    r = offline_client.post(
        "/api/create-portal-session",
        json={"platform": "sleeper"},
    )
    assert r.status_code == 200
    assert r.get_json()["url"] == "https://portal.stripe.test/session_abc"
    assert captured["customer"] == "cus_personal"
    assert "pricing" in captured["return_url"]


def test_portal_endpoint_requires_login(offline_client):
    r = offline_client.post("/api/create-portal-session", json={"platform": "sleeper"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "Not logged in"


def test_portal_endpoint_rejects_bad_platform(offline_client):
    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "user1"
    r = offline_client.post(
        "/api/create-portal-session", json={"platform": "nope"}
    )
    assert r.status_code == 400


def test_portal_endpoint_no_customer_is_graceful_404(offline_client, monkeypatch):
    # No Stripe customer anywhere: must be a 404 with the friendly message,
    # never a 500.
    _mock_subscription(
        monkeypatch,
        lambda user_id, league_id, platform="sleeper": _sub_info(),
    )
    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "freeuser"
    r = offline_client.post(
        "/api/create-portal-session", json={"platform": "sleeper"}
    )
    assert r.status_code == 404
    body = r.get_json()
    assert "No Stripe customer found" in body["error"]


def test_portal_endpoint_league_buyer_only(offline_client, monkeypatch):
    # League mate (not the buyer) must NOT get the buyer's portal session.
    def handler(user_id, league_id, platform="sleeper"):
        if league_id == "lg1":
            return _sub_info(
                has_league_subscription=True,
                subscriber_user_id="buyer1",
                stripe_customer_id="cus_league",
                subscription_type="league",
            )
        return _sub_info()

    _mock_subscription(monkeypatch, handler)
    captured = {}
    _mock_stripe_portal(monkeypatch, captured)

    # The buyer gets a portal session for the shared league row.
    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "buyer1"
    r = offline_client.post(
        "/api/create-portal-session",
        json={"platform": "sleeper", "league_id": "lg1"},
    )
    assert r.status_code == 200
    assert captured["customer"] == "cus_league"

    # A league mate only resolves their own rows: no buyer customer here.
    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "mate2"
    r = offline_client.post(
        "/api/create-portal-session",
        json={"platform": "sleeper", "league_id": "lg1"},
    )
    assert r.status_code == 404
    assert "No Stripe customer found" in r.get_json()["error"]


def test_portal_endpoint_personal_customer_used_for_league_page(
    offline_client, monkeypatch
):
    # Buyer-only applies to the shared league row; the buyer's own personal
    # subscription still resolves on a league-scoped page.
    def handler(user_id, league_id, platform="sleeper"):
        if league_id:
            return _sub_info(has_league_subscription=False)
        return _sub_info(
            stripe_customer_id="cus_personal",
            has_user_subscription=True,
            subscription_type="user",
        )

    _mock_subscription(monkeypatch, handler)
    captured = {}
    _mock_stripe_portal(monkeypatch, captured)

    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "user1"
    r = offline_client.post(
        "/api/create-portal-session",
        json={"platform": "sleeper", "league_id": "lg1"},
    )
    assert r.status_code == 200
    assert captured["customer"] == "cus_personal"


def test_pricing_banner_shows_only_when_customer_resolves(
    offline_client, monkeypatch
):
    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "pro1"

    _mock_subscription(
        monkeypatch,
        lambda user_id, league_id, platform="sleeper": _sub_info(
            stripe_customer_id="cus_personal", has_user_subscription=True
        ),
    )
    r = offline_client.get("/pricing")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "manage-sub-btn" in html
    assert "Manage subscription" in html

    # No customer: banner stays hidden, page still renders fine.
    _mock_subscription(
        monkeypatch, lambda user_id, league_id, platform="sleeper": _sub_info()
    )
    r = offline_client.get("/pricing")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "manage-sub-btn" not in html

    # Guest (not logged in): no banner, no error.
    with offline_client.session_transaction() as sess:
        sess.clear()
    r = offline_client.get("/pricing")
    assert r.status_code == 200
    assert "manage-sub-btn" not in r.get_data(as_text=True)


def test_manage_banner_league_buyer_only(offline_client, monkeypatch):
    def handler(user_id, league_id, platform="sleeper"):
        if league_id == "lg1":
            return _sub_info(
                has_league_subscription=True,
                subscriber_user_id="buyer1",
                stripe_customer_id="cus_league",
            )
        return _sub_info()

    _mock_subscription(monkeypatch, handler)

    with offline_client.application.test_request_context("/sleeper/2026/lg1/pricing"):
        from flask import session

        session["viewer_user_id"] = "buyer1"
        assert "manage-sub-btn" in billing._manage_subscription_banner("lg1", "sleeper")

        session["viewer_user_id"] = "mate2"
        assert billing._manage_subscription_banner("lg1", "sleeper") == ""


def test_manage_banner_never_raises(offline_client, monkeypatch):
    def boom(user_id, league_id, platform="sleeper"):
        raise RuntimeError("db exploded")

    _mock_subscription(monkeypatch, boom)
    with offline_client.application.test_request_context("/pricing"):
        from flask import session

        session["viewer_user_id"] = "user1"
        assert billing._manage_subscription_banner(None, "sleeper") == ""

    with offline_client.session_transaction() as sess:
        sess["viewer_user_id"] = "user1"
    r = offline_client.get("/pricing")
    assert r.status_code == 200
    assert "manage-sub-btn" not in r.get_data(as_text=True)


def test_payment_success_card_has_portal_button(offline_client):
    r = offline_client.get("/pricing?success=1")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'id="sub-portal"' in html
    assert "Manage subscription" in html
    assert "/api/create-portal-session" in html
