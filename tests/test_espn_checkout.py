from types import SimpleNamespace
import urllib.parse

import pytest

# The lightweight CI job intentionally installs pytest without the Flask stack;
# app integration tests must skip during collection there, just like the other
# offline_client tests. The full-stack CI job installs Flask and runs these.
pytest.importorskip("flask")

import routes.billing_bp as billing


def test_rendered_page_exposes_espn_checkout_context(offline_client):
    from app import render_page

    with offline_client.application.test_request_context("/espn/2026/123/waivers"):
        from flask import session
        session["viewer_username"] = "Ryan"
        response = render_page("Waivers", "123", "waivers", "<p>Start/Sit</p>", "espn", 2026)

    html = response.get_data(as_text=True)
    assert 'window.__brctx = {is_logged_in:true' in html
    assert 'platform:"espn",season:2026,leagueId:"123"' in html


@pytest.mark.parametrize("platform", ["sleeper", "espn", "mfl"])
@pytest.mark.parametrize("plan", ["league", "user", "combo"])
def test_checkout_preserves_provider_at_every_plan_entry(
        offline_client, monkeypatch, platform, plan):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    fake_stripe = SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession))
    monkeypatch.setattr(billing, "_stripe", lambda: fake_stripe)
    monkeypatch.setattr(billing, "has_premium_access", lambda *args, **kwargs: False)

    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "Ryan"
        sess["viewer_user_id"] = f"{platform}-owner-7"

    response = offline_client.post("/api/create-checkout-session", json={
        "plan": plan,
        "league_id": "123",
        "platform": platform,
        "season": 2026,
        # paywall.js intentionally returns a completed purchase to the league
        # dashboard (with the subscriber tour), even when checkout began from
        # Start/Sit.
        "return_url": f"/{platform}/2026/123/dashboard?new_subscriber=1",
    })

    assert response.status_code == 200
    assert captured["metadata"]["platform"] == platform
    assert captured["metadata"]["season"] == "2026"
    assert captured["metadata"]["user_id"] == f"{platform}-owner-7"
    assert f"/{platform}/2026/123/dashboard" in urllib.parse.unquote(captured["success_url"])
    assert captured["cancel_url"].endswith(f"/{platform}/2026/123/pricing?canceled=1")


def test_combo_checkout_rejects_double_billing_when_one_component_exists(
        offline_client, monkeypatch):
    fake_stripe = SimpleNamespace(checkout=SimpleNamespace(Session=SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("Stripe must not be called"))
    )))
    monkeypatch.setattr(billing, "_stripe", lambda: fake_stripe)
    monkeypatch.setattr(
        billing, "has_premium_access",
        lambda user_id, league_id, platform="sleeper", account_id=None: bool(league_id),
    )
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "ryan"
        sess["viewer_user_id"] = "sleeper-7"

    response = offline_client.post("/api/create-checkout-session", json={
        "plan": "combo", "league_id": "123", "platform": "sleeper", "season": 2026,
    })

    assert response.status_code == 400
    assert "already" in response.get_json()["error"].lower()


@pytest.mark.parametrize("plan", ["league", "combo"])
def test_league_plans_cannot_charge_without_a_league(offline_client, monkeypatch, plan):
    monkeypatch.setattr(
        billing, "_stripe",
        lambda: (_ for _ in ()).throw(AssertionError("Stripe must not be called")),
    )
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "ryan"

    response = offline_client.post("/api/create-checkout-session", json={
        "plan": plan, "platform": "sleeper", "season": 2026,
    })

    assert response.status_code == 400
    assert "choose a league" in response.get_json()["error"].lower()


def test_subscription_status_uses_stable_id_and_espn_provider(offline_client, monkeypatch):
    calls = []

    def fake_info(user_id, league_id, platform):
        calls.append((user_id, league_id, platform))
        return {
            "has_premium": True, "subscription_type": "user",
            "has_league_subscription": False, "has_user_subscription": True,
            "expires_at": "2027-08-15T00:00:00+00:00",
            "subscriber_user_id": None, "stripe_customer_id": "cus_private",
        }

    monkeypatch.setattr("dashboard_services.subscriptions.get_subscription_info", fake_info)
    monkeypatch.setattr(billing, "has_premium_for_viewer", lambda *args: True)
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "Ryan"
        sess["viewer_user_id"] = "espn-owner-7"

    response = offline_client.get(
        "/api/subscription-status?platform=espn&league_id=123&season=2026"
    )

    assert response.status_code == 200
    assert response.get_json()["has_premium"] is True
    assert "stripe_customer_id" not in response.get_json()
    assert calls == [("espn-owner-7", "123", "espn")]


def test_checkout_allows_google_account_without_sleeper_viewer(offline_client, monkeypatch):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    monkeypatch.setattr(
        billing, "_stripe",
        lambda: SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession)),
    )
    monkeypatch.setattr(billing, "has_premium_access", lambda *args, **kwargs: False)

    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["account_email"] = "user@example.com"

    response = offline_client.post("/api/create-checkout-session", json={
        "plan": "user", "platform": "espn", "season": 2026,
    })

    assert response.status_code == 200
    assert captured["metadata"]["user_id"] == "acct:42"
    assert captured["metadata"]["account_id"] == "42"
    assert captured["metadata"]["platform"] == "espn"
