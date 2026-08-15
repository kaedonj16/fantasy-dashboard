from types import SimpleNamespace

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


def test_checkout_preserves_espn_provider_in_stripe_metadata(offline_client, monkeypatch):
    captured = {}

    class _CheckoutSession:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    fake_stripe = SimpleNamespace(checkout=SimpleNamespace(Session=_CheckoutSession))
    monkeypatch.setattr(billing, "_stripe", lambda: fake_stripe)
    monkeypatch.setattr(billing, "has_premium_for_viewer", lambda *args, **kwargs: False)

    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "Ryan"
        sess["viewer_user_id"] = "espn-owner-7"

    response = offline_client.post("/api/create-checkout-session", json={
        "plan": "user",
        "league_id": "123",
        "platform": "espn",
        "return_url": "/espn/2026/123/waivers?tab=startsit",
    })

    assert response.status_code == 200
    assert captured["metadata"]["platform"] == "espn"
    assert captured["metadata"]["user_id"] == "espn-owner-7"

