"""Stripe Price ID selection for checkout: env Price ID vs ad-hoc fallback.

Covers all six prices (three plans x annual/monthly):
- env var set    -> checkout line item uses {"price": <id>, "quantity": 1}
- env var unset  -> checkout line item uses ad-hoc price_data with the code
                    defaults ($10/$30/$50 annual, $1.49/$4.49/$7.49 monthly)
- whitespace env -> treated as unset (falls back)
Also verifies the checkout session builder passes the chosen line item
through while keeping success/cancel URLs and metadata unchanged.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing


def _clear_price_env(monkeypatch):
    for env in list(billing._ANNUAL_PRICE_ENV.values()) + list(
        billing._MONTHLY_PRICE_ENV.values()
    ):
        monkeypatch.delenv(env, raising=False)


# ── Env var names ─────────────────────────────────────────────────────────────

def test_annual_price_env_names():
    assert billing._ANNUAL_PRICE_ENV == {
        "starter": "STRIPE_PRICE_STARTER_ANNUAL",
        "all_pro": "STRIPE_PRICE_ALL_PRO_ANNUAL",
        "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_ANNUAL",
    }


def test_monthly_price_env_names_match_monthly_plans():
    assert billing._MONTHLY_PRICE_ENV == {
        "starter": "STRIPE_PRICE_STARTER_MONTHLY",
        "all_pro": "STRIPE_PRICE_ALL_PRO_MONTHLY",
        "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_MONTHLY",
    }


def test_price_id_helpers_read_env(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_STARTER_ANNUAL", "price_annual_starter_123")
    monkeypatch.setenv("STRIPE_PRICE_ALL_PRO_MONTHLY", "price_monthly_all_pro_456")
    assert billing._annual_price_id("starter") == "price_annual_starter_123"
    assert billing._monthly_price_id("all_pro") == "price_monthly_all_pro_456"


def test_price_id_helpers_empty_when_unset(monkeypatch):
    _clear_price_env(monkeypatch)
    for plan in billing._ANNUAL_PRICE_ENV:
        assert billing._annual_price_id(plan) == ""
    for plan in billing._MONTHLY_PRICE_ENV:
        assert billing._monthly_price_id(plan) == ""
    assert billing._annual_price_id("nope") == ""
    assert billing._monthly_price_id("nope") == ""


# ── Annual line items ─────────────────────────────────────────────────────────

def test_line_item_annual_prefers_env_price_id(monkeypatch):
    _clear_price_env(monkeypatch)
    monkeypatch.setenv("STRIPE_PRICE_ALL_PRO_ANNUAL", "price_annual_all_pro_abc")
    item = billing._checkout_line_item("all_pro", "year")
    assert item == {"price": "price_annual_all_pro_abc", "quantity": 1}


def test_line_item_annual_env_wins_for_all_plans(monkeypatch):
    _clear_price_env(monkeypatch)
    for plan, env in billing._ANNUAL_PRICE_ENV.items():
        monkeypatch.setenv(env, f"price_annual_{plan}_x")
        item = billing._checkout_line_item(plan, "year")
        assert item == {"price": f"price_annual_{plan}_x", "quantity": 1}
        monkeypatch.delenv(env)


def test_line_item_annual_fallback_uses_code_defaults(monkeypatch):
    _clear_price_env(monkeypatch)
    # Product env vars bind into _STRIPE_PRICES at import time; patch the specs.
    monkeypatch.setitem(billing._STRIPE_PRICES["starter"], "product", "prod_starter_x")
    monkeypatch.setitem(billing._STRIPE_PRICES["all_pro"], "product", "prod_all_pro_x")
    monkeypatch.setitem(billing._STRIPE_PRICES["hall_of_fame"], "product", "prod_hall_of_fame_x")
    expected = {
        "starter": (1000, "prod_starter_x"),
        "all_pro": (3000, "prod_all_pro_x"),
        "hall_of_fame": (5000, "prod_hall_of_fame_x"),
    }
    for plan, (amount, product) in expected.items():
        item = billing._checkout_line_item(plan, "year")
        assert item["quantity"] == 1
        assert "price" not in item
        assert item["price_data"]["currency"] == "usd"
        assert item["price_data"]["unit_amount"] == amount
        assert item["price_data"]["recurring"] == {"interval": "year"}
        assert item["price_data"]["product"] == product


def test_line_item_annual_whitespace_env_falls_back(monkeypatch):
    _clear_price_env(monkeypatch)
    monkeypatch.setenv("STRIPE_PRICE_HALL_OF_FAME_ANNUAL", "   ")
    item = billing._checkout_line_item("hall_of_fame", "year")
    assert "price" not in item
    assert item["price_data"]["unit_amount"] == 5000


# ── Monthly line items ────────────────────────────────────────────────────────

def test_line_item_monthly_prefers_env_price_id(monkeypatch):
    _clear_price_env(monkeypatch)
    monkeypatch.setenv("STRIPE_PRICE_STARTER_MONTHLY", "price_monthly_starter_123")
    item = billing._checkout_line_item("starter", "month")
    assert item == {"price": "price_monthly_starter_123", "quantity": 1}


def test_line_item_monthly_fallback_uses_monthly_defaults(monkeypatch):
    _clear_price_env(monkeypatch)
    # Product env vars bind into _STRIPE_PRICES at import time; patch the specs.
    monkeypatch.setitem(billing._STRIPE_PRICES["starter"], "product", "prod_starter_x")
    monkeypatch.setitem(billing._STRIPE_PRICES["all_pro"], "product", "prod_all_pro_x")
    monkeypatch.setitem(billing._STRIPE_PRICES["hall_of_fame"], "product", "prod_hall_of_fame_x")
    expected = {
        "starter": 149,      # $1.49/mo
        "all_pro": 449,      # $4.49/mo
        "hall_of_fame": 749,  # $7.49/mo
    }
    products = {
        "starter": "prod_starter_x",
        "all_pro": "prod_all_pro_x",
        "hall_of_fame": "prod_hall_of_fame_x",
    }
    for plan, amount in expected.items():
        item = billing._checkout_line_item(plan, "month")
        assert item["quantity"] == 1
        assert "price" not in item
        assert item["price_data"]["currency"] == "usd"
        assert item["price_data"]["unit_amount"] == amount
        assert item["price_data"]["recurring"] == {"interval": "month"}
        assert item["price_data"]["product"] == products[plan]


def test_line_item_interval_defaults_to_year(monkeypatch):
    _clear_price_env(monkeypatch)
    assert billing._checkout_line_item("starter")["price_data"]["recurring"] == {
        "interval": "year"
    }
    assert billing._checkout_line_item("starter", "bogus")["price_data"][
        "recurring"
    ] == {"interval": "year"}
    assert billing._checkout_line_item("starter", "MONTH")["price_data"][
        "recurring"
    ] == {"interval": "month"}


# ── Checkout session builder ──────────────────────────────────────────────────

def _fake_stripe(captured):
    class _Session:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/s/123")

    return SimpleNamespace(checkout=SimpleNamespace(Session=_Session))


def test_checkout_url_uses_env_price_id(monkeypatch):
    from flask import Flask

    _clear_price_env(monkeypatch)
    monkeypatch.setenv("STRIPE_PRICE_STARTER_ANNUAL", "price_annual_starter_123")
    captured = {}
    monkeypatch.setattr(billing, "_stripe", lambda: _fake_stripe(captured))
    tiny = Flask(__name__)
    with tiny.test_request_context("https://example.com/pricing"):
        url, err = billing._stripe_checkout_url(
            "acct:1", {"plan": "starter", "platform": "sleeper", "season": 2026}
        )
    assert err is None
    assert url == "https://checkout.stripe.test/s/123"
    assert captured["line_items"] == [
        {"price": "price_annual_starter_123", "quantity": 1}
    ]
    assert captured["mode"] == "subscription"


def test_checkout_url_fallback_keeps_urls_and_metadata(monkeypatch):
    from flask import Flask

    _clear_price_env(monkeypatch)
    monkeypatch.setitem(billing._STRIPE_PRICES["hall_of_fame"], "product", "prod_hall_of_fame_x")
    captured = {}
    monkeypatch.setattr(billing, "_stripe", lambda: _fake_stripe(captured))
    tiny = Flask(__name__)
    with tiny.test_request_context("https://example.com/pricing"):
        _url, err = billing._stripe_checkout_url(
            "acct:1",
            {
                "plan": "hall_of_fame",
                "platform": "sleeper",
                "season": 2026,
                "league_id": "123",
            },
        )
    assert err is None
    line = captured["line_items"][0]
    assert "price" not in line
    assert line["price_data"]["unit_amount"] == 5000
    assert line["price_data"]["recurring"] == {"interval": "year"}
    assert line["price_data"]["product"] == "prod_hall_of_fame_x"
    assert captured["success_url"].startswith(
        "https://example.com/pricing?success=1&session_id="
    )
    assert "platform=sleeper" in captured["success_url"]
    assert (
        captured["cancel_url"]
        == "https://example.com/sleeper/2026/123/pricing?canceled=1"
    )
    assert captured["metadata"] == {
        "plan": "hall_of_fame",
        "user_id": "acct:1",
        "league_id": "123",
        "platform": "sleeper",
        "season": "2026",
        "interval": "year",
        "account_id": "",
    }
