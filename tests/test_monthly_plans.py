"""Monthly billing: interval plumbing, pricing toggle, and webhook behavior.

Covers the monthly-plans workstream without touching live Stripe:
- save-% math for the annual callout
- interval normalization + line-item construction (env Price ID vs fallback)
- checkout metadata carries the interval
- webhook grants write billing_interval (Stripe price is source of truth)
- invoice.paid renewals sync billing_interval
- pricing page renders the monthly/annual toggle with both prices
- paywall.js forwards the chosen interval to /api/create-checkout-session
- migration 038 adds billing_interval to the three subscription tables
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing
from dashboard_services import subscriptions


class _Cursor:
    def __init__(self, rows=None):
        self.queries = []
        self._rows = list(rows or [])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self._cursor


class _Meta(dict):
    def to_dict(self):
        return dict(self)


def _stripe_event(etype, obj):
    return {"type": etype, "data": {"object": obj}}


def _install_webhook_stripe(monkeypatch, event, sub=None):
    class _SigErr(Exception):
        pass

    fake = SimpleNamespace(
        Webhook=SimpleNamespace(construct_event=lambda *a, **k: event),
        SignatureVerificationError=_SigErr,
        Subscription=SimpleNamespace(
            retrieve=lambda sid: sub or SimpleNamespace(current_period_end=1_900_000_000),
        ),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    return fake


# ── Pure helpers ──────────────────────────────────────────────────────────────

def test_normalize_interval():
    assert billing._normalize_interval("month") == "month"
    assert billing._normalize_interval("MONTH") == "month"
    assert billing._normalize_interval("year") == "year"
    assert billing._normalize_interval(None) == "year"
    assert billing._normalize_interval("") == "year"
    assert billing._normalize_interval("lifetime") == "year"


def test_annual_savings_pct_matches_spec():
    # All three plans are priced so annual saves 44%:
    # starter $10 vs 12x$1.49=$17.88; all_pro $30 vs $53.88; hall_of_fame $50 vs $89.88.
    assert billing._annual_savings_pct("starter") == 44
    assert billing._annual_savings_pct("all_pro") == 44
    assert billing._annual_savings_pct("hall_of_fame") == 44


def test_monthly_defaults_match_spec_cents():
    assert billing._MONTHLY_DEFAULT_UNIT_AMOUNTS == {
        "starter": 149,
        "all_pro": 449,
        "hall_of_fame": 749,
    }


def test_monthly_price_env_names():
    assert billing._MONTHLY_PRICE_ENV == {
        "starter": "STRIPE_PRICE_STARTER_MONTHLY",
        "all_pro": "STRIPE_PRICE_ALL_PRO_MONTHLY",
        "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_MONTHLY",
    }


def test_line_item_annual_unchanged(monkeypatch):
    monkeypatch.setitem(billing._STRIPE_PRICES["starter"], "product", "prod_starter_x")
    item = billing._checkout_line_item("starter", "year")
    assert item["quantity"] == 1
    assert item["price_data"]["unit_amount"] == 1000
    assert item["price_data"]["recurring"] == {"interval": "year"}
    assert item["price_data"]["product"] == "prod_starter_x"
    assert "price" not in item


def test_line_item_monthly_fallback_uses_default_amount(monkeypatch):
    for plan in billing._MONTHLY_PRICE_ENV:
        monkeypatch.delenv(billing._MONTHLY_PRICE_ENV[plan], raising=False)
    monkeypatch.setitem(billing._STRIPE_PRICES["all_pro"], "product", "prod_all_pro_x")
    item = billing._checkout_line_item("all_pro", "month")
    assert item["quantity"] == 1
    assert item["price_data"]["unit_amount"] == 449
    assert item["price_data"]["recurring"] == {"interval": "month"}
    assert item["price_data"]["product"] == "prod_all_pro_x"
    assert "price" not in item


def test_line_item_monthly_prefers_env_price_id(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_HALL_OF_FAME_MONTHLY", "price_monthly_hof_123")
    item = billing._checkout_line_item("hall_of_fame", "month")
    assert item == {"price": "price_monthly_hof_123", "quantity": 1}


def test_line_item_monthly_hall_of_fame_product_data_fallback(monkeypatch):
    monkeypatch.delenv("STRIPE_PRICE_HALL_OF_FAME_MONTHLY", raising=False)
    monkeypatch.setitem(billing._STRIPE_PRICES["hall_of_fame"], "product", "prod_hof_x")
    item = billing._checkout_line_item("hall_of_fame", "month")
    assert item["price_data"]["unit_amount"] == 749
    assert item["price_data"]["recurring"] == {"interval": "month"}
    assert item["price_data"]["product"] == "prod_hof_x"


def test_checkout_metadata_carries_interval(offline_client):
    import app as app_module

    with app_module.app.test_request_context("/"):
        meta = billing._checkout_metadata("starter", "u1", "lg1", "sleeper", 2026, "month")
        assert meta["interval"] == "month"
        assert meta["plan"] == "starter"
        meta2 = billing._checkout_metadata("hall_of_fame", "u1", "", "sleeper", 2026)
        assert meta2["interval"] == "year"


def test_interval_from_subscription_reads_price(monkeypatch):
    monkeypatch.setattr(billing, "_STRIPE_PRODUCT_STARTER", "prod_starter_test")
    sub = {
        "items": {"data": [{
            "price": {
                "product": "prod_starter_test",
                "recurring": {"interval": "month"},
            },
        }]},
    }
    assert billing._interval_from_subscription(sub) == "month"


def test_interval_from_subscription_blank_when_unknown():
    assert billing._interval_from_subscription(None) == ""
    assert billing._interval_from_subscription({"items": {"data": []}}) == ""
    sub = {"items": {"data": [{"price": {"product": "prod_other"}}]}}
    assert billing._interval_from_subscription(sub) == ""


# ── Subscription storage ──────────────────────────────────────────────────────

def test_create_league_subscription_stores_interval(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.create_league_subscription(
        "lg1", "u1", datetime.now(timezone.utc), billing_interval="month",
    ) is True
    joined = "\n".join(sql for sql, _ in cursor.queries)
    assert "billing_interval" in joined
    insert = [p for sql, p in cursor.queries if "INSERT INTO league_subscriptions" in sql][0]
    assert insert[-1] == "month"


def test_create_user_subscription_stores_interval(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.create_user_subscription(
        "u1", datetime.now(timezone.utc), billing_interval="month",
        plan_key="all_pro",
    ) is True
    insert = [p for sql, p in cursor.queries if "INSERT INTO user_subscriptions" in sql][0]
    assert insert[-2] == "month"
    assert insert[-1] == "all_pro"


def test_create_user_league_subscription_stores_interval(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.create_user_league_subscription(
        "u1", "lg1", datetime.now(timezone.utc), billing_interval="month",
    ) is True
    insert = [p for sql, p in cursor.queries if "INSERT INTO user_league_subscriptions" in sql][0]
    assert insert[-1] == "month"


def test_create_functions_normalize_bad_interval(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.create_user_subscription(
        "u1", datetime.now(timezone.utc), billing_interval="lifetime",
    ) is True
    insert = [p for sql, p in cursor.queries if "INSERT INTO user_subscriptions" in sql][0]
    assert insert[-2] == "year"


def test_get_subscription_info_returns_billing_interval(monkeypatch):
    row = {
        "expires_at": datetime.now(timezone.utc),
        "stripe_customer_id": "cus_1",
        "billing_interval": "month",
        "plan_key": "starter",
    }
    cursor = _Cursor(rows=[dict(row)])
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    info = subscriptions.get_subscription_info("u1", None, "sleeper")
    assert info["has_user_subscription"] is True
    assert info["billing_interval"] == "month"
    assert info["user_plan_key"] == "starter"
    assert info["subscription_type"] == "starter"


def test_get_subscription_info_defaults_interval_year(monkeypatch):
    cursor = _Cursor(rows=[])
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    info = subscriptions.get_subscription_info("u1", None, "sleeper")
    assert info["billing_interval"] == "year"


# ── Webhook ───────────────────────────────────────────────────────────────────

def test_webhook_monthly_grant_writes_interval(offline_client, monkeypatch):
    captured = {}

    def fake_create(user_id, expires_at, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(billing, "create_user_subscription", fake_create)

    session = SimpleNamespace(
        metadata=_Meta(
            plan="starter", user_id="u1", league_id="",
            platform="sleeper", interval="month",
        ),
        subscription="sub_m",
        customer="cus_m",
    )
    _install_webhook_stripe(
        monkeypatch, _stripe_event("checkout.session.completed", session),
    )

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["billing_interval"] == "month"


def test_webhook_interval_prefers_stripe_price_over_metadata(offline_client, monkeypatch):
    captured = {}

    def fake_create(user_id, expires_at, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(billing, "create_user_subscription", fake_create)
    monkeypatch.setattr(billing, "_STRIPE_PRODUCT_HALL_OF_FAME", "prod_hof_test")

    session = SimpleNamespace(
        metadata=_Meta(plan="hall_of_fame", user_id="u1", platform="sleeper", interval="year"),
        subscription="sub_p",
        customer="cus_p",
    )
    sub = {
        "items": {"data": [{
            "price": {
                "product": "prod_hof_test",
                "recurring": {"interval": "month"},
            },
        }]},
        "current_period_end": 1_900_000_000,
    }
    _install_webhook_stripe(
        monkeypatch, _stripe_event("checkout.session.completed", session), sub=sub,
    )

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["billing_interval"] == "month"


def test_invoice_paid_syncs_billing_interval(offline_client, monkeypatch):
    executed = []
    cursor = _Cursor()

    def fake_execute(sql, params=None):
        executed.append((sql, params))

    cursor.execute = fake_execute

    import dashboard_services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(cursor))
    monkeypatch.setattr(billing, "_STRIPE_PRODUCT_HALL_OF_FAME", "prod_hof_test")

    sub = {
        "items": {"data": [{
            "price": {
                "product": "prod_hof_test",
                "recurring": {"interval": "month"},
            },
        }]},
        "current_period_end": 1_900_000_000,
    }

    class _FakeSub:
        subscription = "sub_renew"

    fake = SimpleNamespace(
        Webhook=SimpleNamespace(
            construct_event=lambda *a, **k: _stripe_event("invoice.paid", _FakeSub())
        ),
        SignatureVerificationError=type("SigErr", (Exception,), {}),
        Subscription=SimpleNamespace(retrieve=lambda sid: sub),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    interval_updates = [p for sql, p in executed if "billing_interval" in sql]
    assert len(interval_updates) == 3
    assert all(p[1] == "month" for p in interval_updates)
    assert all(p[2] == "sub_renew" for p in interval_updates)


# ── Pricing page + frontend ───────────────────────────────────────────────────

def test_pricing_page_has_billing_toggle(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    assert 'data-billing-interval="month"' in html
    assert 'data-billing-interval="year"' in html
    assert 'class="pricing-plan-grid" data-billing="year"' in html
    assert "brSetBillingInterval" in html
    assert "__brBillingInterval" in html


def test_pricing_page_shows_monthly_prices_and_savings(offline_client):
    html = offline_client.get("/pricing").get_data(as_text=True)
    for price in ("$1.49<span>/mo</span>", "$4.49<span>/mo</span>",
                  "$7.49<span>/mo</span>"):
        assert price in html
    for annual in ("$10<span>/year</span>", "$30<span>/year</span>",
                   "$50<span>/year</span>"):
        assert annual in html
        assert "Save 44% with annual billing" in html


def test_pricing_page_copy_no_longer_annual_only():
    import pathlib
    billing_src = pathlib.Path("routes/billing_bp.py").read_text(encoding="utf-8")
    assert "One annual charge. No monthly-price shorthand." not in billing_src
    assert "Billed monthly or annually." in billing_src
    body = billing_src.split("def _pricing_body")[1].split("def page_league_pro_invite")[0]
    assert "\u2014" not in body


def test_paywall_js_forwards_interval():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    assert "window.__brBillingInterval === 'month' ? 'month' : 'year'" in js
    assert js.count("interval: billingInterval") == 2


def test_migration_038_adds_billing_interval():
    import pathlib
    sql = pathlib.Path("migrations/038_billing_interval.sql").read_text(encoding="utf-8")
    for table in ("league_subscriptions", "user_subscriptions", "user_league_subscriptions"):
        assert table in sql
        assert "billing_interval" in sql
    assert "ADD COLUMN IF NOT EXISTS" in sql


def test_success_page_shows_billing_line_and_manage_link():
    import pathlib
    src = pathlib.Path("routes/billing_bp.py").read_text(encoding="utf-8")
    assert 'id="sub-billing"' in src
    assert "'PRO ' + interval + renews" in src
    # #1966 owns the manage control on the success page (sub-portal button);
    # monthly plans only adds the interval-aware billing line above it.
    assert 'id="sub-portal"' in src
    assert "/api/create-portal-session" in src


def test_paywall_modal_has_per_card_billing_toggle():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    assert "paywall-interval-toggle" in js
    assert 'data-interval="year"' in js
    assert 'data-interval="month"' in js
    assert "data-price-annual" in js
    assert "data-price-monthly" in js
    assert "setPaywallCardInterval" in js
    assert "_resolveBillingInterval" in js
    assert "save 44%" in js


def test_paywall_modal_resolves_card_interval_before_global():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    # initiatePurchase and _initiatePurchaseWithLeague both resolve per-card
    assert js.count("_resolveBillingInterval(btn, type)") == 2
    # per-plan choice survives the guest Google redirect via sessionStorage
    assert "brBillingInterval:' + planKey" in js or 'brBillingInterval:" + planKey' in js


def test_paywall_modal_uses_logo_and_gold_crown():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    assert 'src="/static/Website_Logo_dark.png"' in js
    assert '<p class="home-pro-eyebrow">BR Fantasy PRO</p>' not in js
    css = pathlib.Path("static/paywall.css").read_text(encoding="utf-8")
    assert ".home-pro-banner-logo" in css
    assert "linear-gradient(180deg, #ffedb0" in css


def test_paywall_modal_removed_pick_after_checkout_copy():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    assert "Pick it after checkout" not in js
    assert "Pick them after checkout" not in js


def test_paywall_modal_shows_annual_savings_incentive():
    import pathlib
    js = pathlib.Path("static/paywall.js").read_text(encoding="utf-8")
    assert "paywall-save-line" in js
    assert "Save ${saveLine} a year vs monthly" in js
    assert "Annual would save you ${saveLine} a year" in js
    css = pathlib.Path("static/paywall.css").read_text(encoding="utf-8")
    assert ".paywall-save-line" in css
