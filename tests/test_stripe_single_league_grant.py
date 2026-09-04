"""One League (single_league) must land in user_league_subscriptions."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing
from dashboard_services import subscriptions


class _Cursor:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchone(self):
        return None


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
    """Stripe-like metadata: mapping plus ``to_dict()``."""

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


def test_metadata_dict_uses_to_dict():
    obj = SimpleNamespace(metadata=_Meta(plan="single_league", league_id="lg1", user_id="u1"))
    assert billing._metadata_dict(obj) == {
        "plan": "single_league", "league_id": "lg1", "user_id": "u1",
    }


def test_subscriber_user_id_prefixes_bare_account():
    assert billing._subscriber_user_id({"user_id": "sleeper-7"}) == "sleeper-7"
    assert billing._subscriber_user_id({"account_id": "42"}) == "acct:42"
    assert billing._subscriber_user_id({"user_id": "", "account_id": "acct:9"}) == "acct:9"


def test_plan_from_subscription_matches_single_league_product():
    sub = {
        "items": {
            "data": [{
                "price": {"product": billing._STRIPE_SINGLE_LEAGUE_PRODUCT},
            }],
        },
    }
    assert billing._plan_from_subscription(sub) == "single_league"


def test_create_user_league_ensures_table(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    ok = subscriptions.create_user_league_subscription(
        "u1", "lg1", datetime.now(timezone.utc), platform="sleeper",
        stripe_subscription_id="sub_1", stripe_customer_id="cus_1",
    )
    assert ok is True
    joined = "\n".join(sql for sql, _ in cursor.queries)
    assert "CREATE TABLE IF NOT EXISTS user_league_subscriptions" in joined
    assert "INSERT INTO user_league_subscriptions" in joined


def test_webhook_grants_single_league_from_session_metadata(offline_client, monkeypatch):
    captured = {}

    def fake_create(user_id, league_id, expires_at, platform="sleeper", **kwargs):
        captured.update(
            user_id=user_id, league_id=league_id, platform=platform, **kwargs
        )
        return True

    monkeypatch.setattr(billing, "create_user_league_subscription", fake_create)
    monkeypatch.setattr(billing, "create_league_subscription", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(billing, "create_user_subscription", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    session = SimpleNamespace(
        metadata=_Meta(
            plan="single_league", user_id="sleeper-7", league_id="123",
            platform="sleeper", account_id="42",
        ),
        subscription="sub_sl",
        customer="cus_sl",
    )
    _install_webhook_stripe(
        monkeypatch, _stripe_event("checkout.session.completed", session),
    )

    response = offline_client.post(
        "/api/stripe-webhook",
        data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["user_id"] == "sleeper-7"
    assert captured["league_id"] == "123"
    assert captured["stripe_subscription_id"] == "sub_sl"
    assert captured["stripe_customer_id"] == "cus_sl"


def test_webhook_infers_single_league_from_product_when_plan_missing(
        offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "create_user_league_subscription",
        lambda user_id, league_id, expires_at, platform="sleeper", **kw: captured.update(
            user_id=user_id, league_id=league_id, **kw
        ) or True,
    )

    session = SimpleNamespace(
        metadata=_Meta(user_id="u1", league_id="lg9", platform="espn"),
        subscription="sub_x",
        customer="cus_x",
    )
    sub = {
        "items": {"data": [{"price": {"product": billing._STRIPE_SINGLE_LEAGUE_PRODUCT}}]},
        "current_period_end": 1_900_000_000,
    }
    _install_webhook_stripe(
        monkeypatch, _stripe_event("checkout.session.completed", session), sub=sub,
    )

    response = offline_client.post(
        "/api/stripe-webhook",
        data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["user_id"] == "u1"
    assert captured["league_id"] == "lg9"


def test_subscription_created_grants_from_copied_metadata(offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "create_user_league_subscription",
        lambda user_id, league_id, expires_at, platform="sleeper", **kw: captured.update(
            user_id=user_id, league_id=league_id, platform=platform, **kw
        ) or True,
    )

    sub = SimpleNamespace(
        id="sub_copied",
        customer="cus_copied",
        current_period_end=1_900_000_000,
        metadata=_Meta(
            plan="single_league", user_id="acct:42", league_id="lg2", platform="espn",
        ),
    )
    _install_webhook_stripe(
        monkeypatch, _stripe_event("customer.subscription.created", sub),
    )

    response = offline_client.post(
        "/api/stripe-webhook",
        data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["user_id"] == "acct:42"
    assert captured["league_id"] == "lg2"
    assert captured["platform"] == "espn"
    assert captured["stripe_subscription_id"] == "sub_copied"


def test_success_page_grants_single_league(offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "create_user_league_subscription",
        lambda user_id, league_id, expires_at, platform="sleeper", **kw: captured.update(
            user_id=user_id, league_id=league_id, **kw
        ) or True,
    )

    cs = SimpleNamespace(
        status="complete",
        subscription="sub_ok",
        customer="cus_ok",
        metadata=_Meta(
            plan="single_league", user_id="u1", league_id="lg1", platform="sleeper",
        ),
    )
    fake_stripe = SimpleNamespace(
        checkout=SimpleNamespace(Session=SimpleNamespace(retrieve=lambda sid: cs)),
        Subscription=SimpleNamespace(
            retrieve=lambda sid: SimpleNamespace(current_period_end=1_900_000_000),
        ),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake_stripe)

    response = offline_client.get("/pricing?success=1&session_id=cs_test")
    assert response.status_code == 200
    assert captured["user_id"] == "u1"
    assert captured["league_id"] == "lg1"
    assert captured["stripe_subscription_id"] == "sub_ok"
