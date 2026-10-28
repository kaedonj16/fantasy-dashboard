"""Regression: yearly Stripe grants were credited with 32 days of PRO.

Root cause (proven against stripe 15.6.2): API 2025-03-31+ (Basil)
subscriptions carry ``current_period_end`` ONLY on ``items.data[]``, and real
v15 resource objects are not dicts: ``hasattr(obj, "get")`` is False and
``item.get(...)`` raises AttributeError ("'get' is a dict method, but a
SubscriptionItem is not a dict"). The old ``_subscription_period_end``
hand-rolled getattr/.get/[] access, hit that AttributeError on the items,
swallowed it, and silently fell back to now + 32 days for every real grant.
Plain-dict and SimpleNamespace fakes masked it, because both answer .get.

These tests therefore drive the billing code with objects that reproduce the
v15 access semantics exactly (plus one opportunistic test on a real
``stripe.Subscription.construct_from`` object when a real stripe is
installed), and with plain dicts for the shape that already worked.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import routes.billing_bp as billing


# ── v15-semantics fake ───────────────────────────────────────────────────────

def _wrap(value):
    if isinstance(value, dict):
        return _V15Object(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _plain(value):
    if isinstance(value, _V15Object):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


class _V15Object:
    """A stripe-python v15 API resource, faithfully in miniature.

    Verified against stripe 15.6.2 ``Subscription.construct_from``:
    attribute access and ``obj[key]`` work and wrap nested dicts; dict
    methods are blocked, so ``hasattr(obj, "get")`` is False and
    ``obj.get(...)`` raises AttributeError; ``to_dict()`` works.
    """

    def __init__(self, data):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return _wrap(data[name])
        # v15 raises AttributeError for missing fields AND for dict methods
        # like "get"; both read as "attribute does not exist" to callers.
        raise AttributeError(name)

    def __getitem__(self, key):
        return _wrap(object.__getattribute__(self, "_data")[key])

    def to_dict(self):
        return {k: _plain(v) for k, v in object.__getattribute__(self, "_data").items()}


PERIOD_END = datetime(2027, 9, 11, 13, 51, tzinfo=timezone.utc)
PERIOD_START = datetime(2026, 9, 11, 13, 51, tzinfo=timezone.utc)


def _basil_payload(**overrides):
    """A Basil-shaped subscription payload: NO top-level current_period_end."""
    payload = {
        "id": "sub_basil",
        "object": "subscription",
        "status": "active",
        "cancel_at_period_end": False,
        "customer": "cus_basil",
        "metadata": {
            "plan": "single_league",
            "user_id": "1395899014741786624",
            "league_id": "1382454060321349632",
            "platform": "sleeper",
            "account_id": "36",
            "interval": "year",
        },
        "items": {
            "object": "list",
            "data": [{
                "id": "si_basil",
                "object": "subscription_item",
                "current_period_start": int(PERIOD_START.timestamp()),
                "current_period_end": int(PERIOD_END.timestamp()),
                "price": {
                    "id": "price_basil_year",
                    "object": "price",
                    "product": billing._STRIPE_SINGLE_LEAGUE_PRODUCT,
                    "recurring": {"interval": "year"},
                },
            }],
        },
    }
    payload.update(overrides)
    return payload


def _basil_object(**overrides):
    return _V15Object(_basil_payload(**overrides))


class _Meta(dict):
    """Stripe-like metadata: mapping plus ``to_dict()`` (as in sibling tests)."""

    def to_dict(self):
        return dict(self)


def _install_webhook_stripe(monkeypatch, event, sub=None):
    class _SigErr(Exception):
        pass

    fake = SimpleNamespace(
        Webhook=SimpleNamespace(construct_event=lambda *a, **k: event),
        SignatureVerificationError=_SigErr,
        Subscription=SimpleNamespace(retrieve=lambda sid: sub),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    return fake


# ── helper-level: the exact production failure ───────────────────────────────

def test_v15_fake_reproduces_v15_access_semantics():
    sub = _basil_object()
    assert hasattr(sub, "get") is False
    item = sub["items"]["data"][0]
    with pytest.raises(AttributeError):
        item.get("current_period_end")


def test_period_end_from_v15_object_items():
    assert billing._subscription_period_end(_basil_object()) == PERIOD_END


def test_period_end_from_plain_dict_items():
    assert billing._subscription_period_end(_basil_payload()) == PERIOD_END


def test_period_end_construct_from_real_stripe_object():
    stripe = pytest.importorskip("stripe")
    raw = stripe.Subscription.construct_from(_basil_payload(), "sk_test_fake")
    if isinstance(raw, dict):
        pytest.skip("installed stripe is a stub, not the real v15 library")
    assert billing._subscription_period_end(raw) == PERIOD_END


def test_period_end_prefers_top_level_when_present():
    sub = _basil_payload(current_period_end=int(PERIOD_END.timestamp()))
    assert billing._subscription_period_end(sub) == PERIOD_END


def test_period_end_unreadable_returns_none_never_a_date():
    assert billing._subscription_period_end({}) is None
    assert billing._subscription_period_end(None) is None
    assert billing._subscription_period_end(_V15Object({"id": "sub_x", "items": {"data": []}})) is None


def test_fallback_is_interval_aware_and_loud(caplog):
    now = datetime.now(timezone.utc)
    with caplog.at_level("ERROR"):
        year = billing._period_end_or_fallback({}, interval="year", sub_id="sub_y", source="test")
        month = billing._period_end_or_fallback({}, interval="month", sub_id="sub_m", source="test")
    assert 360 <= (year - now).days <= 366
    assert 30 <= (month - now).days <= 32
    assert "sub_y" in caplog.text and "sub_m" in caplog.text


# ── webhook grant: yearly single_league lands on the item period end ────────

def test_webhook_checkout_completed_yearly_single_league_gets_full_year(
        offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "create_user_league_subscription",
        lambda user_id, league_id, expires_at, platform="sleeper", **kw: captured.update(
            user_id=user_id, league_id=league_id, expires_at=expires_at, **kw
        ) or True,
    )

    session = SimpleNamespace(
        metadata=_Meta(
            plan="single_league", user_id="1395899014741786624",
            league_id="1382454060321349632", platform="sleeper",
            account_id="36", interval="year",
        ),
        subscription="sub_basil",
        customer="cus_basil",
    )
    event = {"type": "checkout.session.completed", "data": {"object": session}}
    _install_webhook_stripe(monkeypatch, event, sub=_basil_object())

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}", headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["league_id"] == "1382454060321349632"
    # The bug granted now + 32 days here. The full year must land instead.
    assert captured["expires_at"] == PERIOD_END


def test_webhook_checkout_completed_unreadable_period_end_uses_year_fallback(
        offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "create_user_league_subscription",
        lambda user_id, league_id, expires_at, platform="sleeper", **kw: captured.update(
            expires_at=expires_at, **kw
        ) or True,
    )

    session = SimpleNamespace(
        metadata=_Meta(
            plan="single_league", user_id="u1", league_id="lg1",
            platform="sleeper", account_id="9", interval="year",
        ),
        subscription="sub_empty",
        customer="cus_empty",
    )
    event = {"type": "checkout.session.completed", "data": {"object": session}}
    _install_webhook_stripe(monkeypatch, event, sub={})

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}", headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    delta = captured["expires_at"] - datetime.now(timezone.utc)
    assert timedelta(days=360) <= delta <= timedelta(days=367)


# ── lifecycle sync: event object carries the period end on its items ────────

def test_lifecycle_event_object_uses_item_period_end(offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "apply_subscription_lifecycle",
        lambda sub_id, **kw: captured.update(sub_id=sub_id, **kw) or {"updated": []},
    )
    event = {
        "type": "customer.subscription.updated",
        "data": {"object": _basil_object(status="active")},
    }
    _install_webhook_stripe(monkeypatch, event)

    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}", headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 200
    assert captured["sub_id"] == "sub_basil"
    assert captured["expires_at"] == PERIOD_END
    assert captured["interval"] == "year"
