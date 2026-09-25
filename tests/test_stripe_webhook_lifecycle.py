"""Webhook-driven subscription entitlements: lifecycle event sync.

Covers customer.subscription.updated / customer.subscription.deleted handling
in /api/stripe-webhook: signature verification, real-time plan/interval/status
sync, cancel-at-period-end semantics, and upgrade/downgrade reconciliation.
invoice.payment_failed is intentionally NOT covered here: PR #1958 owns
dunning for that event.
"""
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

import routes.billing_bp as billing
from dashboard_services import subscriptions


NOW = datetime.now(timezone.utc)
FUTURE = NOW + timedelta(days=30)
PAST = NOW - timedelta(days=30)


class _FakeCursor:
    """Routes SELECTs to canned rows; records every statement."""

    def __init__(self, existing_tables=(), conflicts=None):
        self.existing_tables = set(existing_tables)
        self.conflicts = conflicts or {}
        self.queries = []
        self._last_table = None
        self._last_kind = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        self._last_kind = None
        self._last_table = None
        if "stripe_subscription_id = %s LIMIT 1" in sql and "SELECT id FROM" in sql:
            m = re.search(r"SELECT id FROM (\w+)", sql)
            self._last_kind = "existing"
            self._last_table = m.group(1) if m else None
        elif "SELECT stripe_subscription_id FROM" in sql:
            m = re.search(r"FROM (\w+)", sql)
            self._last_kind = "conflict"
            self._last_table = m.group(1) if m else None

    def fetchone(self):
        if self._last_kind == "existing":
            if self._last_table in self.existing_tables:
                return {"id": 7}
            return None
        if self._last_kind == "conflict":
            return self.conflicts.get(self._last_table)
        return None


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self._cursor


def _install_conn(monkeypatch, cursor):
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _FakeConn(cursor))
    # _ensure_billing_interval memoizes per process; reset for each test.
    monkeypatch.setattr(subscriptions, "_BILLING_INTERVAL_ENSURED", set())


def _sync(monkeypatch, cursor, **kwargs):
    _install_conn(monkeypatch, cursor)
    params = {
        "event_type": "customer.subscription.updated",
        "status": "active",
        "cancel_at_period_end": False,
        "expires_at": FUTURE,
        "plan": "user",
        "interval": "year",
        "user_id": "u1",
        "league_id": "lg1",
        "platform": "sleeper",
    }
    params.update(kwargs)
    return subscriptions.apply_subscription_lifecycle("sub_1", **params)


def _updates_for(cursor, table):
    return [sql for sql, _ in cursor.queries if sql.startswith("UPDATE " + table)]


# ── unit: apply_subscription_lifecycle ────────────────────────────────────────

def test_updated_active_syncs_status_interval_and_period_end(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(monkeypatch, cursor, interval="month")
    updates = _updates_for(cursor, "user_subscriptions")
    assert len(updates) == 1
    assert "subscription_status = 'active'" in updates[0]
    assert "billing_interval" in updates[0]
    params = [p for sql, p in cursor.queries if sql.startswith("UPDATE user_subscriptions")][0]
    assert params[1] == "month"
    assert params[0] == FUTURE
    assert summary["updated"] == ["user_subscriptions"]
    assert summary["canceled"] == []


def test_updated_past_due_keeps_access_during_retry_window(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(monkeypatch, cursor, status="past_due")
    updates = _updates_for(cursor, "user_subscriptions")
    assert len(updates) == 1
    assert "subscription_status = 'active'" in updates[0]
    assert "'canceled'" not in updates[0]
    assert summary["updated"] == ["user_subscriptions"]


def test_updated_unpaid_revokes(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions", "league_subscriptions"))
    summary = _sync(monkeypatch, cursor, status="unpaid", plan="combo",
                    league_id="lg1")
    for table in ("user_subscriptions", "league_subscriptions"):
        updates = _updates_for(cursor, table)
        assert len(updates) == 1
        assert "'canceled'" in updates[0]
    assert sorted(summary["canceled"]) == ["league_subscriptions", "user_subscriptions"]


def test_updated_cancel_at_period_end_keeps_access_until_period_end(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(
        monkeypatch, cursor, status="active", cancel_at_period_end=True,
    )
    updates = _updates_for(cursor, "user_subscriptions")
    assert len(updates) == 1
    assert "subscription_status = 'active'" in updates[0]
    assert "'canceled'" not in updates[0]
    assert summary["updated"] == ["user_subscriptions"]


def test_updated_plan_upgrade_grants_missing_table(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(monkeypatch, cursor, plan="combo", league_id="lg1")
    inserts = [sql for sql, _ in cursor.queries
               if sql.lstrip().startswith("INSERT INTO league_subscriptions")]
    assert len(inserts) == 1
    assert summary["granted"] == ["league_subscriptions"]
    assert summary["updated"] == ["user_subscriptions"]


def test_updated_plan_upgrade_skips_table_owned_by_another_sub(monkeypatch):
    cursor = _FakeCursor(
        existing_tables=("user_subscriptions",),
        conflicts={"league_subscriptions": {"stripe_subscription_id": "sub_other"}},
    )
    summary = _sync(monkeypatch, cursor, plan="combo", league_id="lg1")
    inserts = [sql for sql, _ in cursor.queries
               if sql.lstrip().startswith("INSERT INTO league_subscriptions")]
    assert inserts == []
    assert summary["granted"] == []


def test_updated_plan_downgrade_revokes_dropped_table(monkeypatch):
    cursor = _FakeCursor(
        existing_tables=("user_subscriptions", "league_subscriptions"),
    )
    summary = _sync(monkeypatch, cursor, plan="user")
    league_updates = _updates_for(cursor, "league_subscriptions")
    assert len(league_updates) == 1
    assert "'canceled'" in league_updates[0]
    user_updates = _updates_for(cursor, "user_subscriptions")
    assert "'canceled'" not in user_updates[0]
    assert summary["canceled"] == ["league_subscriptions"]


def test_updated_unknown_plan_only_syncs_clock(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(monkeypatch, cursor, plan="")
    updates = _updates_for(cursor, "user_subscriptions")
    assert len(updates) == 1
    assert "'canceled'" not in updates[0]
    assert summary["canceled"] == []


def test_deleted_immediate_revokes_all_tables(monkeypatch):
    cursor = _FakeCursor(
        existing_tables=("user_subscriptions", "league_subscriptions"),
    )
    summary = _sync(
        monkeypatch, cursor, event_type="customer.subscription.deleted",
        status="canceled", plan="combo",
    )
    for table in ("user_subscriptions", "league_subscriptions"):
        updates = _updates_for(cursor, table)
        assert len(updates) == 1
        assert "'canceled'" in updates[0]
    assert sorted(summary["canceled"]) == ["league_subscriptions", "user_subscriptions"]


def test_deleted_scheduled_cancel_keeps_access_until_period_end(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(
        monkeypatch, cursor, event_type="customer.subscription.deleted",
        status="canceled", cancel_at_period_end=True, expires_at=FUTURE,
    )
    updates = _updates_for(cursor, "user_subscriptions")
    assert len(updates) == 1
    assert "'canceled'" not in updates[0]
    assert "subscription_status = 'active'" in updates[0]
    assert summary["updated"] == ["user_subscriptions"]


def test_deleted_after_period_end_revokes(monkeypatch):
    cursor = _FakeCursor(existing_tables=("user_subscriptions",))
    summary = _sync(
        monkeypatch, cursor, event_type="customer.subscription.deleted",
        status="canceled", cancel_at_period_end=True, expires_at=PAST,
    )
    updates = _updates_for(cursor, "user_subscriptions")
    assert "'canceled'" in updates[0]
    assert summary["canceled"] == ["user_subscriptions"]


def test_sync_with_unknown_sub_id_touches_nothing(monkeypatch):
    cursor = _FakeCursor(existing_tables=())
    summary = _sync(monkeypatch, cursor)
    writes = [sql for sql, _ in cursor.queries if sql.startswith("UPDATE")]
    assert writes == []
    assert summary["updated"] == [] and summary["canceled"] == []


# ── unit: interval helpers ───────────────────────────────────────────────────

def _sub_with_price(interval):
    return {
        "items": {"data": [{
            "price": {
                "product": billing._STRIPE_USER_PRODUCT,
                "recurring": {"interval": interval},
            },
        }]},
    }


def test_interval_from_subscription_reads_price():
    assert billing._interval_from_subscription(_sub_with_price("month")) == "month"
    assert billing._interval_from_subscription(_sub_with_price("year")) == "year"
    assert billing._normalize_interval("MONTH") == "month"
    assert billing._normalize_interval("bogus") == "year"
    assert billing._normalize_interval("") == "year"


def test_interval_from_subscription_ignores_foreign_products():
    sub = {"items": {"data": [{
        "price": {"product": "prod_FOREIGN", "recurring": {"interval": "month"}},
    }]}}
    assert billing._interval_from_subscription(sub) == ""
    assert billing._interval_from_subscription(None) == ""


# ── endpoint: signature verification ──────────────────────────────────────────

def _install_webhook_stripe(monkeypatch, event=None, exc=None):
    class _SigErr(Exception):
        pass

    def _construct(*a, **k):
        if exc == "sigerr":
            raise _SigErr("bad signature")
        if exc is not None:
            raise exc
        return event

    fake = SimpleNamespace(
        Webhook=SimpleNamespace(construct_event=_construct),
        SignatureVerificationError=_SigErr,
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    return _SigErr


def _post(monkeypatch, offline_client, event=None, exc=None):
    _install_webhook_stripe(monkeypatch, event=event, exc=exc)
    return offline_client.post(
        "/api/stripe-webhook",
        data=b"{}",
        headers={"Stripe-Signature": "t"},
    )


def test_webhook_rejects_bad_signature_with_400(offline_client, monkeypatch):
    response = _post(monkeypatch, offline_client, exc="sigerr")
    assert response.status_code == 400


def test_webhook_rejects_malformed_payload_with_400(offline_client, monkeypatch):
    response = _post(monkeypatch, offline_client, exc=ValueError("bad json"))
    assert response.status_code == 400


def test_webhook_requires_secret_400(offline_client, monkeypatch):
    _install_webhook_stripe(monkeypatch)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    response = offline_client.post(
        "/api/stripe-webhook", data=b"{}",
        headers={"Stripe-Signature": "t"},
    )
    assert response.status_code == 400


# ── endpoint: lifecycle dispatch ──────────────────────────────────────────────

def _lifecycle_event(etype, **overrides):
    sub = {
        "id": "sub_1",
        "status": "active",
        "cancel_at_period_end": False,
        "current_period_end": int(FUTURE.timestamp()),
        "metadata": {
            "plan": "user", "user_id": "u1", "league_id": "lg1",
            "platform": "sleeper", "interval": "year",
        },
        "items": {"data": [{
            "current_period_end": int(FUTURE.timestamp()),
            "price": {
                "product": billing._STRIPE_USER_PRODUCT,
                "recurring": {"interval": "year"},
            },
        }]},
    }
    sub.update(overrides)
    return {"type": etype, "data": {"object": sub}}


def test_webhook_updated_dispatches_lifecycle_sync(offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "apply_subscription_lifecycle",
        lambda sub_id, **kw: captured.update(sub_id=sub_id, **kw) or {"updated": []},
    )
    response = _post(
        monkeypatch, offline_client,
        event=_lifecycle_event("customer.subscription.updated", status="past_due"),
    )
    assert response.status_code == 200
    assert captured["sub_id"] == "sub_1"
    assert captured["event_type"] == "customer.subscription.updated"
    assert captured["status"] == "past_due"
    assert captured["plan"] == "user"
    assert captured["interval"] == "year"
    assert captured["user_id"] == "u1"


def test_webhook_deleted_dispatches_lifecycle_sync(offline_client, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        billing, "apply_subscription_lifecycle",
        lambda sub_id, **kw: captured.update(sub_id=sub_id, **kw) or {"canceled": []},
    )
    response = _post(
        monkeypatch, offline_client,
        event=_lifecycle_event(
            "customer.subscription.deleted",
            status="canceled", cancel_at_period_end=True,
        ),
    )
    assert response.status_code == 200
    assert captured["event_type"] == "customer.subscription.deleted"
    assert captured["cancel_at_period_end"] is True


def test_webhook_lifecycle_never_500s_on_sync_error(offline_client, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(billing, "apply_subscription_lifecycle", _boom)
    response = _post(
        monkeypatch, offline_client,
        event=_lifecycle_event("customer.subscription.updated"),
    )
    assert response.status_code == 200


def test_webhook_unknown_event_type_returns_200(offline_client, monkeypatch):
    response = _post(
        monkeypatch, offline_client,
        event={"type": "customer.tax_id.created", "data": {"object": {}}},
    )
    assert response.status_code == 200


def test_webhook_updated_without_sub_id_skips_gracefully(offline_client, monkeypatch):
    called = []

    def _capture(sub_id, **kw):
        called.append(sub_id)
        return {}

    monkeypatch.setattr(billing, "apply_subscription_lifecycle", _capture)
    event = _lifecycle_event("customer.subscription.updated")
    del event["data"]["object"]["id"]
    response = _post(monkeypatch, offline_client, event=event)
    assert response.status_code == 200
    assert called == []
