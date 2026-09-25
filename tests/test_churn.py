"""Churn reduction: dunning, cancel save flow, trial reminders, win-back."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("flask")

from utils import churn, churn_email
import routes.billing_bp as billing


# ── Fake DB ───────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows=None):
        self._rows = rows or []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """Scripted DB: map a SQL substring to rows returned by fetchone/fetchall."""

    def __init__(self, script=None):
        self.script = script or {}
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        for key, rows in self.script.items():
            if key in sql:
                return _FakeResult(rows)
        return _FakeResult([])

    def commit(self):
        pass


def _patch_db(monkeypatch, script=None):
    conn = _FakeConn(script)
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = conn
    ctx.__exit__.return_value = False
    monkeypatch.setattr("dashboard_services.db.get_conn", lambda *a, **k: ctx)
    return conn


# ── Email builders ────────────────────────────────────────────────────────────

def test_dunning_touch1_copy():
    p = churn_email.build_dunning_touch(touch=1, first_name="Al", plan="user")
    assert p["subject"] == "Your PRO payment failed"
    assert "—" not in p["subject"] and "—" not in p["html"]
    assert "Update payment method" in p["html"]
    assert "dunning-touch-1" in p["tags"]


def test_dunning_touch2_copy():
    p = churn_email.build_dunning_touch(touch=2, plan="league")
    assert "needs attention" in p["subject"]
    assert "—" not in p["subject"] and "—" not in p["html"]
    assert "dunning-touch-2" in p["tags"]


def test_trial_reminder_copy():
    p2 = churn_email.build_trial_reminder(days_left=2)
    p1 = churn_email.build_trial_reminder(days_left=1)
    assert p2["subject"] == "Your PRO trial ends in 2 days"
    assert p1["subject"] == "Your PRO trial ends tomorrow"
    assert "trial-reminder-2d" in p2["tags"] and "trial-reminder-1d" in p1["tags"]
    assert "—" not in p1["html"] and "—" not in p2["html"]


def test_winback_copy_has_offer_and_token_link():
    p = churn_email.build_winback(
        first_name="Al", offer_label="20% off your first year back",
        checkout_url="https://x/pro/winback?token=abc",
    )
    assert "comeback offer" in p["subject"].lower()
    assert "20% off your first year back" in p["html"]
    assert "token=abc" in p["html"]
    assert "—" not in p["html"]
    assert "winback" in p["tags"]


# ── Dunning state machine ─────────────────────────────────────────────────────

def test_open_dunning_idempotent_same_episode(monkeypatch):
    conn = _patch_db(monkeypatch, {
        "FROM churn_dunning": [{"touch_count": 1, "resolved_at": None,
                                "first_failed_at": None}],
    })
    assert churn.open_dunning("sub_1", 7, "a@b.com", "user") is False
    assert not any("INSERT INTO churn_dunning" in s for s, _ in conn.statements)


def test_open_dunning_inserts_new_episode(monkeypatch):
    conn = _patch_db(monkeypatch)
    assert churn.open_dunning("sub_1", 7, "a@b.com", "user") is True
    assert any("INSERT INTO churn_dunning" in s for s, _ in conn.statements)


def test_bump_dunning_touch_caps_at_two(monkeypatch):
    _patch_db(monkeypatch, {
        "FROM churn_dunning": [{"touch_count": 2}],
    })
    assert churn.bump_dunning_touch("sub_1") is False


def test_bump_dunning_touch_advances(monkeypatch):
    conn = _patch_db(monkeypatch, {
        "FROM churn_dunning": [{"touch_count": 1}],
    })
    assert churn.bump_dunning_touch("sub_1") is True
    assert any("touch_count = touch_count + 1" in s for s, _ in conn.statements)


# ── Webhook: invoice.payment_failed ───────────────────────────────────────────

class _Meta(dict):
    def to_dict(self):
        return dict(self)


def _webhook_app(monkeypatch, event):
    class _SigErr(Exception):
        pass

    sent = []

    fake_sub = SimpleNamespace(
        metadata=_Meta(plan="user", user_id="acct:42", account_id="42"),
        items={"data": []},
    )

    fake = SimpleNamespace(
        Webhook=SimpleNamespace(construct_event=lambda *a, **k: event),
        SignatureVerificationError=_SigErr,
        Subscription=SimpleNamespace(retrieve=lambda sid: fake_sub),
    )
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr(
        "utils.welcome_email._account_email_row",
        lambda aid: {"id": 42, "email": "al@example.com", "first_name": "Al"},
    )
    monkeypatch.setattr(
        churn_email, "send_dunning_touch",
        lambda **kw: sent.append(kw) or True,
    )
    from flask import Flask

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(billing.billing_bp)
    return app, sent


def test_payment_failed_sends_touch1_once(monkeypatch):
    _patch_db(monkeypatch)  # no existing dunning row, no events
    event = {"type": "invoice.payment_failed",
             "data": {"object": {"subscription": "sub_9"}}}
    app, sent = _webhook_app(monkeypatch, event)
    client = app.test_client()
    r = client.post("/api/stripe-webhook", data="{}",
                    headers={"Stripe-Signature": "t"})
    assert r.status_code == 200
    assert len(sent) == 1
    assert sent[0]["touch"] == 1
    assert sent[0]["email"] == "al@example.com"

    # Second identical webhook: episode already open, no resend.
    sent.clear()
    monkeypatch.setattr(churn, "open_dunning", lambda *a, **k: False)
    r = client.post("/api/stripe-webhook", data="{}",
                    headers={"Stripe-Signature": "t"})
    assert r.status_code == 200
    assert sent == []


# ── Pause / cancel endpoints ──────────────────────────────────────────────────

def _authed_client(monkeypatch, app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["account_id"] = "42"
    return client


def _billing_app(monkeypatch):
    from flask import Flask

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(billing.billing_bp)
    return app


def test_pause_subscription_uses_pause_collection(monkeypatch):
    calls = []

    class _Sub:
        @staticmethod
        def modify(sid, **kw):
            calls.append((sid, kw))
            return {"id": sid}

    fake = SimpleNamespace(Subscription=_Sub)
    monkeypatch.setattr(churn, "_stripe", lambda: fake)
    out = churn.pause_subscription("sub_9", months=2)
    assert out["ok"] is True
    sid, kw = calls[0]
    assert sid == "sub_9"
    pc = kw["pause_collection"]
    assert pc["behavior"] == "mark_uncollectible"
    assert pc["resumes_at"] > 0


def test_cancel_endpoint_records_survey_and_cancels(monkeypatch):
    app = _billing_app(monkeypatch)
    events = []
    monkeypatch.setattr(churn, "record_event",
                        lambda aid, kind, detail=None: events.append((kind, detail)))
    monkeypatch.setattr(
        churn, "active_subscriptions_for_user",
        lambda uid: [{"table": "user_subscriptions", "stripe_subscription_id": "sub_9",
                      "plan": "user", "expires_at": None, "league_id": ""}],
    )
    canceled = []
    monkeypatch.setattr(churn, "cancel_at_period_end",
                        lambda sid: canceled.append(sid) or {"ok": True})
    client = _authed_client(monkeypatch, app)
    r = client.post("/api/cancel-subscription", json={
        "stripe_subscription_id": "sub_9",
        "action": "cancel",
        "reason": "too_expensive",
        "reason_detail": "",
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["canceled"] is True
    assert canceled == ["sub_9"]
    kinds = [k for k, _ in events]
    assert "cancel_survey" in kinds and "cancel_confirmed" in kinds
    survey = next(d for k, d in events if k == "cancel_survey")
    assert survey["reason"] == "too_expensive"


def test_cancel_endpoint_pause_action(monkeypatch):
    app = _billing_app(monkeypatch)
    monkeypatch.setattr(churn, "record_event", lambda *a, **k: True)
    monkeypatch.setattr(
        churn, "active_subscriptions_for_user",
        lambda uid: [{"table": "user_subscriptions", "stripe_subscription_id": "sub_9",
                      "plan": "user", "expires_at": None, "league_id": ""}],
    )
    paused = []
    monkeypatch.setattr(churn, "pause_subscription",
                        lambda sid, months=2: paused.append(sid) or {"ok": True,
                                                                    "resumes_at": 123})
    client = _authed_client(monkeypatch, app)
    r = client.post("/api/cancel-subscription", json={
        "stripe_subscription_id": "sub_9",
        "action": "pause",
        "reason": "skipped",
    })
    assert r.status_code == 200
    assert r.get_json()["paused"] is True
    assert paused == ["sub_9"]


def test_cancel_endpoint_rejects_unowned_sub(monkeypatch):
    app = _billing_app(monkeypatch)
    monkeypatch.setattr(churn, "active_subscriptions_for_user", lambda uid: [])
    client = _authed_client(monkeypatch, app)
    r = client.post("/api/cancel-subscription", json={
        "stripe_subscription_id": "sub_evil", "action": "cancel", "reason": "skipped",
    })
    assert r.status_code == 404


# ── Win-back token ────────────────────────────────────────────────────────────

def test_winback_token_roundtrip_and_tamper(monkeypatch):
    monkeypatch.setenv("WINBACK_TOKEN_SECRET", "s3cret")
    tok = churn.make_winback_token(42)
    assert churn.verify_winback_token(tok) == 42
    assert churn.verify_winback_token(tok + "x") is None
    assert churn.verify_winback_token("bogus") is None


def test_winback_endpoint_applies_coupon(monkeypatch):
    app = _billing_app(monkeypatch)
    monkeypatch.setattr(churn, "verify_winback_token", lambda t: 42)
    monkeypatch.setattr(churn, "account_has_active_sub", lambda aid: False)
    monkeypatch.setattr(churn, "winback_coupon_id", lambda: "coupon_20off")
    monkeypatch.setattr(churn, "record_event", lambda *a, **k: True)
    created = []

    class _Session:
        @staticmethod
        def create(**kw):
            created.append(kw)
            return SimpleNamespace(url="https://checkout.stripe.test/sess")

    fake = SimpleNamespace(checkout=SimpleNamespace(Session=_Session))
    monkeypatch.setattr(billing, "_stripe", lambda: fake)
    client = app.test_client()
    r = client.get("/pro/winback?token=tok")
    assert r.status_code == 302
    assert r.headers["Location"] == "https://checkout.stripe.test/sess"
    kw = created[0]
    assert kw["discounts"] == [{"coupon": "coupon_20off"}]
    assert kw["metadata"]["account_id"] == "42"
    assert kw["metadata"]["winback"] == "1"


def test_winback_endpoint_blocks_active_sub(monkeypatch):
    app = _billing_app(monkeypatch)
    monkeypatch.setattr(churn, "verify_winback_token", lambda t: 42)
    monkeypatch.setattr(churn, "account_has_active_sub", lambda aid: True)
    client = app.test_client()
    r = client.get("/pro/winback?token=tok")
    assert r.status_code == 302
    assert "winback=active" in r.headers["Location"]


# ── Trial integration point: defensive no-op ──────────────────────────────────

def test_find_trials_due_no_table(monkeypatch):
    _patch_db(monkeypatch, {
        "information_schema": [],  # pro_trials does not exist
    })
    assert churn.find_trials_due() == []
    assert churn.trial_days_left_for_account(42) is None


# ── Daily scan: dunning escalation ────────────────────────────────────────────

def test_daily_scan_escalates_dunning(monkeypatch):
    _patch_db(monkeypatch)
    monkeypatch.setattr(churn, "dunning_due_for_escalation", lambda: [
        {"stripe_subscription_id": "sub_9", "account_id": 42,
         "email": "al@example.com", "plan": "user", "touch_count": 1},
    ])
    monkeypatch.setattr(churn, "stripe_sub_is_past_due", lambda sid: True)
    monkeypatch.setattr(churn, "bump_dunning_touch", lambda sid: True)
    monkeypatch.setattr(churn, "_account_row",
                        lambda aid, em="": {"id": 42, "email": "al@example.com",
                                            "first_name": "Al"})
    sent = []
    monkeypatch.setattr("utils.churn_email.send_dunning_touch",
                        lambda **kw: sent.append(kw) or True)
    monkeypatch.setattr(churn, "record_event", lambda *a, **k: True)
    monkeypatch.setattr(churn, "find_trials_due", lambda: [])
    monkeypatch.setattr(churn, "find_winback_candidates", lambda: [])
    monkeypatch.setattr(churn, "winback_coupon_id", lambda: "")
    summary = churn.run_daily_scan()
    assert summary["dunning_touch_2"] == 1
    assert sent[0]["touch"] == 2


def test_daily_scan_resolves_recovered_dunning(monkeypatch):
    _patch_db(monkeypatch)
    monkeypatch.setattr(churn, "dunning_due_for_escalation", lambda: [
        {"stripe_subscription_id": "sub_9", "account_id": 42,
         "email": "al@example.com", "plan": "user", "touch_count": 1},
    ])
    monkeypatch.setattr(churn, "stripe_sub_is_past_due", lambda sid: False)
    resolved = []
    monkeypatch.setattr(churn, "resolve_dunning", lambda sid: resolved.append(sid))
    monkeypatch.setattr(churn, "find_trials_due", lambda: [])
    monkeypatch.setattr(churn, "find_winback_candidates", lambda: [])
    monkeypatch.setattr(churn, "winback_coupon_id", lambda: "")
    summary = churn.run_daily_scan()
    assert summary["dunning_resolved"] == 1
    assert resolved == ["sub_9"]


def test_winback_candidate_skips_active_sub(monkeypatch):
    _patch_db(monkeypatch, {
        "FROM user_subscriptions": [{"account_id": 42, "user_id": "acct:42"}],
        "FROM accounts": [{"id": 42, "email": "al@example.com", "first_name": "Al"}],
    })
    monkeypatch.setattr(churn, "account_has_active_sub", lambda aid: True)
    assert churn.find_winback_candidates() == []
