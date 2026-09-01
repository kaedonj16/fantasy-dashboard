"""Delivery event recording, suppression, and Brevo webhook application."""
from __future__ import annotations

from unittest import mock

from utils.email_events import apply_webhook_payload, HARD_SUPPRESS_EVENTS


class _Conn:
    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        class R:
            def fetchall(self_inner):
                return []
            def fetchone(self_inner):
                return None
        return R()

    def commit(self):
        pass


def test_hard_bounce_suppresses_email():
    conn = _Conn()
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = conn
    ctx.__exit__.return_value = False
    with mock.patch("utils.email_events.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn", return_value=ctx):
        out = apply_webhook_payload({
            "event": "hardBounce",
            "email": "bounce@example.com",
            "message-id": "mid-1",
        })
    assert out["ok"] is True
    joined = " ".join(s[0] for s in conn.statements)
    assert "email_suppressions" in joined
    assert "hardbounce" in HARD_SUPPRESS_EVENTS or "hard_bounce" in HARD_SUPPRESS_EVENTS


def test_soft_bounce_does_not_suppress():
    conn = _Conn()
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = conn
    ctx.__exit__.return_value = False
    with mock.patch("utils.email_events.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn", return_value=ctx):
        apply_webhook_payload({
            "event": "softBounce",
            "email": "temp@example.com",
            "message-id": "mid-2",
        })
    joined = " ".join(s[0] for s in conn.statements)
    assert "email_suppressions" not in joined
    assert "bounced_at" in joined or "UPDATE email_delivery_events" in joined


def test_delivered_updates_timestamp():
    conn = _Conn()
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = conn
    ctx.__exit__.return_value = False
    with mock.patch("utils.email_events.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn", return_value=ctx):
        apply_webhook_payload({
            "event": "delivered",
            "email": "ok@example.com",
            "messageId": "abc-123",
        })
    sql = conn.statements[0][0]
    assert "delivered_at" in sql
    assert "abc-123" in conn.statements[0][1]
