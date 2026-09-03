"""Lightweight email delivery observability and bounce suppression.

The weekly send does not depend on webhooks being configured. Events are
recorded at send time; later Brevo callbacks (delivered/opened/clicked/bounce)
update the same row when a webhook secret is set.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA_READY = False

HARD_SUPPRESS_EVENTS = frozenset({
    "hardbounce", "hard_bounce", "blocked", "spam", "complaint", "invalid",
    "unsubscribed",
})
SOFT_EVENTS = frozenset({"softbounce", "soft_bounce", "deferred"})


def ensure_schema(conn=None) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY and conn is None:
        return

    def _run(c):
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS email_delivery_events (
                id SERIAL PRIMARY KEY,
                account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
                email TEXT,
                email_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                provider_message_id TEXT,
                platform TEXT,
                league_id TEXT,
                season INTEGER,
                iso_week TEXT,
                status TEXT NOT NULL,
                error_category TEXT,
                error_detail TEXT,
                sent_at TIMESTAMPTZ,
                delivered_at TIMESTAMPTZ,
                opened_at TIMESTAMPTZ,
                clicked_at TIMESTAMPTZ,
                bounced_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        c.execute(
            """CREATE INDEX IF NOT EXISTS email_delivery_events_account_week_idx
               ON email_delivery_events (account_id, email_type, iso_week)"""
        )
        c.execute(
            """CREATE INDEX IF NOT EXISTS email_delivery_events_message_id_idx
               ON email_delivery_events (provider_message_id)"""
        )
        c.execute(
            """CREATE INDEX IF NOT EXISTS email_delivery_events_email_idx
               ON email_delivery_events (email)"""
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS email_suppressions (
                email TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                provider TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        try:
            c.commit()
        except Exception:
            pass

    try:
        if conn is not None:
            _run(conn)
        else:
            from dashboard_services.db import get_conn
            with get_conn() as c:
                _run(c)
        _SCHEMA_READY = True
    except Exception:
        logger.debug("[email-events] ensure_schema failed", exc_info=True)
        raise


def record_send(
    *,
    account_id: Optional[int],
    email: str,
    email_type: str,
    provider: str,
    provider_message_id: Optional[str] = None,
    platform: Optional[str] = None,
    league_id: Optional[str] = None,
    season: Optional[int] = None,
    iso_week: Optional[str] = None,
    status: str = "sent",
    error_category: Optional[str] = None,
    error_detail: Optional[str] = None,
) -> None:
    """Insert one delivery row. Never stores message body content."""
    try:
        from dashboard_services.db import get_conn
        now = datetime.now(tz=timezone.utc)
        sent_at = now if status in ("sent", "delivered") else None
        with get_conn() as conn:
            ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO email_delivery_events (
                    account_id, email, email_type, provider, provider_message_id,
                    platform, league_id, season, iso_week, status,
                    error_category, error_detail, sent_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    int(account_id) if account_id is not None else None,
                    (email or "").strip().lower() or None,
                    email_type,
                    provider or "unknown",
                    provider_message_id,
                    platform,
                    league_id,
                    int(season) if season is not None else None,
                    iso_week,
                    status,
                    error_category,
                    (error_detail or "")[:500] or None,
                    sent_at,
                ),
            )
            conn.commit()
    except Exception:
        logger.debug("[email-events] record_send failed", exc_info=True)


def is_suppressed(email: str) -> bool:
    addr = (email or "").strip().lower()
    if not addr:
        return False
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            ensure_schema(conn)
            row = conn.execute(
                "SELECT reason FROM email_suppressions WHERE email = %s",
                (addr,),
            ).fetchone()
        return bool(row)
    except Exception:
        logger.debug("[email-events] is_suppressed failed", exc_info=True)
        return False


def suppress_email(email: str, reason: str, provider: str = "brevo") -> None:
    addr = (email or "").strip().lower()
    if not addr:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO email_suppressions (email, reason, provider, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (email) DO UPDATE
                    SET reason = EXCLUDED.reason,
                        provider = EXCLUDED.provider,
                        updated_at = now()
                """,
                (addr, (reason or "hard_bounce")[:80], provider),
            )
            conn.commit()
    except Exception:
        logger.debug("[email-events] suppress_email failed", exc_info=True)


def _event_name(payload: dict) -> str:
    ev = payload.get("event") or payload.get("event-type") or payload.get("type") or ""
    return str(ev).strip().lower().replace("-", "").replace("_", "")


def apply_webhook_payload(payload: dict, *, provider: str = "brevo") -> dict:
    """Update delivery rows (and suppression) from one Brevo-style event.

    Returns a small result dict. Never raises.
    """
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "invalid"}
    event = _event_name(payload)
    email = str(payload.get("email") or "").strip().lower()
    message_id = (
        payload.get("message-id")
        or payload.get("messageId")
        or payload.get("message_id")
        or payload.get("id")
    )
    message_id = str(message_id).strip() if message_id else None
    now = datetime.now(tz=timezone.utc)

    status_map = {
        "delivered": ("delivered", "delivered_at"),
        "opened": ("opened", "opened_at"),
        "uniqueopened": ("opened", "opened_at"),
        "click": ("clicked", "clicked_at"),
        "clicked": ("clicked", "clicked_at"),
        "hardbounce": ("bounced", "bounced_at"),
        "softbounce": ("bounced", "bounced_at"),
        "blocked": ("bounced", "bounced_at"),
        "spam": ("complained", "bounced_at"),
        "complaint": ("complained", "bounced_at"),
        "invalid": ("bounced", "bounced_at"),
        "unsubscribed": ("unsubscribed", None),
    }
    mapped = status_map.get(event)
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            ensure_schema(conn)
            if mapped:
                new_status, ts_col = mapped
                sets = ["status = %s"]
                params: list[Any] = [new_status]
                if ts_col:
                    sets.append(f"{ts_col} = COALESCE({ts_col}, %s)")
                    params.append(now)
                where = []
                if message_id:
                    where.append("provider_message_id = %s")
                    params.append(message_id)
                if email:
                    where.append("email = %s")
                    params.append(email)
                if where:
                    sql = (
                        "UPDATE email_delivery_events SET "
                        + ", ".join(sets)
                        + " WHERE "
                        + " AND ".join(where)
                    )
                    conn.execute(sql, tuple(params))
            if event in {e.replace("_", "") for e in HARD_SUPPRESS_EVENTS} and email:
                conn.execute(
                    """
                    INSERT INTO email_suppressions (email, reason, provider, updated_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (email) DO UPDATE
                        SET reason = EXCLUDED.reason,
                            provider = EXCLUDED.provider,
                            updated_at = now()
                    """,
                    (email, event[:80], provider),
                )
                if event == "unsubscribed":
                    _opt_out_by_email(conn, email)
            conn.commit()
        return {"ok": True, "event": event, "email": bool(email)}
    except Exception as exc:
        logger.warning("[email-events] webhook apply failed: %s", type(exc).__name__)
        return {"ok": False, "reason": "db"}


def _opt_out_by_email(conn, email: str) -> None:
    """Brevo unsubscribed → disable weekly_digest for matching accounts."""
    try:
        rows = conn.execute(
            "SELECT id FROM accounts WHERE lower(email) = %s",
            (email,),
        ).fetchall() or []
        from utils.email_preferences import set_enabled, WEEKLY_DIGEST
        for row in rows:
            aid = row.get("id") if isinstance(row, dict) else row[0]
            if aid:
                set_enabled(int(aid), False, WEEKLY_DIGEST, conn=conn)
    except Exception:
        logger.debug("[email-events] opt-out by email failed", exc_info=True)
