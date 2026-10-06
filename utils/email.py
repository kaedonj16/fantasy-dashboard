"""Consolidated utils module: email.

transactional email (delivery, events, notifications, preferences, welcome, churn)

Merged from: utils/email_delivery.py, utils/email_events.py, utils/email_notifications.py, utils/email_preferences.py, utils/welcome_email.py, utils/churn_email.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/email_delivery.py
# ======================================================================

"""Provider-independent transactional email sender.

Callers use ``send_email(...)`` and do not care whether Brevo or SMTP delivers
the message. Brevo is the primary production provider; SMTP remains a temporary
fallback when no Brevo API key is configured.

Never log API keys, cookies, or raw provider auth headers.
"""

import json
import logging
import os
import re
import smtplib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
DEFAULT_TIMEOUT_SEC = 15
_MAX_LOG_BODY = 400


@dataclass
class SendResult:
    """Outcome of one send attempt. ``ok`` is True only when the provider accepted."""

    ok: bool
    provider: str = "none"
    message_id: Optional[str] = None
    error: Optional[str] = None
    error_category: Optional[str] = None
    status_code: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def _primary_domain() -> str:
    pd = (os.environ.get("PRIMARY_DOMAIN") or "").strip().lower()
    if pd.startswith("www."):
        pd = pd[4:]
    return pd or "brfantasyfootball.com"


def brevo_config() -> dict[str, str]:
    """Public (non-secret) Brevo sender settings. API key is never returned."""
    domain = _primary_domain()
    sender_email = (
        (os.environ.get("BREVO_SENDER_EMAIL") or "").strip()
        or (os.environ.get("EMAIL_USER") or "").strip()
        or f"noreply@{domain}"
    )
    sender_name = (os.environ.get("BREVO_SENDER_NAME") or "").strip() or "BR Fantasy"
    reply_to = (
        (os.environ.get("BREVO_REPLY_TO_EMAIL") or "").strip()
        or (os.environ.get("CONTACT_EMAIL") or "").strip()
        or (os.environ.get("EMAIL_USER") or "").strip()
        or sender_email
    )
    return {
        "sender_email": sender_email,
        "sender_name": sender_name,
        "reply_to": reply_to,
    }


def _brevo_api_key() -> str:
    return (os.environ.get("BREVO_API_KEY") or "").strip()


def is_brevo_configured() -> bool:
    return bool(_brevo_api_key())


def smtp_config() -> dict[str, Any]:
    return {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "email_user": os.getenv("EMAIL_USER"),
        "email_password": os.getenv("EMAIL_PASSWORD"),
    }


def is_smtp_configured() -> bool:
    cfg = smtp_config()
    return bool(cfg["email_user"] and cfg["email_password"])


def is_configured() -> bool:
    """True when any outbound provider can send user-facing mail."""
    return is_brevo_configured() or is_smtp_configured()


def active_provider() -> str:
    if is_brevo_configured():
        return "brevo"
    if is_smtp_configured():
        return "smtp"
    return "none"


def html_to_text(html: str) -> str:
    """Small HTML→text fallback (no external deps)."""
    text = re.sub(r"<\s*br\s*/?>", "\n", html or "", flags=re.I)
    text = re.sub(r"</\s*(p|div|tr|h[1-6]|li)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_provider_text(text: str) -> str:
    """Strip secrets before logging a provider response."""
    if not text:
        return ""
    out = re.sub(r"xkeysib-[A-Za-z0-9_-]+", "[redacted-key]", text)
    out = re.sub(r"(?i)(api[-_]?key|authorization|password)\s*[:=]\s*\S+", r"\1=[redacted]", out)
    return out[:_MAX_LOG_BODY]


def _category_for_status(status: Optional[int], body: str = "") -> str:
    if status == 429:
        return "rate_limited"
    if status is not None and 400 <= status < 500:
        lowered = (body or "").lower()
        if "invalid" in lowered and "email" in lowered:
            return "invalid_recipient"
        return "provider"
    if status is not None and status >= 500:
        return "provider"
    return "provider"


def send_email(
    to: str,
    subject: str,
    html: str,
    text: Optional[str] = None,
    unsubscribe_url: Optional[str] = None,
    tags: Optional[list] = None,
    *,
    reply_to: Optional[str] = None,
    sender_email: Optional[str] = None,
    sender_name: Optional[str] = None,
    timeout: Optional[float] = None,
) -> SendResult:
    """Send one HTML email. Brevo first; SMTP only when Brevo is not configured."""
    to_email = (to or "").strip()
    if not to_email or "@" not in to_email:
        return SendResult(ok=False, provider=active_provider(), error="invalid recipient",
                          error_category="invalid_recipient")
    if not is_configured():
        logger.info("[email] sender not configured; skipping send")
        return SendResult(ok=False, provider="none", error="not configured",
                          error_category="not_configured")

    if is_brevo_configured():
        return _send_via_brevo(
            to_email, subject, html, text=text, unsubscribe_url=unsubscribe_url,
            tags=tags, reply_to=reply_to, sender_email=sender_email,
            sender_name=sender_name, timeout=timeout,
        )
    return _send_via_smtp(
        to_email, subject, html, text=text, unsubscribe_url=unsubscribe_url,
        reply_to=reply_to,
    )


def _send_via_brevo(
    to_email: str,
    subject: str,
    html: str,
    *,
    text: Optional[str],
    unsubscribe_url: Optional[str],
    tags: Optional[list],
    reply_to: Optional[str],
    sender_email: Optional[str],
    sender_name: Optional[str],
    timeout: Optional[float],
) -> SendResult:
    cfg = brevo_config()
    payload: dict[str, Any] = {
        "sender": {
            "email": sender_email or cfg["sender_email"],
            "name": sender_name or cfg["sender_name"],
        },
        "to": [{"email": to_email}],
        "subject": subject or "",
        "htmlContent": html or "",
        "textContent": text or html_to_text(html or ""),
    }
    rt = (reply_to or cfg["reply_to"] or "").strip()
    if rt:
        payload["replyTo"] = {"email": rt}
    headers: dict[str, str] = {}
    if unsubscribe_url:
        headers["List-Unsubscribe"] = f"<{unsubscribe_url}>"
        headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    if headers:
        payload["headers"] = headers
    clean_tags = []
    for t in tags or []:
        s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(t or "").strip())[:50]
        if s:
            clean_tags.append(s)
    if clean_tags:
        payload["tags"] = clean_tags[:10]

    api_key = _brevo_api_key()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BREVO_API_URL,
        data=body,
        method="POST",
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": api_key,
        },
    )
    wait = float(timeout if timeout is not None else DEFAULT_TIMEOUT_SEC)
    try:
        with urllib.request.urlopen(req, timeout=wait) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = int(getattr(resp, "status", 200) or 200)
            data = _parse_json(raw)
            message_id = _extract_message_id(data)
            logger.info(
                "[email] brevo accepted to=%s status=%s message_id=%s",
                _mask_email(to_email), status, message_id or "-",
            )
            return SendResult(
                ok=True, provider="brevo", message_id=message_id, status_code=status,
            )
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
        finally:
            try:
                exc.close()
            except Exception:
                pass
        status = int(getattr(exc, "code", 0) or 0)
        category = _category_for_status(status, raw)
        logger.warning(
            "[email] brevo rejected to=%s status=%s category=%s body=%s",
            _mask_email(to_email), status, category, _sanitize_provider_text(raw),
        )
        return SendResult(
            ok=False, provider="brevo", error=_sanitize_provider_text(raw) or f"HTTP {status}",
            error_category=category, status_code=status,
        )
    except Exception as exc:
        logger.warning(
            "[email] brevo request failed to=%s err=%s",
            _mask_email(to_email), type(exc).__name__,
        )
        return SendResult(
            ok=False, provider="brevo", error=type(exc).__name__,
            error_category="provider",
        )


def _send_via_smtp(
    to_email: str,
    subject: str,
    html: str,
    *,
    text: Optional[str],
    unsubscribe_url: Optional[str],
    reply_to: Optional[str],
) -> SendResult:
    cfg = smtp_config()
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = cfg["email_user"]
        msg["To"] = to_email
        msg["Subject"] = subject
        if reply_to:
            msg["Reply-To"] = reply_to
        if unsubscribe_url:
            msg["List-Unsubscribe"] = f"<{unsubscribe_url}>"
            msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        msg.attach(MIMEText(text or html_to_text(html), "plain"))
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"], timeout=DEFAULT_TIMEOUT_SEC) as server:
            server.starttls()
            server.login(cfg["email_user"], cfg["email_password"])
            server.send_message(msg)
        logger.info("[email] smtp accepted to=%s", _mask_email(to_email))
        return SendResult(ok=True, provider="smtp")
    except Exception as exc:
        logger.warning(
            "[email] smtp send failed to=%s err=%s",
            _mask_email(to_email), type(exc).__name__,
        )
        return SendResult(
            ok=False, provider="smtp", error=type(exc).__name__,
            error_category="provider",
        )


def _parse_json(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_message_id(data: dict) -> Optional[str]:
    for key in ("messageId", "message_id", "id"):
        val = data.get(key)
        if val:
            return str(val)
    return None


def _mask_email(email: str) -> str:
    s = (email or "").strip()
    if "@" not in s:
        return s[:2] + "…" if s else ""
    local, _, domain = s.partition("@")
    if len(local) <= 2:
        shown = local[:1] + "…"
    else:
        shown = local[:2] + "…"
    return f"{shown}@{domain}"


def retry_after_seconds(result: SendResult, *, default: float = 2.0) -> float:
    """Backoff hint after a rate-limit. Does not sleep."""
    if result.error_category == "rate_limited":
        return default
    return 0.0


def sleep_briefly(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(min(float(seconds), 30.0))


# ======================================================================
# From utils/email_events.py
# ======================================================================

"""Lightweight email delivery observability and bounce suppression.

The weekly send does not depend on webhooks being configured. Events are
recorded at send time; later Brevo callbacks (delivered/opened/clicked/bounce)
update the same row when a webhook secret is set.
"""

import logging
from datetime import datetime, timezone


_EMAIL_EVENTS_SCHEMA_READY = False

HARD_SUPPRESS_EVENTS = frozenset({
    "hardbounce", "hard_bounce", "blocked", "spam", "complaint", "invalid",
    "unsubscribed",
})
SOFT_EVENTS = frozenset({"softbounce", "soft_bounce", "deferred"})


def ensure_email_events_schema(conn=None) -> None:
    global _EMAIL_EVENTS_SCHEMA_READY
    if _EMAIL_EVENTS_SCHEMA_READY and conn is None:
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
        _EMAIL_EVENTS_SCHEMA_READY = True
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
            ensure_email_events_schema(conn)
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
            ensure_email_events_schema(conn)
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
            ensure_email_events_schema(conn)
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
            ensure_email_events_schema(conn)
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
        for row in rows:
            aid = row.get("id") if isinstance(row, dict) else row[0]
            if aid:
                set_enabled(int(aid), False, WEEKLY_DIGEST, conn=conn)
    except Exception:
        logger.debug("[email-events] opt-out by email failed", exc_info=True)


# ======================================================================
# From utils/email_notifications.py
# ======================================================================

"""
Email notification utilities for cron job failures and important events.
"""

import traceback


def get_email_config():
    """Get email configuration from environment variables."""
    return {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'email_user': os.getenv('EMAIL_USER'),
        'email_password': os.getenv('EMAIL_PASSWORD'),
        'recipient_email': os.getenv('RECIPIENT_EMAIL'),
    }


def is_email_configured():
    """Check if email configuration is properly set up."""
    config = get_email_config()
    return all([config['email_user'], config['email_password'], config['recipient_email']])


def is_sender_configured():
    """True when outbound-mail *sender* creds are set (recipient not required).

    Brevo is primary. SMTP remains a fallback when no Brevo API key is set.
    ``is_email_configured`` also requires RECIPIENT_EMAIL, which is only for the
    admin error digest.
    """
    return is_configured()


def send_html_email(to_email: str, subject: str, html_body: str,
                    text_body: str = None, unsubscribe_url: str = None) -> bool:
    """Send one HTML email via the shared delivery layer (Brevo, else SMTP)."""
    return bool(send_email(
        to_email, subject, html_body, text=text_body,
        unsubscribe_url=unsubscribe_url,
    ))


def send_error_email(subject: str, error_message: str, context: dict = None):
    """
    Send an error notification email.
    
    Args:
        subject: Email subject line
        error_message: Main error message
        context: Additional context information (optional)
    """
    if not is_email_configured():
        print("[email] Email not configured, skipping notification")
        return False
    
    config = get_email_config()
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config['email_user']
        msg['To'] = config['recipient_email']
        msg['Subject'] = f"[Fantasy Dashboard Alert] {subject}"
        
        # Build email body
        body_parts = [
            f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Environment: {os.getenv('PYTHON_ENV', 'development')}",
            "",
            f"Error: {error_message}",
            ""
        ]
        
        if context:
            body_parts.append("Context:")
            body_parts.append("-" * 20)
            for key, value in context.items():
                if isinstance(value, dict):
                    body_parts.append(f"{key}:")
                    for k, v in value.items():
                        body_parts.append(f"  {k}: {v}")
                else:
                    body_parts.append(f"{key}: {value}")
            body_parts.append("")
        
        # Add traceback if available
        if traceback.format_exc() != 'NoneType: None\n':
            body_parts.append("Traceback:")
            body_parts.append("-" * 20)
            body_parts.append(traceback.format_exc())
        
        body = "\n".join(body_parts)
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['email_user'], config['email_password'])
            server.send_message(msg)
        
        print(f"[email] Error notification sent to {config['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"[email] Failed to send error notification: {e}")
        return False


def send_cron_failure_notification(error: Exception, context: dict = None):
    """
    Send a notification when cron_daily fails.
    
    Args:
        error: The exception that occurred
        context: Additional context information
    """
    context = context or {}
    context.update({
        'script': 'cron_daily.py',
        'error_type': type(error).__name__,
    })
    
    return send_error_email(
        subject="Cron Daily Job Failed",
        error_message=str(error),
        context=context
    )




# ======================================================================
# From utils/email_preferences.py
# ======================================================================

"""Extensible account notification preferences.

Postgres remains the source of truth for whether someone should receive a given
email type. ``accounts.email_opt_out`` is preserved as a legacy fallback for
``weekly_digest`` so existing unsubscribes keep working.
"""

import logging


WEEKLY_DIGEST = "weekly_digest"
ONBOARDING = "onboarding"
KNOWN_TYPES = (
    WEEKLY_DIGEST,
    ONBOARDING,
    "waiver_report",
    "trade_alerts",
    "player_alerts",
    "product_updates",
)

# Types that send unless the user opts out (default enabled with no preference row).
_OPT_OUT_TYPES = frozenset({WEEKLY_DIGEST, ONBOARDING})

_EMAIL_PREFERENCES_SCHEMA_READY = False


def ensure_email_preferences_schema(conn=None) -> None:
    """Create preference storage. Safe to call repeatedly."""
    global _EMAIL_PREFERENCES_SCHEMA_READY
    if _EMAIL_PREFERENCES_SCHEMA_READY and conn is None:
        return

    def _run(c):
        c.execute(
            "ALTER TABLE accounts ADD COLUMN IF NOT EXISTS email_opt_out BOOLEAN DEFAULT FALSE"
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS account_notification_preferences (
                account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                channel TEXT NOT NULL DEFAULT 'email',
                notification_type TEXT NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (account_id, channel, notification_type)
            )
            """
        )
        c.execute(
            """CREATE INDEX IF NOT EXISTS account_notification_preferences_type_idx
               ON account_notification_preferences (notification_type, enabled)"""
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
        _EMAIL_PREFERENCES_SCHEMA_READY = True
    except Exception:
        logger.debug("[email-prefs] ensure_schema failed", exc_info=True)
        raise


def is_enabled(
    account_id: int,
    notification_type: str = WEEKLY_DIGEST,
    *,
    email_opt_out: Optional[bool] = None,
    conn=None,
) -> bool:
    """True when the account should receive this notification type.

    Preference row wins. If none exists, ``weekly_digest`` falls back to
    ``NOT email_opt_out`` (existing users keep receiving mail until they
    unsubscribe). ``onboarding`` (signup / PRO welcome) also defaults on.
    Unknown future types default to disabled until opted in.
    """
    ntype = (notification_type or WEEKLY_DIGEST).strip().lower()
    row = None
    try:
        if conn is not None:
            ensure_email_preferences_schema(conn)
            row = conn.execute(
                """SELECT enabled FROM account_notification_preferences
                   WHERE account_id = %s AND channel = 'email' AND notification_type = %s""",
                (int(account_id), ntype),
            ).fetchone()
        else:
            from dashboard_services.db import get_conn
            with get_conn() as c:
                ensure_email_preferences_schema(c)
                row = c.execute(
                    """SELECT enabled FROM account_notification_preferences
                       WHERE account_id = %s AND channel = 'email' AND notification_type = %s""",
                    (int(account_id), ntype),
                ).fetchone()
    except Exception:
        logger.debug("[email-prefs] is_enabled query failed", exc_info=True)
        row = None
    if row is not None:
        val = row.get("enabled") if isinstance(row, dict) else row[0]
        return bool(val)
    if ntype == WEEKLY_DIGEST:
        if email_opt_out is None:
            email_opt_out = _legacy_opt_out(account_id)
        return not bool(email_opt_out)
    if ntype in _OPT_OUT_TYPES:
        return True
    return False


def set_enabled(
    account_id: int,
    enabled: bool,
    notification_type: str = WEEKLY_DIGEST,
    *,
    conn=None,
) -> bool:
    """Upsert one preference. Returns True on success."""
    ntype = (notification_type or WEEKLY_DIGEST).strip().lower()
    try:
        if conn is not None:
            ensure_email_preferences_schema(conn)
            conn.execute(
                """
                INSERT INTO account_notification_preferences
                    (account_id, channel, notification_type, enabled, updated_at)
                VALUES (%s, 'email', %s, %s, now())
                ON CONFLICT (account_id, channel, notification_type)
                DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
                """,
                (int(account_id), ntype, bool(enabled)),
            )
            try:
                conn.commit()
            except Exception:
                pass
            return True
        from dashboard_services.db import get_conn
        with get_conn() as c:
            ensure_email_preferences_schema(c)
            c.execute(
                """
                INSERT INTO account_notification_preferences
                    (account_id, channel, notification_type, enabled, updated_at)
                VALUES (%s, 'email', %s, %s, now())
                ON CONFLICT (account_id, channel, notification_type)
                DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
                """,
                (int(account_id), ntype, bool(enabled)),
            )
            try:
                c.commit()
            except Exception:
                pass
        return True
    except Exception as exc:
        logger.warning("[email-prefs] set_enabled failed: %s", exc)
        return False


def unsubscribe_weekly_digest(account_id: int) -> bool:
    """Opt out of weekly digest only. Does not disable future email categories."""
    return set_enabled(int(account_id), False, WEEKLY_DIGEST)


def unsubscribe_onboarding(account_id: int) -> bool:
    """Opt out of signup / PRO welcome (and other onboarding) emails."""
    return set_enabled(int(account_id), False, ONBOARDING)


def unsubscribe_type(account_id: int, notification_type: str) -> bool:
    """Opt out of one notification type."""
    ntype = (notification_type or WEEKLY_DIGEST).strip().lower()
    if ntype == WEEKLY_DIGEST:
        return unsubscribe_weekly_digest(int(account_id))
    if ntype == ONBOARDING:
        return unsubscribe_onboarding(int(account_id))
    return set_enabled(int(account_id), False, ntype)


def _legacy_opt_out(account_id: int) -> bool:
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT email_opt_out FROM accounts WHERE id = %s",
                (int(account_id),),
            ).fetchone()
        if not row:
            return False
        return bool(row.get("email_opt_out") if isinstance(row, dict) else row[0])
    except Exception:
        return False


# ======================================================================
# From utils/welcome_email.py
# ======================================================================

"""Signup and PRO onboarding emails.

Fired once when a Google account is first created, and once when a PRO plan is
granted. Reuses the weekly digest chrome (with brand logos) and the shared
Brevo/SMTP delivery layer. Opt-out is the ``onboarding`` preference type,
independent of weekly_digest.
"""

import logging
from html import escape


_SIGNUP_STATE = "signup_welcome_sent:"  # + account_id
_PRO_STATE = "pro_welcome_sent:"  # + account_id





def brand_asset_url(filename: str) -> str:
    name = (filename or "").lstrip("/")
    return f"{_base_url()}/static/{name}"


def _logo_urls() -> dict[str, str]:
    """Absolute URLs for email-safe brand marks (light header, full-color)."""
    return {
        "logo": brand_asset_url("BR_Logo.png"),
        "mark": brand_asset_url("BR_Mark.png"),
        "site": brand_asset_url("Website_Logo.png"),
        "app": brand_asset_url("app-icon-192.png"),
    }




def _unsub_url(account_id: int) -> Optional[str]:
    from utils.weekly_email import make_unsub_token

    token = make_unsub_token(int(account_id), ONBOARDING)
    if not token:
        return None
    return f"{_base_url()}/email/unsubscribe?token={token}"


def _section(title: str, body_html: str) -> str:
    t = escape(title, quote=False)
    return (
        f'<h3 class="em-h" style="margin:22px 0 10px;font-size:11px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#0f2747;">{t}</h3>'
        f'<div class="em-t" style="margin:0;font-size:15px;color:#122d4b;line-height:1.55;">{body_html}</div>'
    )


def _lead(body_html: str) -> str:
    """A single lead/closing paragraph. Caller supplies safe inline HTML."""
    return (
        f'<p class="em-t" style="margin:0 0 18px;font-size:15px;color:#122d4b;line-height:1.6;">'
        f"{body_html}</p>"
    )


def _link_label(title: str, href: str = "") -> str:
    label = escape(title, quote=False)
    if href:
        return (
            f'<a class="em-cta-a" href="{escape(href, quote=True)}" style="color:#3b82f6;'
            f'text-decoration:none;font-weight:700;">{label}</a>'
        )
    return f'<strong class="em-t" style="color:#122d4b;">{label}</strong>'


def _step(num: int, title: str, detail: str, href: str = "") -> str:
    """A numbered step: accent circle + title + one line of detail."""
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td style="width:34px;vertical-align:top;padding:0 12px 16px 0;">'
        f'<div style="width:26px;height:26px;border-radius:50%;background:#122d4b;'
        f'color:#ffffff;font-size:13px;font-weight:800;line-height:26px;text-align:center;">'
        f"{int(num)}</div></td>"
        f'<td style="vertical-align:top;padding:0 0 16px;">'
        f'<div style="font-size:15px;line-height:1.4;">{_link_label(title, href)}</div>'
        f'<div class="em-t3" style="margin-top:3px;font-size:13px;color:#6b7280;line-height:1.5;">'
        f"{escape(detail, quote=False)}</div>"
        f"</td></tr></table>"
    )


def _feature(title: str, detail: str, href: str = "", first: bool = False) -> str:
    """A compact divided list row (no heavy card border)."""
    border = "" if first else "border-top:1px solid #e8eef4;"
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">'
        f'<tr><td style="padding:12px 0;{border}">'
        f'<div style="font-size:15px;line-height:1.4;">{_link_label(title, href)}</div>'
        f'<div class="em-t3" style="margin-top:2px;font-size:13px;color:#6b7280;line-height:1.5;">'
        f"{escape(detail, quote=False)}</div>"
        f"</td></tr></table>"
    )


def _hero_banner(logos: dict[str, str], eyebrow: str = "") -> str:
    """Light accent strip under the greeting.

    The masthead already carries the wordmark, so this no longer repeats the
    logo. It is just an eyebrow label on a tinted, accent-ruled bar.
    """
    eye = escape(eyebrow, quote=False) if eyebrow else ""
    if not eye:
        return ""
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="margin:0 0 16px;background:#eef4fb;border:1px solid #e8eef4;'
        f'border-left:3px solid #122d4b;border-radius:10px;">'
        f'<tr><td style="padding:12px 16px;">'
        f'<div style="font-size:13px;font-weight:800;letter-spacing:.08em;'
        f'text-transform:uppercase;color:#0f2747;" class="em-h">{eye}</div>'
        f"</td></tr></table>"
    )



def build_signup_welcome(
    *,
    first_name: Optional[str] = None,
    dash_url: str = "",
    unsub_href: str = "{UNSUB}",
) -> dict:
    """Return ``{subject, html, tags}`` for a new-account welcome email."""
    from utils.digest import email_shell, greeting_html

    logos = _logo_urls()
    base = _base_url()
    dash = (dash_url or base).rstrip("/") or base
    pricing = f"{base}/pricing"
    rankings = f"{base}/rankings/dynasty"
    trade = f"{base}/trade"
    trade_values = f"{base}/dynasty-trade-value-chart"

    parts = [
        greeting_html(first_name),
        _hero_banner(logos, eyebrow="Your front office is ready"),
        _lead(
            "Welcome to <strong>BR Fantasy</strong>, the front office for your dynasty, "
            "redraft, and keeper leagues. Connect a league and the whole site builds "
            "itself around your roster."
        ),
        _section(
            "Start here",
            _step(
                1,
                "Connect your league",
                "Sign in with Sleeper, ESPN, Yahoo, MFL, or Fleaflicker. It takes about "
                "two minutes, and Google keeps your leagues and settings synced on "
                "every device.",
                dash,
            )
            + _step(
                2,
                "Open your dashboard",
                "Land in the league you just connected. Activity, waivers, standings, "
                "and start/sit all load around your team.",
                dash,
            )
            + _step(
                3,
                "Set your lineup with Start/Sit",
                "Weekly start scores rank your roster, including K and DST when your "
                "league uses them, so you set the strongest lineup in seconds.",
                dash,
            ),
        ),
        _section(
            "Free tools worth trying first",
            _feature(
                "Trade Calculator",
                "Grade any deal with BR values and format controls, then share a link "
                "with your league.",
                trade,
                first=True,
            )
            + _feature(
                "Player Rankings",
                "Filter by position and format, then open any player for metrics, game "
                "logs, and value history.",
                rankings,
            )
            + _feature(
                "Dynasty Trade Value Chart",
                "A public value chart for quick fairness checks, even before you link "
                "a league.",
                trade_values,
            ),
        ),
        _lead(
            "Every Tuesday we send a personalized digest for your main league: start/sit, "
            "waivers, and value moves. When you want deeper tools like trade suggestions, "
            "playoff sims, and breakout detection, PRO starts at $10/year "
            f'(<a href="{escape(pricing, quote=True)}" class="em-cta-a" style="color:#3b82f6;font-weight:700;'
            'text-decoration:none;">see plans</a>).'
        ),
    ]

    html = email_shell(
        "".join(parts),
        subtitle="Welcome to BR Fantasy",
        dash_url=dash,
        cta_label="Open BR Fantasy →",
        unsub_href=unsub_href,
        logo_url=logos["logo"],
        brand_mark_url="",
        footer_kind="onboarding",
        header_theme="light",
    )
    hi = (first_name or "").strip() or "there"
    return {
        "subject": f"Welcome to BR Fantasy, {hi}",
        "html": html,
        "tags": ["signup-welcome", "onboarding"],
    }


def build_pro_welcome(
    *,
    first_name: Optional[str] = None,
    plan: str = "user",
    platform: str = "",
    season: Optional[int] = None,
    league_id: str = "",
    dash_url: str = "",
    unsub_href: str = "{UNSUB}",
) -> dict:
    """Return ``{subject, html, tags}`` for a new PRO subscription welcome."""
    from utils.digest import email_shell, greeting_html

    logos = _logo_urls()
    base = _base_url()
    plan_key = (plan or "user").strip().lower()
    plan_label = _PLAN_LABELS.get(plan_key, "PRO")
    plat = (platform or "sleeper").strip().lower() or "sleeper"
    season_i = int(season) if season else None
    lid = (league_id or "").strip()

    if dash_url:
        dash = dash_url.rstrip("/")
    elif lid and season_i:
        dash = f"{base}/{plat}/{season_i}/{lid}/dashboard"
    else:
        dash = base

    if lid and season_i:
        root = f"{base}/{plat}/{season_i}/{lid}"
        trade_sugg = f"{root}/trade?tab=suggestions"
        trade_intel = f"{root}/trade?tab=intel"
        breakouts = f"{root}/breakouts"
        draft = f"{root}/draft"
        weekly = f"{root}/weekly"
        teams = f"{root}/teams"
        dashboard = f"{root}/dashboard"
    else:
        trade_sugg = f"{base}/pricing"
        trade_intel = breakouts = draft = weekly = teams = dashboard = dash

    plan_blurb = {
        "starter": (
            "Starter PRO unlocks premium tools for the one league you pick. "
            "You can change your pick anytime from your PRO settings."
        ),
        "all_pro": (
            "All-Pro PRO unlocks premium tools for up to 5 leagues you pick. "
            "You can change your picks anytime from your PRO settings."
        ),
        "hall_of_fame": (
            "Hall of Fame PRO follows you across every league on your Google account, "
            "ideal if you manage multiple teams."
        ),
        # Grandfathered retired plans (no longer sold).
        "single_league": (
            "One League PRO unlocks premium tools for the league you chose at checkout. "
            "Other leagues stay on the free tier unless you upgrade."
        ),
        "user": (
            "Personal PRO follows you across every league on your Google account, "
            "ideal if you manage multiple teams."
        ),
        "league": (
            "League PRO is shared with every manager in the league you purchased for. "
            "Send them the invite link from League Health / pricing so they can claim access."
        ),
        "combo": (
            "League + Personal PRO covers shared access for one league plus Personal PRO "
            "on all of your other teams."
        ),
    }.get(plan_key, "Your PRO plan is active.")

    feats = [
        (
            "Trade Intelligence",
            "Real dynasty trade frequency and market values, one click into the calculator.",
            trade_intel,
        ),
        (
            "Trade Targets",
            "Roster-fit targets from teams that need your surplus, mixed across positions.",
            "",
        ),
        (
            "Breakout Engine",
            "Opportunity and vacated targets ranked with historical comps.",
            breakouts,
        ),
        (
            "Front Office Report",
            "Generate an AI read on roster construction, trade lanes, and your standings path.",
            dashboard,
        ),
        (
            "Custom Draft Board and Deep Dive",
            "Pin, mute, and reorder into the Draft Room, then replay your picks afterward.",
            draft,
        ),
        (
            "Roster Grades and Playoff Outlook",
            "Letter grades and competitive windows on Teams; playoff odds and late-season clinch Outlook on Standings.",
            teams,
        ),
    ]
    if plan_key in ("user", "combo", "all_pro", "hall_of_fame"):
        feats.append(
            (
                "Cross-league This Week's Moves",
                "Lineup and injury actions ranked across every linked league so nothing slips.",
                "",
            )
        )
    toolkit = "".join(
        _feature(t, d, h, first=(i == 0)) for i, (t, d, h) in enumerate(feats)
    )

    parts = [
        greeting_html(first_name),
        _hero_banner(logos, eyebrow=f"{plan_label} unlocked"),
        _lead(
            f"Thanks for going <strong>{escape(plan_label, quote=False)}</strong>. "
            f"{escape(plan_blurb, quote=False)} Here is where to start."
        ),
        _section(
            "Do this first",
            _step(
                1,
                "Open Trade Suggestions",
                "Pick Contending, Rebuilding, Consolidate, or Distribute. Each package "
                "runs a full post-trade playoff sim, so the Win% and playoff-odds shifts "
                "are real.",
                trade_sugg,
            )
            + _step(
                2,
                "Pressure-test a deal with Playoff Impact",
                "Run any trade through the calculator for playoff odds, projected wins and "
                "PPG, plus a plain-language verdict. Dynasty leagues also get Future Outlook "
                "(pick odds, roster age, prime years).",
                dashboard,
            )
            + _step(
                3,
                "Share your Weekly Recap",
                "Generate the AI storyline after your week and drop the share card in "
                "your league chat.",
                weekly,
            ),
        ),
        _section("The rest of your PRO toolkit", toolkit),
        _lead(
            "Your plan renews yearly through Stripe. Manage your payment method or cancel "
            "from Pricing, then Manage billing. You are also on the Tuesday digest for "
            "your main league, which you can opt out of without losing PRO."
        ),
    ]

    html = email_shell(
        "".join(parts),
        subtitle=f"Welcome to {plan_label}",
        dash_url=trade_sugg if "trade" in trade_sugg else dash,
        cta_label="Open Trade Suggestions →",
        unsub_href=unsub_href,
        logo_url=logos["logo"],
        brand_mark_url="",
        footer_kind="onboarding",
        header_theme="light",
    )
    return {
        "subject": f"Your {plan_label} is ready",
        "html": html,
        "tags": ["pro-welcome", "onboarding", f"plan-{plan_key}"],
    }


def _claim_once(key: str, value: str = "1") -> bool:
    """Insert app_state key; True only if this caller won the claim."""
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO app_state (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO NOTHING",
                (key, value),
            )
            conn.commit()
            rc = getattr(cur, "rowcount", None)
            if rc is not None:
                return int(rc) == 1
            # Driver without rowcount: treat a fresh insert as claimed only when
            # the stored value matches what we just wrote (best-effort).
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = %s", (key,)
            ).fetchone()
            val = row.get("value") if isinstance(row, dict) else (row[0] if row else None)
            return val == value
    except Exception:
        logger.debug("[welcome-email] claim_once failed key=%s", key, exc_info=True)
        return False


def _release_claim(key: str) -> None:
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            conn.execute("DELETE FROM app_state WHERE key = %s", (key,))
            conn.commit()
    except Exception:
        logger.debug("[welcome-email] release_claim failed key=%s", key, exc_info=True)


def _account_email_row(account_id: int) -> Optional[dict]:
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                "SELECT id, email, first_name FROM accounts WHERE id = %s",
                (int(account_id),),
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        logger.debug("[welcome-email] account lookup failed", exc_info=True)
        return None


def resolve_account_from_subscriber(
    user_id: str = "",
    account_id: Optional[int] = None,
) -> Optional[dict]:
    """Map Stripe ``user_id`` / ``acct:<id>`` metadata to an accounts row."""
    if account_id:
        return _account_email_row(int(account_id))
    uid = (user_id or "").strip()
    if uid.startswith("acct:"):
        try:
            return _account_email_row(int(uid.split(":", 1)[1]))
        except (TypeError, ValueError):
            return None
    if uid.isdigit():
        # Bare account id sometimes stored in metadata.
        row = _account_email_row(int(uid))
        if row:
            return row
    return None


def _should_send_welcome(account_id: int, email: str) -> tuple[bool, str]:

    if not email or "@" not in email:
        return False, "no_email"
    if is_suppressed(email):
        return False, "suppressed"
    if not is_enabled(int(account_id), ONBOARDING):
        return False, "opted_out"
    return True, "ok"


def _deliver_welcome_email(
    *,
    account_id: int,
    email: str,
    payload: dict,
    email_type: str,
    unsub: str,
    state_key: str,
) -> bool:

    if not is_configured():
        logger.info("[welcome-email] sender not configured; skip type=%s account=%s", email_type, account_id)
        _release_claim(state_key)
        return False

    html = (payload.get("html") or "").replace("{UNSUB}", unsub)
    result = send_email(
        email,
        payload.get("subject") or "BR Fantasy",
        html,
        unsubscribe_url=unsub,
        tags=payload.get("tags") or ["onboarding"],
    )
    if result.ok:
        record_send(
            account_id=int(account_id),
            email=email,
            email_type=email_type,
            provider=result.provider,
            provider_message_id=result.message_id,
            status="sent",
        )
        return True
    logger.warning(
        "[welcome-email] send failed type=%s account=%s provider=%s err=%s",
        email_type, account_id, result.provider, (result.error or "")[:200],
    )
    record_send(
        account_id=int(account_id),
        email=email,
        email_type=email_type,
        provider=result.provider or "none",
        provider_message_id=result.message_id,
        status="failed",
        error_category=result.error_category,
        error_detail=result.error,
    )
    _release_claim(state_key)
    return False


def send_signup_welcome(
    account_id: int,
    *,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    dash_url: str = "",
    force: bool = False,
) -> bool:
    """Send the new-account welcome once. Returns True if accepted by the provider."""
    row = _account_email_row(int(account_id)) if not email else {
        "id": int(account_id), "email": email, "first_name": first_name,
    }
    if not row and email:
        row = {"id": int(account_id), "email": email, "first_name": first_name}
    if not row:
        return False
    to = (row.get("email") or email or "").strip()
    name = first_name if first_name is not None else row.get("first_name")
    ok, reason = _should_send_welcome(int(account_id), to)
    if not ok:
        logger.info("[welcome-email] signup skip account=%s reason=%s", account_id, reason)
        return False

    state_key = f"{_SIGNUP_STATE}{int(account_id)}"
    if not force and not _claim_once(state_key):
        logger.info("[welcome-email] signup already claimed account=%s", account_id)
        return False

    unsub = _unsub_url(int(account_id))
    if not unsub:
        logger.error("[welcome-email] cannot mint onboarding unsub token; skip signup")
        _release_claim(state_key)
        return False

    payload = build_signup_welcome(
        first_name=name, dash_url=dash_url or _base_url(), unsub_href=unsub,
    )
    return _deliver_welcome_email(
        account_id=int(account_id),
        email=to,
        payload=payload,
        email_type="signup_welcome",
        unsub=unsub,
        state_key=state_key,
    )


def send_pro_welcome(
    account_id: int,
    *,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    plan: str = "user",
    platform: str = "",
    season: Optional[int] = None,
    league_id: str = "",
    dash_url: str = "",
    force: bool = False,
) -> bool:
    """Send the PRO welcome once per account (idempotent across webhook + success page)."""
    row = _account_email_row(int(account_id)) if not email else None
    if row is None and email:
        row = {"id": int(account_id), "email": email, "first_name": first_name}
    if not row:
        row = _account_email_row(int(account_id))
    if not row:
        return False
    to = (email or row.get("email") or "").strip()
    name = first_name if first_name is not None else row.get("first_name")
    ok, reason = _should_send_welcome(int(account_id), to)
    if not ok:
        logger.info("[welcome-email] pro skip account=%s reason=%s", account_id, reason)
        return False

    state_key = f"{_PRO_STATE}{int(account_id)}"
    if not force and not _claim_once(state_key):
        logger.info("[welcome-email] pro already claimed account=%s", account_id)
        return False

    unsub = _unsub_url(int(account_id))
    if not unsub:
        logger.error("[welcome-email] cannot mint onboarding unsub token; skip pro")
        _release_claim(state_key)
        return False

    payload = build_pro_welcome(
        first_name=name,
        plan=plan,
        platform=platform,
        season=season,
        league_id=league_id,
        dash_url=dash_url,
        unsub_href=unsub,
    )
    return _deliver_welcome_email(
        account_id=int(account_id),
        email=to,
        payload=payload,
        email_type="pro_welcome",
        unsub=unsub,
        state_key=state_key,
    )


# ======================================================================
# From utils/churn_email.py
# ======================================================================

"""Churn-reduction transactional emails (dunning, trial reminders, win-back).

Follows the ``utils.welcome_email`` pattern: ``build_*`` returns
``{subject, html, tags}`` and ``send_*`` handles suppression, provider send,
and delivery-event recording. These are billing account notices (not
marketing), so they are not gated on notification preferences; hard bounces
are still respected.

Templates (flagged for review in the PR body):
  - dunning_touch_1 / dunning_touch_2: failed renewal payment
  - trial_reminder_2d / trial_reminder_1d: trial expiring (integration point:
    utils.churn.find_trials_due; no-op until the trial workstream lands)
  - winback: one-time comeback offer with a Stripe coupon
"""

import logging


_PLAN_LABELS = {
    # Current catalog (for sale).
    "starter": "Starter PRO",
    "all_pro": "All-Pro PRO",
    "hall_of_fame": "Hall of Fame PRO",
    # Retired from sale but grandfathered for existing subscribers.
    "single_league": "One League PRO",
    "user": "Personal PRO",
    "league": "League PRO",
    "combo": "League + Personal PRO",
    # Current catalog.
    "starter": "Starter PRO",
    "all_pro": "All-Pro PRO",
    "hall_of_fame": "Hall of Fame PRO",
}


def _base_url() -> str:
    import os

    return (os.environ.get("SITE_BASE_URL") or "https://brfantasyfootball.com").rstrip("/")


def _logos() -> dict[str, str]:

    return {"logo": brand_asset_url("BR_Logo.png")}


def _cta_shell(
    *,
    greeting_first_name: Optional[str],
    eyebrow: str,
    subtitle: str,
    lead_html: str,
    body_html: str = "",
    cta_label: str,
    cta_url: str,
    tags: list[str],
) -> dict:
    from utils.digest import email_shell, greeting_html

    logos = _logos()
    parts = [
        greeting_html(greeting_first_name),
        _hero_banner(logos, eyebrow=eyebrow),
        _lead(lead_html),
    ]
    if body_html:
        parts.append(body_html)
    html = email_shell(
        "".join(parts),
        subtitle=subtitle,
        dash_url=cta_url,
        cta_label=cta_label,
        unsub_href="",
        logo_url=logos["logo"],
        brand_mark_url="",
        footer_kind="billing",
        header_theme="light",
    )
    return {"html": html, "tags": tags}


def build_dunning_touch(
    *,
    touch: int = 1,
    first_name: Optional[str] = None,
    plan: str = "",
    billing_url: str = "",
) -> dict:
    """Failed renewal payment: touch 1 (immediate) and touch 2 (day 3)."""
    plan_label = _PLAN_LABELS.get((plan or "").strip().lower(), "PRO")
    base = _base_url()
    url = (billing_url or f"{base}/pricing#pro-billing").strip()
    if touch >= 2:
        subject = "Quick reminder: your PRO payment needs attention"
        eyebrow = "Payment still needs attention"
        lead = (
            f"We still could not charge your card for <strong>{escape(plan_label, quote=False)}</strong>. "
            "Stripe will keep retrying, but updating your payment method now is the fastest way to keep "
            "your premium tools uninterrupted."
        )
        tags = ["dunning-touch-2", "billing"]
    else:
        subject = "Your PRO payment failed"
        eyebrow = "Payment failed"
        lead = (
            f"Your renewal payment for <strong>{escape(plan_label, quote=False)}</strong> did not go through. "
            "Your card may have expired or your bank may have declined the charge. "
            "Stripe will retry automatically, but you can fix it in under a minute."
        )
        tags = ["dunning-touch-1", "billing"]
    payload = _cta_shell(
        greeting_first_name=first_name,
        eyebrow=eyebrow,
        subtitle="PRO payment failed",
        lead_html=lead,
        cta_label="Update payment method",
        cta_url=url,
        tags=tags,
    )
    payload["subject"] = subject
    return payload


def build_trial_reminder(
    *,
    days_left: int = 2,
    first_name: Optional[str] = None,
    plan: str = "",
    pricing_url: str = "",
) -> dict:
    """Trial expiring: 2-day and 1-day reminders."""
    days_left = 1 if int(days_left or 2) <= 1 else 2
    plan_label = _PLAN_LABELS.get((plan or "").strip().lower(), "PRO")
    base = _base_url()
    url = (pricing_url or f"{base}/pricing").strip()
    when = "tomorrow" if days_left == 1 else "in 2 days"
    payload = _cta_shell(
        greeting_first_name=first_name,
        eyebrow="Trial ending",
        subtitle=f"Your PRO trial ends {when}",
        lead_html=(
            f"Your <strong>{escape(plan_label, quote=False)}</strong> trial ends {when}. "
            "Keep Trade Intel, Breakouts, the Front Office Report, and the rest of PRO "
            "without missing a beat."
        ),
        cta_label="Keep PRO",
        cta_url=url,
        tags=[f"trial-reminder-{days_left}d", "billing"],
    )
    payload["subject"] = (
        "Your PRO trial ends tomorrow" if days_left == 1 else "Your PRO trial ends in 2 days"
    )
    return payload


def build_winback(
    *,
    first_name: Optional[str] = None,
    offer_label: str = "",
    checkout_url: str = "",
) -> dict:
    """One-time comeback offer for lapsed PRO users."""
    offer = (offer_label or "").strip() or "20% off your first year back"
    payload = _cta_shell(
        greeting_first_name=first_name,
        eyebrow="A comeback offer",
        subtitle="We saved you a seat",
        lead_html=(
            "It has been a while since you had PRO, and the toolbox has grown: "
            "Breakout Engine, Trade Intel, the Front Office Report, and playoff "
            "simulations on every trade. Come back today and take "
            f"<strong>{escape(offer, quote=False)}</strong>."
        ),
        body_html=(
            '<p class="em-t3" style="margin:0 0 18px;font-size:13px;color:#6b7280;line-height:1.6;">'
            "This is a one-time offer link, just for your account. "
            "If you already resubscribed, ignore this email.</p>"
        ),
        cta_label="Claim the offer",
        cta_url=checkout_url,
        tags=["winback", "billing"],
    )
    payload["subject"] = "A comeback offer for your PRO"
    return payload


def _should_send_churn(email: str) -> tuple[bool, str]:

    if not email or "@" not in email:
        return False, "no_email"
    if is_suppressed(email):
        return False, "suppressed"
    return True, "ok"


def _deliver_churn_email(
    *,
    account_id: Optional[int],
    email: str,
    payload: dict,
    email_type: str,
) -> bool:
    """Send one churn email. Returns True when the provider accepted it."""

    if not is_configured():
        logger.info("[churn-email] sender not configured; skip type=%s", email_type)
        return False

    result = send_email(
        email,
        payload.get("subject") or "BR Fantasy",
        payload.get("html") or "",
        tags=payload.get("tags") or ["billing"],
    )
    record_send(
        account_id=int(account_id) if account_id else None,
        email=email,
        email_type=email_type,
        provider=result.provider or "none",
        provider_message_id=result.message_id,
        status="sent" if result.ok else "failed",
        error_category=result.error_category,
        error_detail=result.error,
    )
    if not result.ok:
        logger.warning(
            "[churn-email] send failed type=%s provider=%s err=%s",
            email_type, result.provider, (result.error or "")[:200],
        )
    return result.ok


def send_dunning_touch(
    *,
    account_id: Optional[int],
    email: str,
    first_name: Optional[str] = None,
    plan: str = "",
    touch: int = 1,
) -> bool:
    ok, reason = _should_send_churn(email)
    if not ok:
        logger.info("[churn-email] dunning skip reason=%s", reason)
        return False
    payload = build_dunning_touch(touch=touch, first_name=first_name, plan=plan)
    return _deliver_churn_email(
        account_id=account_id, email=email, payload=payload,
        email_type=f"dunning_touch_{touch}",
    )


def send_trial_reminder(
    *,
    account_id: Optional[int],
    email: str,
    first_name: Optional[str] = None,
    days_left: int = 2,
    plan: str = "",
) -> bool:
    ok, reason = _should_send_churn(email)
    if not ok:
        logger.info("[churn-email] trial reminder skip reason=%s", reason)
        return False
    payload = build_trial_reminder(days_left=days_left, first_name=first_name, plan=plan)
    return _deliver_churn_email(
        account_id=account_id, email=email, payload=payload,
        email_type=f"trial_reminder_{2 if int(days_left or 2) > 1 else 1}d",
    )


def send_winback(
    *,
    account_id: Optional[int],
    email: str,
    first_name: Optional[str] = None,
    token: str = "",
) -> bool:
    from utils.churn import winback_offer_label

    ok, reason = _should_send_churn(email)
    if not ok:
        logger.info("[churn-email] winback skip reason=%s", reason)
        return False
    base = _base_url()
    url = f"{base}/pro/winback?token={token}" if token else f"{base}/pricing"
    payload = build_winback(first_name=first_name, offer_label=winback_offer_label(),
                            checkout_url=url)
    return _deliver_churn_email(
        account_id=account_id, email=email, payload=payload, email_type="winback",
    )
