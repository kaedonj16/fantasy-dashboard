"""Provider-independent transactional email sender.

Callers use ``send_email(...)`` and do not care whether Brevo or SMTP delivers
the message. Brevo is the primary production provider; SMTP remains a temporary
fallback when no Brevo API key is configured.

Never log API keys, cookies, or raw provider auth headers.
"""
from __future__ import annotations

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
