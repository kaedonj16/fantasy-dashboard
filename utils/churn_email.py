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
from __future__ import annotations

import logging
from html import escape
from typing import Optional

logger = logging.getLogger(__name__)

_PLAN_LABELS = {
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
    from utils.welcome_email import brand_asset_url

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
    from utils.digest_sections import email_shell, greeting_html
    from utils.welcome_email import _hero_banner, _lead

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
            '<p style="margin:0 0 18px;font-size:13px;color:#6b7280;line-height:1.6;">'
            "This is a one-time offer link, just for your account. "
            "If you already resubscribed, ignore this email.</p>"
        ),
        cta_label="Claim the offer",
        cta_url=checkout_url,
        tags=["winback", "billing"],
    )
    payload["subject"] = "A comeback offer for your PRO"
    return payload


def _should_send(email: str) -> tuple[bool, str]:
    from utils.email_events import is_suppressed

    if not email or "@" not in email:
        return False, "no_email"
    if is_suppressed(email):
        return False, "suppressed"
    return True, "ok"


def _deliver(
    *,
    account_id: Optional[int],
    email: str,
    payload: dict,
    email_type: str,
) -> bool:
    """Send one churn email. Returns True when the provider accepted it."""
    from utils.email_delivery import is_configured, send_email
    from utils.email_events import record_send

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
    ok, reason = _should_send(email)
    if not ok:
        logger.info("[churn-email] dunning skip reason=%s", reason)
        return False
    payload = build_dunning_touch(touch=touch, first_name=first_name, plan=plan)
    return _deliver(
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
    ok, reason = _should_send(email)
    if not ok:
        logger.info("[churn-email] trial reminder skip reason=%s", reason)
        return False
    payload = build_trial_reminder(days_left=days_left, first_name=first_name, plan=plan)
    return _deliver(
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

    ok, reason = _should_send(email)
    if not ok:
        logger.info("[churn-email] winback skip reason=%s", reason)
        return False
    base = _base_url()
    url = f"{base}/pro/winback?token={token}" if token else f"{base}/pricing"
    payload = build_winback(first_name=first_name, offer_label=winback_offer_label(),
                            checkout_url=url)
    return _deliver(
        account_id=account_id, email=email, payload=payload, email_type="winback",
    )
