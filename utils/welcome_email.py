"""Signup and PRO onboarding emails.

Fired once when a Google account is first created, and once when a PRO plan is
granted. Reuses the weekly digest chrome (with brand logos) and the shared
Brevo/SMTP delivery layer. Opt-out is the ``onboarding`` preference type,
independent of weekly_digest.
"""
from __future__ import annotations

import logging
import os
from html import escape
from typing import Optional

logger = logging.getLogger(__name__)

_SIGNUP_STATE = "signup_welcome_sent:"  # + account_id
_PRO_STATE = "pro_welcome_sent:"  # + account_id

_PLAN_LABELS = {
    "single_league": "One League PRO",
    "user": "Personal PRO",
    "league": "League PRO",
    "combo": "League + Personal PRO",
}


def _base_url() -> str:
    return (os.environ.get("SITE_BASE_URL") or "https://brfantasyfootball.com").rstrip("/")


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
    from utils.email_preferences import ONBOARDING
    from utils.weekly_email import make_unsub_token

    token = make_unsub_token(int(account_id), ONBOARDING)
    if not token:
        return None
    return f"{_base_url()}/email/unsubscribe?token={token}"


def _section(title: str, body_html: str) -> str:
    t = escape(title, quote=False)
    return (
        f'<h3 style="margin:22px 0 10px;font-size:11px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#334155;">{t}</h3>'
        f'<div style="margin:0;font-size:14px;color:#0f172a;line-height:1.55;">{body_html}</div>'
    )


def _lead(body_html: str) -> str:
    """A single lead/closing paragraph. Caller supplies safe inline HTML."""
    return (
        f'<p style="margin:0 0 18px;font-size:15px;color:#0f172a;line-height:1.6;">'
        f"{body_html}</p>"
    )


def _link_label(title: str, href: str = "") -> str:
    label = escape(title, quote=False)
    if href:
        return (
            f'<a href="{escape(href, quote=True)}" style="color:#1d4ed8;'
            f'text-decoration:none;font-weight:700;">{label}</a>'
        )
    return f'<strong style="color:#0f172a;">{label}</strong>'


def _step(num: int, title: str, detail: str, href: str = "") -> str:
    """A numbered step: accent circle + title + one line of detail."""
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td style="width:34px;vertical-align:top;padding:0 12px 16px 0;">'
        f'<div style="width:26px;height:26px;border-radius:50%;background:#2563eb;'
        f'color:#ffffff;font-size:13px;font-weight:800;line-height:26px;text-align:center;">'
        f"{int(num)}</div></td>"
        f'<td style="vertical-align:top;padding:0 0 16px;">'
        f'<div style="font-size:15px;line-height:1.4;">{_link_label(title, href)}</div>'
        f'<div style="margin-top:3px;font-size:13px;color:#475569;line-height:1.5;">'
        f"{escape(detail, quote=False)}</div>"
        f"</td></tr></table>"
    )


def _feature(title: str, detail: str, href: str = "", first: bool = False) -> str:
    """A compact divided list row (no heavy card border)."""
    border = "" if first else "border-top:1px solid #e6ebf2;"
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">'
        f'<tr><td style="padding:12px 0;{border}">'
        f'<div style="font-size:14px;line-height:1.4;">{_link_label(title, href)}</div>'
        f'<div style="margin-top:2px;font-size:13px;color:#475569;line-height:1.5;">'
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
        f'style="margin:0 0 16px;background:#eff4ff;border:1px solid #dbe6fb;'
        f'border-left:3px solid #2563eb;border-radius:10px;">'
        f'<tr><td style="padding:12px 16px;">'
        f'<div style="font-size:12px;font-weight:800;letter-spacing:.08em;'
        f'text-transform:uppercase;color:#1d4ed8;">{eye}</div>'
        f"</td></tr></table>"
    )



def build_signup_welcome(
    *,
    first_name: Optional[str] = None,
    dash_url: str = "",
    unsub_href: str = "{UNSUB}",
) -> dict:
    """Return ``{subject, html, tags}`` for a new-account welcome email."""
    from utils.digest_sections import email_shell, greeting_html

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
                "two minutes, and Google keeps your watchlist and settings synced on "
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
                "Star the players you follow",
                "Your watchlist flags value moves and injuries the next time you "
                "come back.",
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
            "playoff sims, and breakout detection, PRO starts at $5/year "
            f'(<a href="{escape(pricing, quote=True)}" style="color:#1d4ed8;font-weight:700;'
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
    from utils.digest_sections import email_shell, greeting_html

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
            "Send them the invite link from Commissioner / pricing so they can claim access."
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
            "A full AI read on roster construction, trade lanes, and your standings path.",
            dashboard,
        ),
        (
            "Custom Draft Board and Deep Dive",
            "Pin, mute, and reorder into the Draft Room, then replay your picks afterward.",
            draft,
        ),
        (
            "Roster Grades and Playoff Scenarios",
            "Letter grades, competitive window, playoff odds, and late-season magic numbers.",
            teams,
        ),
    ]
    if plan_key in ("user", "combo"):
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
                "PPG, roster age, and a plain-language verdict.",
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


def _should_send(account_id: int, email: str) -> tuple[bool, str]:
    from utils.email_events import is_suppressed
    from utils.email_preferences import ONBOARDING, is_enabled

    if not email or "@" not in email:
        return False, "no_email"
    if is_suppressed(email):
        return False, "suppressed"
    if not is_enabled(int(account_id), ONBOARDING):
        return False, "opted_out"
    return True, "ok"


def _deliver(
    *,
    account_id: int,
    email: str,
    payload: dict,
    email_type: str,
    unsub: str,
    state_key: str,
) -> bool:
    from utils.email_delivery import is_configured, send_email
    from utils.email_events import record_send

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
    ok, reason = _should_send(int(account_id), to)
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
    return _deliver(
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
    ok, reason = _should_send(int(account_id), to)
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
    return _deliver(
        account_id=int(account_id),
        email=to,
        payload=payload,
        email_type="pro_welcome",
        unsub=unsub,
        state_key=state_key,
    )
