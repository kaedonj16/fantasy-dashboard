"""Signup and PRO onboarding emails.

Fired once when a Google account is first created, and once when a PRO plan is
granted. Reuses the weekly digest chrome (with brand logos) and the shared
Brevo/SMTP delivery layer. Opt-out is the ``onboarding`` preference type —
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
    """Absolute URLs for email-safe brand + platform marks (dark header / light body)."""
    return {
        "logo": brand_asset_url("BR_Logo_dark.png"),
        "mark": brand_asset_url("BR_Mark_dark.png"),
        "sleeper": brand_asset_url("sleeper-logo.png"),
        "espn": brand_asset_url("espn-logo.png"),
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
        f'<h3 style="margin:22px 0 8px;font-size:11px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#334155;">{t}</h3>'
        f'<div style="margin:0;font-size:14px;color:#0f172a;line-height:1.55;">{body_html}</div>'
    )


def _bullet(title: str, detail: str, href: str = "") -> str:
    label = escape(title, quote=False)
    body = escape(detail, quote=False)
    if href:
        label = (
            f'<a href="{escape(href, quote=True)}" style="color:#1d4ed8;text-decoration:none;'
            f'font-weight:700;">{label}</a>'
        )
    else:
        label = f'<strong style="color:#0f172a;">{label}</strong>'
    return (
        f'<div style="margin:0 0 12px;padding:12px 14px;background:#ffffff;border:1px solid #e6ebf2;'
        f'border-radius:10px;">'
        f'<div style="font-size:14px;line-height:1.35;">{label}</div>'
        f'<div style="margin-top:4px;font-size:13px;color:#475569;line-height:1.45;">{body}</div>'
        f"</div>"
    )


def _platform_row(logos: dict[str, str]) -> str:
    cells = []
    for key, label in (("sleeper", "Sleeper"), ("espn", "ESPN")):
        src = logos.get(key) or ""
        if not src:
            continue
        cells.append(
            f'<td style="padding:0 10px 0 0;vertical-align:middle;">'
            f'<img src="{escape(src, quote=True)}" alt="{escape(label, quote=True)}" '
            f'width="28" height="28" style="display:block;border:0;border-radius:6px;'
            f'width:28px;height:28px;object-fit:contain;" />'
            f"</td>"
        )
    text_plats = "Yahoo · MFL · Fleaflicker"
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" style="margin:8px 0 0;">'
        f"<tr>{''.join(cells)}"
        f'<td style="vertical-align:middle;font-size:12px;color:#64748b;font-weight:600;">'
        f"{escape(text_plats, quote=False)}</td></tr></table>"
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
    rankings = f"{base}/rankings"
    trade_values = f"{base}/trade-values"
    compare = f"{base}/compare"
    prospects = f"{base}/prospects"

    parts = [
        greeting_html(first_name),
        (
            '<p style="margin:0 0 8px;font-size:15px;color:#0f172a;line-height:1.55;">'
            "Welcome to <strong>BR Fantasy</strong> — the front office managers use to "
            "run dynasty, redraft, and keeper leagues. Below is how people actually use "
            "the site week to week, plus the free features to open first.</p>"
        ),
        _section(
            "1. Connect your leagues (2 minutes)",
            "From the home page, sign in with Sleeper username, ESPN league ID, Yahoo "
            "OAuth, MFL, or Fleaflicker. Google keeps your watchlist, digests, and "
            "settings synced on every device. Multi-league managers live in "
            "<strong>My Leagues</strong> and jump with the league switcher."
            + _platform_row(logos)
            + _bullet(
                "Open your dashboard",
                "Land in the league you just connected — activity, waivers, and standings "
                "context load with your roster.",
                dash,
            ),
        ),
        _section(
            "2. Ways managers use BR Fantasy",
            ""
            + _bullet(
                "Morning scan (in season)",
                "Open the dashboard for Since Last Visit, then Waivers & Start/Sit and "
                "Matchups. Star anyone you're tracking so value and injury alerts stick.",
            )
            + _bullet(
                "Trade desk anytime",
                "Build a deal in the Trade Calculator, share the link in chat, and check "
                "counter-suggestions before you send. Free on every connected league.",
            )
            + _bullet(
                "Offseason / draft prep",
                "Use Rankings, Prospects, Draft Room mocks, and the Cheat Sheet (CSV + "
                "live Sleeper sync). Keeper leagues get Keeper Assistant with auto costs.",
            )
            + _bullet(
                "League history & bragging rights",
                "Standings, Awards, Graphs, and History turn multi-year leagues into a "
                "story — share cards when you want the group chat to notice.",
            )
            + _bullet(
                "Weekly email digest",
                "We email a personalized start/sit, waiver, and value recap for your "
                "primary league every Tuesday. Opt out from that footer anytime — it "
                "doesn't turn off this welcome mail.",
            ),
        ),
        _section(
            "3. Free features to try today",
            ""
            + _bullet(
                "Trade Calculator",
                "Grade both sides with BR values, format controls (teams, PPR, Superflex), "
                "AI analysis, and shareable trade links.",
                dash,
            )
            + _bullet(
                "Player Rankings & search",
                "Filter by position and format, sort by value/age/PPG, and open any player "
                "modal for metrics, game logs, value history, and trade comps.",
                rankings,
            )
            + _bullet(
                "Dynasty Trade Value Chart",
                "Public value chart you can use even before a league is linked — great for "
                "quick fairness checks.",
                trade_values,
            )
            + _bullet(
                "Player Compare",
                "Side-by-side stats and metrics when you're stuck between two names.",
                compare,
            )
            + _bullet(
                "Prospect Rankings",
                "Rookie production, athleticism, draft capital, and comps for the active class.",
                prospects,
            )
            + _bullet(
                "Watchlist",
                "Star players once; get value-move and injury flags when you return "
                "(synced when signed in).",
            )
            + _bullet(
                "Draft Room & Cheat Sheet",
                "Mock any format, connect a live Sleeper/ESPN draft, print/export a cheat "
                "sheet, and review draft history after the fact.",
            )
            + _bullet(
                "Waivers, Start/Sit & Schedule Assistant",
                "Ranked free-agent targets, weekly start scores (including K/DST when your "
                "league uses them), and matchup difficulty across a week range.",
            )
            + _bullet(
                "Matchups hub & Redzone",
                "Optimal lineup, scout report, power rankings, SOS, streaming options, "
                "and a live red-zone tracker during games.",
            )
            + _bullet(
                "Teams, Standings & Activity",
                "Deep team tabs, standings, and a transaction feed plus NFL headlines — "
                "your league's command center.",
            ),
        ),
        _section(
            "4. When you're ready for PRO",
            "PRO adds Trade Suggestions with playoff-odds impact, Trade Targets & Intel, "
            "Playoff Impact sims, Breakout Engine, Front Office Report, Weekly Recap "
            "storylines, Custom Draft Board, Draft Deep Dive, Roster Intel, and "
            "cross-league This Week's Moves. Plans: One League $5/yr, Personal $10/yr, "
            "League $15/yr, League + Personal $20/yr — see "
            f'<a href="{escape(pricing, quote=True)}" style="color:#1d4ed8;font-weight:700;'
            'text-decoration:none;">Pricing</a>.'
        ),
        (
            '<p style="margin:20px 0 0;font-size:13px;color:#64748b;line-height:1.5;">'
            "Tip: after you connect a league, replay the short site tour from Settings "
            "anytime. Questions? Reply to this email or open Support in the footer.</p>"
        ),
    ]

    html = email_shell(
        "".join(parts),
        subtitle="Welcome to BR Fantasy",
        dash_url=dash,
        cta_label="Open BR Fantasy →",
        unsub_href=unsub_href,
        logo_url=logos["logo"],
        brand_mark_url=logos["mark"],
        footer_kind="onboarding",
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
            "Personal PRO follows you across every league on your Google account — "
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

    parts = [
        greeting_html(first_name),
        (
            '<p style="margin:0 0 4px;font-size:12px;font-weight:800;letter-spacing:.08em;'
            'text-transform:uppercase;color:#1d4ed8;">PRO unlocked</p>'
            f'<p style="margin:0 0 12px;font-size:15px;color:#0f172a;line-height:1.55;">'
            f"Thanks for going <strong>{escape(plan_label, quote=False)}</strong>. "
            f"{escape(plan_blurb, quote=False)} Here's how to put it to work — not just "
            "a feature list, but when to open each tool.</p>"
        ),
        _section(
            "How to use PRO this week",
            ""
            + _bullet(
                "1. Open Trade Suggestions first",
                "Pick Contending, Rebuilding, Consolidate, or Distribute. Each package "
                "runs a full post-trade playoff sim so Win% and playoff-odds shifts are real.",
                trade_sugg,
            )
            + _bullet(
                "2. Pressure-test a deal with Playoff Impact",
                "Before you accept anything, run Playoff Impact on the calculator: playoff "
                "odds, projected wins/PPG, top-3 pick odds, roster age, and a plain-language "
                "verdict (Win-Now, Building, Balanced…).",
            )
            + _bullet(
                "3. Fill holes with Trade Targets",
                "Roster-fit targets from teams that need your surplus — mixed across "
                "positions, not just the top four names at a weak spot.",
            )
            + _bullet(
                "4. Scan Breakouts & Roster Intel",
                "Breakout Engine ranks opportunity + vacated targets with historical comps. "
                "Teams → Roster Intel tags Core / Sell High / Buy Window / Breakout Hold.",
                breakouts,
            )
            + _bullet(
                "5. Share the Weekly Recap",
                "Generate the AI storyline after your week and drop the share card in the "
                "league chat — PRO content that makes you look like the commissioner.",
                weekly,
            )
            + _bullet(
                "Replay the in-app PRO tour",
                "Short welcome overlay after checkout; reopen anytime from Settings → "
                "PRO welcome.",
            ),
        ),
        _section(
            "Full PRO toolkit (and when to use it)",
            ""
            + _bullet(
                "Trade Suggestions",
                "Use when you want packages built for your archetype with playoff-odds impact baked in.",
                trade_sugg,
            )
            + _bullet(
                "Trade Intelligence",
                "Use when you need real dynasty trade frequency, market values, and "
                "one-click load into the calculator.",
                trade_intel,
            )
            + _bullet(
                "Playoff Impact",
                "Use on every non-trivial trade — Monte Carlo on playoff odds, wins, PPG, "
                "draft capital, age, and prime years left.",
            )
            + _bullet(
                "Breakout Engine",
                "Use in offseason and early season to find opportunity before the wire heats up.",
                breakouts,
            )
            + _bullet(
                "Front Office Report",
                "Use for a full AI read on roster construction, trade lanes, and standings path "
                "(in-season hub + offseason generate).",
                dashboard,
            )
            + _bullet(
                "Weekly Recap",
                "Use after each week for an AI storyline plus shareable OG image.",
                weekly,
            )
            + _bullet(
                "Custom Draft Board & Deep Dive",
                "Use before and during drafts: pin/mute/reorder (follows you into Draft Room); "
                "Deep Dive replays Decision Score vs the remaining pool.",
                draft,
            )
            + _bullet(
                "Roster Grades, Archetypes & Playoff Scenarios",
                "Use under Teams for letter grades, competitive window, playoff odds, and "
                "clinch/elimination magic numbers late season.",
                teams,
            )
            + _bullet(
                "Cross-league This Week's Moves (Personal / Combo)",
                "Use My Leagues when you run multiple teams — lineup and injury actions "
                "ranked across every linked league so nothing slips.",
            ),
        ),
        _section(
            "Still free (and worth combining with PRO)",
            "Trade Calculator & share links, Rankings, Watchlist alerts, Waivers/Start-Sit, "
            "Matchups, Redzone, Awards/History, and the Tuesday digest all stay available. "
            "PRO layers decision quality on top — it doesn't replace the free workflow."
        ),
        _section(
            "Billing & sharing",
            "Subscriptions renew yearly through Stripe. Manage payment method, invoices, "
            "or cancel from Pricing → Manage billing while signed in. League PRO buyers "
            "keep a copyable invite link under Commissioner so teammates can claim access."
        ),
        (
            '<p style="margin:20px 0 0;font-size:13px;color:#64748b;line-height:1.5;">'
            "You're also on the Tuesday weekly digest for your primary league — opt out "
            "from any digest footer without losing PRO or these onboarding emails.</p>"
        ),
    ]

    html = email_shell(
        "".join(parts),
        subtitle=f"Welcome to {plan_label}",
        dash_url=trade_sugg if "trade" in trade_sugg else dash,
        cta_label="Open Trade Suggestions →",
        unsub_href=unsub_href,
        logo_url=logos["logo"],
        brand_mark_url=logos["mark"],
        footer_kind="onboarding",
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
