"""
Billing / subscription routes.

Routes: /pricing, /api/create-checkout-session, /api/stripe-webhook,
        /api/subscription-status, /api/pro-signup/pending, /pro/resume-checkout
Also handles: /<platform>/<season>/<league_id>/pricing
"""
from __future__ import annotations

import html
import json
import logging
import os
import urllib.parse
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request, session

from dashboard_services.subscriptions import (
    add_pro_league_slot,
    apply_subscription_lifecycle,
    create_league_subscription,
    create_user_league_subscription,
    create_user_subscription,
    find_active_personal_plan,
    get_pro_league_slots,
    has_any_user_league_subscription,
    has_premium_access,
    has_premium_for_viewer,
    has_user_league_subscription,
    set_pro_league_slots,
    slot_cap_for_plan,
)

billing_bp = Blueprint("billing", __name__)
logger = logging.getLogger(__name__)

_STRIPE_LEAGUE_PRODUCT = "prod_USjDJYPhNGnmvM"
_STRIPE_USER_PRODUCT   = "prod_USjDRuVDcwH1xb"
_STRIPE_COMBO_PRODUCT  = "prod_UT5DaCA4u6hWgb"
_STRIPE_SINGLE_LEAGUE_PRODUCT = (
    os.environ.get("STRIPE_SINGLE_LEAGUE_PRODUCT", "").strip()
    or "prod_VB4fRiQaKCTcu9"
)


def _stripe():
    """Lazy-import stripe so missing package doesn't break the whole blueprint."""
    import stripe as _s
    _s.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    return _s


def _stripe_field(obj, key, default=None):
    """Read a field from a Stripe object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _stripe_id(value) -> str:
    """Coerce a Stripe id that may arrive as a string or expanded object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(_stripe_field(value, "id", "") or "")


def _metadata_dict(obj) -> dict:
    """Normalize Stripe metadata to a plain str→str dict.

    ``dict(stripe_object.metadata)`` is unreliable across stripe-python
    versions; Checkout success already uses ``to_dict()``. The webhook must
    do the same or ``plan`` / ``league_id`` come through empty and One League
    never writes ``user_league_subscriptions``.
    """
    raw = _stripe_field(obj, "metadata", None)
    if not raw:
        return {}
    if hasattr(raw, "to_dict"):
        try:
            raw = raw.to_dict()
        except Exception:
            pass
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except Exception:
            return {}
    out = {}
    for key, value in raw.items():
        if key is None:
            continue
        out[str(key)] = "" if value is None else str(value)
    return out


def _product_plan_map() -> dict:
    mapping = {
        _STRIPE_LEAGUE_PRODUCT: "league",
        _STRIPE_USER_PRODUCT: "user",
        _STRIPE_COMBO_PRODUCT: "combo",
        _STRIPE_SINGLE_LEAGUE_PRODUCT: "single_league",
    }
    # New-catalog products (env-configured). Lets webhook plan inference work
    # for ad-hoc price_data checkouts as well as dashboard Price IDs.
    for product_id, plan in (
        (_STRIPE_PRODUCT_STARTER, "starter"),
        (_STRIPE_PRODUCT_ALL_PRO, "all_pro"),
        (_STRIPE_PRODUCT_HALL_OF_FAME, "hall_of_fame"),
    ):
        if product_id:
            mapping[product_id] = plan
    return mapping


def _plan_from_subscription(sub) -> str:
    """Infer plan from the Stripe product or price on a subscription's items.

    Product inference covers ad-hoc price_data checkouts and legacy product
    env vars. Price-ID inference covers dashboard catalog Prices created from
    the six new env vars, where the Stripe product may not be env-configured.
    """
    if not sub:
        return ""
    items = _stripe_field(sub, "items")
    data = _stripe_field(items, "data") or []
    product_map = _product_plan_map()
    # Map configured Stripe Price ID values (not env var names) to plans.
    price_map = {}
    for plan in _STRIPE_PRICES:
        for price_id in (_annual_price_id(plan), _monthly_price_id(plan)):
            if price_id:
                price_map[price_id] = plan
    for item in data:
        price = _stripe_field(item, "price") or {}
        price_id = _stripe_id(price)
        plan = price_map.get(price_id)
        if plan:
            return plan
        product = _stripe_field(price, "product")
        product_id = _stripe_id(product)
        plan = product_map.get(product_id)
        if plan:
            return plan
    return ""


_BILLING_INTERVALS = ("month", "year")


def _normalize_interval(value) -> str:
    """Coerce a client/supplied interval to 'month' or 'year' (default year)."""
    interval = str(value or "year").strip().lower()
    return interval if interval in _BILLING_INTERVALS else "year"


def _interval_from_subscription(sub) -> str:
    """Read the billing interval from a Stripe subscription's price.

    Interval-agnostic: whatever Stripe says the current price renews on,
    the DB stores. Returns "" when the price is not expanded or the product
    is not one of ours, so callers can fall back to checkout metadata.
    """
    if not sub:
        return ""
    items = _stripe_field(sub, "items")
    data = _stripe_field(items, "data") or []
    product_map = _product_plan_map()
    for item in data:
        price = _stripe_field(item, "price") or {}
        product = _stripe_field(price, "product")
        if not product_map.get(_stripe_id(product)):
            continue
        recurring = _stripe_field(price, "recurring") or {}
        interval = str(_stripe_field(recurring, "interval") or "").strip().lower()
        if interval in _BILLING_INTERVALS:
            return interval
    return ""


def _subscriber_user_id(meta: dict) -> str:
    """Prefer checkout ``user_id``; fall back to ``acct:<account_id>``."""
    user_id = str(meta.get("user_id") or "").strip()
    if user_id:
        return user_id
    account_id = str(meta.get("account_id") or "").strip()
    if not account_id:
        return ""
    return account_id if account_id.startswith("acct:") else f"acct:{account_id}"


def _checkout_metadata(plan: str, user_id: str, league_id: str, platform: str, season: int,
                       interval: str = "year") -> dict:
    return {
        "plan": plan,
        "user_id": user_id,
        "league_id": league_id,
        "platform": platform,
        "season": str(season),
        "interval": _normalize_interval(interval),
        "account_id": str(session.get("account_id") or ""),
    }


def _apply_plan_grant(
    plan: str,
    user_id: str,
    league_id: str,
    platform: str,
    expires_at,
    sub_id: str,
    cust_id: str,
    *,
    source: str,
    account_id: str = "",
    season: str = "",
    interval: str = "year",
) -> None:
    """Write the entitlement row(s) for a paid plan. Idempotent upserts."""
    plan = (plan or "").strip()
    user_id = (user_id or "").strip()
    league_id = (league_id or "").strip()
    platform = (platform or "sleeper").strip() or "sleeper"
    sub_id = _stripe_id(sub_id) or None
    cust_id = _stripe_id(cust_id) or None
    interval = _normalize_interval(interval)
    if not plan:
        logger.info(
            "[stripe] %s missing plan metadata user=%s league=%s",
            source, user_id, league_id,
        )
        return

    granted = False
    handled = False
    if plan in _LEGACY_PRICES:
        # Grandfathered plans: entitlement writes unchanged.
        handled = True
        if plan in ("league", "combo") and league_id:
            ok = create_league_subscription(
                league_id, user_id or "", expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
                platform=platform,
                billing_interval=interval,
            )
            granted = granted or bool(ok)
            logger.info(
                "[stripe] %s league subscription %s for league=%s user=%s expires=%s",
                source, "created" if ok else "FAILED", league_id, user_id, expires_at,
            )
        if plan in ("user", "combo") and user_id:
            ok = create_user_subscription(
                user_id, expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
                platform=platform,
                billing_interval=interval,
                plan_key=plan,
            )
            granted = granted or bool(ok)
            logger.info(
                "[stripe] %s user subscription %s for user=%s expires=%s",
                source, "created" if ok else "FAILED", user_id, expires_at,
            )
        if plan == "single_league":
            if user_id and league_id:
                ok = create_user_league_subscription(
                    user_id, league_id, expires_at,
                    stripe_subscription_id=sub_id,
                    stripe_customer_id=cust_id,
                    platform=platform,
                    billing_interval=interval,
                )
                granted = granted or bool(ok)
                logger.info(
                    "[stripe] %s single-league subscription %s for user=%s league=%s expires=%s",
                    source, "created" if ok else "FAILED", user_id, league_id, expires_at,
                )
            else:
                logger.warning(
                    "[stripe] %s single_league grant skipped: user_id=%r league_id=%r",
                    source, user_id, league_id,
                )
    elif plan in _STRIPE_PRICES and user_id:
        handled = True
        ok = create_user_subscription(
            user_id, expires_at,
            stripe_subscription_id=sub_id,
            stripe_customer_id=cust_id,
            platform=platform,
            billing_interval=interval,
            plan_key=plan,
        )
        granted = granted or bool(ok)
        logger.info(
            "[stripe] %s %s subscription %s for user=%s expires=%s",
            source, plan, "created" if ok else "FAILED", user_id, expires_at,
        )
        # Seed the first slot when checkout carried a league (e.g. bought from
        # a league page). The buyer can change slots anytime via the API.
        if ok and slot_cap_for_plan(plan) and league_id:
            add_pro_league_slot(
                user_id, platform, league_id,
                stripe_subscription_id=sub_id,
            )
    if not handled:
        logger.warning(
            "[stripe] %s unhandled plan=%s league=%s user=%s",
            source, plan, league_id, user_id,
        )

    if granted and plan in _KNOWN_PLANS:
        try:
            _maybe_send_pro_welcome(
                plan=plan,
                user_id=user_id,
                account_id=account_id,
                league_id=league_id,
                platform=platform,
                season=season,
            )
        except Exception:
            logger.warning("[stripe] pro welcome email failed", exc_info=True)


def _maybe_send_pro_welcome(
    *,
    plan: str,
    user_id: str,
    account_id: str = "",
    league_id: str = "",
    platform: str = "",
    season: str = "",
) -> None:
    """Best-effort PRO onboarding email after a successful grant."""
    from utils.welcome_email import resolve_account_from_subscriber, send_pro_welcome

    aid = None
    raw_aid = (account_id or "").strip()
    if raw_aid:
        try:
            aid = int(raw_aid.replace("acct:", "") if raw_aid.startswith("acct:") else raw_aid)
        except (TypeError, ValueError):
            aid = None
    row = resolve_account_from_subscriber(user_id=user_id, account_id=aid)
    if not row:
        logger.info("[stripe] pro welcome skipped: no account for user=%s", user_id)
        return
    season_i = None
    try:
        season_i = int(season) if season else None
    except (TypeError, ValueError):
        season_i = None
    send_pro_welcome(
        int(row["id"]),
        email=row.get("email"),
        first_name=row.get("first_name"),
        plan=plan,
        platform=platform or "sleeper",
        season=season_i,
        league_id=league_id or "",
    )


_STRIPE_PRODUCT_STARTER = os.environ.get("STRIPE_PRODUCT_STARTER", "").strip()
_STRIPE_PRODUCT_ALL_PRO = os.environ.get("STRIPE_PRODUCT_ALL_PRO", "").strip()
_STRIPE_PRODUCT_HALL_OF_FAME = os.environ.get("STRIPE_PRODUCT_HALL_OF_FAME", "").strip()


# ── Plan catalog ──────────────────────────────────────────────────────────────
# Purchasable plans. Amounts are the ad-hoc fallback used when no dashboard
# Price ID is configured (see _ANNUAL_PRICE_ENV / _MONTHLY_PRICE_ENV below);
# a configured Price ID always wins.
_STRIPE_PRICES = {
    "starter": {
        "unit_amount": 1000,  # $10/year
        "product": _STRIPE_PRODUCT_STARTER,
        "product_name": "BR Fantasy Starter PRO",
    },
    "all_pro": {
        "unit_amount": 3000,  # $30/year
        "product": _STRIPE_PRODUCT_ALL_PRO,
        "product_name": "BR Fantasy All-Pro PRO",
    },
    "hall_of_fame": {
        "unit_amount": 5000,  # $50/year
        "product": _STRIPE_PRODUCT_HALL_OF_FAME,
        "product_name": "BR Fantasy Hall of Fame PRO",
    },
}

# Retired from sale but GRANDFATHERED: existing subscribers keep renewing and
# keep their entitlements. Used ONLY for grandfathered renewals,
# checkout-session resumption (sessions staged before the cutover), webhooks,
# and entitlement checks. New buyers can never purchase these keys: checkout
# validates against _STRIPE_PRICES only.
_LEGACY_PRICES = {
    "league": {"unit_amount": 3500, "product": _STRIPE_LEAGUE_PRODUCT},
    "user":   {"unit_amount": 2000, "product": _STRIPE_USER_PRODUCT},
    "combo":  {"unit_amount": 4500, "product": _STRIPE_COMBO_PRODUCT},
    "single_league": {
        "unit_amount": 1000,
        "product": _STRIPE_SINGLE_LEAGUE_PRODUCT,
        "product_name": "BR Fantasy Single League PRO",
    },
}

# Every plan key the backend will ever honor in webhooks, grants, and
# entitlement checks (new + grandfathered).
_KNOWN_PLANS = frozenset(_STRIPE_PRICES) | frozenset(_LEGACY_PRICES)

# League-slot caps for the purchasable plans. None = unlimited
# (hall_of_fame is PRO everywhere, same as the grandfathered user plan).
# Mirrored in dashboard_services/subscriptions.py as _SLOT_CAP_BY_PLAN.
_PLAN_LEAGUE_CAP = {
    "starter": 1,
    "all_pro": 5,
    "hall_of_fame": None,
}

# Dashboard-created Stripe Price IDs. When the env var for a plan + interval
# is set, checkout uses that Price ID; when unset, checkout falls back to
# ad-hoc price_data with the code defaults, so nothing breaks before the
# dashboard Prices are wired up. An env-provided Price ID always wins.
_ANNUAL_PRICE_ENV = {
    "starter": "STRIPE_PRICE_STARTER_ANNUAL",
    "all_pro": "STRIPE_PRICE_ALL_PRO_ANNUAL",
    "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_ANNUAL",
}

# Monthly billing defaults (unit_amount, cents). ADJUSTABLE: edit here, or
# create monthly Prices in the Stripe Dashboard and point the env vars below
# at them -- an env-provided Price ID always wins over these defaults.
_MONTHLY_DEFAULT_UNIT_AMOUNTS = {
    "starter": 149,       # $1.49/mo
    "all_pro": 449,       # $4.49/mo
    "hall_of_fame": 749,  # $7.49/mo
}
_MONTHLY_PRICE_ENV = {
    "starter": "STRIPE_PRICE_STARTER_MONTHLY",
    "all_pro": "STRIPE_PRICE_ALL_PRO_MONTHLY",
    "hall_of_fame": "STRIPE_PRICE_HALL_OF_FAME_MONTHLY",
}

# Legacy Price ID env vars + monthly defaults. Kept only so checkout sessions
# staged before the cutover (resume flow) keep working if Kaedon created
# dashboard Prices for the old catalog. The old monthly Price IDs
# ($2.99/$4.99/$5.99) never need creating now.
_LEGACY_ANNUAL_PRICE_ENV = {
    "league": "STRIPE_PRICE_LEAGUE_ANNUAL",
    "user": "STRIPE_PRICE_USER_ANNUAL",
    "combo": "STRIPE_PRICE_COMBO_ANNUAL",
    "single_league": "STRIPE_PRICE_SINGLE_LEAGUE_ANNUAL",
}
_LEGACY_MONTHLY_DEFAULT_UNIT_AMOUNTS = {
    "league": 499,         # $4.99/mo
    "user": 299,           # $2.99/mo
    "combo": 599,          # $5.99/mo
    "single_league": 149,  # $1.49/mo
}
_LEGACY_MONTHLY_PRICE_ENV = {
    "league": "STRIPE_PRICE_LEAGUE_MONTHLY",
    "user": "STRIPE_PRICE_USER_MONTHLY",
    "combo": "STRIPE_PRICE_COMBO_MONTHLY",
    "single_league": "STRIPE_PRICE_SINGLE_LEAGUE_MONTHLY",
}

_BILLING_INTERVALS = ("month", "year")


def _normalize_interval(value) -> str:
    """Coerce a client/supplied interval to 'month' or 'year' (default year)."""
    interval = str(value or "year").strip().lower()
    return interval if interval in _BILLING_INTERVALS else "year"


def _price_env_name(plan: str, interval: str) -> str:
    """Env var holding the dashboard Price ID for a plan + interval.

    New-catalog vars first, legacy vars second (resume flow for sessions
    staged before the cutover). Empty when the plan is unknown.
    """
    if _normalize_interval(interval) == "month":
        return _MONTHLY_PRICE_ENV.get(plan, "") or _LEGACY_MONTHLY_PRICE_ENV.get(plan, "")
    return _ANNUAL_PRICE_ENV.get(plan, "") or _LEGACY_ANNUAL_PRICE_ENV.get(plan, "")


def _annual_price_id(plan: str) -> str:
    """Stripe Price ID for the annual tier, from env (empty when unset)."""
    env_name = _ANNUAL_PRICE_ENV.get(plan, "") or _LEGACY_ANNUAL_PRICE_ENV.get(plan, "")
    return os.environ.get(env_name, "").strip() if env_name else ""


def _monthly_price_id(plan: str) -> str:
    """Stripe Price ID for the monthly tier, from env (empty when unset)."""
    env_name = _MONTHLY_PRICE_ENV.get(plan, "") or _LEGACY_MONTHLY_PRICE_ENV.get(plan, "")
    return os.environ.get(env_name, "").strip() if env_name else ""


def _plan_price_spec(plan: str) -> dict:
    """Price spec for any known plan (new catalog first, legacy second)."""
    spec = _STRIPE_PRICES.get(plan) or _LEGACY_PRICES.get(plan)
    if not spec:
        raise KeyError(f"unknown plan: {plan}")
    return spec


def _monthly_default_amount(plan: str) -> int:
    return _MONTHLY_DEFAULT_UNIT_AMOUNTS.get(
        plan, _LEGACY_MONTHLY_DEFAULT_UNIT_AMOUNTS.get(plan, 0)
    )


def _checkout_line_item(plan: str, interval: str = "year") -> dict:
    """Build the Stripe Checkout line item for a plan + interval.

    A dashboard-created Price ID (env) always wins over ad-hoc price_data.
    Annual falls back to the existing ad-hoc price_data with the annual
    defaults above; monthly falls back to ad-hoc price_data with the monthly
    defaults. Works for the new catalog and (resume flow only) legacy plans.
    """
    interval = _normalize_interval(interval)
    spec = _plan_price_spec(plan)
    if interval == "month":
        price_id = _monthly_price_id(plan)
        unit_amount = _monthly_default_amount(plan)
    else:
        price_id = _annual_price_id(plan)
        unit_amount = spec["unit_amount"]
    if price_id:
        return {"price": price_id, "quantity": 1}
    price_data = {
        "currency": "usd",
        "unit_amount": unit_amount,
        "recurring": {"interval": interval},
    }
    if spec.get("product"):
        price_data["product"] = spec["product"]
    else:
        price_data["product_data"] = {
            "name": spec.get("product_name") or "BR Fantasy PRO",
        }
    return {"price_data": price_data, "quantity": 1}


def _annual_savings_pct(plan: str) -> int:
    """Whole-percent saved by annual vs 12x the monthly default."""
    annual = _plan_price_spec(plan)["unit_amount"]
    monthly_equiv = 12 * _monthly_default_amount(plan)
    if monthly_equiv <= annual:
        return 0
    return round(100 * (monthly_equiv - annual) / monthly_equiv)
# No purchasable plan requires a league at checkout: starter/all_pro buyers
# pick their slot league(s) after purchase (a league_id passed at checkout
# just seeds the first slot), and hall_of_fame is PRO everywhere.
_LEAGUE_REQUIRED_PLANS = frozenset()
_MEMBERSHIP_REQUIRED_PLANS = frozenset()

# Grandfathered plans keep their old checkout-time requirements in the
# resume flow (sessions staged before the cutover).
_LEGACY_LEAGUE_REQUIRED_PLANS = frozenset({"league", "combo", "single_league"})
_LEGACY_MEMBERSHIP_REQUIRED_PLANS = frozenset({"league", "combo", "single_league"})

_SUPPORTED_PLATFORMS = {"sleeper", "espn", "yahoo", "mfl", "fleaflicker"}


def _request_platform(payload=None) -> str:
    """Resolve the provider without silently turning an ESPN flow into Sleeper."""
    payload = payload if isinstance(payload, dict) else {}
    return str(
        payload.get("platform") or request.values.get("platform")
        or session.get("viewer_platform") or session.get("last_platform")
        or "sleeper"
    ).strip().lower()


def _safe_local_url(value: str, fallback: str) -> str:
    """Allow same-site absolute/local redirects, rejecting protocol-relative URLs."""
    from utils.safe_url import safe_local_url
    return safe_local_url(value, fallback, host_url=request.host_url)


def _checkout_user_id() -> str | None:
    """Stable checkout identity: provider viewer, else Google account."""
    return (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )


def pending_checkout_resume_path() -> str | None:
    """Return the resume URL when a home-page PRO signup is staged."""
    pending = session.get("pending_checkout")
    if isinstance(pending, dict) and pending.get("plan"):
        return "/pro/resume-checkout"
    return None


def _normalize_pending_checkout(data: dict) -> tuple[dict | None, str | None]:
    """Validate a home PRO signup payload. Returns (pending, error).

    Only the current catalog is stagable: legacy keys can never be purchased
    by new buyers (grandfathered sessions resume via /pro/resume-checkout).
    """
    plan = str((data or {}).get("plan") or "").strip()
    if plan not in _STRIPE_PRICES:
        return None, "Pick a plan to continue."
    platform = _request_platform(data)
    if platform not in _SUPPORTED_PLATFORMS:
        return None, "Choose a supported platform."
    league_id = str((data or {}).get("league_id") or "").strip()
    if plan in _LEAGUE_REQUIRED_PLANS and not league_id:
        return None, "Enter your league info to continue."
    try:
        season = int((data or {}).get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return None, "Invalid season."
    username = str((data or {}).get("username") or "").strip() or None
    team_id = str((data or {}).get("team_id") or "").strip() or None
    name = str((data or {}).get("name") or "").strip() or None
    return {
        "plan": plan,
        "platform": platform,
        "league_id": league_id,
        "season": season,
        "username": username,
        "team_id": team_id,
        "name": name,
    }, None


def _require_google_to_subscribe():
    """New subscriptions require a Google account. Sleeper-only identity is not enough."""
    if session.get("account_id"):
        return None
    return jsonify({"error": "Sign in with Google to subscribe."}), 401


# ── Helpers ───────────────────────────────────────────────────────────────────

def _subscription_period_end(sub) -> "datetime":
    """Return the subscription's current period end as a UTC datetime.

    Stripe API 2025-03-31+ (stripe-python v15) moved ``current_period_end`` off
    the Subscription object onto each subscription *item*. This reads whichever
    location is present and falls back to a 32-day grant if neither is found.
    """
    ts = getattr(sub, "current_period_end", None)
    if not ts:
        try:
            items = (sub.get("items") if hasattr(sub, "get") else sub["items"])
            data = items["data"] if items else []
            ts = max((it.get("current_period_end") or 0) for it in data) or None
        except Exception:
            ts = None
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.now(timezone.utc) + timedelta(days=32)


def _try_grant_from_stripe_success() -> None:
    """
    When a user returns from Stripe checkout, verify the session server-side
    and grant the subscription immediately. This is a reliable fallback for
    when the webhook is delayed or misconfigured.
    """
    if request.args.get("success") != "1":
        return
    checkout_session_id = request.args.get("session_id", "").strip()
    if not checkout_session_id:
        return
    try:
        cs = _stripe().checkout.Session.retrieve(checkout_session_id)
        if cs.status != "complete":
            return

        meta      = _metadata_dict(cs)
        plan      = (meta.get("plan") or "").strip()
        user_id   = _subscriber_user_id(meta)
        platform  = meta.get("platform") or "sleeper"
        league_id = meta.get("league_id") or ""
        sub_id    = _stripe_id(_stripe_field(cs, "subscription"))
        cust_id   = _stripe_id(_stripe_field(cs, "customer"))

        try:
            sub        = _stripe().Subscription.retrieve(sub_id) if sub_id else None
            expires_at = (
                _subscription_period_end(sub)
                if sub else datetime.now(timezone.utc) + timedelta(days=366)
            )
            if plan not in _KNOWN_PLANS:
                plan = _plan_from_subscription(sub)
        except Exception:
            expires_at = datetime.now(timezone.utc) + timedelta(days=366)
            sub = None

        if plan not in _KNOWN_PLANS:
            return
        if plan in ("user", "hall_of_fame", "starter", "all_pro") and not user_id:
            return
        if plan in ("league", "single_league") and not league_id:
            return
        if plan == "combo" and not league_id and not user_id:
            return

        # Interval from checkout metadata; trust Stripe's price when the
        # metadata predates monthly billing.
        interval = (meta.get("interval") or "").strip().lower()
        if interval not in ("month", "year") and sub is not None:
            interval = _interval_from_subscription(sub) or interval
        interval = _normalize_interval(interval)

        _apply_plan_grant(
            plan, user_id, league_id, platform, expires_at, sub_id, cust_id,
            source="success-page",
            account_id=str(meta.get("account_id") or ""),
            season=str(meta.get("season") or ""),
            interval=interval,
        )
    except Exception:
        logger.exception("[stripe] success-page session verification failed")


def _pricing_feature_item(icon: str, label: str, *, free: bool = False) -> str:
    tier = "pricing-feature-free" if free else "pricing-feature-pro"
    return (
        f'<div class="pricing-feature-item {tier}">'
        f'<span class="pricing-feature-icon" aria-hidden="true"><i class="fa-solid {html.escape(icon)}"></i></span>'
        f'<span class="pricing-feature-label">{label}</span>'
        f"</div>"
    )


def _pricing_features_grid(items: list[tuple[str, str]], *, free: bool = False) -> str:
    cells = "".join(_pricing_feature_item(icon, label, free=free) for icon, label in items)
    return f'<div class="pricing-features-grid">{cells}</div>'


_PRO_FEATURES = [
    ("fa-handshake", "Roster-based trade suggestions"),
    ("fa-chart-line", "Trade Intel feed &amp; history"),
    ("fa-wand-magic-sparkles", "AI trade analysis &amp; counters"),
    ("fa-trophy", "Playoff-impact simulations"),
    ("fa-fire", "Breakout Engine"),
    ("fa-briefcase", "Front Office Report"),
    ("fa-newspaper", "Premium AI weekly recap storyline"),
    ("fa-layer-group", "Cross-league This week’s moves"),
    ("fa-clipboard-list", "Custom Draft Board"),
    ("fa-arrow-trend-up", "Trend Scout"),
    ("fa-magnifying-glass-chart", "Draft Deep Dive"),
]

_FREE_FEATURES = [
    ("fa-calculator", "Trade calculator &amp; Sleeper comps"),
    ("fa-table", "Advanced Metrics"),
    ("fa-gavel", "Auction Values"),
    ("fa-file-csv", "Live cheat-sheet overlay &amp; CSV"),
]


_MANAGE_PLAN_LABELS = {
    "starter": "Starter PRO",
    "all_pro": "All-Pro PRO",
    "hall_of_fame": "Hall of Fame PRO",
    # Grandfathered labels for existing subscribers.
    "user": "Personal PRO",
    "single_league": "One League PRO",
    "league": "League PRO",
    "combo": "League + Personal PRO",
}

_CANCEL_REASON_LABELS = [
    ("too_expensive", "Too expensive"),
    ("not_using", "Not using it enough"),
    ("missing_feature", "Missing a feature I need"),
    ("switched_tool", "Switched to another tool"),
    ("seasonal", "Seasonal: I will be back"),
    ("other", "Other"),
]


def _pricing_manage_card() -> str:
    """'Your PRO' management card for signed-in subscribers on /pricing.

    Shows plan + renewal, a Manage payment method button (Stripe portal), and
    a Cancel PRO button that opens the survey + pause-offer modal first.
    Returns "" for guests and non-subscribers.
    """
    from flask import session as _session
    from utils import churn as _churn

    user_id = (
        _session.get("viewer_user_id")
        or _session.get("viewer_username")
        or (("acct:" + str(_session.get("account_id")).strip()) if _session.get("account_id") else None)
    )
    if not user_id:
        return ""
    subs = _churn.active_subscriptions_for_user(user_id)
    if not subs:
        return ""

    def _fmt_date(value) -> str:
        try:
            if hasattr(value, "strftime"):
                return value.strftime("%b %d, %Y")
        except Exception:
            pass
        return str(value or "")

    rows = []
    slot_blocks = []
    for i, sub in enumerate(subs):
        plan = str(sub.get("plan") or "")
        label = _MANAGE_PLAN_LABELS.get(plan, "PRO")
        league = html.escape(str(sub.get("league_id") or ""), quote=True)
        league_note = f" for league <code>{league}</code>" if league else ""
        sub_id = html.escape(str(sub.get("stripe_subscription_id") or ""), quote=True)
        renew = html.escape(_fmt_date(sub.get("expires_at")), quote=False)
        rows.append(f"""
        <div class="pro-billing-row" style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 0;border-top:1px solid var(--border);">
          <div>
            <div style="font-weight:700;font-size:15px;">{html.escape(label, quote=False)}{league_note}</div>
            <div style="font-size:13px;color:var(--text-muted);">Renews {renew}. Cancel or pause anytime.</div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button type="button" class="btn btn-secondary" onclick="brOpenPortal(this)">Manage payment method</button>
            <button type="button" class="btn" style="border:1px solid #fca5a5;color:#b91c1c;background:transparent;"
                    onclick="brOpenCancelModal('{sub_id}', '{renew}')">Cancel PRO</button>
          </div>
        </div>""")
        # Slot-capped plans (Starter / All-Pro): let the buyer pick which
        # leagues get PRO, up to the plan cap. Hall of Fame is unlimited, so
        # it gets a short note instead of a picker.
        cap = slot_cap_for_plan(plan)
        if plan == "hall_of_fame":
            slot_blocks.append("""
        <div class="br-slots-static" style="padding:14px 0 6px;border-top:1px solid var(--border);">
          <div style="font-weight:700;font-size:15px;">Your PRO leagues</div>
          <p style="margin:4px 0 0;font-size:13px;color:var(--text-muted);">Hall of Fame covers unlimited leagues. Nothing to manage here.</p>
        </div>""")
        elif cap:
            block_id = f"brSlots{i}"
            plat = html.escape(str(sub.get("platform") or "sleeper"), quote=True)
            slot_blocks.append(f"""
        <div class="br-slots" id="{block_id}" data-platform="{plat}"
             style="padding:14px 0 6px;border-top:1px solid var(--border);">
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;justify-content:space-between;">
            <div style="font-weight:700;font-size:15px;">Your PRO leagues</div>
            <span class="br-slots-count" style="font-size:13px;color:var(--text-muted);">Loading slots&hellip;</span>
          </div>
          <p style="margin:4px 0 10px;font-size:13px;color:var(--text-muted);">
            {html.escape(label, quote=False)} covers PRO for you in {cap}
            league{'s' if cap != 1 else ''}. Pick which ones below. You can
            change this anytime.</p>
          <div class="br-slots-list" style="display:flex;flex-direction:column;gap:8px;"></div>
          <div style="display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap;">
            <button type="button" class="btn btn-primary"
                    onclick="brSaveProLeagues('{block_id}')">Save leagues</button>
            <span class="br-slots-note" style="font-size:13px;color:var(--text-muted);"></span>
          </div>
        </div>""")

    reason_radios = "".join(
        f"""<label style="display:flex;gap:10px;align-items:flex-start;padding:10px 12px;border:1px solid var(--border);border-radius:10px;cursor:pointer;font-size:14px;">
          <input type="radio" name="br-cancel-reason" value="{key}" style="margin-top:3px;"/>
          <span>{label}</span></label>"""
        for key, label in _CANCEL_REASON_LABELS
    )

    return f"""
    <section class="pricing-section" id="pro-billing" aria-label="Manage your PRO">
      <div class="card" style="max-width:760px;margin:0 auto;">
        <div class="card-body" style="padding:26px 28px;">
          <h2 style="margin:0 0 4px;font-size:20px;">Your PRO</h2>
          <p style="margin:0 0 6px;font-size:14px;color:var(--text-muted);">Update your card, pause, or cancel. No phone calls, no dark patterns.</p>
          {''.join(rows)}
          {''.join(slot_blocks)}
          <div id="brBillingMsg" style="display:none;margin-top:12px;font-size:14px;"></div>
        </div>
      </div>
    </section>
    <div id="brCancelModal" hidden
         style="position:fixed;inset:0;z-index:10001;background:rgba(15,23,42,.55);align-items:center;justify-content:center;padding:20px;">
      <div role="dialog" aria-modal="true" aria-labelledby="brCancelTitle"
           style="background:var(--card);border:1px solid var(--border);border-radius:16px;max-width:480px;width:100%;padding:28px;box-shadow:0 24px 60px rgba(0,0,0,.3);">
        <div id="brCancelStep1">
          <h3 id="brCancelTitle" style="margin:0 0 6px;font-size:19px;">Sorry to see you go</h3>
          <p style="margin:0 0 16px;font-size:14px;color:var(--text-muted);">Mind telling us why? Optional. It helps us build better.</p>
          <div style="display:flex;flex-direction:column;gap:8px;">{reason_radios}</div>
          <textarea id="brCancelDetail" rows="2" placeholder="Anything else? (optional)"
                    style="width:100%;box-sizing:border-box;margin-top:10px;padding:10px 12px;border:1px solid var(--border);border-radius:10px;font-size:14px;background:var(--card);color:var(--text);"></textarea>
          <div style="display:flex;gap:10px;margin-top:18px;justify-content:flex-end;">
            <button type="button" class="btn btn-secondary" onclick="brCancelReason('skipped')">Skip</button>
            <button type="button" class="btn btn-primary" onclick="brCancelReason('chosen')">Continue</button>
          </div>
        </div>
        <div id="brCancelStep2" hidden>
          <h3 style="margin:0 0 6px;font-size:19px;">Wait: pause instead?</h3>
          <p style="margin:0 0 16px;font-size:14px;color:var(--text-muted);line-height:1.55;">
            Keep every PRO tool free for 2 months. Billing pauses now and resumes automatically.
            No need to cancel and resubscribe later.</p>
          <div style="display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap;">
            <button type="button" class="btn btn-secondary" onclick="brCancelGoFinal()">Continue to cancel</button>
            <button type="button" class="btn btn-primary" onclick="brCancelFinal('pause')">Pause PRO for 2 months</button>
          </div>
        </div>
        <div id="brCancelStep3" hidden>
          <h3 style="margin:0 0 6px;font-size:19px;">Cancel PRO?</h3>
          <p style="margin:0 0 18px;font-size:14px;color:var(--text-muted);line-height:1.55;">
            You will keep PRO until <strong id="brCancelUntil"></strong>, then it turns off.
            This cannot be undone automatically.</p>
          <div style="display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap;">
            <button type="button" class="btn btn-primary" onclick="brCloseCancelModal()">Keep PRO</button>
            <button type="button" class="btn" style="border:1px solid #fca5a5;color:#b91c1c;background:transparent;"
                    onclick="brCancelFinal('cancel')">Confirm cancellation</button>
          </div>
        </div>
        <div id="brCancelDone" hidden>
          <h3 style="margin:0 0 6px;font-size:19px;" id="brCancelDoneTitle">Done</h3>
          <p style="margin:0 0 18px;font-size:14px;color:var(--text-muted);line-height:1.55;" id="brCancelDoneMsg"></p>
          <div style="display:flex;justify-content:flex-end;">
            <button type="button" class="btn btn-primary" onclick="brCloseCancelModal()">Close</button>
          </div>
        </div>
      </div>
    </div>
    <script>
    (function() {{
      var subId = '', reason = 'skipped', detail = '';
      window.brOpenPortal = function(btn) {{
        btn.disabled = true;
        fetch('/api/create-portal-session', {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{}})
        }}).then(function(r) {{ return r.json(); }}).then(function(d) {{
          if (d.url) {{ window.location.href = d.url; }}
          else {{
            btn.disabled = false;
            brBillingNote(d.error || 'Could not open billing. Please try again.', true);
          }}
        }}).catch(function() {{
          btn.disabled = false;
          brBillingNote('Could not open billing. Please try again.', true);
        }});
      }};
      window.brBillingNote = function(msg, isErr) {{
        var el = document.getElementById('brBillingMsg');
        if (!el) return;
        el.style.display = 'block';
        el.style.color = isErr ? '#b91c1c' : '#166534';
        el.textContent = msg;
      }};
      window.brOpenCancelModal = function(sid, until) {{
        subId = sid; reason = 'skipped'; detail = '';
        document.getElementById('brCancelUntil').textContent = until || 'the end of your billing period';
        ['brCancelStep1','brCancelStep2','brCancelStep3','brCancelDone'].forEach(function(id, i) {{
          document.getElementById(id).hidden = (i !== 0);
        }});
        var modal = document.getElementById('brCancelModal');
        modal.hidden = false;
        modal.style.display = 'flex';
      }};
      window.brCloseCancelModal = function() {{
        var modal = document.getElementById('brCancelModal');
        modal.hidden = true;
        modal.style.display = 'none';
      }};
      window.brCancelReason = function(how) {{
        if (how === 'chosen') {{
          var sel = document.querySelector('input[name="br-cancel-reason"]:checked');
          reason = sel ? sel.value : 'skipped';
        }} else {{
          reason = 'skipped';
        }}
        var ta = document.getElementById('brCancelDetail');
        detail = ta ? ta.value : '';
        document.getElementById('brCancelStep1').hidden = true;
        document.getElementById('brCancelStep2').hidden = false;
      }};
      window.brCancelGoFinal = function() {{
        document.getElementById('brCancelStep2').hidden = true;
        document.getElementById('brCancelStep3').hidden = false;
      }};
      window.brCancelFinal = function(action) {{
        fetch('/api/cancel-subscription', {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{stripe_subscription_id: subId, action: action, reason: reason, reason_detail: detail}})
        }}).then(function(r) {{ return r.json().then(function(d) {{ return {{ok: r.ok, d: d}}; }}); }}).then(function(res) {{
          var title = document.getElementById('brCancelDoneTitle');
          var msg = document.getElementById('brCancelDoneMsg');
          ['brCancelStep1','brCancelStep2','brCancelStep3'].forEach(function(id) {{
            document.getElementById(id).hidden = true;
          }});
          document.getElementById('brCancelDone').hidden = false;
          if (res.ok && res.d.ok) {{
            if (action === 'pause') {{
              title.textContent = 'PRO paused';
              msg.textContent = 'Your PRO is paused for 2 months. Billing resumes automatically. Nothing else to do.';
            }} else {{
              title.textContent = 'PRO canceled';
              msg.textContent = 'Your PRO stays on until ' + (document.getElementById('brCancelUntil').textContent || 'the end of your billing period') + '. Thanks for being with us.';
            }}
          }} else {{
            title.textContent = 'Something went wrong';
            msg.textContent = (res.d && res.d.error) || 'Please try again or manage billing to finish this.';
          }}
        }}).catch(function() {{
          window.brCloseCancelModal();
          brBillingNote('Something went wrong. Please try again.', true);
        }});
      }};
      var modal = document.getElementById('brCancelModal');
      if (modal) modal.addEventListener('click', function(e) {{
        if (e.target === modal) window.brCloseCancelModal();
      }});

      // PRO league slots (Starter / All-Pro): pick which leagues get PRO.
      window.brSaveProLeagues = function(blockId) {{
        var block = document.getElementById(blockId);
        if (!block) return;
        var note = block.querySelector('.br-slots-note');
        var ids = Array.prototype.map.call(
          block.querySelectorAll('input.br-slot-check:checked'),
          function(c) {{ return c.value; }}
        );
        var plat = block.getAttribute('data-platform') || 'sleeper';
        note.textContent = 'Saving...';
        fetch('/api/billing/pro-leagues?platform=' + encodeURIComponent(plat), {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{league_ids: ids}})
        }}).then(function(r) {{ return r.json().then(function(d) {{ return {{ok: r.ok, d: d}}; }}); }}).then(function(res) {{
          if (res.ok) {{
            var n = (res.d.league_ids || []).length;
            note.textContent = 'Saved. PRO applies in ' + n + ' league' + (n === 1 ? '' : 's') + '.';
          }} else {{
            note.textContent = (res.d && res.d.error) || 'Could not save. Please try again.';
          }}
        }}).catch(function() {{
          note.textContent = 'Could not save. Please try again.';
        }});
      }};
      function brSlotsEnforceCap(block, cap) {{
        var list = block.querySelector('.br-slots-list');
        var note = block.querySelector('.br-slots-note');
        var countEl = block.querySelector('.br-slots-count');
        var checked = list.querySelectorAll('input.br-slot-check:checked').length;
        Array.prototype.forEach.call(
          list.querySelectorAll('input.br-slot-check:not(:checked)'),
          function(c) {{ c.disabled = cap > 0 && checked >= cap; }}
        );
        // Persistent "N of M slots used" count; the note stays free for the
        // first-run nudge, save confirmations, and error messages.
        if (countEl) countEl.textContent = checked + ' of ' + cap + ' slots used';
        if (note) {{
          if (cap > 0 && checked >= cap) {{
            note.setAttribute('data-cap-msg', '1');
            note.textContent = 'Your plan covers up to ' + cap +
              (cap === 1 ? ' league.' : ' leagues.') +
              ' Uncheck a league to free a slot first.';
          }} else if (note.getAttribute('data-cap-msg')) {{
            note.removeAttribute('data-cap-msg');
            note.textContent = checked === 0
              ? 'No leagues picked yet. PRO is not active anywhere until you save at least one.'
              : '';
          }}
        }}
      }}
      function brSlotsInitOne(block) {{
        if (block.dataset.brSlotsInit) return;
        block.dataset.brSlotsInit = '1';
        var plat = block.getAttribute('data-platform') || 'sleeper';
        var list = block.querySelector('.br-slots-list');
        var note = block.querySelector('.br-slots-note');
        Promise.all([
          fetch('/api/billing/pro-leagues?platform=' + encodeURIComponent(plat))
            .then(function(r) {{ return r.json(); }}),
          fetch('/api/my-leagues').then(function(r) {{ return r.json(); }})
            .catch(function() {{ return {{leagues: []}}; }})
        ]).then(function(parts) {{
          var info = parts[0] || {{}};
          if (info.error) {{
            note.textContent = info.error;
            return;
          }}
          var cap = info.cap || 0;
          var saved = info.league_ids || [];
          var leagues = ((parts[1] || {{}}).leagues || [])
            .filter(function(l) {{ return (l.platform || 'sleeper') === plat && l.league_id; }});
          var seen = {{}};
          var rows = 0;
          list.innerHTML = '';
          function addSlotRow(id, labelText, isChecked) {{
            if (!id || seen[id]) return;
            seen[id] = 1;
            rows++;
            var label = document.createElement('label');
            label.style.cssText = 'display:flex;gap:10px;align-items:center;padding:10px 12px;border:1px solid var(--border);border-radius:10px;cursor:pointer;font-size:14px;';
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.className = 'br-slot-check';
            cb.value = id;
            cb.checked = !!isChecked;
            cb.addEventListener('change', function() {{ brSlotsEnforceCap(block, cap); }});
            label.appendChild(cb);
            var sp = document.createElement('span');
            sp.textContent = labelText;
            label.appendChild(sp);
            list.appendChild(label);
          }}
          leagues.forEach(function(l) {{
            addSlotRow(l.league_id, l.label || l.name || l.league_id,
              saved.indexOf(l.league_id) !== -1);
          }});
          // Saved slot leagues that are no longer in the league list stay
          // visible (and checked) so the buyer can uncheck them to free slots.
          (saved || []).forEach(function(id) {{
            id = String(id || '').trim();
            addSlotRow(id, 'League ' + id + ' (not in your saved leagues)', true);
          }});
          if (!rows) {{
            list.innerHTML = '<p style="font-size:13px;color:var(--text-muted);margin:0;">No leagues found on this platform yet. Connect a league first.</p>';
          }}
          brSlotsEnforceCap(block, cap);
          if (!saved.length) {{
            note.textContent = 'No leagues picked yet. PRO is not active anywhere until you save at least one.';
          }}
        }}).catch(function() {{
          list.innerHTML = '<p style="font-size:13px;color:var(--text-muted);margin:0;">Could not load your leagues. Please refresh.</p>';
        }});
      }}
      Array.prototype.forEach.call(document.querySelectorAll('.br-slots'), brSlotsInitOne);
    }})();
    </script>
    """


def _pricing_body(league_id: str | None = None, platform: str = "sleeper") -> str:
    from flask import session as _session
    plan      = request.args.get("plan", "")
    success   = request.args.get("success") == "1"
    canceled  = request.args.get("canceled") == "1"
    return_to = request.args.get("return_to", "").strip()
    platform  = (platform or "").strip().lower()
    if platform not in _SUPPORTED_PLATFORMS:
        platform = "sleeper"

    if success:
        # Build a proper destination from Stripe session metadata if return_to is missing
        session_id = request.args.get("session_id", "").strip()
        if not return_to and session_id:
            try:
                from datetime import datetime as _dt
                cs   = _stripe().checkout.Session.retrieve(session_id)
                meta = cs.metadata.to_dict() if cs.metadata else {}
                league_id_meta = meta.get("league_id", "")
                if league_id_meta:
                    season = int(meta.get("season") or _dt.now().year)
                    platform = meta.get("platform") or "sleeper"
                    return_to = f"/{platform}/{season}/{league_id_meta}/dashboard?new_subscriber=1&welcome=personal"
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        # Sanitize before embedding in HTML/JS -- checkout already filters return_url,
        # but a crafted /pricing?success=1&return_to=https://evil link must not redirect.
        return_to = _safe_local_url(return_to, "/pricing")
        safe_return = html.escape(return_to) if return_to else ""
        viewer_user_id = (
            _session.get("viewer_user_id")
            or _session.get("viewer_username")
            or (("acct:" + str(_session.get("account_id")).strip()) if _session.get("account_id") else "")
        )
        return f"""
    <div class="card central" style="max-width:560px;text-align:center;">
      <div class="card-body" style="padding:48px 32px;">
        <div id="sub-icon" style="font-size:56px;margin-bottom:20px;">
          <i class="fa-solid fa-circle-check" style="color:#22c55e;"></i>
        </div>
        <h2 id="sub-heading" style="margin:0 0 10px;font-size:24px;">Payment confirmed!</h2>
        <p id="sub-msg" style="color:var(--text-muted);margin:0 0 28px;">
          Activating your premium access&hellip;
        </p>
        <p id="sub-billing" style="display:none;color:var(--text);font-weight:700;margin:-18px 0 28px;font-size:15px;"></p>
        <div id="sub-spinner" style="margin:0 auto 16px;width:32px;height:32px;border:3px solid #e5e7eb;border-top-color:#2563eb;border-radius:50%;animation:paywall-spin .8s linear infinite;"></div>
        <div id="sub-invite" style="display:none;text-align:left;margin:0 0 20px;padding:16px;border:1px solid var(--border);border-radius:12px;background:var(--bg-alt, #f8fafc);">
          <div style="font-size:14px;font-weight:700;margin-bottom:6px;">PRO is on for your league</div>
          <p style="margin:0 0 12px;font-size:13px;color:var(--text-muted);line-height:1.5;">
            Share this link so every manager can sign in and unlock the same tools.
          </p>
          <div style="display:flex;gap:8px;align-items:stretch;">
            <input id="sub-invite-url" type="text" readonly
              style="flex:1;min-width:0;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--card);font-size:12px;color:var(--text);"/>
            <button type="button" id="sub-invite-copy"
              style="flex-shrink:0;padding:10px 14px;border-radius:8px;border:none;background:#2563eb;color:#fff;font-weight:700;font-size:13px;cursor:pointer;">
              Copy invite
            </button>
          </div>
          <p id="sub-invite-copied" style="display:none;margin:8px 0 0;font-size:12px;color:#16a34a;">Invite link copied.</p>
        </div>
        <a id="sub-return" href="{safe_return or '/pricing'}" style="display:none;margin-top:8px;padding:12px 28px;border-radius:9px;background:linear-gradient(135deg,#122d4b,#2563eb);color:white;font-weight:700;text-decoration:none;font-size:15px;">Continue to dashboard</a>
        <button type="button" id="sub-portal" style="display:none;margin-top:8px;padding:12px 28px;border-radius:9px;border:1px solid var(--border);background:var(--card);color:var(--text);font-weight:700;font-size:15px;cursor:pointer;">Manage subscription</button>
      </div>
    </div>
    <script>
    (function() {{
      var returnTo = {json.dumps(return_to)};
      var userId   = {json.dumps(viewer_user_id)};
      var platform = {json.dumps(request.args.get("platform") or "")};
      var attempts = 0, maxAttempts = 8;

      var leagueId = '';
      var season = '';
      try {{
        if (returnTo) {{
          var parts = new URL(returnTo, window.location.origin).pathname.split('/').filter(Boolean);
          if (parts.length >= 3) {{
            platform = parts[0];
            season = parts[1];
            leagueId = parts[2];
          }} else if (parts.length >= 1) {{
            platform = parts[0];
          }}
        }}
      }} catch(e) {{}}

      var params = [];
      if (userId)   params.push('user_id='   + encodeURIComponent(userId));
      if (leagueId) params.push('league_id=' + encodeURIComponent(leagueId));
      if (platform) params.push('platform=' + encodeURIComponent(platform));
      if (season)   params.push('season=' + encodeURIComponent(season));
      var statusUrl = '/api/subscription-status' + (params.length ? '?' + params.join('&') : '');

      function inviteUrl() {{
        if (!leagueId || !season || !platform) return '';
        return window.location.origin + '/invite/' + encodeURIComponent(platform) + '/'
          + encodeURIComponent(season) + '/' + encodeURIComponent(leagueId);
      }}

      function showInvitePanel() {{
        var panel = document.getElementById('sub-invite');
        var input = document.getElementById('sub-invite-url');
        var copyBtn = document.getElementById('sub-invite-copy');
        var url = inviteUrl();
        if (!panel || !input || !url) return false;
        input.value = url;
        panel.style.display = 'block';
        if (copyBtn && !copyBtn.dataset.bound) {{
          copyBtn.dataset.bound = '1';
          copyBtn.addEventListener('click', function() {{
            var done = function() {{
              var note = document.getElementById('sub-invite-copied');
              if (note) note.style.display = 'block';
              copyBtn.textContent = 'Copied';
            }};
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(url).then(done).catch(function() {{
                input.select(); document.execCommand('copy'); done();
              }});
            }} else {{
              input.select(); document.execCommand('copy'); done();
            }}
          }});
        }}
        return true;
      }}

      function redirect() {{
        window.location.href = returnTo || '/pricing';
      }}

      function billingLabel(d) {{
        var interval = d.billing_interval === 'month' ? 'Monthly' : 'Annual';
        var renews = '';
        if (d.expires_at) {{
          try {{
            renews = ', renews ' + new Date(d.expires_at).toLocaleDateString(
              'en-US', {{ month: 'short', day: 'numeric' }});
          }} catch (e) {{}}
        }}
        return 'PRO ' + interval + renews;
      }}

      function finishActive(msg, leaguePlan, d) {{
        document.getElementById('sub-spinner').style.display = 'none';
        document.getElementById('sub-msg').textContent = msg;
        var bill = document.getElementById('sub-billing');
        if (bill && d) {{
          bill.textContent = billingLabel(d);
          bill.style.display = 'block';
        }}
        var btn = document.getElementById('sub-return');
        var showedInvite = showInvitePanel();
        if (btn) btn.style.display = 'inline-block';
        var portalBtn = document.getElementById('sub-portal');
        if (portalBtn) {{
          portalBtn.style.display = 'inline-block';
          if (!portalBtn.dataset.bound) {{
            portalBtn.dataset.bound = '1';
            portalBtn.addEventListener('click', function() {{
              portalBtn.disabled = true;
              var body = {{platform: platform}};
              if (leagueId) body.league_id = leagueId;
              fetch('/api/create-portal-session', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(body)
              }})
              .then(function(r) {{ return r.json().then(function(d) {{ return {{status: r.status, body: d}}; }}); }})
              .then(function(res) {{
                if (res.status === 200 && res.body && res.body.url) {{ window.location.href = res.body.url; return; }}
                document.getElementById('sub-msg').textContent =
                  (res.body && res.body.error) || 'Could not open the subscription portal. Please try again.';
                portalBtn.disabled = false;
              }})
              .catch(function() {{
                document.getElementById('sub-msg').textContent = 'Could not open the subscription portal. Please try again.';
                portalBtn.disabled = false;
              }});
            }});
          }}
        }}
        // League buyers already got the invite moment here -- tag the dashboard
        // welcome as the feature CTA (no second invite), and suppress the
        // floating invite banner on arrival.
        if (leaguePlan) {{
          try {{ sessionStorage.setItem('br_skip_league_pro_banner', '1'); }} catch (e) {{}}
          try {{
            if (returnTo) {{
              var u = new URL(returnTo, window.location.origin);
              u.searchParams.set('new_subscriber', '1');
              u.searchParams.set('welcome', 'league');
              returnTo = u.pathname + u.search;
              if (btn) btn.setAttribute('href', returnTo);
            }}
          }} catch (e) {{}}
        }}
        if (!showedInvite) setTimeout(redirect, 800);
      }}

      function activate() {{
        attempts++;
        fetch(statusUrl)
          .then(function(r) {{ return r.json(); }})
          .then(function(d) {{
            if (d.has_premium) {{
              var leaguePlan = !!(d.has_league_subscription);
              finishActive(leaguePlan
                ? 'PRO is active for your league.'
                : 'Premium is active - taking you there now!', leaguePlan, d);
              if (!leaguePlan) setTimeout(redirect, 800);
            }} else if (attempts < maxAttempts) {{
              setTimeout(activate, 1000);
            }} else {{
              // Grant may be on its way via webhook - show continue anyway
              finishActive('Access granted! If features take a moment to appear, try refreshing.', !!leagueId, d);
              if (!leagueId) setTimeout(redirect, 2000);
            }}
          }})
          .catch(function() {{
            if (attempts < maxAttempts) setTimeout(activate, 1000);
            else setTimeout(redirect, 1000);
          }});
      }}

      // Start quickly - grant was applied server-side before page rendered
      setTimeout(activate, 400);
    }})();
    </script>
    """

    selected_plan = plan if plan in {"starter", "all_pro", "hall_of_fame"} else ""
    plans = [
        ("starter", "Starter", "$10/year", "$1.49/mo", "PRO for you in 1 league of your choice. Pick it after checkout and change it anytime.", "Choose Starter", False),
        ("all_pro", "All-Pro", "$30/year", "$4.49/mo", "PRO for you in up to 5 leagues. Pick them after checkout and change them anytime.", "Choose All-Pro", True),
        ("hall_of_fame", "Hall of Fame", "$50/year", "$7.49/mo", "PRO for you in every league you play. Your league mates are not upgraded.", "Choose Hall of Fame", False),
    ]

    def _plan_leagues_label(plan_key: str) -> str:
        """League-count line under the plan name (from the backend slot cap)."""
        cap = slot_cap_for_plan(plan_key)
        if not cap:
            return "Unlimited leagues"
        return f"{cap} league{'s' if cap != 1 else ''}"

    plan_cards = "".join(
        f'''<article class="pricing-option{' featured' if recommended else ''}{' is-selected' if selected_plan == key else ''}" data-plan-card="{key}">
          <div class="pricing-header"><h3>{name}</h3>{'<span class="pricing-badge">Recommended</span>' if recommended else ''}</div>
          <p class="pricing-leagues">{_plan_leagues_label(key)}</p>
          <div class="pricing-price">
            <span class="pp-price pp-annual">{annual_price.replace('/year', '<span>/year</span>')}</span>
            <span class="pp-price pp-monthly">{monthly_price.replace('/mo', '<span>/mo</span>')}</span>
          </div>
          <p class="pp-save">Save {_annual_savings_pct(key)}% with annual billing</p>
          <p class="pricing-desc">{coverage}</p>
          <button type="button" class="btn {'btn-primary' if recommended else 'btn-secondary'} paywall-cta" onclick="initiatePurchase('{key}', this)">{cta}</button>
        </article>'''
        for key, name, annual_price, monthly_price, coverage, cta, recommended in plans
    )
    canceled_banner = """
    <div class="pricing-alert" role="status"><i class="fa-solid fa-circle-xmark" aria-hidden="true"></i>
      Checkout was canceled. You have not been charged.
    </div>""" if canceled else ""

    # Manage-subscription banner: only renders when a Stripe customer resolves
    # for the viewer (buyer-only on shared league rows). Silent otherwise.
    portal_banner = _manage_subscription_banner(league_id, platform)

    # PRO free trial: one-click start from the pricing page. ?trial=<flag> is
    # set by /pro-trial/start after it runs.
    trial_flag = request.args.get("trial", "").strip()
    trial_notice = ""
    if trial_flag == "started":
        trial_notice = """
    <div class="pricing-alert pricing-alert-success" role="status"><i class="fa-solid fa-circle-check" aria-hidden="true"></i>
      Your 7-day PRO trial is on. Full PRO access, no card required.
    </div>"""
    elif trial_flag == "already_used":
        trial_notice = """
    <div class="pricing-alert" role="status"><i class="fa-solid fa-circle-info" aria-hidden="true"></i>
      This account already used its free trial. Pick a plan below to keep PRO.
    </div>"""
    elif trial_flag == "already-pro":
        trial_notice = """
    <div class="pricing-alert pricing-alert-success" role="status"><i class="fa-solid fa-circle-check" aria-hidden="true"></i>
      You already have PRO access. Nothing to start.
    </div>"""
    elif trial_flag == "error":
        trial_notice = """
    <div class="pricing-alert" role="status"><i class="fa-solid fa-circle-xmark" aria-hidden="true"></i>
      The trial could not start. Please try again.
    </div>"""

    # Trial CTA card: shown to guests and to signed-in users who never used
    # the trial and are not already PRO. Active trials get a status card.
    trial_state = None
    trial_is_pro = False
    try:
        from dashboard_services import subscriptions as _subs
        _trial_acct = _session.get("account_id")
        _plat = (request.args.get("platform") or "sleeper").strip().lower()
        if _plat not in _SUPPORTED_PLATFORMS:
            _plat = "sleeper"
        trial_is_pro = bool(_subs.has_premium_for_viewer(
            _session.get("viewer_username"), _session.get("viewer_user_id"),
            None, _plat, None,
        ))
        if _trial_acct:
            trial_state = _subs.get_trial_state_for_keys(
                [f"acct:{_trial_acct}", str(_trial_acct)])
    except Exception:
        logger.debug("pricing trial state failed", exc_info=True)

    trial_active = bool(trial_state and trial_state.get("active"))
    trial_used = bool(trial_state and trial_state.get("used"))
    trial_section = ""
    if trial_active:
        _days = trial_state.get("days_left") or 1
        _unit = "day" if _days == 1 else "days"
        trial_section = f"""
      <section class="pricing-section" aria-label="PRO trial status">
        <div class="pricing-trial-card pricing-trial-active">
          <div>
            <h2>Your PRO trial is active</h2>
            <p>Trial ends in {_days} {_unit}. Full PRO is on while it lasts.</p>
          </div>
          <span class="pricing-trial-count" aria-hidden="true">{_days}d</span>
        </div>
      </section>"""
    elif not trial_used and not trial_is_pro:
        trial_section = """
      <section class="pricing-section" aria-label="PRO free trial">
        <div class="pricing-trial-card">
          <div>
            <h2>Try PRO free for 7 days</h2>
            <p>Full PRO access across everything. No card required. One trial per account.</p>
          </div>
          <a class="btn btn-primary" href="/pro-trial/start?next=/pricing">Start free trial</a>
        </div>
      </section>"""

    winback_status = request.args.get("winback", "")
    winback_banner = ""
    if winback_status in ("invalid", "active", "unavailable", "error"):
        _wb_msgs = {
            "invalid": "That offer link is invalid or expired.",
            "active": "You already have active PRO. No need for the comeback offer.",
            "unavailable": "That offer is no longer available.",
            "error": "Something went wrong starting checkout. Please try again.",
        }
        winback_banner = f"""
    <div class="pricing-alert" role="status"><i class="fa-solid fa-circle-exclamation" aria-hidden="true"></i>
      {html.escape(_wb_msgs[winback_status], quote=False)}
    </div>"""
    return f"""
    <main class="pricing-page">
      {canceled_banner}
      {trial_notice}
      {portal_banner}
      {winback_banner}
      {_pricing_manage_card()} (feat(churn): dunning, cancel save flow, trial reminders, win-back)
      <header class="pricing-hero">
        <span class="pricing-eyebrow">BR Fantasy PRO</span>
        <h1>Make the next move with confidence.</h1>
        <p>Turn your roster, market activity, and league outlook into clearer trade, waiver, weekly, and draft decisions.</p>
      </header>
      {trial_section}

      <section class="pricing-section pricing-plans" aria-labelledby="pricing-plans-title">
        <div class="pricing-section-heading"><h2 id="pricing-plans-title">Choose your PRO plan</h2><p>Billed monthly or annually. Annual billing saves 44%.</p></div>
        <div class="billing-toggle" role="group" aria-label="Billing interval">
          <button type="button" class="billing-toggle-btn" data-billing-interval="month" aria-pressed="false">Monthly</button>
          <button type="button" class="billing-toggle-btn is-active" data-billing-interval="year" aria-pressed="true">Annual<span class="billing-toggle-save">save 44%</span></button>
        </div>
        <div class="pricing-plan-grid" data-billing="year">{plan_cards}</div>
        <aside class="pricing-proof" aria-label="What managers say">
          <blockquote>THATS ACTUALLY SO SICK BRO</blockquote>
          <p>Jayden Waddell, Pittsburgh Pilots, on the weekly recap</p>
        </aside>
        <p class="pricing-auth-note"><i class="fa-brands fa-google" aria-hidden="true"></i> Google sign-in is required to subscribe and keep access with your account.</p>
      </section>
      <script>
      (function () {{
        // Guest-safe: no network, just swaps the displayed prices and the
        // interval paywall.js sends to /api/create-checkout-session.
        function setBillingInterval(interval) {{
          interval = interval === 'month' ? 'month' : 'year';
          window.__brBillingInterval = interval;
          document.querySelectorAll('[data-billing-interval]').forEach(function (btn) {{
            var active = btn.getAttribute('data-billing-interval') === interval;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
          }});
          document.querySelectorAll('.pricing-plan-grid').forEach(function (grid) {{
            grid.setAttribute('data-billing', interval);
          }});
        }}
        window.brSetBillingInterval = setBillingInterval;
        document.querySelectorAll('[data-billing-interval]').forEach(function (btn) {{
          btn.addEventListener('click', function () {{
            setBillingInterval(btn.getAttribute('data-billing-interval'));
          }});
        }});
        setBillingInterval(window.__brBillingInterval || 'year');
      }})();
      </script>

      <section class="pricing-section" aria-labelledby="pro-includes-title">
        <div class="pricing-section-heading"><h2 id="pro-includes-title">What PRO includes</h2><p>Decision support built around the teams and leagues your plan covers.</p></div>
        <div class="pricing-benefit-groups">
          <article><i class="fa-solid fa-handshake" aria-hidden="true"></i><h3>Trade decisions</h3><p>Roster-based suggestions, Trade Intel, AI trade analysis and counters, and playoff-impact simulations.</p></article>
          <article><i class="fa-solid fa-fire" aria-hidden="true"></i><h3>Player discovery</h3><p>Breakout Engine opportunity signals, historical peers, and confidence-adjusted projections.</p></article>
          <article><i class="fa-solid fa-calendar-week" aria-hidden="true"></i><h3>Weekly guidance</h3><p>Front Office Report, the premium AI weekly recap storyline, and cross-league “This week’s moves.” The cross-league digest requires Hall of Fame coverage across your leagues.</p></article>
          <article><i class="fa-solid fa-clipboard-list" aria-hidden="true"></i><h3>Draft tools</h3><p>Custom Draft Board, Trend Scout, and Draft Deep Dive.</p></article>
        </div>
      </section>

      <section class="pricing-section" aria-labelledby="previews-title">
        <div class="pricing-section-heading"><h2 id="previews-title">See the value in context</h2><p>Illustrative examples only. Not your league results or current player recommendations.</p></div>
        <div class="pricing-preview-grid">
          <article class="pricing-preview"><span>Sample preview</span><i class="fa-solid fa-right-left" aria-hidden="true"></i><h3>Roster-fit trade</h3><strong>Turn surplus WR depth into a starting RB</strong><p>Your receiver room can absorb the loss; the return fills the clearest weekly lineup gap without concentrating too much value in one asset.</p></article>
          <article class="pricing-preview"><span>Sample preview</span><i class="fa-solid fa-arrow-trend-up" aria-hidden="true"></i><h3>Breakout analysis</h3><strong>Opportunity: expanding · Confidence: medium</strong><p>Vacated targets and a clearer route to snaps create upside, while a small sample keeps confidence measured.</p></article>
          <article class="pricing-preview"><span>Sample preview</span><i class="fa-solid fa-newspaper" aria-hidden="true"></i><h3>Weekly storyline</h3><strong>A narrow win reshapes the race</strong><p>Your flex delivered late, moving the team above .500 as next week’s matchup puts the final playoff spot in reach.</p></article>
        </div>
      </section>

      <section class="pricing-section" aria-labelledby="free-title">
        <div class="pricing-section-heading"><h2 id="free-title">What remains free</h2><p>Upgrade only when the premium guidance fits your game.</p></div>
        {_pricing_features_grid(_FREE_FEATURES, free=True)}
        <p class="pricing-free-note">ADP rankings, basic player data, and the non-storyline sections of Weekly Recap also remain free.</p>
      </section>

      <section class="pricing-section pricing-faq" aria-labelledby="faq-title">
        <div class="pricing-section-heading"><h2 id="faq-title">Plan coverage FAQ</h2></div>
        <details><summary>How do Starter and All-Pro league slots work?</summary><p>After checkout you pick which leagues get PRO: 1 league on Starter, up to 5 on All-Pro. You can change your picks anytime from your PRO settings, and PRO applies only to you in those leagues.</p></details>
        <details><summary>Does Hall of Fame cover my league mates?</summary><p>No. Hall of Fame gives only you PRO in every league you play. Your league mates need their own plan.</p></details>
        <details><summary>How does monthly billing work?</summary><p>Choose Monthly above any plan. You are billed each month instead of once a year, and you can switch intervals or cancel from the manage-subscription link after checkout. Interval changes use Stripe's default proration.</p></details>
        <details><summary>Is the whole Weekly Recap premium?</summary><p>No. The AI-written storyline is premium; the recap’s other available sections remain free.</p></details>
        <details><summary>How does PRO billing work?</summary><p>PRO renews automatically at the then-current price until you cancel. Annual plans bill once a year; monthly plans bill each month.</p></details>
        <details><summary>How do I cancel?</summary><p>Cancel anytime through the subscription management link in your account. Your PRO access continues until the end of the current billing term.</p></details>
        <details><summary>Can I get a refund?</summary><p>Charges are non-refundable except as required by law. Canceling stops future renewals but does not refund the current term.</p></details>
      </section>
    </main>
    """

# ── Pricing pages ─────────────────────────────────────────────────────────────

# ── League PRO invite landing ─────────────────────────────────────────────────

@billing_bp.route("/invite/<platform>/<int:season>/<league_id>")
def page_league_pro_invite(platform: str, season: int, league_id: str):
    """Shareable invite for a league-plan unlock.

    Signed-in visitors go straight to the league dashboard. Guests get a short
    landing that stores the target league and points them at Identify / Connect.
    """
    from flask import redirect
    from utils.league_invite import (
        dashboard_after_invite,
        league_invite_path,
        normalize_invite_platform,
    )

    platform = normalize_invite_platform(platform)
    league_id = str(league_id or "").strip()
    if not league_id or platform not in _SUPPORTED_PLATFORMS:
        return redirect("/pricing")

    dest = dashboard_after_invite(platform, season, league_id)
    # Remember the target so Identify / Connect can land here after sign-in.
    session["invite_platform"] = platform
    session["invite_season"] = int(season)
    session["invite_league_id"] = league_id
    session["last_platform"] = platform
    session["last_season"] = int(season)
    session["last_league_id"] = league_id

    signed_in = bool(
        session.get("account_id")
        or session.get("viewer_username")
        or session.get("viewer_user_id")
    )
    if signed_in:
        return redirect(dest)

    from app import render_page

    invite_path = league_invite_path(platform, season, league_id)
    # Platform-specific connect CTAs; Sleeper can identify by username on home.
    if platform == "sleeper":
        primary_href = "/?invite=1"
        primary_label = "Sign in with Sleeper"
        secondary = (
            '<a href="/auth/google?next='
            + html.escape(invite_path, quote=True)
            + '" style="display:inline-block;margin-top:10px;font-size:13px;color:var(--accent);">'
            "Or continue with Google</a>"
        )
    elif platform == "espn":
        primary_href = f"/espn/{season}/{league_id}/dashboard"
        primary_label = "Connect ESPN league"
        secondary = ""
    else:
        primary_href = f"/{platform}/{season}/{league_id}/dashboard"
        primary_label = f"Open {platform.upper()} league"
        secondary = ""

    body = f"""
    <div class="card central" style="max-width:520px;text-align:center;">
      <div class="card-body" style="padding:40px 28px;">
        <div style="font-size:40px;margin-bottom:14px;"><i class="fa-solid fa-unlock" style="color:#2563eb;"></i></div>
        <h1 style="margin:0 0 10px;font-size:22px;">Your league unlocked PRO</h1>
        <p style="margin:0 0 22px;color:var(--text-muted);font-size:14px;line-height:1.55;">
          A league mate already paid for shared premium. Sign in as a manager in this
          league to use Trade Intel, Breakouts, Front Office, and the rest of PRO.
        </p>
        <a href="{html.escape(primary_href, quote=True)}"
           style="display:inline-block;padding:12px 22px;border-radius:9px;background:linear-gradient(135deg,#122d4b,#2563eb);color:#fff;font-weight:700;text-decoration:none;font-size:14px;">
          {html.escape(primary_label)}
        </a>
        {secondary}
      </div>
    </div>
    """
    return render_page(
        "League PRO Invite | BR Fantasy",
        None, "pricing", body, platform, season,
        description="Join your league's shared BR Fantasy PRO access.",
        noindex=True,
        lite_js=True,
    )


@billing_bp.route("/<platform>/<int:season>/<league_id>/pricing")
def page_pricing(platform: str, season: int, league_id: str):
    from app import render_page
    _try_grant_from_stripe_success()
    body_html = _pricing_body(league_id, platform)
    # active="pricing" keeps AdSense off this checkout/utility page.
    # lite_js: guests get public.js + seo_lite.css (pricing is card chrome only);
    # signed-in visitors still receive full app.js / dashboard.css via render_page.
    return render_page(
        "Pricing", league_id, "pricing", body_html, platform, season,
        lite_js=True,
    )


@billing_bp.route("/pricing")
def page_pricing_guest():
    from app import get_nfl_state, render_page
    _try_grant_from_stripe_success()
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    platform = _request_platform()
    if platform not in _SUPPORTED_PLATFORMS:
        platform = "sleeper"
    body_html = _pricing_body(None, platform)
    return render_page(
        "Pricing", None, "pricing", body_html, platform, current_season,
        lite_js=True,
    )


@billing_bp.route("/api/pro-signup/pending", methods=["POST"])
def pro_signup_pending():
    """Stage plan + league from the home-page PRO wizard, then send guests to Google."""
    data = request.get_json(force=True) or {}
    pending, error = _normalize_pending_checkout(data)
    if error or not pending:
        return jsonify({"ok": False, "error": error or "Could not save this signup."}), 400
    session["pending_checkout"] = {
        "plan": pending["plan"],
        "platform": pending["platform"],
        "league_id": pending["league_id"],
        "season": pending["season"],
    }
    session["pending_link"] = {
        "platform": pending["platform"],
        "league_id": pending["league_id"],
        "season": pending["season"],
        "team_id": pending.get("team_id"),
        "name": pending.get("name"),
        "username": pending.get("username"),
    }
    return jsonify({
        "ok": True,
        "auth_url": "/auth/google?intent=onboarding&next=/pro/resume-checkout",
    })


@billing_bp.route("/pro/resume-checkout")
def resume_pro_checkout():
    """After Google (or Yahoo) returns, open Stripe for the staged home PRO signup."""
    from flask import redirect

    pending = session.get("pending_checkout")
    if not isinstance(pending, dict) or not pending.get("plan"):
        return redirect("/pricing")

    if not session.get("account_id"):
        return redirect("/auth/google?intent=onboarding&next=/pro/resume-checkout")
    user_id = _checkout_user_id()
    if not user_id:
        return redirect("/auth/google?intent=onboarding&next=/pro/resume-checkout")

    plan = str(pending.get("plan") or "").strip()
    league_id = str(pending.get("league_id") or "").strip()
    platform = str(pending.get("platform") or "sleeper").strip().lower()
    try:
        season = int(pending.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    if plan not in _KNOWN_PLANS or platform not in _SUPPORTED_PLATFORMS:
        session.pop("pending_checkout", None)
        return redirect("/pricing")
    if plan in _LEGACY_LEAGUE_REQUIRED_PLANS and not league_id:
        return redirect("/pricing")

    if plan in _LEGACY_MEMBERSHIP_REQUIRED_PLANS and league_id:
        from dashboard_services.subscriptions import viewer_is_league_member
        member_id = session.get("viewer_user_id") or session.get("viewer_username")
        if not viewer_is_league_member(member_id, league_id, platform, season):
            return redirect("/pricing?canceled=1")

    return_url = (
        f"/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/dashboard?new_subscriber=1&welcome=personal"
        if league_id else "/pricing?success=1"
    )
    payload = {
        "plan": plan,
        "league_id": league_id,
        "platform": platform,
        "season": season,
        "return_url": return_url,
    }
    url, error = _stripe_checkout_url(
        user_id, payload, _allow_legacy=plan in _LEGACY_PRICES,
    )
    if url:
        session.pop("pending_checkout", None)
        return redirect(url)
    logger.warning("[checkout] resume failed: %s", error)
    return redirect("/pricing?canceled=1")


# ── PRO free trial ────────────────────────────────────────────────────────────
# One click from the paywall/pricing page. No card: Stripe is not involved at
# all (there is nothing to charge). Guests bounce through Google sign-in first
# (the trial key is the Google account, same rule as paid checkout), then land
# back here and the trial starts automatically.

@billing_bp.route("/pro-trial/start")
def pro_trial_start():
    """Start the 7-day no-card PRO trial, then redirect to `next`.

    `next` carries one of ?trial=started | already_used | already-pro | error.
    """
    from flask import redirect
    from dashboard_services.subscriptions import (
        has_premium_for_viewer,
        start_pro_trial_for_account,
    )

    next_url = _safe_local_url(request.args.get("next") or "", "/pricing")
    account_id = session.get("account_id")
    if not account_id:
        inner = "/pro-trial/start?next=" + urllib.parse.quote(next_url, safe="")
        return redirect(
            "/auth/google?intent=onboarding&next=" + urllib.parse.quote(inner, safe="")
        )
    try:
        acct = int(account_id)
    except (TypeError, ValueError):
        return redirect(_with_trial_flag(next_url, "error"))

    platform = _request_platform()
    if platform not in _SUPPORTED_PLATFORMS:
        platform = "sleeper"
    if has_premium_for_viewer(
        session.get("viewer_username"), session.get("viewer_user_id"),
        None, platform, None,
    ):
        # Already PRO (paid plan or live trial): nothing to start.
        return redirect(_with_trial_flag(next_url, "already-pro"))

    result = start_pro_trial_for_account(acct)
    return redirect(_with_trial_flag(next_url, result.get("code") or "error"))


def _with_trial_flag(url: str, flag: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}trial={flag}"


def _stripe_checkout_url(user_id: str, payload: dict, _allow_legacy: bool = False) -> tuple[str | None, str | None]:
    """Create a Stripe Checkout session. Returns (url, error).

    Only the current catalog is purchasable. Legacy plans are rejected
    unless ``_allow_legacy`` is set (the resume flow for sessions staged
    before the cutover); the flag is internal to this module.
    """
    plan = str(payload.get("plan") or "").strip()
    league_id = str(payload.get("league_id") or "").strip()
    platform = _request_platform(payload)
    try:
        season = int(payload.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return None, "Invalid season"
    return_url = str(payload.get("return_url") or "").strip()
    if plan not in _STRIPE_PRICES and not (_allow_legacy and plan in _LEGACY_PRICES):
        return None, "Invalid plan"
    if platform not in _SUPPORTED_PLATFORMS:
        return None, "Invalid platform"

    base_url = request.host_url.rstrip("/")
    return_url = _safe_local_url(return_url, "")
    success_url = base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}"
    if return_url:
        success_url += "&return_to=" + urllib.parse.quote(return_url, safe="")
    success_url += "&platform=" + urllib.parse.quote(platform, safe="")
    if league_id:
        cancel_url = f"{base_url}/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/pricing?canceled=1"
    else:
        cancel_url = base_url + "/pricing?canceled=1&platform=" + urllib.parse.quote(platform, safe="")
    interval = _normalize_interval(payload.get("interval"))
    line_item = _checkout_line_item(plan, interval)
    meta = _checkout_metadata(plan, user_id, league_id, platform, season, interval)
    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[line_item],
            automatic_tax={"enabled": True},
            success_url=success_url,
            cancel_url=cancel_url,
            metadata=meta,
            subscription_data={"metadata": meta},
        )
        return checkout.url, None
    except Exception:
        logger.exception("[stripe] checkout session error")
        return None, "Internal error"


# ── Stripe API endpoints ──────────────────────────────────────────────────────

@billing_bp.route("/api/create-checkout-session", methods=["POST"])
def create_checkout_session():
    # New subscriptions require a Google account. Existing rows keyed by a
    # username remain readable through the entitlement resolver. Google-only
    # managers have account_id without a Sleeper viewer id.
    blocked = _require_google_to_subscribe()
    if blocked:
        return blocked
    user_id = (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )
    logger.info("[checkout] Request from user: %s", user_id)
    if not user_id:
        return jsonify({"error": "Sign in with Google to subscribe."}), 401

    payload    = request.get_json(force=True)
    plan       = str(payload.get("plan") or "").strip()
    league_id  = str(payload.get("league_id") or "").strip()
    return_url = str(payload.get("return_url") or "").strip()
    platform   = _request_platform(payload)
    interval   = _normalize_interval(payload.get("interval"))
    try:
        season = int(payload.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid season"}), 400
    
    logger.info("[checkout] Request payload: plan=%s, league_id=%s, return_url=%s", plan, league_id, return_url)

    if plan not in _STRIPE_PRICES:
        logger.info("[checkout] Invalid plan: %s, available plans: %s", plan, list(_STRIPE_PRICES.keys()))
        return jsonify({"error": "Invalid plan"}), 400
    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"error": "Invalid platform"}), 400
    if plan in _LEAGUE_REQUIRED_PLANS and not league_id:
        return jsonify({"error": "Choose a league before purchasing this plan."}), 400

    # League/combo/single_league require membership. Shared plans unlock
    # co-managers; single_league is buyer-only but still must be a real league
    # the buyer belongs to.
    if plan in _MEMBERSHIP_REQUIRED_PLANS and league_id:
        from dashboard_services.subscriptions import viewer_is_league_member
        member_id = session.get("viewer_user_id") or session.get("viewer_username")
        if not viewer_is_league_member(member_id, league_id, platform, season):
            return jsonify({
                "error": "You must be a member of this league to purchase a league plan."
            }), 403

    # A hall_of_fame purchase is redundant only when the buyer already has
    # unlimited personal PRO (grandfathered user plan or hall_of_fame).
    # Slot-plan holders CAN upgrade to hall_of_fame (the grant upserts their
    # row, same as the old single_league -> user upgrade path).
    # starter/all_pro are redundant when any personal row is active (stacking
    # two personal subscriptions would double-bill) or when a grandfathered
    # single-league row already covers one league.
    username = session.get("viewer_username")
    stable_id = session.get("viewer_user_id")
    account_id = session.get("account_id")
    _identity_keys = [k for k in dict.fromkeys([
        user_id,
        f"acct:{account_id}" if account_id else "",
        str(account_id) if account_id else "",
        stable_id or "",
        username or "",
    ]) if k]
    _personal = find_active_personal_plan(_identity_keys, platform)
    _personal_plan = (_personal or {}).get("plan_key")
    _has_personal_unlimited = _personal_plan in ("user", "hall_of_fame")
    _has_personal_any = _personal_plan is not None
    _has_legacy_single_any = has_any_user_league_subscription(
        user_id, platform, account_id=account_id, user_keys=_identity_keys,
    )
    duplicate = (
        (plan == "hall_of_fame" and _has_personal_unlimited)
        or (plan in ("starter", "all_pro")
            and (_has_personal_any or _has_legacy_single_any))
    )
    logger.info("[checkout] Existing personal plan=%s legacy_single=%s",
                _personal_plan, _has_legacy_single_any)
    if duplicate:
        return jsonify({"error": "You already have this premium subscription."}), 400

    line_item = _checkout_line_item(plan, interval)

    base_url   = request.host_url.rstrip("/")
    return_url = _safe_local_url(return_url, "")

    success_url = base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}"
    if return_url:
        success_url += "&return_to=" + urllib.parse.quote(return_url, safe="")
    success_url += "&platform=" + urllib.parse.quote(platform, safe="")

    if league_id:
        cancel_url = f"{base_url}/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/pricing?canceled=1"
    else:
        cancel_url = base_url + "/pricing?canceled=1&platform=" + urllib.parse.quote(platform, safe="")

    line_item = _checkout_line_item(plan, interval)

    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[line_item],
            automatic_tax={"enabled": True},
            success_url=success_url,
            cancel_url=cancel_url,
            metadata=_checkout_metadata(plan, user_id, league_id, platform, season, interval),
            # Copy onto the Subscription so invoice / subscription.created
            # events can grant One League even if session metadata is dropped.
            subscription_data={
                "metadata": _checkout_metadata(plan, user_id, league_id, platform, season, interval),
            },
        )
        return jsonify({"url": checkout.url})
    except Exception:
        logger.exception("[stripe] checkout session error")
        return jsonify({"error": "Internal error"}), 500


def _handle_subscription_lifecycle_event(s, etype: str) -> None:
    """Sync entitlement rows with customer.subscription.updated/deleted.

    Reads plan, interval, status, and cancel_at_period_end off the event's
    subscription object (no extra Stripe API calls) and reconciles the
    entitlement tables via apply_subscription_lifecycle. invoice.payment_failed
    is intentionally not handled here: PR #1958 owns dunning for that event,
    and the entitlement side is covered because a permanently failed
    subscription surfaces here as unpaid/canceled.
    """
    sub_id = _stripe_id(_stripe_field(s, "id"))
    if not sub_id:
        logger.warning("[stripe] %s without subscription id; skipping", etype)
        return
    status = str(_stripe_field(s, "status") or "").strip().lower()
    cancel_at_period_end = bool(_stripe_field(s, "cancel_at_period_end"))
    try:
        expires_at = _subscription_period_end(s)
    except Exception:
        expires_at = datetime.now(timezone.utc) + timedelta(days=32)
    meta = _metadata_dict(s)
    plan = (meta.get("plan") or "").strip()
    if plan not in _KNOWN_PLANS:
        plan = _plan_from_subscription(s)
    interval = _normalize_interval(
        _interval_from_subscription(s) or meta.get("interval") or "year"
    )
    user_id = _subscriber_user_id({
        "user_id": meta.get("user_id") or meta.get("account_id") or "",
        "account_id": meta.get("account_id") or "",
    })
    platform = (meta.get("platform") or "sleeper").strip() or "sleeper"
    league_id = (meta.get("league_id") or "").strip()
    try:
        result = apply_subscription_lifecycle(
            sub_id,
            event_type=etype,
            status=status,
            cancel_at_period_end=cancel_at_period_end,
            expires_at=expires_at,
            plan=plan,
            interval=interval,
            user_id=user_id,
            league_id=league_id,
            platform=platform,
            plan_key=plan,
        )
    except Exception:
        logger.exception("[stripe] %s lifecycle sync failed sub=%s", etype, sub_id)
        return
    logger.info(
        "[stripe] %s lifecycle sync sub=%s status=%s plan=%s interval=%s result=%s",
        etype, sub_id, status, plan or "-", interval, result,
    )


def _handle_payment_failed(sub_id: str) -> None:
    """Dunning touch 1 on a failed renewal payment. Best-effort; never raises."""
    try:
        from utils import churn as _churn
        from utils import churn_email as _churn_email
        from utils.welcome_email import resolve_account_from_subscriber

        try:
            sub = _stripe().Subscription.retrieve(sub_id)
        except Exception:
            logger.warning("[stripe] payment_failed: could not retrieve sub=%s", sub_id)
            return
        meta = _metadata_dict(sub)
        plan = (meta.get("plan") or "").strip() or _plan_from_subscription(sub)
        user_id = _subscriber_user_id({
            "user_id": meta.get("user_id") or "",
            "account_id": meta.get("account_id") or "",
        })
        account_id = None
        if user_id.startswith("acct:"):
            try:
                account_id = int(user_id.split(":", 1)[1])
            except (TypeError, ValueError):
                account_id = None
        row = resolve_account_from_subscriber(user_id=user_id, account_id=account_id)
        if not row or not (row.get("email") or "").strip():
            logger.info("[stripe] payment_failed: no account email for sub=%s user=%s",
                       sub_id, user_id)
            return
        aid = row.get("id")
        if _churn.open_dunning(sub_id, aid, row["email"].strip(), plan):
            sent = _churn_email.send_dunning_touch(
                account_id=aid, email=row["email"].strip(),
                first_name=row.get("first_name"), plan=plan, touch=1,
            )
            _churn.record_event(aid, "dunning_touch_1",
                               {"sub_id": sub_id, "plan": plan, "sent": bool(sent)})
            logger.info("[stripe] payment_failed dunning touch 1 sub=%s sent=%s", sub_id, sent)
    except Exception:
        logger.exception("[stripe] _handle_payment_failed error")


@billing_bp.route("/api/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")
    secret  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    if not secret:
        logger.error("[stripe] STRIPE_WEBHOOK_SECRET not set - webhook will always fail signature check")
        return "", 400

    try:
        event = _stripe().Webhook.construct_event(payload, sig, secret)
    except ValueError as e:
        logger.error("[stripe] webhook bad payload: %s", e)
        return "", 400
    except _stripe().SignatureVerificationError as e:
        logger.error("[stripe] webhook signature mismatch: %s", e)
        return "", 400

    etype = event["type"]
    # Observability: every verified event type is logged, handled or not.
    logger.info("[stripe] webhook event type=%s", etype)

    if etype in ("checkout.session.completed", "customer.subscription.created"):
        s         = event["data"]["object"]
        meta      = _metadata_dict(s)
        plan      = (meta.get("plan") or "").strip()
        # Prefer checkout user_id; Google-only sessions store acct identity here.
        user_id   = meta.get("user_id") or meta.get("account_id")
        user_id   = _subscriber_user_id({
            "user_id": user_id or "",
            "account_id": meta.get("account_id") or "",
        })
        platform  = meta.get("platform") or "sleeper"
        league_id = meta.get("league_id") or ""
        sub = None
        if etype == "customer.subscription.created":
            sub_id = _stripe_id(_stripe_field(s, "id"))
            cust_id = _stripe_id(_stripe_field(s, "customer"))
            sub = s
            try:
                expires_at = _subscription_period_end(sub)
            except Exception:
                expires_at = datetime.now(timezone.utc) + timedelta(days=32)
            if plan not in _KNOWN_PLANS:
                plan = _plan_from_subscription(sub)
        else:
            sub_id    = _stripe_id(_stripe_field(s, "subscription"))
            cust_id   = _stripe_id(_stripe_field(s, "customer"))
            try:
                sub = _stripe().Subscription.retrieve(sub_id) if sub_id else None
                expires_at = _subscription_period_end(sub) if sub else (
                    datetime.now(timezone.utc) + timedelta(days=32)
                )
                if plan not in _KNOWN_PLANS:
                    plan = _plan_from_subscription(sub)
            except Exception:
                expires_at = datetime.now(timezone.utc) + timedelta(days=32)

        # Interval-agnostic: Stripe's price is the source of truth (covers
        # interval switches made in the billing portal), checkout metadata
        # is the fallback for unexpanded prices.
        interval = _interval_from_subscription(sub) or (meta.get("interval") or "").strip().lower()
        interval = _normalize_interval(interval)

        _apply_plan_grant(
            plan, user_id, league_id, platform, expires_at, sub_id, cust_id,
            source=f"webhook:{etype}",
            account_id=str(meta.get("account_id") or ""),
            season=str(meta.get("season") or ""),
            interval=interval,
        )

    elif etype == "invoice.paid":
        s      = event["data"]["object"]
        sub_id = s.subscription
        if sub_id:
            try:
                from utils import churn as _churn
                _churn.resolve_dunning(_stripe_id(sub_id))
            except Exception:
                logger.debug("[stripe] resolve dunning after invoice.paid failed", exc_info=True)
            try:
                sub        = _stripe().Subscription.retrieve(sub_id)
                expires_at = _subscription_period_end(sub)
                # Interval-agnostic: renewals just extend the period, and the
                # interval column follows whatever price the subscription now
                # runs on (e.g. a monthly->annual switch in the portal).
                interval = _interval_from_subscription(sub) or "year"
                from dashboard_services.db import get_conn
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE league_subscriptions SET expires_at=%s, billing_interval=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, interval, sub_id),
                        )
                        cur.execute(
                            "UPDATE user_subscriptions SET expires_at=%s, billing_interval=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, interval, sub_id),
                        )
                        cur.execute(
                            "UPDATE user_league_subscriptions SET expires_at=%s, billing_interval=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, interval, sub_id),
                        )
            except Exception as e:
                logger.exception("[stripe] invoice.paid renewal error: %s", e)

    elif etype == "invoice.payment_failed":
        # Dunning: Stripe keeps retrying per its own schedule; we add the comms.
        # Touch 1 fires here (idempotent per failure episode); the daily churn
        # scan escalates to touch 2 after ~3 days while still past due, then
        # goes quiet. The in-app banner keeps nudging in the meantime.
        s      = event["data"]["object"]
        sub_id = _stripe_id(_stripe_field(s, "subscription"))
        if sub_id:
            _handle_payment_failed(sub_id)

    elif etype in ("customer.subscription.deleted", "customer.subscription.updated"):
        s = event["data"]["object"]
        _handle_subscription_lifecycle_event(s, etype)
        # Churn bookkeeping from #1958: entitlement cancellation itself is owned
        # by apply_subscription_lifecycle above; this only resolves dunning and
        # records the analytics event.
        if _stripe_field(s, "status") in ("canceled", "unpaid", "past_due"):
            try:
                from utils import churn as _churn
                _churn.resolve_dunning(_stripe_id(_stripe_field(s, "id")))
                if _stripe_field(s, "status") == "canceled":
                    _churn.record_event(None, "subscription_ended",
                                        {"sub_id": _stripe_id(_stripe_field(s, "id"))})
            except Exception:
                logger.debug("[stripe] churn bookkeeping on cancel failed", exc_info=True)

    return "", 200


# ── Subscription status API ───────────────────────────────────────────────────

@billing_bp.route("/api/subscription-status")
def api_subscription_status():
    """Check if user has premium access for a league."""
    from dashboard_services.subscriptions import get_subscription_info

    # Identity is taken from the session, never from a client-supplied user_id,
    # so a caller cannot enumerate other users' subscription details.
    username = session.get("viewer_username")
    stable_id = session.get("viewer_user_id")
    league_id = request.args.get("league_id")
    platform = _request_platform()

    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"has_premium": False, "subscription_type": None,
                        "error": "Invalid platform"}), 400

    try:
        # Current subscriptions use the immutable provider id. Check the legacy
        # handle only when needed so older rows remain manageable and visible.
        sub_info = get_subscription_info(stable_id or username, league_id, platform)
        if username and stable_id and not sub_info.get("has_user_subscription"):
            legacy = get_subscription_info(username, None, platform)
            if legacy.get("has_user_subscription"):
                sub_info["has_user_subscription"] = True
                sub_info["has_premium"] = True
                sub_info["subscription_type"] = (
                    "combo" if sub_info.get("has_league_subscription") else "user"
                )
                if not sub_info.get("expires_at"):
                    sub_info["expires_at"] = legacy.get("expires_at")
                    sub_info["billing_interval"] = legacy.get("billing_interval") or "year"
                sub_info["stripe_customer_id"] = (
                    sub_info.get("stripe_customer_id") or legacy.get("stripe_customer_id")
                )
        # Detailed league rows alone are not an entitlement for an unrelated
        # Sleeper user; use the same membership/account-aware gate as pages.
        sub_info["has_premium"] = has_premium_for_viewer(
            username, stable_id, league_id, platform, request.args.get("season"),
        )
        from dashboard_services.subscriptions import needs_google_link_for_pro, pro_require_google
        from utils.league_invite import is_league_plan_buyer, league_invite_path
        sub_info["needs_google_link"] = needs_google_link_for_pro(username, stable_id, platform)
        sub_info["pro_require_google"] = pro_require_google()
        buyer_id = sub_info.get("subscriber_user_id")
        viewer_ids = {
            str(username or "").strip(),
            str(stable_id or "").strip(),
            (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else ""),
            str(session.get("account_id") or "").strip(),
        }
        sub_info["is_league_buyer"] = is_league_plan_buyer(viewer_ids, buyer_id)
        try:
            season_i = int(request.args.get("season") or session.get("last_season") or 0)
        except (TypeError, ValueError):
            season_i = 0
        if sub_info.get("has_league_subscription") and league_id and season_i:
            sub_info["invite_path"] = league_invite_path(platform, season_i, league_id)
        else:
            sub_info["invite_path"] = None
        # PRO free-trial state for the client (banner countdown, paywall CTA,
        # one-time "trial ended" nudge). Trial gating itself is server-side in
        # has_premium_for_viewer; these flags are display-only.
        from dashboard_services.subscriptions import (
            _pro_trial_session_keys,
            get_trial_state_for_keys,
        )
        trial_state = get_trial_state_for_keys(_pro_trial_session_keys())
        sub_info["trial_active"] = trial_state["active"]
        sub_info["trial_ends_at"] = trial_state["ends_at"]
        sub_info["trial_days_left"] = trial_state["days_left"]
        sub_info["trial_ended"] = trial_state["just_ended"]
        sub_info["trial_available"] = (
            bool(session.get("account_id"))
            and not trial_state["used"]
            and not sub_info.get("has_premium")
        )
        # Strip internal/PII fields - the client only needs entitlement flags.
        for _k in ("stripe_customer_id", "subscriber_user_id"):
            sub_info.pop(_k, None)
        return jsonify(sub_info)
    except Exception:
        logger.exception("[api_subscription_status] Error")
        return jsonify({"has_premium": False, "subscription_type": None, "error": "Internal error"}), 500


# ── PRO league slots ──────────────────────────────────────────────────────────
# Slot-capped plans (starter: 1 league, all_pro: 5 leagues) let the buyer pick
# which leagues get PRO. These endpoints manage that selection; the cap is
# enforced server-side and every league must belong to the buyer.


def _slot_platform() -> str:
    platform = _request_platform()
    return platform if platform in _SUPPORTED_PLATFORMS else "sleeper"


def _slot_checkout_keys(user_id: str) -> list:
    """Checkout identities to look up: the checkout id plus Google account keys."""
    keys = [user_id]
    acct = session.get("account_id")
    if acct:
        for key in (f"acct:{acct}", str(acct)):
            if key not in keys:
                keys.append(key)
    for key in (session.get("viewer_user_id"), session.get("viewer_username")):
        if key and key not in keys:
            keys.append(key)
    return keys


@billing_bp.route("/api/billing/pro-leagues", methods=["GET"])
def api_pro_leagues_get():
    """Return the buyer's selected PRO leagues and the plan's slot cap.

    {"plan": "all_pro", "cap": 5, "league_ids": [...]}.
    For unlimited plans (hall_of_fame, grandfathered user) cap is null and
    league_ids is empty: slots are not needed.
    """
    user_id = _checkout_user_id()
    if not user_id:
        return jsonify({"error": "Sign in with Google to manage PRO leagues."}), 401
    platform = _slot_platform()
    plan = find_active_personal_plan(_slot_checkout_keys(user_id), platform)
    if not plan:
        return jsonify({"error": "No active PRO plan."}), 404
    plan_key = plan.get("plan_key") or ""
    cap = slot_cap_for_plan(plan_key)
    league_ids = get_pro_league_slots(plan["user_key"], platform) if cap else []
    return jsonify({"plan": plan_key, "cap": cap, "league_ids": league_ids})


@billing_bp.route("/api/billing/pro-leagues", methods=["POST"])
def api_pro_leagues_set():
    """Replace the buyer's selected PRO leagues.

    Body: {"league_ids": ["id1", "id2"]}. The plan cap is enforced server-side
    and every league must belong to the buyer, otherwise 400.
    """
    user_id = _checkout_user_id()
    if not user_id:
        return jsonify({"error": "Sign in with Google to manage PRO leagues."}), 401
    data = request.get_json(silent=True) or {}
    league_ids = data.get("league_ids")
    if not isinstance(league_ids, list):
        return jsonify({"error": "league_ids must be a list."}), 400
    platform = _slot_platform()
    plan = find_active_personal_plan(_slot_checkout_keys(user_id), platform)
    if not plan:
        return jsonify({"error": "No active PRO plan."}), 404
    plan_key = plan.get("plan_key") or ""
    cap = slot_cap_for_plan(plan_key)
    if not cap:
        return jsonify({
            "error": "Your plan covers every league. League slots are not needed.",
        }), 400
    cleaned = []
    for lid in league_ids:
        lid = str(lid or "").strip()
        if not lid:
            return jsonify({"error": "League ids must not be blank."}), 400
        if lid not in cleaned:
            cleaned.append(lid)
    if len(cleaned) > cap:
        return jsonify({
            "error": f"Your plan covers {cap} league{'s' if cap != 1 else ''}.",
        }), 400
    # Every league must belong to the buyer (fail closed).
    from dashboard_services.subscriptions import viewer_is_league_member
    try:
        season = int(session.get("last_season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    member_id = session.get("viewer_user_id") or session.get("viewer_username")
    not_member = [
        lid for lid in cleaned
        if not viewer_is_league_member(member_id, lid, platform, season)
    ]
    if not_member:
        return jsonify({"error": "One or more leagues is not on your account."}), 400
    try:
        saved = set_pro_league_slots(
            plan["user_key"], platform, cleaned,
            stripe_subscription_id=plan.get("stripe_subscription_id"),
        )
    except ValueError:
        # Safety net only: every validation above already ran. Log the detail,
        # return a generic message (site audit: billing never leaks str(e)).
        logger.exception("[pro-leagues] slot save failed")
        return jsonify({"error": "Could not save your league selection. Please try again."}), 400
    return jsonify({"ok": True, "plan": plan_key, "cap": cap, "league_ids": saved})


# ── Stripe Customer Portal ────────────────────────────────────────────────────

def _portal_viewer_ids() -> set:
    """Identity set used to match the viewer against a league plan's buyer."""
    return {
        str(session.get("viewer_username") or "").strip(),
        str(session.get("viewer_user_id") or "").strip(),
        (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else ""),
        str(session.get("account_id") or "").strip(),
    }


def _resolve_billing_customer_id(user_id, league_id=None, platform="sleeper"):
    """Resolve the Stripe customer id the current viewer may manage.

    Buyer-only for shared league rows: league mates must never be handed the
    buyer's portal session, so a league row's customer is only used when the
    viewer matches the plan's subscriber_user_id. Personal and single-league
    rows always belong to the viewer, so no buyer check is needed there.
    Returns None when no customer resolves. Never raises.
    """
    from dashboard_services.subscriptions import get_subscription_info
    from utils.league_invite import is_league_plan_buyer

    if platform not in _SUPPORTED_PLATFORMS:
        return None

    sub_info = get_subscription_info(user_id, league_id, platform)
    customer_id = None
    if league_id and sub_info.get("has_league_subscription"):
        # Shared league row: only the buyer may manage the plan.
        if is_league_plan_buyer(_portal_viewer_ids(), sub_info.get("subscriber_user_id")):
            customer_id = sub_info.get("stripe_customer_id")
    else:
        customer_id = sub_info.get("stripe_customer_id")
    # Fall back to user-only lookup (personal plan) if league lookup has no customer
    if not customer_id and league_id:
        user_sub = get_subscription_info(user_id, None, platform)
        customer_id = user_sub.get("stripe_customer_id")
    if not customer_id and session.get("viewer_username") and session.get("viewer_user_id"):
        legacy_sub = get_subscription_info(session.get("viewer_username"), None, platform)
        customer_id = legacy_sub.get("stripe_customer_id")
    if not customer_id and session.get("account_id"):
        acct_key = "acct:" + str(session.get("account_id")).strip()
        if user_id != acct_key:
            acct_sub = get_subscription_info(acct_key, None, platform)
            customer_id = acct_sub.get("stripe_customer_id")
        if not customer_id:
            bare_sub = get_subscription_info(str(session.get("account_id")).strip(), None, platform)
            customer_id = bare_sub.get("stripe_customer_id")
    return customer_id


def _portal_open_js(button_id: str, err_id: str, platform: str, league_id=None) -> str:
    """Inline JS: POST to /api/create-portal-session and redirect to the portal URL."""
    payload_parts = ["platform: " + json.dumps(platform or "")]
    if league_id:
        payload_parts.append("league_id: " + json.dumps(str(league_id)))
    payload = "{" + ", ".join(payload_parts) + "}"
    return (
        "<script>\n"
        "(function() {\n"
        "  var btn = document.getElementById(" + json.dumps(button_id) + ");\n"
        "  var err = document.getElementById(" + json.dumps(err_id) + ");\n"
        "  if (!btn || btn.dataset.portalBound) return;\n"
        "  btn.dataset.portalBound = '1';\n"
        "  btn.addEventListener('click', function() {\n"
        "    btn.disabled = true;\n"
        "    fetch('/api/create-portal-session', {\n"
        "      method: 'POST',\n"
        "      headers: {'Content-Type': 'application/json'},\n"
        "      body: JSON.stringify(" + payload + ")\n"
        "    })\n"
        "    .then(function(r) { return r.json().then(function(d) { return {status: r.status, body: d}; }); })\n"
        "    .then(function(res) {\n"
        "      if (res.status === 200 && res.body && res.body.url) { window.location.href = res.body.url; return; }\n"
        "      var msg = (res.body && res.body.error) || 'Could not open the subscription portal. Please try again.';\n"
        "      if (err) { err.textContent = msg; err.style.display = 'inline'; }\n"
        "      btn.disabled = false;\n"
        "    })\n"
        "    .catch(function() {\n"
        "      if (err) { err.textContent = 'Could not open the subscription portal. Please try again.'; err.style.display = 'inline'; }\n"
        "      btn.disabled = false;\n"
        "    });\n"
        "  });\n"
        "})();\n"
        "</script>"
    )


def _manage_subscription_banner(league_id=None, platform: str = "sleeper") -> str:
    """Render a 'Manage subscription' banner when a Stripe customer resolves.

    Buyer-only for shared league rows. Silent (no banner, no exception) for
    guests, free users, and league mates, so the pricing page never 500s.
    """
    try:
        # Mirror checkout identity: Google-only managers have account_id without
        # a Sleeper viewer id, and their Stripe rows are keyed as acct:<id>.
        user_id = (
            session.get("viewer_user_id")
            or session.get("viewer_username")
            or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
        )
        if not user_id:
            return ""
        if _resolve_billing_customer_id(user_id, league_id, platform) is None:
            return ""
    except Exception:
        logger.debug("manage-subscription banner lookup failed", exc_info=True)
        return ""
    return (
        '<div class="pricing-alert pricing-alert-success" role="status" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        '<i class="fa-solid fa-credit-card" aria-hidden="true"></i>'
        '<span>You already have PRO.</span>'
        '<button type="button" id="manage-sub-btn" class="btn btn-secondary" style="padding:6px 14px;font-size:13px;">Manage subscription</button>'
        '<span id="manage-sub-err" style="display:none;"></span>'
        "</div>"
    ) + _portal_open_js("manage-sub-btn", "manage-sub-err", platform, league_id)


@billing_bp.route("/api/create-portal-session", methods=["POST"])
def api_create_portal_session():
    """Create a Stripe billing portal session so users can manage subscriptions."""
    # Mirror checkout identity: Google-only managers have account_id without a
    # Sleeper viewer id, and their Stripe rows are keyed as acct:<id>.
    user_id = (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )
    league_id = request.json.get("league_id") if request.is_json else request.form.get("league_id")
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    platform = _request_platform(payload)

    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"error": "Invalid platform"}), 400

    try:
        customer_id = _resolve_billing_customer_id(user_id, league_id, platform)
        if not customer_id:
            return jsonify({"error": "No Stripe customer found for your account. Contact support if you believe this is an error."}), 404

        return_url = request.json.get("return_url") if request.is_json else request.form.get("return_url")
        return_url = _safe_local_url(
            return_url,
            request.host_url.rstrip("/") + f"/pricing?platform={urllib.parse.quote(platform, safe='')}",
        )

        portal_session = _stripe().billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return jsonify({"url": portal_session.url})
    except Exception:
        logger.exception("[api_create_portal_session] Error")
        return jsonify({"error": "Internal error"}), 500


# ── Cancellation save flow ────────────────────────────────────────────────────
# The pricing page's "Cancel PRO" opens a survey modal first; the final confirm
# lands here. "Pause PRO for 2 months" is offered BEFORE the final confirm.
# Pause uses Stripe's pause_collection with resumes_at (chosen over a coupon:
# one API call, auto-resumes, subscription stays active so entitlements keep
# working, nothing to reconcile later).

def _cancel_owned_sub(user_id: str, sub_id: str) -> dict | None:
    """Return the user's active sub row for sub_id, or None when not owned."""
    from utils import churn as _churn

    sub_id = (sub_id or "").strip()
    if not sub_id:
        return None
    for row in _churn.active_subscriptions_for_user(user_id):
        if row.get("stripe_subscription_id") == sub_id:
            return row
    return None


@billing_bp.route("/api/pause-subscription", methods=["POST"])
def api_pause_subscription():
    """Pause PRO for 2 months (Stripe pause_collection, auto-resumes)."""
    from utils import churn as _churn

    user_id = _checkout_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    sub_id = str(data.get("stripe_subscription_id") or "").strip()
    owned = _cancel_owned_sub(user_id, sub_id)
    if not owned:
        return jsonify({"error": "Subscription not found for your account."}), 404

    result = _churn.pause_subscription(sub_id, months=_churn.PAUSE_MONTHS)
    if not result.get("ok"):
        return jsonify({"error": result.get("error") or "Could not pause."}), 502

    account_id = None
    if user_id.startswith("acct:"):
        try:
            account_id = int(user_id.split(":", 1)[1])
        except (TypeError, ValueError):
            account_id = None
    _churn.record_event(account_id, "pause_accepted", {
        "sub_id": sub_id,
        "plan": owned.get("plan"),
        "reason": str(data.get("reason") or "")[:40] or None,
        "resumes_at": result.get("resumes_at"),
    })
    return jsonify({"ok": True, "resumes_at": result.get("resumes_at")})


@billing_bp.route("/api/cancel-subscription", methods=["POST"])
def api_cancel_subscription():
    """Record the cancel survey, then pause or cancel at period end.

    Body: {stripe_subscription_id, action: "pause"|"cancel",
           reason: <one of CANCEL_REASONS or "skipped">, reason_detail: <text>}
    The survey is optional: "skipped" is a valid reason.
    """
    from utils import churn as _churn

    user_id = _checkout_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    sub_id = str(data.get("stripe_subscription_id") or "").strip()
    owned = _cancel_owned_sub(user_id, sub_id)
    if not owned:
        return jsonify({"error": "Subscription not found for your account."}), 404

    action = str(data.get("action") or "cancel").strip().lower()
    reason = str(data.get("reason") or "skipped").strip().lower()
    if reason not in _churn.CANCEL_REASONS and reason != "skipped":
        reason = "other"
    detail = str(data.get("reason_detail") or "").strip()[:500]

    account_id = None
    if user_id.startswith("acct:"):
        try:
            account_id = int(user_id.split(":", 1)[1])
        except (TypeError, ValueError):
            account_id = None
    _churn.record_event(account_id, "cancel_survey", {
        "sub_id": sub_id,
        "plan": owned.get("plan"),
        "reason": reason,
        "reason_detail": detail or None,
        "action": action,
    })

    if action == "pause":
        result = _churn.pause_subscription(sub_id, months=_churn.PAUSE_MONTHS)
        if not result.get("ok"):
            return jsonify({"error": result.get("error") or "Could not pause."}), 502
        _churn.record_event(account_id, "pause_accepted", {
            "sub_id": sub_id, "plan": owned.get("plan"), "reason": reason,
            "resumes_at": result.get("resumes_at"),
        })
        return jsonify({"ok": True, "paused": True, "resumes_at": result.get("resumes_at")})

    result = _churn.cancel_at_period_end(sub_id)
    if not result.get("ok"):
        return jsonify({"error": result.get("error") or "Could not cancel."}), 502
    _churn.record_event(account_id, "cancel_confirmed",
                        {"sub_id": sub_id, "plan": owned.get("plan"), "reason": reason})
    expires = owned.get("expires_at")
    try:
        expires_s = expires.isoformat() if hasattr(expires, "isoformat") else str(expires or "")
    except Exception:
        expires_s = ""
    return jsonify({"ok": True, "canceled": True, "access_until": expires_s})


# ── Win-back checkout ─────────────────────────────────────────────────────────
# One-time offer email links here with a signed token (no login needed). The
# endpoint re-checks "no active sub" and applies the WINBACK_COUPON_ID Stripe
# coupon at checkout. Coupon + offer label are configured in the Stripe
# dashboard / env (see PR body).

@billing_bp.route("/pro/winback")
def pro_winback():
    from flask import redirect
    from utils import churn as _churn

    token = request.args.get("token", "")
    account_id = _churn.verify_winback_token(token)
    if not account_id:
        return redirect("/pricing?winback=invalid")
    if _churn.account_has_active_sub(account_id):
        return redirect("/pricing?winback=active")
    coupon = _churn.winback_coupon_id()
    if not coupon:
        logger.warning("[winback] WINBACK_COUPON_ID not set")
        return redirect("/pricing?winback=unavailable")

    # Win-back sells the current all-leagues plan (hall_of_fame, the
    # successor to the retired Personal plan) with the configured coupon.
    line_item = _checkout_line_item("hall_of_fame", "year")
    base_url = request.host_url.rstrip("/")
    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[line_item],
            discounts=[{"coupon": coupon}],
            success_url=base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}&winback=1",
            cancel_url=base_url + "/pricing?canceled=1",
            metadata={
                "plan": "hall_of_fame",
                "user_id": f"acct:{account_id}",
                "account_id": str(account_id),
                "winback": "1",
            },
            subscription_data={
                "metadata": {
                    "plan": "hall_of_fame",
                    "user_id": f"acct:{account_id}",
                    "account_id": str(account_id),
                    "winback": "1",
                },
            },
        )
        _churn.record_event(account_id, "winback_checkout_started", {"coupon": coupon})
        return redirect(checkout.url)
    except Exception:
        logger.exception("[winback] checkout session error")
        return redirect("/pricing?winback=error")
