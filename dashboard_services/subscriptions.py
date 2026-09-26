"""
Subscription management for league and user-based premium access.

Premium Features:
- AI insights
- Breakout candidates
- Viewable advanced metrics

Subscription types (current catalog):
- starter: personal PRO for 1 selected league (league slots, buyer-only)
- all_pro: personal PRO for up to 5 selected leagues (league slots, buyer-only)
- hall_of_fame: personal PRO across all of the buyer's leagues

Grandfathered (no longer sold, renewals and entitlements keep working):
- league: shared PRO for every manager in one league
- user: personal PRO across all of a user's leagues
- single_league: personal PRO for one selected league only (buyer-only)
- combo: league + user
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional, Dict, Any, List

import logging

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


# Tables carrying the billing_interval column (migration 038). The runner stops
# on the first failing migration file, so a non-idempotent earlier file can
# leave this column missing even after deploy; the ensure below is the same
# belt-and-braces pattern as _ensure_user_league_subscriptions_table.
_BILLING_INTERVAL_TABLES = (
    "league_subscriptions",
    "user_subscriptions",
    "user_league_subscriptions",
)
_BILLING_INTERVAL_ENSURED: set = set()


def _ensure_billing_interval(cur, table: str) -> None:
    """ADD COLUMN IF NOT EXISTS billing_interval, once per table per process."""
    if table in _BILLING_INTERVAL_ENSURED:
        return
    if table not in _BILLING_INTERVAL_TABLES:
        raise ValueError(f"unexpected subscription table: {table}")
    cur.execute(
        f"ALTER TABLE {table} "
        "ADD COLUMN IF NOT EXISTS billing_interval TEXT NOT NULL DEFAULT 'year'"
    )
    _BILLING_INTERVAL_ENSURED.add(table)


# League-slot caps for the slot-capped catalog plans (migration 039).
# hall_of_fame and the grandfathered 'user' plan are PRO everywhere, so they
# have no cap (None). Mirrored in routes/billing_bp.py as _PLAN_LEAGUE_CAP;
# keep the two in sync.
_SLOT_CAP_BY_PLAN: Dict[str, int] = {
    "starter": 1,
    "all_pro": 5,
}

_PLAN_KEY_ENSURED = False


def _ensure_plan_key(cur) -> None:
    """ADD COLUMN IF NOT EXISTS plan_key on user_subscriptions (idempotent)."""
    global _PLAN_KEY_ENSURED
    if _PLAN_KEY_ENSURED:
        return
    cur.execute(
        "ALTER TABLE user_subscriptions ADD COLUMN IF NOT EXISTS plan_key TEXT"
    )
    _PLAN_KEY_ENSURED = True


_PRO_LEAGUE_SLOTS_DDL = """
CREATE TABLE IF NOT EXISTS pro_league_slots (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'sleeper',
    league_id TEXT NOT NULL,
    stripe_subscription_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_pro_league_slot UNIQUE (user_id, platform, league_id)
)
"""


def _ensure_pro_league_slots_table(cur) -> None:
    """Create pro_league_slots if migration 039 never applied (idempotent)."""
    cur.execute(_PRO_LEAGUE_SLOTS_DDL)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pro_league_slots_user "
        "ON pro_league_slots (user_id, platform)"
    )


def slot_cap_for_plan(plan_key: Optional[str]) -> Optional[int]:
    """League-slot cap for a catalog plan: int, or None when unlimited."""
    return _SLOT_CAP_BY_PLAN.get((plan_key or "").strip().lower())


def pro_require_google() -> bool:
    """Hard cutover: user-plan PRO requires a Google ``account_id`` session.

    Soft dual-read (default): bare Sleeper viewer id/username can still unlock
    a personal subscription so existing buyers aren't locked out mid-migration.
    Set ``PRO_REQUIRE_GOOGLE=1`` after the link-Google notice period.
    """
    return os.environ.get("PRO_REQUIRE_GOOGLE", "").strip().lower() in ("1", "true", "yes")


def _session_account_id() -> Optional[int]:
    try:
        from flask import session as _session, has_request_context as _hrc
        if not _hrc():
            return None
        acct = _session.get("account_id")
        return int(acct) if acct not in (None, "") else None
    except Exception:
        return None


def viewer_has_legacy_user_subscription(
    viewer_username: Optional[str],
    viewer_user_id: Optional[str],
    platform: str = "sleeper",
) -> bool:
    """True when the Sleeper viewer identity has an active personal subscription.

    Used for the "Link Google to secure PRO" prompt -- independent of whether
    soft dual-read still grants access.
    """
    platform = platform or "sleeper"
    if viewer_user_id and has_premium_access(viewer_user_id, None, platform):
        return True
    if viewer_username and has_premium_access(viewer_username, None, platform):
        return True
    return False


def needs_google_link_for_pro(
    viewer_username: Optional[str] = None,
    viewer_user_id: Optional[str] = None,
    platform: str = "sleeper",
) -> bool:
    """Username-only session holding a user-plan sub that should link Google."""
    if _session_account_id():
        return False
    return viewer_has_legacy_user_subscription(viewer_username, viewer_user_id, platform)


def premium_required(fn):
    """Decorator that enforces premium access on a Flask route.

    Identity comes from the server-side session (``viewer_username`` /
    ``viewer_user_id``) - never from the client - while ``league_id`` /
    ``platform`` / ``season`` are read from the request (query string, form, or
    JSON body). A league subscription is only honored when the viewer is an
    actual member of that league, so a tampered ``league_id`` cannot unlock
    premium. Returns a 403 paywall response otherwise.
    """
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        from flask import request, session, jsonify

        data = request.get_json(silent=True) if request.is_json else None
        data = data or {}
        league_id = data.get("league_id") or request.values.get("league_id")
        platform = (data.get("platform") or request.values.get("platform")
                    or "sleeper")
        season = data.get("season") or request.values.get("season")

        if not has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            league_id, platform, season,
        ):
            return jsonify({"paywall": True, "error": "Premium required"}), 403
        return fn(*args, **kwargs)

    return _wrapper


def _account_user_keys(account_id: int) -> List[str]:
    """Subscription keys that may be stored for a Google account."""
    return [f"acct:{account_id}", str(account_id)]


def has_user_league_subscription(
    user_id: Optional[str],
    league_id: Optional[str],
    platform: str = "sleeper",
    account_id: Optional[int] = None,
) -> bool:
    """True when this user (or linked account) bought single-league PRO for league_id.

    Buyer-only -- co-managers are not entitled via this table.
    """
    if not league_id:
        return False
    if not user_id and not account_id:
        return False

    platform = platform or "sleeper"
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if user_id:
                    cur.execute("""
                        SELECT 1 FROM user_league_subscriptions
                        WHERE user_id = %s
                          AND platform = %s
                          AND league_id = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (user_id, platform, league_id, now))
                    if cur.fetchone():
                        return True

                if account_id:
                    keys = _account_user_keys(int(account_id))
                    cur.execute("""
                        SELECT 1 FROM user_league_subscriptions
                        WHERE user_id = ANY(%s)
                          AND platform = %s
                          AND league_id = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (keys, platform, league_id, now))
                    if cur.fetchone():
                        return True

                    # Linked platform identities on this Google account.
                    cur.execute("""
                        SELECT 1
                        FROM user_league_subscriptions uls
                        JOIN account_identities ai
                          ON ai.platform = uls.platform
                         AND (ai.platform_user_id = uls.user_id OR ai.handle = uls.user_id)
                        WHERE ai.account_id = %s
                          AND uls.platform = %s
                          AND uls.league_id = %s
                          AND uls.subscription_status = 'active'
                          AND uls.expires_at > %s
                        LIMIT 1
                    """, (account_id, platform, league_id, now))
                    if cur.fetchone():
                        return True

                return False
    except Exception as e:
        logger.error("[subscriptions] Error checking single-league access: %s", e)
        return False


def has_any_user_league_subscription(
    user_id: Optional[str],
    platform: str = "sleeper",
    account_id: Optional[int] = None,
    user_keys: Optional[List[str]] = None,
) -> bool:
    """True when the user holds any active legacy single-league row (any league).

    ``user_keys`` widens the direct identity check to every alias the
    checkout flow uses (stable viewer id, username, acct keys); the legacy
    row was granted under whichever identity paid for it.
    """
    keys = [k for k in dict.fromkeys([*(user_keys or []), user_id or ""]) if k]
    if not keys and not account_id:
        return False
    platform = platform or "sleeper"
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_user_league_subscriptions_table(cur)
                if keys:
                    cur.execute("""
                        SELECT 1 FROM user_league_subscriptions
                        WHERE user_id = ANY(%s)
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (keys, platform, now))
                    if cur.fetchone():
                        return True
                if account_id:
                    keys = _account_user_keys(int(account_id))
                    cur.execute("""
                        SELECT 1 FROM user_league_subscriptions
                        WHERE user_id = ANY(%s)
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (keys, platform, now))
                    if cur.fetchone():
                        return True
                return False
    except Exception as e:
        logger.error("[subscriptions] Error checking any single-league access: %s", e)
        return False


def has_premium_access(
    user_id: Optional[str],
    league_id: Optional[str],
    platform: str = "sleeper",
    account_id: Optional[int] = None,
) -> bool:
    """
    Check if a user has premium access for a specific league.

    Premium access is granted if ANY of:
    1. The league has an active subscription (league-based, shared)
    2. The user has an active subscription (user-based, covers all leagues)
    3. account_id is given and any platform identity linked to that account has
       an active user subscription (account-based, spans platforms)
    4. The user (or linked account) has a single-league subscription for this
       league_id (buyer-only; requires league_id)

    (3)/(4) are strictly additive: they only ever grant access, never remove it.

    Args:
        user_id: Sleeper username or user ID
        league_id: League ID
        platform: Platform name (default: 'sleeper')
        account_id: Standalone account id (optional; enables the account-based check)

    Returns:
        True if user has premium access, False otherwise
    """
    if not user_id and not league_id and not account_id:
        return False

    now = datetime.now(timezone.utc)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Belt-and-braces: plan_key / slots may be missing when
                # migration 039 has not applied yet.
                _ensure_plan_key(cur)
                _ensure_pro_league_slots_table(cur)
                # Check league subscription first (if league_id provided)
                if league_id:
                    cur.execute("""
                        SELECT 1 FROM league_subscriptions
                        WHERE league_id = %s
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (league_id, platform, now))

                    if cur.fetchone():
                        return True

                # Check user subscription (if user_id provided)
                if user_id:
                    cur.execute("""
                        SELECT plan_key FROM user_subscriptions
                        WHERE user_id = %s
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (user_id, platform, now))
                    _urow = cur.fetchone()
                    if _urow:
                        _upk = ((_urow.get("plan_key") if isinstance(_urow, dict)
                                 else _urow[0]) or "").strip().lower()
                        if _upk in _SLOT_CAP_BY_PLAN:
                            # Slot-capped plan (starter/all_pro): PRO only in
                            # the selected leagues, never account-wide.
                            if league_id and _slot_row_active(
                                cur, user_id, platform, league_id, now,
                            ):
                                return True
                        else:
                            # hall_of_fame / grandfathered 'user' / legacy NULL
                            # plan_key: personal PRO across all leagues.
                            return True

                    # PRO free trial (additive, like a personal user plan).
                    # trial_active_for_keys fails closed on DB errors and is
                    # per-request memoized; the gate below never grants on error.
                    if trial_active_for_keys([user_id]):
                        return True

                # Account-based (additive): premium on any linked platform
                # identity covers the whole account, across platforms.
                # Slot-capped plans (starter/all_pro) only grant the leagues in
                # their slots; unlimited plans (hall_of_fame, grandfathered
                # 'user') grant account-wide.
                if account_id:
                    cur.execute("""
                        SELECT us.user_id, us.plan_key
                        FROM user_subscriptions us
                        JOIN account_identities ai
                          ON ai.platform = us.platform
                         AND (ai.platform_user_id = us.user_id OR ai.handle = us.user_id)
                        WHERE ai.account_id = %s
                          AND us.subscription_status = 'active'
                          AND us.expires_at > %s
                    """, (account_id, now))
                    for _r in (cur.fetchall() or []):
                        _uk = _r.get("user_id") if isinstance(_r, dict) else _r[0]
                        _pk = ((_r.get("plan_key") if isinstance(_r, dict)
                                else _r[1]) or "").strip().lower()
                        if _pk in _SLOT_CAP_BY_PLAN:
                            if league_id and _slot_row_active(
                                cur, _uk, platform, league_id, now,
                            ):
                                return True
                        else:
                            return True

                    # Google-only checkout (no platform identity yet) stores the
                    # subscription against acct:<id> (or the bare account id).
                    cur.execute("""
                        SELECT user_id, plan_key FROM user_subscriptions
                        WHERE user_id IN (%s, %s)
                          AND subscription_status = 'active'
                          AND expires_at > %s
                    """, (f"acct:{account_id}", str(account_id), now))
                    for _r in (cur.fetchall() or []):
                        _uk = _r.get("user_id") if isinstance(_r, dict) else _r[0]
                        _pk = ((_r.get("plan_key") if isinstance(_r, dict)
                                else _r[1]) or "").strip().lower()
                        if _pk in _SLOT_CAP_BY_PLAN:
                            if league_id and _slot_row_active(
                                cur, _uk, platform, league_id, now,
                            ):
                                return True
                        else:
                            return True

                    # PRO free trial: trial rows are keyed acct:<id>, exactly
                    # like Google-only checkout rows above. A live trial grants
                    # full PRO, the same scope as a personal user plan.
                    if trial_active_for_keys([f"acct:{account_id}", str(account_id)]):
                        return True

    except Exception as e:
        logger.error("[subscriptions] Error checking premium access: %s", e)
        # Fail closed: on any error, deny premium rather than grant it.
        return False

    # Buyer-only single-league plan (own connection -- avoid nesting get_conn).
    if league_id and has_user_league_subscription(
        user_id, league_id, platform, account_id=account_id,
    ):
        return True

    # Slot-capped plans (starter/all_pro): PRO only in the selected leagues.
    if league_id and has_pro_league_slot(
        user_id, league_id, platform, account_id=account_id,
    ):
        return True

    return False


# ── League membership (guards the shared league-plan entitlement) ─────────────

_MEMBER_CACHE: Dict[Any, Any] = {}
_MEMBER_TTL = 600  # seconds


def _viewer_league_ids(viewer_user_id: str, season: int) -> set:
    """Return the set of Sleeper league_ids the user belongs to (cached)."""
    ck = (str(viewer_user_id), int(season))
    hit = _MEMBER_CACHE.get(ck)
    if hit and (time.time() - hit[1]) < _MEMBER_TTL:
        return hit[0]
    from dashboard_services.api import get_sleeper_user_leagues
    raw = get_sleeper_user_leagues(str(viewer_user_id), int(season)) or []
    ids = {str(lg.get("league_id")) for lg in raw if lg.get("league_id")}
    _MEMBER_CACHE[ck] = (ids, time.time())
    return ids


def viewer_is_league_member(
    viewer_user_id: Optional[str], league_id: Optional[str],
    platform: str = "sleeper", season: Optional[int] = None,
) -> bool:
    """Whether the given viewer actually belongs to the league.

    League-plan premium is shared across a league, so we must confirm the
    requester is a member before honoring it - otherwise anyone who knows a
    paid league's (non-secret) id would unlock premium for free.

    Sleeper membership is the live user-leagues list. ESPN/Yahoo/MFL membership
    is the durable ``user_leagues`` row on the signed-in account. Fail closed
    when we cannot confirm.
    """
    if not league_id:
        return False
    plat = (platform or "sleeper").strip().lower()
    if plat != "sleeper":
        try:
            from flask import has_request_context, session
            if not has_request_context():
                return False
            account_id = session.get("account_id")
            if not account_id:
                return False
            from dashboard_services.accounts import list_user_leagues
            lid = str(league_id)
            season_i = int(season) if season not in (None, "") else None
            for lg in list_user_leagues(int(account_id)):
                if str(lg.get("platform") or "").lower() != plat:
                    continue
                if str(lg.get("league_id")) != lid:
                    continue
                if season_i is None or int(lg.get("season") or 0) == season_i:
                    return True
            return False
        except Exception:
            return False
    if not viewer_user_id:
        return False
    try:
        season = int(season or datetime.now().year)
        return str(league_id) in _viewer_league_ids(viewer_user_id, season)
    except Exception:
        # Fail closed on the league path; a user with their own subscription is
        # unaffected (that is checked separately).
        return False


def has_premium_for_viewer(
    viewer_username: Optional[str], viewer_user_id: Optional[str],
    league_id: Optional[str], platform: str = "sleeper",
    season: Optional[int] = None,
) -> bool:
    """Premium gate that is safe against ``league_id`` tampering.

    Grant order:
      1. Google ``account_id`` personal all-leagues plan (linked / ``acct:`` rows)
      2. Shared league plan for verified members
      3. Single-league personal plan for this league (buyer-only; account or
         soft dual-read identity)
      4. Legacy Sleeper user-plan via viewer id/username -- soft dual-read only
         (disabled when ``PRO_REQUIRE_GOOGLE=1``)
    """
    platform = platform or "sleeper"
    # Per-request memoization: render_page (every server-rendered page) plus some
    # handlers call this 1-2x per request, each hitting the DB. Cache the result
    # on flask.g so a page render costs at most one premium lookup.
    _cache = None
    _require_google = pro_require_google()
    _acct = _session_account_id()
    _key = (viewer_username, viewer_user_id, league_id, platform, str(season),
            _acct, _require_google)
    try:
        from flask import g, has_request_context
        if has_request_context():
            _cache = getattr(g, "_premium_cache", None)
            if _cache is None:
                _cache = {}
                g._premium_cache = _cache
            if _key in _cache:
                return _cache[_key]
    except Exception:
        _cache = None

    result = False

    # Account-based (primary for personal all-leagues plans after Google link).
    if _acct and has_premium_access(None, None, platform, account_id=_acct):
        result = True

    # League subscription only for actual members (shared plan -- membership is
    # the guard; Google is not required).
    if not result and league_id and has_premium_access(None, league_id, platform) \
            and viewer_is_league_member(viewer_user_id, league_id, platform, season):
        result = True

    # Single-league personal plan (buyer-only for this league). Prefer the
    # Google account key; soft dual-read still honors Sleeper viewer ids.
    # Covers both the grandfathered single_league rows and the new
    # slot-capped plans (starter/all_pro) via pro_league_slots.
    if not result and league_id:
        if _acct and (
            has_user_league_subscription(
                viewer_user_id or viewer_username, league_id, platform,
                account_id=_acct,
            )
            or has_pro_league_slot(
                viewer_user_id or viewer_username, league_id, platform,
                account_id=_acct,
            )
        ):
            result = True
        elif not _require_google:
            if viewer_user_id and has_user_league_subscription(
                viewer_user_id, league_id, platform,
            ):
                result = True
            elif viewer_username and has_user_league_subscription(
                viewer_username, league_id, platform,
            ):
                result = True
            elif (viewer_user_id or viewer_username) and has_pro_league_slot(
                viewer_user_id or viewer_username, league_id, platform,
            ):
                result = True

    # Legacy Sleeper username/id personal subscription.
    # Soft dual-read: still honor so buyers aren't locked out before linking.
    # Hard cutover (PRO_REQUIRE_GOOGLE): skip -- thieves can't unlock PRO by
    # typing a username, and real buyers restore access by linking Google.
    if not result and not _require_google:
        if viewer_user_id and has_premium_access(viewer_user_id, None, platform):
            result = True
        elif viewer_username and has_premium_access(viewer_username, None, platform):
            result = True

    if _cache is not None:
        _cache[_key] = result
    return result


def get_subscription_info(user_id: Optional[str], league_id: Optional[str], platform: str = "sleeper") -> Dict[
    str, Any]:
    """
    Get detailed subscription information for a user/league.

    Returns:
        {
            "has_premium": bool,
            "subscription_type": "league" | "user" | "combo" | "single_league" | None,
            "has_league_subscription": bool,
            "has_user_subscription": bool,
            "has_single_league_subscription": bool,
            "expires_at": datetime | None,
            "subscriber_user_id": str | None  # Only for league subscriptions
        }
    """
    result = {
        "has_premium": False,
        "subscription_type": None,
        "has_league_subscription": False,
        "has_user_subscription": False,
        "has_single_league_subscription": False,
        "expires_at": None,
        "billing_interval": "year",
        "subscriber_user_id": None,
        "stripe_customer_id": None,
        "user_plan_key": None,
    }

    now = datetime.now(timezone.utc)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for _table in _BILLING_INTERVAL_TABLES:
                    _ensure_billing_interval(cur, _table)
                _ensure_plan_key(cur)
                if league_id:
                    cur.execute("""
                        SELECT expires_at, subscriber_user_id, stripe_customer_id, billing_interval
                        FROM league_subscriptions
                        WHERE league_id = %s
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (league_id, platform, now))
                    row = cur.fetchone()
                    if row:
                        result["has_league_subscription"] = True
                        result["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else None
                        result["billing_interval"] = row.get("billing_interval") or "year"
                        result["subscriber_user_id"] = row["subscriber_user_id"]
                        result["stripe_customer_id"] = row.get("stripe_customer_id")

                if user_id:
                    cur.execute("""
                        SELECT expires_at, stripe_customer_id, billing_interval, plan_key
                        FROM user_subscriptions
                        WHERE user_id = %s
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (user_id, platform, now))
                    row = cur.fetchone()
                    if row:
                        result["has_user_subscription"] = True
                        _upk = ((row.get("plan_key") if isinstance(row, dict)
                                 else row[3]) or "").strip().lower()
                        if _upk:
                            result["user_plan_key"] = _upk
                        if not result["expires_at"]:
                            result["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else None
                            result["billing_interval"] = row.get("billing_interval") or "year"
                        if not result["stripe_customer_id"]:
                            result["stripe_customer_id"] = row.get("stripe_customer_id")

                    if league_id:
                        cur.execute("""
                            SELECT expires_at, stripe_customer_id, billing_interval
                            FROM user_league_subscriptions
                            WHERE user_id = %s
                              AND platform = %s
                              AND league_id = %s
                              AND subscription_status = 'active'
                              AND expires_at > %s
                            LIMIT 1
                        """, (user_id, platform, league_id, now))
                        row = cur.fetchone()
                        if row:
                            result["has_single_league_subscription"] = True
                            if not result["expires_at"]:
                                result["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else None
                                result["billing_interval"] = row.get("billing_interval") or "year"
                            if not result["stripe_customer_id"]:
                                result["stripe_customer_id"] = row.get("stripe_customer_id")

        has_league = result["has_league_subscription"]
        has_user = result["has_user_subscription"]
        has_single = result["has_single_league_subscription"]
        if has_league and has_user:
            result["subscription_type"] = "combo"
        elif has_league:
            result["subscription_type"] = "league"
        elif has_user:
            # The real catalog plan when known (starter/all_pro/hall_of_fame);
            # legacy rows read as "user".
            result["subscription_type"] = result.get("user_plan_key") or "user"
        elif has_single:
            result["subscription_type"] = "single_league"

        result["has_premium"] = has_league or has_user or has_single
        return result

    except Exception as e:
        logger.error("[subscriptions] Error getting subscription info: %s", e)
        return result


def create_league_subscription(
        league_id: str,
        subscriber_user_id: str,
        expires_at: datetime,
        platform: str = "sleeper",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        billing_interval: str = "year",
) -> bool:
    """Create or update a league subscription."""
    billing_interval = (billing_interval or "year").strip().lower()
    if billing_interval not in ("month", "year"):
        billing_interval = "year"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_billing_interval(cur, "league_subscriptions")
                cur.execute("""
                    INSERT INTO league_subscriptions (
                        league_id, platform, subscriber_user_id,
                        subscription_status, stripe_subscription_id,
                        stripe_customer_id, expires_at, billing_interval
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (platform, league_id) DO UPDATE SET
                        subscriber_user_id = EXCLUDED.subscriber_user_id,
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        billing_interval = EXCLUDED.billing_interval,
                        updated_at = NOW()
                """, (
                    league_id, platform, subscriber_user_id,
                    'active', stripe_subscription_id,
                    stripe_customer_id, expires_at, billing_interval
                ))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error creating league subscription: %s", e)
        return False


def create_user_subscription(
        user_id: str,
        expires_at: datetime,
        platform: str = "sleeper",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        billing_interval: str = "year",
        plan_key: str = "",
) -> bool:
    """Create or update a user subscription.

    ``plan_key`` records the catalog plan ('starter' | 'all_pro' |
    'hall_of_fame', or the grandfathered 'user'). Empty/NULL keeps the legacy
    unlimited-personal semantics.
    """
    billing_interval = (billing_interval or "year").strip().lower()
    if billing_interval not in ("month", "year"):
        billing_interval = "year"
    plan_key = (plan_key or "").strip().lower() or None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_billing_interval(cur, "user_subscriptions")
                _ensure_plan_key(cur)
                cur.execute("""
                    INSERT INTO user_subscriptions (
                        user_id, platform, subscription_status,
                        stripe_subscription_id, stripe_customer_id, expires_at,
                        billing_interval, plan_key
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform) DO UPDATE SET
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        billing_interval = EXCLUDED.billing_interval,
                        plan_key = COALESCE(EXCLUDED.plan_key, user_subscriptions.plan_key),
                        updated_at = NOW()
                """, (
                    user_id, platform, 'active',
                    stripe_subscription_id, stripe_customer_id, expires_at,
                    billing_interval, plan_key,
                ))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error creating user subscription: %s", e)
        return False


_USER_LEAGUE_SUBS_DDL = """
CREATE TABLE IF NOT EXISTS user_league_subscriptions (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'sleeper',
    league_id TEXT NOT NULL,
    subscription_status TEXT NOT NULL DEFAULT 'active',
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_user_league_status CHECK (
        subscription_status IN ('active', 'canceled', 'expired')
    ),
    UNIQUE (user_id, platform, league_id)
)
"""


def _ensure_user_league_subscriptions_table(cur) -> None:
    """Create the One League table if migration 032 never applied.

    ``scripts/run_migrations.py`` stops on the first failing file, so a
    non-idempotent earlier migration can leave this table missing. Checkout
    still succeeds; the grant then fails and the buyer never appears in the DB.
    """
    cur.execute(_USER_LEAGUE_SUBS_DDL)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_league_subs_lookup "
        "ON user_league_subscriptions(user_id, platform, league_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_league_subs_expires "
        "ON user_league_subscriptions(expires_at)"
    )


def create_user_league_subscription(
        user_id: str,
        league_id: str,
        expires_at: datetime,
        platform: str = "sleeper",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        billing_interval: str = "year",
) -> bool:
    """Create or update a buyer-only single-league subscription."""
    billing_interval = (billing_interval or "year").strip().lower()
    if billing_interval not in ("month", "year"):
        billing_interval = "year"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_user_league_subscriptions_table(cur)
                _ensure_billing_interval(cur, "user_league_subscriptions")
                cur.execute("""
                    INSERT INTO user_league_subscriptions (
                        user_id, platform, league_id, subscription_status,
                        stripe_subscription_id, stripe_customer_id, expires_at,
                        billing_interval
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform, league_id) DO UPDATE SET
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        billing_interval = EXCLUDED.billing_interval,
                        updated_at = NOW()
                """, (
                    user_id, platform, league_id, 'active',
                    stripe_subscription_id, stripe_customer_id, expires_at,
                    billing_interval,
                ))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error creating single-league subscription: %s", e)
        return False


# ── PRO league slots (slot-capped plans: starter / all_pro) ───────────────────
# A subscription row holds up to N selected league ids. Slots grant PRO only
# for the selected leagues, and only while the owning subscription row is
# active. The cap is enforced server-side on every write.


def find_active_personal_plan(
    user_keys: List[str],
    platform: str = "sleeper",
) -> Optional[Dict[str, Any]]:
    """Return the caller's active personal subscription row, if any.

    ``user_keys`` are the checkout identities to look up (e.g. viewer id,
    ``acct:<id>``). Returns ``{"user_key", "plan_key", "expires_at",
    "stripe_subscription_id"}`` or None. A NULL plan_key reads as the
    grandfathered 'user' plan (unlimited personal PRO). Slot-capped plans are
    preferred when several rows are active.
    """
    keys = [k for k in (user_keys or []) if k]
    if not keys:
        return None
    platform = platform or "sleeper"
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_plan_key(cur)
                cur.execute("""
                    SELECT user_id, plan_key, expires_at, stripe_subscription_id
                    FROM user_subscriptions
                    WHERE user_id = ANY(%s)
                      AND platform = %s
                      AND subscription_status = 'active'
                      AND expires_at > %s
                """, (keys, platform, now))
                rows = cur.fetchall() or []
    except Exception as e:
        logger.error("[subscriptions] Error finding personal plan: %s", e)
        return None
    if not rows:
        return None

    def _rank(row) -> tuple:
        pk = ((row.get("plan_key") if isinstance(row, dict) else row[1]) or "").strip().lower()
        return (0 if pk in _SLOT_CAP_BY_PLAN else 1, pk)

    def _as_dict(row) -> dict:
        if isinstance(row, dict):
            d = dict(row)
        else:
            d = {
                "user_id": row[0], "plan_key": row[1],
                "expires_at": row[2], "stripe_subscription_id": row[3],
            }
        pk = (d.get("plan_key") or "").strip().lower() or "user"
        d["plan_key"] = pk
        return d

    best = min((_as_dict(r) for r in rows), key=_rank)
    return {
        "user_key": best.get("user_id"),
        "plan_key": best.get("plan_key"),
        "expires_at": best.get("expires_at"),
        "stripe_subscription_id": best.get("stripe_subscription_id"),
    }


def get_pro_league_slots(
    user_id: str,
    platform: str = "sleeper",
) -> List[str]:
    """Return the league ids assigned to the user's PRO slots (active plan only)."""
    user_id = (user_id or "").strip()
    if not user_id:
        return []
    platform = platform or "sleeper"
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_pro_league_slots_table(cur)
                _ensure_plan_key(cur)
                cur.execute("""
                    SELECT pls.league_id
                    FROM pro_league_slots pls
                    JOIN user_subscriptions us
                      ON us.user_id = pls.user_id
                     AND us.platform = pls.platform
                     AND us.subscription_status = 'active'
                     AND us.expires_at > %s
                     AND us.plan_key = ANY(%s)
                    WHERE pls.user_id = %s
                      AND pls.platform = %s
                    ORDER BY pls.league_id
                """, (now, list(_SLOT_CAP_BY_PLAN), user_id, platform))
                return [r["league_id"] if isinstance(r, dict) else r[0]
                        for r in (cur.fetchall() or [])]
    except Exception as e:
        logger.error("[subscriptions] Error reading PRO league slots: %s", e)
        return []


def set_pro_league_slots(
    user_id: str,
    platform: str,
    league_ids: List[str],
    stripe_subscription_id: Optional[str] = None,
) -> List[str]:
    """Replace the user's PRO league slot set. Enforces the plan cap.

    Raises ValueError when the user has no active slot-capped plan, the cap
    is exceeded, or a league id is blank. Slots for leagues outside the new
    set are deleted.
    """
    user_id = (user_id or "").strip()
    platform = (platform or "sleeper").strip() or "sleeper"
    if not user_id:
        raise ValueError("No user.")
    plan = find_active_personal_plan([user_id], platform)
    plan_key = (plan or {}).get("plan_key") or ""
    cap = slot_cap_for_plan(plan_key)
    if cap is None:
        raise ValueError(
            "League slots are only for Starter and All-Pro plans."
            if plan else "No active PRO plan."
        )
    cleaned: List[str] = []
    for lid in league_ids or []:
        lid = str(lid or "").strip()
        if not lid:
            raise ValueError("League ids must not be blank.")
        if lid not in cleaned:
            cleaned.append(lid)
    if len(cleaned) > cap:
        raise ValueError(
            f"Your plan covers {cap} league{'s' if cap != 1 else ''}."
        )
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_pro_league_slots_table(cur)
                if cleaned:
                    cur.execute("""
                        INSERT INTO pro_league_slots
                            (user_id, platform, league_id, stripe_subscription_id,
                             created_at, updated_at)
                        SELECT %s, %s, lid, %s, %s, %s
                        FROM UNNEST(%s::text[]) AS lid
                        ON CONFLICT (user_id, platform, league_id) DO UPDATE SET
                            stripe_subscription_id = COALESCE(
                                EXCLUDED.stripe_subscription_id,
                                pro_league_slots.stripe_subscription_id),
                            updated_at = NOW()
                    """, (user_id, platform, stripe_subscription_id, now, now, cleaned))
                    cur.execute("""
                        DELETE FROM pro_league_slots
                        WHERE user_id = %s AND platform = %s
                          AND NOT (league_id = ANY(%s))
                    """, (user_id, platform, cleaned))
                else:
                    cur.execute("""
                        DELETE FROM pro_league_slots
                        WHERE user_id = %s AND platform = %s
                    """, (user_id, platform))
        return cleaned
    except ValueError:
        raise
    except Exception as e:
        logger.error("[subscriptions] Error writing PRO league slots: %s", e)
        raise ValueError("Could not save league slots.") from e


def add_pro_league_slot(
    user_id: str,
    platform: str,
    league_id: str,
    stripe_subscription_id: Optional[str] = None,
) -> bool:
    """Add one league to the user's slots when the cap allows. Best-effort."""
    user_id = (user_id or "").strip()
    league_id = (league_id or "").strip()
    if not user_id or not league_id:
        return False
    try:
        current = get_pro_league_slots(user_id, platform)
        if league_id in current:
            return True
        set_pro_league_slots(
            user_id, platform, current + [league_id],
            stripe_subscription_id=stripe_subscription_id,
        )
        return True
    except ValueError as e:
        logger.info("[subscriptions] slot seed skipped user=%s league=%s: %s",
                   user_id, league_id, e)
        return False
    except Exception:
        logger.warning("[subscriptions] slot seed failed", exc_info=True)
        return False


def _slot_row_active(cur, user_key: str, platform: str, league_id: str, now) -> bool:
    """Cursor-level slot check: active slot-plan row covering league_id."""
    cur.execute("""
        SELECT 1
        FROM pro_league_slots pls
        JOIN user_subscriptions us
          ON us.user_id = pls.user_id
         AND us.platform = pls.platform
         AND us.subscription_status = 'active'
         AND us.expires_at > %s
         AND us.plan_key = ANY(%s)
        WHERE pls.user_id = %s
          AND pls.platform = %s
          AND pls.league_id = %s
        LIMIT 1
    """, (now, list(_SLOT_CAP_BY_PLAN), user_key, platform, league_id))
    return bool(cur.fetchone())


def has_pro_league_slot(
    user_id: Optional[str],
    league_id: Optional[str],
    platform: str = "sleeper",
    account_id: Optional[int] = None,
) -> bool:
    """True when the user holds an active slot-capped plan covering league_id."""
    if not league_id or (not user_id and not account_id):
        return False
    platform = platform or "sleeper"
    keys = []
    if user_id:
        keys.append(user_id)
    if account_id:
        keys.extend(_account_user_keys(int(account_id)))
    if not keys:
        return False
    now = datetime.now(timezone.utc)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_pro_league_slots_table(cur)
                _ensure_plan_key(cur)
                cur.execute("""
                    SELECT 1
                    FROM pro_league_slots pls
                    JOIN user_subscriptions us
                      ON us.user_id = pls.user_id
                     AND us.platform = pls.platform
                     AND us.subscription_status = 'active'
                     AND us.expires_at > %s
                     AND us.plan_key = ANY(%s)
                    WHERE pls.user_id = ANY(%s)
                      AND pls.platform = %s
                      AND pls.league_id = %s
                    LIMIT 1
                """, (now, list(_SLOT_CAP_BY_PLAN), keys, platform, league_id))
                return bool(cur.fetchone())
    except Exception as e:
        logger.error("[subscriptions] Error checking PRO league slot: %s", e)
        return False


def cancel_subscription(subscription_id: str, subscription_type: str = "league") -> bool:
    """Cancel a subscription (set status to 'canceled')."""
    try:
        if subscription_type == "league":
            table = "league_subscriptions"
        elif subscription_type == "single_league":
            table = "user_league_subscriptions"
        else:
            table = "user_subscriptions"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {table}
                    SET subscription_status = 'canceled',
                        updated_at = NOW()
                    WHERE stripe_subscription_id = %s
                """, (subscription_id,))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error canceling subscription: %s", e)
        return False


# ── Stripe subscription lifecycle sync ────────────────────────────────────────
# Driven by the customer.subscription.updated / customer.subscription.deleted
# webhook events. The event object already carries everything needed (status,
# cancel_at_period_end, current period end), so no extra Stripe API calls.

# Plan -> entitlement tables that plan owns. combo owns two rows (shared league
# PRO plus personal user PRO); the new catalog plans and every other plan own
# exactly one. Grandfathered plans keep working through renewals and webhooks.
_PLAN_LIFECYCLE_TABLES = {
    "league": ("league_subscriptions",),
    "user": ("user_subscriptions",),
    "single_league": ("user_league_subscriptions",),
    "combo": ("league_subscriptions", "user_subscriptions"),
    "starter": ("user_subscriptions",),
    "all_pro": ("user_subscriptions",),
    "hall_of_fame": ("user_subscriptions",),
}

# Stripe statuses that keep PRO access working. past_due is deliberately here:
# Stripe is still retrying the payment (dunning nudges the buyer), so access
# continues through the retry window instead of dropping on the first failure.
_ACTIVE_LIKE_STATUSES = ("active", "trialing", "past_due")

# Stripe statuses that end PRO access immediately.
_REVOKED_STATUSES = ("canceled", "unpaid", "incomplete_expired")


def _cancel_lifecycle_rows(cur, table: str, sub_id: str) -> None:
    cur.execute(
        f"UPDATE {table} SET subscription_status = 'canceled',"
        " updated_at = NOW() WHERE stripe_subscription_id = %s",
        (sub_id,),
    )


def _grant_lifecycle_table(
    cur,
    table: str,
    *,
    user_id: str,
    league_id: str,
    platform: str,
    period_end,
    interval: str,
    sub_id: str,
    plan_key: str = "",
) -> bool:
    """Write the missing entitlement row for a plan upgrade.

    Never steals a row another subscription owns: when the natural key already
    has an active row for a different subscription id, the grant is skipped.
    Returns True when a row was written.
    """
    if table == "league_subscriptions":
        if not league_id:
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: no league_id for sub=%s",
                sub_id,
            )
            return False
        cur.execute(
            "SELECT stripe_subscription_id FROM league_subscriptions"
            " WHERE platform = %s AND league_id = %s"
            " AND subscription_status = 'active' AND expires_at > NOW() LIMIT 1",
            (platform, league_id),
        )
        row = cur.fetchone()
        if row and (row.get("stripe_subscription_id") or "") not in ("", sub_id):
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: league %s already"
                " has an active subscription",
                league_id,
            )
            return False
        return bool(create_league_subscription(
            league_id, user_id, period_end, platform=platform,
            stripe_subscription_id=sub_id, billing_interval=interval,
        ))
    if table == "user_subscriptions":
        if not user_id:
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: no user_id for sub=%s",
                sub_id,
            )
            return False
        cur.execute(
            "SELECT stripe_subscription_id FROM user_subscriptions"
            " WHERE user_id = %s AND platform = %s"
            " AND subscription_status = 'active' AND expires_at > NOW() LIMIT 1",
            (user_id, platform),
        )
        row = cur.fetchone()
        if row and (row.get("stripe_subscription_id") or "") not in ("", sub_id):
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: user %s already"
                " has an active subscription",
                user_id,
            )
            return False
        return bool(create_user_subscription(
            user_id, period_end, platform=platform,
            stripe_subscription_id=sub_id, billing_interval=interval,
            plan_key=plan_key,
        ))
    if table == "user_league_subscriptions":
        if not user_id or not league_id:
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: missing identity for sub=%s",
                sub_id,
            )
            return False
        cur.execute(
            "SELECT stripe_subscription_id FROM user_league_subscriptions"
            " WHERE user_id = %s AND platform = %s AND league_id = %s"
            " AND subscription_status = 'active' AND expires_at > NOW() LIMIT 1",
            (user_id, platform, league_id),
        )
        row = cur.fetchone()
        if row and (row.get("stripe_subscription_id") or "") not in ("", sub_id):
            logger.warning(
                "[subscriptions] lifecycle upgrade skipped: user %s league %s"
                " already has an active subscription",
                user_id, league_id,
            )
            return False
        return bool(create_user_league_subscription(
            user_id, league_id, period_end, platform=platform,
            stripe_subscription_id=sub_id, billing_interval=interval,
        ))
    return False


def apply_subscription_lifecycle(
    sub_id: str,
    *,
    event_type: str,
    status: str,
    cancel_at_period_end: bool,
    expires_at,
    plan: str = "",
    interval: str = "year",
    user_id: str = "",
    league_id: str = "",
    platform: str = "sleeper",
    plan_key: str = "",
) -> dict:
    """Reconcile entitlement rows with a subscription lifecycle webhook event.

    Handles ``customer.subscription.updated`` and
    ``customer.subscription.deleted``:

    - updated, status active/trialing: rows go ``active`` with the fresh
      period end and billing interval (this also recovers a subscription that
      went past due and then paid). A plan change grants newly covered tables
      and revokes tables the plan no longer covers.
    - updated with ``cancel_at_period_end``: access is kept until the period
      end; the later ``deleted`` event revokes.
    - updated, status past_due: access is kept while Stripe retries; the
      dunning flow nudges the buyer to fix payment in the meantime.
    - updated, status unpaid/canceled/incomplete_expired: revoke immediately.
    - deleted: an immediate cancel revokes at once; a scheduled
      (cancel_at_period_end) cancel keeps access until the period end.

    Returns a summary dict of what changed. Never raises for unknown plans or
    missing rows; DB errors are logged and swallowed so the webhook stays 2xx.
    """
    summary = {
        "sub_id": (sub_id or "").strip(),
        "event": event_type,
        "status": (status or "").strip().lower(),
        "updated": [],
        "granted": [],
        "canceled": [],
    }
    sub_id = summary["sub_id"]
    status = summary["status"]
    if not sub_id:
        return summary
    plan = (plan or "").strip()
    interval = (interval or "year").strip().lower()
    if interval not in ("month", "year"):
        interval = "year"
    platform = (platform or "sleeper").strip() or "sleeper"
    user_id = (user_id or "").strip()
    league_id = (league_id or "").strip()
    # plan_key rides along for user_subscriptions rows so slot-capped plans
    # keep their cap through webhook-driven grants. Defaults to the plan when
    # the caller did not pass one explicitly.
    plan_key = (plan_key or plan or "").strip().lower()

    now = datetime.now(timezone.utc)
    period_end = expires_at if isinstance(expires_at, datetime) else None
    if period_end is not None and period_end.tzinfo is None:
        period_end = period_end.replace(tzinfo=timezone.utc)

    # A scheduled cancel keeps access until the period actually ends, for both
    # the updated event that announces it and an early/out-of-order deleted.
    keep_until_period_end = (
        bool(cancel_at_period_end)
        and period_end is not None
        and period_end > now
    )
    revoke_now = (
        event_type == "customer.subscription.deleted" and not keep_until_period_end
    ) or (
        event_type == "customer.subscription.updated" and status in _REVOKED_STATUSES
    )

    desired_tables = _PLAN_LIFECYCLE_TABLES.get(plan, ())

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for table in _BILLING_INTERVAL_TABLES:
                    _ensure_billing_interval(cur, table)
                _ensure_plan_key(cur)
                existing = set()
                for table in _BILLING_INTERVAL_TABLES:
                    cur.execute(
                        f"SELECT id FROM {table}"
                        " WHERE stripe_subscription_id = %s LIMIT 1",
                        (sub_id,),
                    )
                    if cur.fetchone():
                        existing.add(table)

                for table in _BILLING_INTERVAL_TABLES:
                    if table not in existing:
                        continue
                    plan_downgraded = (
                        event_type == "customer.subscription.updated"
                        and bool(desired_tables)
                        and table not in desired_tables
                    )
                    if revoke_now or plan_downgraded:
                        _cancel_lifecycle_rows(cur, table, sub_id)
                        summary["canceled"].append(table)
                    elif keep_until_period_end or status not in _REVOKED_STATUSES:
                        if keep_until_period_end or status in _ACTIVE_LIKE_STATUSES:
                            if table == "user_subscriptions" and plan_key:
                                # Plan changes (e.g. starter -> all_pro) land
                                # here: sync plan_key so slot caps and
                                # entitlement checks follow the new plan.
                                cur.execute(
                                    "UPDATE user_subscriptions"
                                    " SET subscription_status = 'active',"
                                    " expires_at = %s, billing_interval = %s,"
                                    " plan_key = %s, updated_at = NOW()"
                                    " WHERE stripe_subscription_id = %s",
                                    (period_end, interval, plan_key, sub_id),
                                )
                            else:
                                cur.execute(
                                    f"UPDATE {table} SET subscription_status = 'active',"
                                    " expires_at = %s, billing_interval = %s,"
                                    " updated_at = NOW()"
                                    " WHERE stripe_subscription_id = %s",
                                    (period_end, interval, sub_id),
                                )
                        else:
                            # Unknown status (e.g. incomplete, paused): sync the
                            # clock but do not flip access either way.
                            cur.execute(
                                f"UPDATE {table} SET expires_at = %s,"
                                " billing_interval = %s, updated_at = NOW()"
                                " WHERE stripe_subscription_id = %s",
                                (period_end, interval, sub_id),
                            )
                        summary["updated"].append(table)

                # Plan upgrade: grant tables the new plan covers that have no
                # row for this subscription yet.
                if (
                    event_type == "customer.subscription.updated"
                    and not revoke_now
                    and status in ("active", "trialing")
                    and desired_tables
                ):
                    for table in desired_tables:
                        if table in existing:
                            continue
                        if _grant_lifecycle_table(
                            cur, table, user_id=user_id, league_id=league_id,
                            platform=platform, period_end=period_end,
                            interval=interval, sub_id=sub_id,
                            plan_key=plan_key,
                        ):
                            summary["granted"].append(table)
    except Exception:
        logger.exception("[subscriptions] lifecycle sync failed sub=%s", sub_id)
    return summary


# ── PRO free trial ────────────────────────────────────────────────────────────
# Adjustable defaults (also documented in the PR body):
#   PRO_TRIAL_DAYS          trial length, days (default 7; env PRO_TRIAL_DAYS)
#   PRO_TRIAL_REQUIRE_CARD  whether starting a trial needs a payment method
#                           (default False: no-card, one-click signup)
#   Trial scope is FULL PRO: a trial grants everything a Hall of Fame plan
#   grants (every PRO feature, all of the holder's leagues).
#   One trial per user ever: enforced by UNIQUE(user_key) on pro_trials.
#   Re-claim via a brand-new Google account is out of scope.

PRO_TRIAL_DAYS = int(os.environ.get("PRO_TRIAL_DAYS", "7") or 7)
PRO_TRIAL_REQUIRE_CARD = os.environ.get("PRO_TRIAL_REQUIRE_CARD", "").strip().lower() in (
    "1", "true", "yes",
)


_PRO_TRIALS_DDL = """
CREATE TABLE IF NOT EXISTS pro_trials (
    id SERIAL PRIMARY KEY,
    user_key TEXT NOT NULL UNIQUE,
    account_id BIGINT,
    trial_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trial_ends_at TIMESTAMPTZ NOT NULL,
    subscription_status TEXT NOT NULL DEFAULT 'active',
    ended_notified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_pro_trial_status CHECK (
        subscription_status IN ('active', 'expired')
    )
)
"""


def _ensure_pro_trials_table(cur) -> None:
    """Create pro_trials if a migration never applied (idempotent)."""
    cur.execute(_PRO_TRIALS_DDL)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pro_trials_ends_at ON pro_trials(trial_ends_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pro_trials_account_id ON pro_trials(account_id)"
    )


def _pro_trial_session_keys() -> List[str]:
    """Identity keys the trial system recognizes, best first.

    Claim keys are always the Google account keys (``acct:<id>`` / bare id);
    viewer ids are included as a read fallback so legacy rows still gate.
    """
    keys: List[str] = []
    acct = _session_account_id()
    if acct:
        keys += [f"acct:{acct}", str(acct)]
    try:
        from flask import session as _session, has_request_context as _hrc
        if _hrc():
            for raw in (_session.get("viewer_user_id"), _session.get("viewer_username")):
                key = str(raw or "").strip()
                if key and key not in keys:
                    keys.append(key)
    except Exception:
        pass
    return keys


_MISSING = object()


def _trial_cache_get(key):
    try:
        from flask import g, has_request_context
        if has_request_context():
            cache = getattr(g, "_trial_cache", None)
            if cache is None:
                cache = {}
                g._trial_cache = cache
            return cache, cache.get(key, _MISSING)
    except Exception:
        pass
    return None, _MISSING


def _trial_row_for_keys(cur, keys: List[str]) -> Optional[dict]:
    """Newest trial row matching any key, or None. Never raises."""
    keys = [str(k) for k in (keys or []) if str(k or "").strip()]
    if not keys:
        return None
    placeholders = ", ".join(["%s"] * len(keys))
    cur.execute(
        f"SELECT user_key, account_id, trial_started_at, trial_ends_at, "
        f"subscription_status, ended_notified FROM pro_trials "
        f"WHERE user_key IN ({placeholders}) "
        f"ORDER BY trial_started_at DESC LIMIT 1",
        tuple(keys),
    )
    return cur.fetchone()


def trial_active_for_keys(keys: List[str]) -> bool:
    """True when any of the keys holds an unexpired, active trial.

    Fail closed: any DB error denies the trial grant (never the reverse).
    Per-request memoized so the premium gate costs at most one trial query.
    """
    keys = [str(k) for k in (keys or []) if str(k or "").strip()]
    if not keys:
        return False
    cache_key = ("active", tuple(sorted(set(keys))))
    cache, hit = _trial_cache_get(cache_key)
    if hit is not _MISSING:
        return bool(hit)

    result = False
    try:
        now = datetime.now(timezone.utc)
        with get_conn() as conn:
            with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(cache_key[1]))
                cur.execute(
                    f"SELECT 1 FROM pro_trials "
                    f"WHERE user_key IN ({placeholders}) "
                    f"AND subscription_status = 'active' "
                    f"AND trial_ends_at > %s LIMIT 1",
                    (*cache_key[1], now),
                )
                result = bool(cur.fetchone())
    except Exception:
        logger.debug("[subscriptions] trial active check failed", exc_info=True)
        result = False

    if cache is not None:
        cache[cache_key] = result
    return result


def start_pro_trial_for_account(account_id: int) -> Dict[str, Any]:
    """Start the one free trial for a Google account.

    Returns {"ok", "code", "message", "ends_at", "days"} where code is one of
    "started", "already_used", "error". Idempotent under races: a unique
    violation on the second concurrent INSERT resolves to "already_used".
    """
    user_key = f"acct:{account_id}"
    now = datetime.now(timezone.utc)
    ends_at = now + timedelta(days=PRO_TRIAL_DAYS)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                _ensure_pro_trials_table(cur)
                if _trial_row_for_keys(cur, [user_key, str(account_id)]):
                    return {
                        "ok": False, "code": "already_used",
                        "message": "This account already used its free trial.",
                        "ends_at": None, "days": PRO_TRIAL_DAYS,
                    }
                cur.execute(
                    "INSERT INTO pro_trials "
                    "(user_key, account_id, trial_started_at, trial_ends_at, subscription_status) "
                    "VALUES (%s, %s, %s, %s, 'active')",
                    (user_key, account_id, now, ends_at),
                )
        logger.info("[trial] started for account_id=%s ends=%s", account_id, ends_at.isoformat())
        return {
            "ok": True, "code": "started",
            "message": f"PRO trial started. Full PRO for {PRO_TRIAL_DAYS} days, no card required.",
            "ends_at": ends_at.isoformat(), "days": PRO_TRIAL_DAYS,
        }
    except Exception as exc:
        # Concurrent double-claim: the loser hits the UNIQUE(user_key) wall.
        pgcode = getattr(exc, "pgcode", "") or ""
        name = type(exc).__name__
        if pgcode == "23505" or "UniqueViolation" in name or "IntegrityError" in name:
            logger.info("[trial] concurrent claim resolved as already_used account_id=%s", account_id)
            return {
                "ok": False, "code": "already_used",
                "message": "This account already used its free trial.",
                "ends_at": None, "days": PRO_TRIAL_DAYS,
            }
        logger.error("[subscriptions] Error starting PRO trial: %s", exc)
        return {
            "ok": False, "code": "error",
            "message": "Could not start the trial. Please try again.",
            "ends_at": None, "days": PRO_TRIAL_DAYS,
        }


def get_trial_state_for_keys(keys: List[str]) -> Dict[str, Any]:
    """Describe trial state for the given keys.

    Returns {"active", "used", "ends_at" (iso), "days_left", "just_ended",
    "user_key"}. Lapsed trials are lazily flipped to 'expired' here; the
    one-time "trial ended" nudge fires via "just_ended" exactly once (the
    ended_notified flag is consumed on that read).
    """
    import math

    state: Dict[str, Any] = {
        "active": False, "used": False, "ends_at": None, "days_left": 0,
        "just_ended": False, "user_key": None,
    }
    keys = [str(k) for k in (keys or []) if str(k or "").strip()]
    if not keys:
        return state
    try:
        now = datetime.now(timezone.utc)
        with get_conn() as conn:
            with conn.cursor() as cur:
                row = _trial_row_for_keys(cur, keys)
                if not row:
                    return state
                state["used"] = True
                state["user_key"] = row.get("user_key")
                ends_at = row.get("trial_ends_at")
                status = row.get("subscription_status") or "active"
                if status == "active" and ends_at and ends_at > now:
                    state["active"] = True
                    state["ends_at"] = ends_at.isoformat()
                    remaining = (ends_at - now).total_seconds()
                    state["days_left"] = max(1, int(math.ceil(remaining / 86400)))
                    return state
                if status == "active":
                    # Clean downgrade: flip to expired so the gate denies PRO
                    # from here on; the nudge fires once.
                    cur.execute(
                        "UPDATE pro_trials SET subscription_status = 'expired', "
                        "ended_notified = TRUE, updated_at = NOW() "
                        "WHERE user_key = %s AND subscription_status = 'active'",
                        (row.get("user_key"),),
                    )
                    state["just_ended"] = not bool(row.get("ended_notified"))
                elif not row.get("ended_notified"):
                    # Already expired (e.g. flipped outside this path) but the
                    # nudge never fired: fire it once now.
                    cur.execute(
                        "UPDATE pro_trials SET ended_notified = TRUE, "
                        "updated_at = NOW() WHERE user_key = %s",
                        (row.get("user_key"),),
                    )
                    state["just_ended"] = True
    except Exception:
        logger.debug("[subscriptions] trial state check failed", exc_info=True)
    return state
