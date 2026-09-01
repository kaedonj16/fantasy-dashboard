"""
Subscription management for league and user-based premium access.

Premium Features:
- AI insights
- Breakout candidates
- Viewable advanced metrics

Subscription types:
- league: shared PRO for every manager in one league
- user: personal PRO across all of a user's leagues
- single_league: personal PRO for one selected league only (buyer-only)
- combo: league + user
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, List

import logging

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


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

    Used for the "Link Google to secure PRO" prompt — independent of whether
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

    Buyer-only — co-managers are not entitled via this table.
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
                        SELECT 1 FROM user_subscriptions
                        WHERE user_id = %s
                          AND platform = %s
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (user_id, platform, now))

                    if cur.fetchone():
                        return True

                # Account-based (additive): premium on any linked platform
                # identity covers the whole account, across platforms.
                if account_id:
                    cur.execute("""
                        SELECT 1
                        FROM user_subscriptions us
                        JOIN account_identities ai
                          ON ai.platform = us.platform
                         AND (ai.platform_user_id = us.user_id OR ai.handle = us.user_id)
                        WHERE ai.account_id = %s
                          AND us.subscription_status = 'active'
                          AND us.expires_at > %s
                        LIMIT 1
                    """, (account_id, now))

                    if cur.fetchone():
                        return True

                    # Google-only checkout (no platform identity yet) stores the
                    # subscription against acct:<id> (or the bare account id).
                    cur.execute("""
                        SELECT 1 FROM user_subscriptions
                        WHERE user_id IN (%s, %s)
                          AND subscription_status = 'active'
                          AND expires_at > %s
                        LIMIT 1
                    """, (f"acct:{account_id}", str(account_id), now))

                    if cur.fetchone():
                        return True

    except Exception as e:
        logger.error("[subscriptions] Error checking premium access: %s", e)
        # Fail closed: on any error, deny premium rather than grant it.
        return False

    # Buyer-only single-league plan (own connection — avoid nesting get_conn).
    if league_id and has_user_league_subscription(
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
      4. Legacy Sleeper user-plan via viewer id/username — soft dual-read only
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

    # League subscription only for actual members (shared plan — membership is
    # the guard; Google is not required).
    if not result and league_id and has_premium_access(None, league_id, platform) \
            and viewer_is_league_member(viewer_user_id, league_id, platform, season):
        result = True

    # Single-league personal plan (buyer-only for this league). Prefer the
    # Google account key; soft dual-read still honors Sleeper viewer ids.
    if not result and league_id:
        if _acct and has_user_league_subscription(
            viewer_user_id or viewer_username, league_id, platform, account_id=_acct,
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

    # Legacy Sleeper username/id personal subscription.
    # Soft dual-read: still honor so buyers aren't locked out before linking.
    # Hard cutover (PRO_REQUIRE_GOOGLE): skip — thieves can't unlock PRO by
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
        "subscriber_user_id": None,
        "stripe_customer_id": None,
    }

    now = datetime.now(timezone.utc)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if league_id:
                    cur.execute("""
                        SELECT expires_at, subscriber_user_id, stripe_customer_id
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
                        result["subscriber_user_id"] = row["subscriber_user_id"]
                        result["stripe_customer_id"] = row.get("stripe_customer_id")

                if user_id:
                    cur.execute("""
                        SELECT expires_at, stripe_customer_id
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
                        if not result["expires_at"]:
                            result["expires_at"] = row["expires_at"].isoformat() if row["expires_at"] else None
                        if not result["stripe_customer_id"]:
                            result["stripe_customer_id"] = row.get("stripe_customer_id")

                    if league_id:
                        cur.execute("""
                            SELECT expires_at, stripe_customer_id
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
            result["subscription_type"] = "user"
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
        stripe_customer_id: Optional[str] = None
) -> bool:
    """Create or update a league subscription."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO league_subscriptions (
                        league_id, platform, subscriber_user_id,
                        subscription_status, stripe_subscription_id,
                        stripe_customer_id, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (platform, league_id) DO UPDATE SET
                        subscriber_user_id = EXCLUDED.subscriber_user_id,
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = NOW()
                """, (
                    league_id, platform, subscriber_user_id,
                    'active', stripe_subscription_id,
                    stripe_customer_id, expires_at
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
        stripe_customer_id: Optional[str] = None
) -> bool:
    """Create or update a user subscription."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_subscriptions (
                        user_id, platform, subscription_status,
                        stripe_subscription_id, stripe_customer_id, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform) DO UPDATE SET
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = NOW()
                """, (
                    user_id, platform, 'active',
                    stripe_subscription_id, stripe_customer_id, expires_at
                ))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error creating user subscription: %s", e)
        return False


def create_user_league_subscription(
        user_id: str,
        league_id: str,
        expires_at: datetime,
        platform: str = "sleeper",
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
) -> bool:
    """Create or update a buyer-only single-league subscription."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_league_subscriptions (
                        user_id, platform, league_id, subscription_status,
                        stripe_subscription_id, stripe_customer_id, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform, league_id) DO UPDATE SET
                        subscription_status = EXCLUDED.subscription_status,
                        stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                        stripe_customer_id = EXCLUDED.stripe_customer_id,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = NOW()
                """, (
                    user_id, platform, league_id, 'active',
                    stripe_subscription_id, stripe_customer_id, expires_at,
                ))
        return True
    except Exception as e:
        logger.error("[subscriptions] Error creating single-league subscription: %s", e)
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
