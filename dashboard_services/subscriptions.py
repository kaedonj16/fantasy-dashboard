"""
Subscription management for league and user-based premium access.

Premium Features:
- AI insights
- Breakout candidates
- Viewable advanced metrics
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any

from dashboard_services.db import get_conn


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


def has_premium_access(user_id: Optional[str], league_id: Optional[str], platform: str = "sleeper") -> bool:
    """
    Check if a user has premium access for a specific league.

    Premium access is granted if EITHER:
    1. The league has an active subscription (league-based)
    2. The user has an active subscription (user-based, covers all leagues)

    Args:
        user_id: Sleeper username or user ID
        league_id: League ID
        platform: Platform name (default: 'sleeper')

    Returns:
        True if user has premium access, False otherwise
    """
    if not user_id and not league_id:
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

                return False

    except Exception as e:
        print(f"[subscriptions] Error checking premium access: {e}")
        # Fail closed: on any error, deny premium rather than grant it.
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

    Membership is only verifiable for Sleeper; for other platforms we do not
    block (return True) so those flows keep working.
    """
    if not league_id:
        return False
    if (platform or "sleeper") != "sleeper":
        return True
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

    Grants access if the viewer has their own user subscription (valid
    anywhere), or the league has a subscription AND the viewer is a verified
    member of that league.
    """
    platform = platform or "sleeper"
    # Own subscription works everywhere.
    if viewer_username and has_premium_access(viewer_username, None, platform):
        return True
    # League subscription only for actual members.
    if league_id and has_premium_access(None, league_id, platform):
        if viewer_is_league_member(viewer_user_id, league_id, platform, season):
            return True
    return False


def get_subscription_info(user_id: Optional[str], league_id: Optional[str], platform: str = "sleeper") -> Dict[
    str, Any]:
    """
    Get detailed subscription information for a user/league.

    Returns:
        {
            "has_premium": bool,
            "subscription_type": "league" | "user" | "combo" | None,
            "has_league_subscription": bool,
            "has_user_subscription": bool,
            "expires_at": datetime | None,
            "subscriber_user_id": str | None  # Only for league subscriptions
        }
    """
    result = {
        "has_premium": False,
        "subscription_type": None,
        "has_league_subscription": False,
        "has_user_subscription": False,
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

        has_league = result["has_league_subscription"]
        has_user = result["has_user_subscription"]
        if has_league and has_user:
            result["subscription_type"] = "combo"
        elif has_league:
            result["subscription_type"] = "league"
        elif has_user:
            result["subscription_type"] = "user"

        result["has_premium"] = has_league or has_user
        return result

    except Exception as e:
        print(f"[subscriptions] Error getting subscription info: {e}")
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
                    ON CONFLICT (league_id) DO UPDATE SET
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
        print(f"[subscriptions] Error creating league subscription: {e}")
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
        print(f"[subscriptions] Error creating user subscription: {e}")
        return False


def cancel_subscription(subscription_id: str, subscription_type: str = "league") -> bool:
    """Cancel a subscription (set status to 'canceled')."""
    try:
        table = "league_subscriptions" if subscription_type == "league" else "user_subscriptions"
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
        print(f"[subscriptions] Error canceling subscription: {e}")
        return False
