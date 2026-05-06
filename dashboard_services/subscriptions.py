"""
Subscription management for league and user-based premium access.

Premium Features:
- AI insights
- Breakout candidates
- Viewable advanced metrics
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any

from dashboard_services.db import get_conn


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
        # Fail open - allow access on error to prevent breaking the app
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
        "subscriber_user_id": None
    }

    now = datetime.now(timezone.utc)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if league_id:
                    cur.execute("""
                        SELECT expires_at, subscriber_user_id
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

                if user_id:
                    cur.execute("""
                        SELECT expires_at
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
