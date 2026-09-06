"""Extensible account notification preferences.

Postgres remains the source of truth for whether someone should receive a given
email type. ``accounts.email_opt_out`` is preserved as a legacy fallback for
``weekly_digest`` so existing unsubscribes keep working.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

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

_SCHEMA_READY = False


def ensure_schema(conn=None) -> None:
    """Create preference storage. Safe to call repeatedly."""
    global _SCHEMA_READY
    if _SCHEMA_READY and conn is None:
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
        _SCHEMA_READY = True
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
            ensure_schema(conn)
            row = conn.execute(
                """SELECT enabled FROM account_notification_preferences
                   WHERE account_id = %s AND channel = 'email' AND notification_type = %s""",
                (int(account_id), ntype),
            ).fetchone()
        else:
            from dashboard_services.db import get_conn
            with get_conn() as c:
                ensure_schema(c)
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
            ensure_schema(conn)
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
            ensure_schema(c)
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
