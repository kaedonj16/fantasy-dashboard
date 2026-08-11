"""Standalone user accounts + cross-platform league memberships.

A single person is one row in ``accounts`` (identified by an OAuth provider
subject + email). Their Sleeper / Yahoo / ESPN connections attach to that
account via ``account_identities``, and every league they belong to — on any
platform, whether or not they've subscribed to notifications for it — is one row
in ``user_leagues``. My Leagues, the league switcher, and push all read from
``user_leagues`` so the set of "my leagues" is platform-agnostic.

This layer is additive: it sits alongside the existing Sleeper-session identity
and does not (yet) change how billing or push key off the Sleeper user id. The
migration of those onto ``account_id`` is a later, deliberate step.

DB access mirrors the rest of the app: raw SQL through ``dashboard_services.db``.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_TABLES_READY = False


def init_accounts_tables() -> None:
    """Create the accounts / identities / leagues tables once per process."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id            SERIAL PRIMARY KEY,
                email         TEXT UNIQUE,
                google_sub    TEXT UNIQUE,
                apple_sub     TEXT UNIQUE,
                created_at    TIMESTAMPTZ DEFAULT now(),
                last_login_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS account_identities (
                id               SERIAL PRIMARY KEY,
                account_id       INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                platform         TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                handle           TEXT,
                created_at       TIMESTAMPTZ DEFAULT now(),
                UNIQUE (platform, platform_user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_leagues (
                id         SERIAL PRIMARY KEY,
                account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                platform   TEXT NOT NULL,
                league_id  TEXT NOT NULL,
                season     INTEGER,
                team_id    TEXT,
                name       TEXT,
                added_at   TIMESTAMPTZ DEFAULT now(),
                UNIQUE (account_id, platform, league_id, season)
            )
            """
        )
        conn.commit()
    _TABLES_READY = True


def upsert_google_account(google_sub: str, email: Optional[str]) -> Optional[int]:
    """Find or create the account for a Google identity; return its id.

    Matches on ``google_sub`` first, then falls back to ``email`` so a person who
    already has an account (e.g. a future Apple sign-in with the same address)
    isn't duplicated. Bumps ``last_login_at`` on every call.
    """
    if not google_sub:
        return None
    google_sub = str(google_sub).strip()
    email = (str(email).strip().lower() or None) if email else None
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM accounts WHERE google_sub = %s", (google_sub,)
        ).fetchone()
        if not row and email:
            row = conn.execute(
                "SELECT id FROM accounts WHERE email = %s", (email,)
            ).fetchone()
        if row:
            acct_id = row["id"]
            conn.execute(
                "UPDATE accounts SET last_login_at = now(), "
                "google_sub = COALESCE(google_sub, %s), "
                "email = COALESCE(email, %s) WHERE id = %s",
                (google_sub, email, acct_id),
            )
        else:
            acct_id = conn.execute(
                "INSERT INTO accounts (email, google_sub) VALUES (%s, %s) RETURNING id",
                (email, google_sub),
            ).fetchone()["id"]
        conn.commit()
        return acct_id


def link_platform_identity(
    account_id: int, platform: str, platform_user_id: str, handle: Optional[str] = None
) -> None:
    """Attach a platform identity (Sleeper user id / Yahoo guid) to an account."""
    if not (account_id and platform and platform_user_id):
        return
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        # A given platform identity belongs to one account; re-point it (and
        # refresh the handle) if the same person signs in from a new account.
        conn.execute(
            """
            INSERT INTO account_identities (account_id, platform, platform_user_id, handle)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (platform, platform_user_id) DO UPDATE
                SET account_id = EXCLUDED.account_id,
                    handle     = COALESCE(EXCLUDED.handle, account_identities.handle)
            """,
            (account_id, platform, str(platform_user_id), handle),
        )
        conn.commit()


def add_user_league(
    account_id: int,
    platform: str,
    league_id: str,
    season: Optional[int] = None,
    team_id: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Record a league membership for an account (idempotent)."""
    if not (account_id and platform and league_id):
        return
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_leagues (account_id, platform, league_id, season, team_id, name)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (account_id, platform, league_id, season) DO UPDATE
                SET team_id = COALESCE(EXCLUDED.team_id, user_leagues.team_id),
                    name    = COALESCE(EXCLUDED.name,    user_leagues.name)
            """,
            (account_id, platform, str(league_id), season, team_id, name),
        )
        conn.commit()


def remove_user_league(
    account_id: int, platform: str, league_id: str, season: Optional[int] = None
) -> None:
    if not (account_id and platform and league_id):
        return
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        if season is None:
            conn.execute(
                "DELETE FROM user_leagues WHERE account_id = %s AND platform = %s AND league_id = %s",
                (account_id, platform, str(league_id)),
            )
        else:
            conn.execute(
                "DELETE FROM user_leagues WHERE account_id = %s AND platform = %s "
                "AND league_id = %s AND season = %s",
                (account_id, platform, str(league_id), season),
            )
        conn.commit()


def list_user_leagues(account_id: int) -> list[dict]:
    """Every league linked to an account, across platforms. Newest first."""
    if not account_id:
        return []
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT platform, league_id, season, team_id, name, added_at "
            "FROM user_leagues WHERE account_id = %s ORDER BY added_at DESC",
            (account_id,),
        ).fetchall()
    return [
        {
            "platform": r["platform"],
            "league_id": r["league_id"],
            "season": r["season"],
            "team_id": r["team_id"],
            "name": r["name"],
        }
        for r in rows
    ]
