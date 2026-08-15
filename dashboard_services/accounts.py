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

import json
import logging
import os
import base64
import hashlib
import secrets
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
        conn.execute("ALTER TABLE accounts ADD COLUMN IF NOT EXISTS first_name TEXT")
        conn.execute("ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_platform TEXT")
        conn.execute("ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_league_id TEXT")
        conn.execute("ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_season INTEGER")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS account_auth_identities (
                id SERIAL PRIMARY KEY,
                account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                auth_provider TEXT NOT NULL,
                auth_provider_subject TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE (auth_provider, auth_provider_subject)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pending_provider_connections (
                token_hash TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                connection_method TEXT NOT NULL,
                league_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                league_name TEXT,
                encrypted_credentials TEXT NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )"""
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fantasy_provider_connections (
                id                    SERIAL PRIMARY KEY,
                account_id            INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                provider              TEXT NOT NULL,
                connection_method     TEXT NOT NULL,
                encrypted_credentials TEXT,
                status                TEXT NOT NULL DEFAULT 'connected',
                last_authenticated_at TIMESTAMPTZ,
                created_at            TIMESTAMPTZ DEFAULT now(),
                updated_at            TIMESTAMPTZ DEFAULT now(),
                last_synced_at        TIMESTAMPTZ,
                last_successful_sync_at TIMESTAMPTZ,
                last_error_code       TEXT,
                credential_expires_at TIMESTAMPTZ,
                UNIQUE (account_id, provider, connection_method)
            )
            """
        )
        for col, defn in (
            ("last_synced_at", "TIMESTAMPTZ"),
            ("last_successful_sync_at", "TIMESTAMPTZ"),
            ("last_error_code", "TEXT"),
            ("credential_expires_at", "TIMESTAMPTZ"),
        ):
            conn.execute(f"ALTER TABLE fantasy_provider_connections ADD COLUMN IF NOT EXISTS {col} {defn}")
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
                provider_connection_id INTEGER REFERENCES fantasy_provider_connections(id) ON DELETE SET NULL,
                added_at   TIMESTAMPTZ DEFAULT now(),
                UNIQUE (account_id, platform, league_id, season)
            )
            """
        )
        conn.execute(
            """ALTER TABLE user_leagues ADD COLUMN IF NOT EXISTS
               provider_connection_id INTEGER REFERENCES fantasy_provider_connections(id)
               ON DELETE SET NULL"""
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS account_league_visits (
                account_id     INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                platform       TEXT NOT NULL,
                league_id      TEXT NOT NULL,
                season         INTEGER NOT NULL,
                roster_id      TEXT,
                last_visit_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                roster_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb,
                PRIMARY KEY (account_id, platform, league_id, season)
            )
            """
        )
        conn.commit()
    _TABLES_READY = True


def _encrypt_provider_credentials(credentials: dict) -> str:
    """Encrypt provider secrets using a deployment-specific key."""
    secret = os.getenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if not secret:
        raise RuntimeError("Private provider connections are not configured.")
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key).encrypt(json.dumps(credentials).encode("utf-8")).decode("ascii")


def _decrypt_provider_credentials(encrypted: str) -> Optional[dict]:
    secret = os.getenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if not secret:
        return None
    from cryptography.fernet import Fernet, InvalidToken
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    try:
        raw = Fernet(key).decrypt(encrypted.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except (InvalidToken, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_espn_league_credentials(account_id: int, league_id: str, season: int) -> Optional[dict]:
    """Return decrypted credentials only to backend provider code."""
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT c.encrypted_credentials FROM user_leagues l
               JOIN fantasy_provider_connections c ON c.id = l.provider_connection_id
               WHERE l.account_id = %s AND l.platform = 'espn' AND l.league_id = %s
                 AND l.season = %s AND c.status = 'connected'""",
            (account_id, str(league_id), int(season)),
        ).fetchone()
    if not row or not row["encrypted_credentials"]:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    if credentials is None:
        logger.warning("Unable to decrypt stored ESPN credentials for connection")
    return credentials


def stage_private_espn_connection(
    league_id: str, season: int, name: str, swid: str, espn_s2: str,
) -> str:
    """Store validated onboarding secrets behind a short-lived opaque token."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    encrypted = _encrypt_provider_credentials({"swid": swid, "espn_s2": espn_s2})
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM pending_provider_connections WHERE expires_at < now()")
        conn.execute(
            """INSERT INTO pending_provider_connections
               (token_hash,provider,connection_method,league_id,season,league_name,
                encrypted_credentials,expires_at)
               VALUES (%s,'espn','private',%s,%s,%s,%s,now()+interval '15 minutes')""",
            (token_hash, str(league_id), int(season), name, encrypted),
        )
        conn.commit()
    return token


def consume_private_espn_connection(token: str) -> Optional[dict]:
    """Atomically consume one staged connection after Google authentication."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """DELETE FROM pending_provider_connections WHERE token_hash=%s
               AND expires_at >= now() RETURNING league_id,season,league_name,encrypted_credentials""",
            (token_hash,),
        ).fetchone()
        conn.commit()
    if not row:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    if not credentials:
        return None
    return {
        "league_id": row["league_id"], "season": row["season"],
        "name": row["league_name"], **credentials,
    }


def peek_private_espn_connection(token: str, league_id: str, season: int) -> Optional[dict]:
    """Read an unexpired staged connection for an anonymous dashboard session."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT league_id,season,league_name,encrypted_credentials
               FROM pending_provider_connections WHERE token_hash=%s AND expires_at>=now()
               AND league_id=%s AND season=%s""",
            (token_hash, str(league_id), int(season)),
        ).fetchone()
    if not row:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    return ({"league_id": row["league_id"], "season": row["season"],
             "name": row["league_name"], **credentials} if credentials else None)


def add_espn_league_connection(
    account_id: int, league_id: str, season: int, name: str,
    connection_method: str, *, swid: Optional[str] = None, espn_s2: Optional[str] = None,
) -> None:
    """Atomically persist a validated ESPN league and optional encrypted auth."""
    if connection_method not in ("public", "private"):
        raise ValueError("Invalid ESPN connection method.")
    encrypted = None
    if connection_method == "private":
        if not swid or not espn_s2:
            raise ValueError("Private ESPN connections require both credentials.")
        encrypted = _encrypt_provider_credentials({"swid": swid, "espn_s2": espn_s2})
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        connection_id = None
        if connection_method == "private":
            connection_id = conn.execute(
                """INSERT INTO fantasy_provider_connections
                       (account_id, provider, connection_method, encrypted_credentials,
                        status, last_authenticated_at)
                   VALUES (%s, 'espn', 'private', %s, 'connected', now())
                   ON CONFLICT (account_id, provider, connection_method) DO UPDATE SET
                       encrypted_credentials = EXCLUDED.encrypted_credentials,
                       status = 'connected', last_authenticated_at = now(), updated_at = now()
                   RETURNING id""",
                (account_id, encrypted),
            ).fetchone()["id"]
        conn.execute(
            """INSERT INTO user_leagues
                   (account_id, platform, league_id, season, name, provider_connection_id)
               VALUES (%s, 'espn', %s, %s, %s, %s)
               ON CONFLICT (account_id, platform, league_id, season) DO UPDATE SET
                   name = EXCLUDED.name,
                   provider_connection_id = EXCLUDED.provider_connection_id""",
            (account_id, str(league_id), int(season), name, connection_id),
        )
        conn.commit()


def consume_league_visit(
    account_id: int,
    platform: str,
    league_id: str,
    season: int,
    roster_id: Optional[str],
    roster_snapshot: list[dict],
) -> Optional[dict]:
    """Atomically advance and return an account's prior league-visit state.

    The row lock makes the digest account-scoped rather than browser-scoped: if
    two signed-in devices visit in sequence, only the first can consume the same
    activity window.  The roster snapshot travels with the timestamp so value
    and injury changes follow the account across devices too.
    """
    if not (account_id and platform and league_id):
        return None
    init_accounts_tables()
    from dashboard_services.db import get_conn

    key = (account_id, str(platform), str(league_id), int(season))
    with get_conn() as conn:
        previous = conn.execute(
            """
            SELECT roster_id, last_visit_at, roster_snapshot
            FROM account_league_visits
            WHERE account_id = %s AND platform = %s AND league_id = %s AND season = %s
            FOR UPDATE
            """,
            key,
        ).fetchone()
        conn.execute(
            """
            INSERT INTO account_league_visits
                (account_id, platform, league_id, season, roster_id, last_visit_at, roster_snapshot)
            VALUES (%s, %s, %s, %s, %s, now(), %s::jsonb)
            ON CONFLICT (account_id, platform, league_id, season) DO UPDATE
                SET roster_id = EXCLUDED.roster_id,
                    last_visit_at = now(),
                    roster_snapshot = EXCLUDED.roster_snapshot
            """,
            (*key, str(roster_id) if roster_id else None, json.dumps(roster_snapshot or [])),
        )
        conn.commit()
    return dict(previous) if previous else None


def upsert_google_account(
    google_sub: str, email: Optional[str], first_name: Optional[str] = None,
) -> Optional[int]:
    """Resolve a Google subject to exactly one canonical application account."""
    if not google_sub:
        return None
    google_sub = str(google_sub).strip()
    email = (str(email).strip().lower() or None) if email else None
    first_name = (str(first_name).strip() or None) if first_name else None
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT a.id FROM accounts a JOIN account_auth_identities i ON i.account_id=a.id
               WHERE i.auth_provider='google' AND i.auth_provider_subject=%s""",
            (google_sub,),
        ).fetchone()
        if not row:
            row = conn.execute("SELECT id FROM accounts WHERE google_sub=%s", (google_sub,)).fetchone()
        if not row and email:
            row = conn.execute(
                """SELECT a.id FROM accounts a WHERE a.email=%s AND NOT EXISTS (
                   SELECT 1 FROM account_auth_identities i
                   WHERE i.account_id=a.id AND i.auth_provider='google')""",
                (email,),
            ).fetchone()
        if row:
            acct_id = row["id"]
            conn.execute(
                """UPDATE accounts SET last_login_at=now(), google_sub=COALESCE(google_sub,%s),
                   email=COALESCE(email,%s), first_name=COALESCE(%s,first_name) WHERE id=%s""",
                (google_sub, email, first_name, acct_id),
            )
        else:
            acct_id = conn.execute(
                "INSERT INTO accounts (email,google_sub,first_name) VALUES (%s,%s,%s) RETURNING id",
                (email, google_sub, first_name),
            ).fetchone()["id"]
        conn.execute(
            """INSERT INTO account_auth_identities
               (account_id,auth_provider,auth_provider_subject) VALUES (%s,'google',%s)
               ON CONFLICT (auth_provider,auth_provider_subject) DO NOTHING""",
            (acct_id, google_sub),
        )
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


def list_account_platform_ids(account_id: int, platform: str) -> list[str]:
    """Platform user ids (e.g. Sleeper user_ids) linked to this account for a
    platform. Used to validate stored leagues against the live platform list."""
    if not (account_id and platform):
        return []
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT platform_user_id FROM account_identities "
            "WHERE account_id = %s AND platform = %s",
            (account_id, str(platform)),
        ).fetchall()
    return [str(r["platform_user_id"]) for r in rows if r.get("platform_user_id")]


def list_user_leagues(account_id: int) -> list[dict]:
    """Every league linked to an account, across platforms. Newest first."""
    if not account_id:
        return []
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT l.platform,l.league_id,l.season,l.team_id,l.name,l.added_at,
                      c.status AS connection_status,c.last_synced_at,
                      c.last_successful_sync_at,c.last_error_code
               FROM user_leagues l LEFT JOIN fantasy_provider_connections c
                 ON c.id=l.provider_connection_id
               WHERE l.account_id=%s ORDER BY l.added_at DESC""",
            (account_id,),
        ).fetchall()
    return [
        {
            "platform": r["platform"],
            "league_id": r["league_id"],
            "season": r["season"],
            "team_id": r["team_id"],
            "name": r["name"],
            "connection_status": r.get("connection_status") or "connected",
            "last_synced_at": r.get("last_synced_at"),
            "last_successful_sync_at": r.get("last_successful_sync_at"),
            "last_error_code": r.get("last_error_code"),
        }
        for r in rows
    ]


def resolve_account_viewer_for_league(
    account_id: int, platform: str, league_id: str, season: int,
    users: list[dict], rosters: list[dict],
) -> Optional[dict]:
    """Resolve an account's stable identity to its team in one saved league.

    The league-scoped ``team_id`` wins when present. Otherwise a stable platform
    user id is matched to the roster owner and the resulting team is persisted.
    Display names are never used to establish ownership.
    """
    if not (account_id and platform and league_id and season):
        return None
    init_accounts_tables()
    from dashboard_services.db import get_conn
    key = (account_id, platform, str(league_id), int(season))
    with get_conn() as conn:
        membership = conn.execute(
            """SELECT team_id FROM user_leagues WHERE account_id=%s AND platform=%s
               AND league_id=%s AND season=%s""", key,
        ).fetchone()
        if not membership:
            return None
        identity_rows = conn.execute(
            """SELECT platform_user_id,handle FROM account_identities
               WHERE account_id=%s AND platform=%s""", (account_id, platform),
        ).fetchall()

    identities = {str(row["platform_user_id"]): row.get("handle") for row in identity_rows}
    stored_team_id = str(membership.get("team_id") or "")
    roster = next((r for r in rosters or []
                   if stored_team_id and str(r.get("roster_id") or "") == stored_team_id), None)
    if roster is None and identities:
        roster = next((r for r in rosters or []
                       if str(r.get("owner_id") or "") in identities), None)
    if roster is None:
        return None

    roster_id = str(roster.get("roster_id") or "")
    owner_id = str(roster.get("owner_id") or "")
    user = next((u for u in users or []
                 if str(u.get("user_id") or "") == owner_id), None) or {}
    user_meta, roster_meta = user.get("metadata") or {}, roster.get("metadata") or {}
    username = user.get("username") or identities.get(owner_id) or user.get("display_name") or owner_id
    team_name = (roster_meta.get("team_name") or user_meta.get("team_name")
                 or user.get("display_name") or username or f"Roster {roster_id}")
    if roster_id and roster_id != stored_team_id:
        with get_conn() as conn:
            conn.execute(
                """UPDATE user_leagues SET team_id=%s WHERE account_id=%s AND platform=%s
                   AND league_id=%s AND season=%s""", (roster_id, *key),
            )
            conn.commit()
    return {"viewer_username": username, "viewer_user_id": owner_id or None,
            "viewer_roster_id": roster_id or None, "viewer_team_name": team_name}


def set_last_active_league(account_id: int, platform: str, league_id: str, season: int) -> bool:
    """Record activity only for an account-owned saved league."""
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        owned = conn.execute(
            """SELECT 1 FROM user_leagues WHERE account_id=%s AND platform=%s
               AND league_id=%s AND season=%s""",
            (account_id, platform, str(league_id), int(season)),
        ).fetchone()
        if not owned:
            return False
        conn.execute(
            """UPDATE accounts SET last_active_platform=%s,last_active_league_id=%s,
               last_active_season=%s WHERE id=%s""",
            (platform, str(league_id), int(season), account_id),
        )
        conn.commit()
    return True


def get_post_login_destination(account_id: int) -> Optional[str]:
    """Choose a saved destination using database metadata only."""
    leagues = list_user_leagues(account_id)
    if not leagues:
        return None
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        account = conn.execute(
            "SELECT last_active_platform,last_active_league_id,last_active_season FROM accounts WHERE id=%s",
            (account_id,),
        ).fetchone()
    chosen = None
    if account and account.get("last_active_league_id"):
        chosen = next((league for league in leagues
                       if league["platform"] == account["last_active_platform"]
                       and str(league["league_id"]) == str(account["last_active_league_id"])
                       and int(league["season"] or 0) == int(account["last_active_season"] or 0)), None)
    if chosen is None and len(leagues) == 1:
        chosen = leagues[0]
    if chosen:
        return f"/{chosen['platform']}/{chosen['season']}/{chosen['league_id']}/dashboard"
    return "/portfolio"


def mark_espn_connection_status(
    account_id: int, league_id: str, season: int, status: str,
    error_code: Optional[str] = None,
) -> None:
    """Update provider status through an account-owned league association."""
    if status not in ("connected", "reauth_required", "sync_error", "disconnected"):
        raise ValueError("Invalid provider connection status")
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """UPDATE fantasy_provider_connections c SET status=%s,last_error_code=%s,updated_at=now()
               FROM user_leagues l WHERE l.provider_connection_id=c.id AND l.account_id=%s
               AND l.platform='espn' AND l.league_id=%s AND l.season=%s""",
            (status, (error_code or "")[:64] or None, account_id, str(league_id), int(season)),
        )
        conn.commit()


def replace_espn_credentials(
    account_id: int, league_id: str, season: int, swid: str, espn_s2: str,
) -> bool:
    """Replace credentials only through an account-owned league connection."""
    encrypted = _encrypt_provider_credentials({"swid": swid, "espn_s2": espn_s2})
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """UPDATE fantasy_provider_connections c SET encrypted_credentials=%s,
               status='connected',last_error_code=NULL,last_authenticated_at=now(),updated_at=now()
               FROM user_leagues l WHERE l.provider_connection_id=c.id AND l.account_id=%s
               AND l.platform='espn' AND l.league_id=%s AND l.season=%s RETURNING c.id""",
            (encrypted, account_id, str(league_id), int(season)),
        ).fetchone()
        conn.commit()
    return bool(row)


def owns_user_league(account_id: int, platform: str, league_id: str, season: int) -> bool:
    """Authorization primitive for authenticated league mutations."""
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        return bool(conn.execute(
            """SELECT 1 FROM user_leagues WHERE account_id=%s AND platform=%s
               AND league_id=%s AND season=%s""",
            (account_id, platform, str(league_id), int(season)),
        ).fetchone())
