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
_ENCRYPTION_FALLBACK_LOGGED = False


class ProviderCredentialConfigurationError(RuntimeError):
    """Raised when the server has no stable secret for provider credentials."""


def _provider_credential_secret() -> str:
    """Return a stable credential-encryption secret without exposing its value.

    ``PROVIDER_CREDENTIAL_ENCRYPTION_KEY`` keeps provider credentials isolated
    when configured. Existing deployments already require a stable
    ``FLASK_SECRET_KEY``, so use it as a backwards-compatible fallback rather
    than failing anonymous private-league onboarding after ESPN validation.
    """
    global _ENCRYPTION_FALLBACK_LOGGED
    secret = os.getenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if secret:
        return secret
    secret = os.getenv("FLASK_SECRET_KEY", "").strip()
    if secret:
        if not _ENCRYPTION_FALLBACK_LOGGED:
            logger.warning(
                "PROVIDER_CREDENTIAL_ENCRYPTION_KEY is unset; using the stable FLASK_SECRET_KEY fallback"
            )
            _ENCRYPTION_FALLBACK_LOGGED = True
        return secret
    raise ProviderCredentialConfigurationError(
        "No provider credential encryption key is configured."
    )


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
                credential_expires_at TIMESTAMPTZ
            )
            """
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS fantasy_provider_connections_account_provider_idx
               ON fantasy_provider_connections (account_id, provider, connection_method)"""
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
    secret = _provider_credential_secret()
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key).encrypt(json.dumps(credentials).encode("utf-8")).decode("ascii")


def _decrypt_provider_credentials(encrypted: str) -> Optional[dict]:
    try:
        secret = _provider_credential_secret()
    except ProviderCredentialConfigurationError:
        return None
    from cryptography.fernet import Fernet, InvalidToken
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    try:
        raw = Fernet(key).decrypt(encrypted.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except (InvalidToken, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_provider_league_credentials(
    account_id: int, provider: str, league_id: str, season: int,
) -> Optional[dict]:
    """Return decrypted credentials only to backend provider code."""
    provider = str(provider or "").strip().lower()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT c.encrypted_credentials FROM user_leagues l
               JOIN fantasy_provider_connections c ON c.id = l.provider_connection_id
               WHERE l.account_id = %s AND l.platform = %s AND l.league_id = %s
                 AND l.season = %s AND c.status = 'connected'""",
            (account_id, provider, str(league_id), int(season)),
        ).fetchone()
        # ESPN league IDs persist across seasons; My Leagues bumps the display
        # year without rewriting the saved row, so fall back to the latest
        # connected credentials for that league.
        if (not row or not row["encrypted_credentials"]) and provider == "espn":
            row = conn.execute(
                """SELECT c.encrypted_credentials FROM user_leagues l
                   JOIN fantasy_provider_connections c ON c.id = l.provider_connection_id
                   WHERE l.account_id = %s AND l.platform = %s AND l.league_id = %s
                     AND c.status = 'connected'
                   ORDER BY l.season DESC LIMIT 1""",
                (account_id, provider, str(league_id)),
            ).fetchone()
    if not row or not row["encrypted_credentials"]:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    if credentials is None:
        logger.warning("Unable to decrypt stored %s credentials for connection", provider)
    return credentials


def get_espn_league_credentials(account_id: int, league_id: str, season: int) -> Optional[dict]:
    """ESPN-specific alias for get_provider_league_credentials."""
    return get_provider_league_credentials(account_id, "espn", league_id, season)


def get_any_espn_account_credentials(account_id: int) -> Optional[dict]:
    """Any connected ESPN SWID/espn_s2 on this Google account.

    Private ESPN leagues need cookies. Users often paste them once when linking
    the first league; later ESPN leagues on the same account should be able to
    reuse that login for Redzone / live fetches.
    """
    if not account_id:
        return None
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT c.encrypted_credentials FROM user_leagues l
               JOIN fantasy_provider_connections c ON c.id = l.provider_connection_id
               WHERE l.account_id = %s AND l.platform = 'espn'
                 AND c.status = 'connected' AND c.encrypted_credentials IS NOT NULL
               ORDER BY l.season DESC, c.updated_at DESC NULLS LAST
               LIMIT 1""",
            (int(account_id),),
        ).fetchone()
    if not row or not row["encrypted_credentials"]:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    if credentials is None:
        logger.warning("Unable to decrypt stored espn credentials for account fallback")
    return credentials


def stage_private_provider_connection(
    provider: str, league_id: str, season: int, name: str, credentials: dict,
) -> str:
    """Store validated onboarding secrets behind a short-lived opaque token."""
    provider = str(provider or "").strip().lower()
    if not provider or not isinstance(credentials, dict) or not credentials:
        raise ValueError("Private provider connections require credentials.")
    # Never persist passwords — callers must pass derived tokens/cookies/keys only.
    safe = {k: v for k, v in credentials.items()
            if k.lower() not in {"password", "passwd", "pass"} and v}
    if not safe:
        raise ValueError("Private provider connections require credentials.")
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    encrypted = _encrypt_provider_credentials(safe)
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM pending_provider_connections WHERE expires_at < now()")
        conn.execute(
            """INSERT INTO pending_provider_connections
               (token_hash,provider,connection_method,league_id,season,league_name,
                encrypted_credentials,expires_at)
               VALUES (%s,%s,'private',%s,%s,%s,%s,now()+interval '15 minutes')""",
            (token_hash, provider, str(league_id), int(season), name, encrypted),
        )
        conn.commit()
    return token


def stage_private_espn_connection(
    league_id: str, season: int, name: str, swid: str, espn_s2: str,
) -> str:
    """Store validated ESPN onboarding secrets behind a short-lived opaque token."""
    return stage_private_provider_connection(
        "espn", league_id, season, name, {"swid": swid, "espn_s2": espn_s2},
    )


def consume_private_provider_connection(token: str) -> Optional[dict]:
    """Atomically consume one staged connection after Google authentication."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """DELETE FROM pending_provider_connections WHERE token_hash=%s
               AND expires_at >= now()
               RETURNING provider,league_id,season,league_name,encrypted_credentials""",
            (token_hash,),
        ).fetchone()
        conn.commit()
    if not row:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    if not credentials:
        return None
    return {
        "provider": row["provider"], "league_id": row["league_id"],
        "season": row["season"], "name": row["league_name"], **credentials,
    }


def consume_private_espn_connection(token: str) -> Optional[dict]:
    """ESPN-compatible consume; drops the provider key for historical callers."""
    pending = consume_private_provider_connection(token)
    if not pending:
        return None
    pending.pop("provider", None)
    return pending


def peek_private_provider_connection(
    token: str, provider: str, league_id: str, season: int,
) -> Optional[dict]:
    """Read an unexpired staged connection for an anonymous dashboard session."""
    if not token:
        return None
    provider = str(provider or "").strip().lower()
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT league_id,season,league_name,encrypted_credentials
               FROM pending_provider_connections WHERE token_hash=%s AND expires_at>=now()
               AND provider=%s AND league_id=%s AND season=%s""",
            (token_hash, provider, str(league_id), int(season)),
        ).fetchone()
    if not row:
        return None
    credentials = _decrypt_provider_credentials(row["encrypted_credentials"])
    return ({"league_id": row["league_id"], "season": row["season"],
             "name": row["league_name"], **credentials} if credentials else None)


def peek_private_espn_connection(token: str, league_id: str, season: int) -> Optional[dict]:
    return peek_private_provider_connection(token, "espn", league_id, season)


def add_provider_league_connection(
    account_id: int, provider: str, league_id: str, season: int, name: str,
    connection_method: str, *, credentials: Optional[dict] = None,
    team_id: Optional[str] = None,
) -> None:
    """Atomically persist a validated league and optional encrypted auth.

    Private credentials are stored per league via ``provider_connection_id``.
    Passwords must never appear in ``credentials``.
    """
    provider = str(provider or "").strip().lower()
    if connection_method not in ("public", "private"):
        raise ValueError("Invalid provider connection method.")
    encrypted = None
    if connection_method == "private":
        if not isinstance(credentials, dict) or not credentials:
            raise ValueError("Private provider connections require credentials.")
        safe = {k: v for k, v in credentials.items()
                if k.lower() not in {"password", "passwd", "pass"} and v}
        if not safe:
            raise ValueError("Private provider connections require credentials.")
        encrypted = _encrypt_provider_credentials(safe)
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        connection_id = None
        if connection_method == "private":
            existing = conn.execute(
                """SELECT c.id FROM user_leagues l
                   JOIN fantasy_provider_connections c ON c.id = l.provider_connection_id
                   WHERE l.account_id=%s AND l.platform=%s AND l.league_id=%s AND l.season=%s""",
                (account_id, provider, str(league_id), int(season)),
            ).fetchone()
            if existing:
                connection_id = conn.execute(
                    """UPDATE fantasy_provider_connections SET encrypted_credentials=%s,
                           status='connected', last_authenticated_at=now(), updated_at=now(),
                           last_error_code=NULL
                       WHERE id=%s RETURNING id""",
                    (encrypted, existing["id"]),
                ).fetchone()["id"]
            else:
                # ESPN historically shared one private cookie row per account.
                # Reuse that shared row when present so existing leagues keep working.
                if provider == "espn":
                    shared = conn.execute(
                        """SELECT id FROM fantasy_provider_connections
                           WHERE account_id=%s AND provider='espn' AND connection_method='private'
                           LIMIT 1""",
                        (account_id,),
                    ).fetchone()
                    if shared:
                        connection_id = conn.execute(
                            """UPDATE fantasy_provider_connections SET encrypted_credentials=%s,
                                   status='connected', last_authenticated_at=now(), updated_at=now(),
                                   last_error_code=NULL
                               WHERE id=%s RETURNING id""",
                            (encrypted, shared["id"]),
                        ).fetchone()["id"]
                if connection_id is None:
                    connection_id = conn.execute(
                        """INSERT INTO fantasy_provider_connections
                               (account_id, provider, connection_method, encrypted_credentials,
                                status, last_authenticated_at)
                           VALUES (%s, %s, 'private', %s, 'connected', now())
                           RETURNING id""",
                        (account_id, provider, encrypted),
                    ).fetchone()["id"]
        conn.execute(
            """INSERT INTO user_leagues
                   (account_id, platform, league_id, season, name, team_id, provider_connection_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (account_id, platform, league_id, season) DO UPDATE SET
                   name = EXCLUDED.name,
                   team_id = COALESCE(EXCLUDED.team_id, user_leagues.team_id),
                   provider_connection_id = COALESCE(
                       EXCLUDED.provider_connection_id, user_leagues.provider_connection_id
                   )""",
            (account_id, provider, str(league_id), int(season), name, team_id, connection_id),
        )
        conn.commit()
    # Fleaflicker private login stores the owner id (metadata.flea_owner_id), not
    # the team/roster id. Persist it as a platform identity so saved leagues can
    # resolve "your team" on reconnect even when team_id was never picked.
    if provider == "fleaflicker" and credentials:
        flea_uid = str(credentials.get("flea_user_id") or "").strip()
        if flea_uid:
            link_platform_identity(account_id, "fleaflicker", flea_uid)


def add_espn_league_connection(
    account_id: int, league_id: str, season: int, name: str,
    connection_method: str, *, swid: Optional[str] = None, espn_s2: Optional[str] = None,
) -> None:
    """Atomically persist a validated ESPN league and optional encrypted auth."""
    credentials = None
    if connection_method == "private":
        if not swid or not espn_s2:
            raise ValueError("Private ESPN connections require both credentials.")
        credentials = {"swid": swid, "espn_s2": espn_s2}
    add_provider_league_connection(
        account_id, "espn", league_id, season, name, connection_method,
        credentials=credentials,
    )
    # The SWID is the ESPN owner id on rosters. Storing it as a platform
    # identity lets My Leagues and the dashboard resolve "your team" even when
    # the user connected cookies without picking a roster in the link modal.
    if swid:
        normalized = str(swid).strip()
        if normalized and not (normalized.startswith("{") and normalized.endswith("}")):
            normalized = "{" + normalized.strip("{}") + "}"
        if normalized:
            link_platform_identity(account_id, "espn", normalized)

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
) -> tuple[Optional[int], bool]:
    """Resolve a Google subject to exactly one canonical application account.

    Returns ``(account_id, created)``. ``created`` is True only when a new
    ``accounts`` row was inserted (true first-time signup).
    """
    if not google_sub:
        return None, False
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
        created = False
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
            created = True
        conn.execute(
            """INSERT INTO account_auth_identities
               (account_id,auth_provider,auth_provider_subject) VALUES (%s,'google',%s)
               ON CONFLICT (auth_provider,auth_provider_subject) DO NOTHING""",
            (acct_id, google_sub),
        )
        conn.commit()
        return acct_id, created


def link_platform_identity(
    account_id: int, platform: str, platform_user_id: str, handle: Optional[str] = None
) -> str:
    """Attach a platform identity (Sleeper user id / Yahoo guid) to an account.

    Returns:
      ``linked``   — newly inserted
      ``already``  — already owned by this account (handle refreshed)
      ``conflict`` — identity belongs to a *different* account (not stolen)
      ``noop``     — missing args

    Never re-points an identity from another account: that would let a thief
    who typed someone else's Sleeper username + their own Google claim PRO.
    """
    if not (account_id and platform and platform_user_id):
        return "noop"
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT account_id FROM account_identities
            WHERE platform = %s AND platform_user_id = %s
            LIMIT 1
            """,
            (platform, str(platform_user_id)),
        ).fetchone()
        if row:
            existing = int(row["account_id"])
            if existing != int(account_id):
                logger.warning(
                    "[accounts] refuse identity steal: %s:%s owned by acct %s, not %s",
                    platform, platform_user_id, existing, account_id,
                )
                return "conflict"
            if handle:
                conn.execute(
                    """
                    UPDATE account_identities SET handle = COALESCE(%s, handle)
                    WHERE platform = %s AND platform_user_id = %s
                    """,
                    (handle, platform, str(platform_user_id)),
                )
                conn.commit()
            return "already"
        conn.execute(
            """
            INSERT INTO account_identities (account_id, platform, platform_user_id, handle)
            VALUES (%s, %s, %s, %s)
            """,
            (account_id, platform, str(platform_user_id), handle),
        )
        conn.commit()
        return "linked"


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


def get_saved_league_name(account_id, platform, league_id) -> str:
    """The stored display name for one saved league, newest season first."""
    if not (account_id and platform and league_id):
        return ""
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """SELECT name FROM user_leagues
               WHERE account_id=%s AND platform=%s AND league_id=%s
               ORDER BY season DESC LIMIT 1""",
            (int(account_id), str(platform).lower(), str(league_id)),
        ).fetchone()
    return str((row or {}).get("name") or "").strip()


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


def resolve_account_leagues(account_id: int, enrichments=None, current_season=None) -> list[dict]:
    """Return the durable, cross-platform portfolio for one app account.

    App-account authentication is established only by Google. Callers must pass
    the ``account_id`` already present in the authenticated session; this helper
    never looks up an account from a provider identity.

    ``user_leagues`` is authoritative. Optional provider records may enrich an
    existing association, but discovery alone never re-attaches an explicitly
    unlinked league, and sparse values never erase stored account/team metadata.
    Entries are deduplicated by ``platform + league_id`` so the contract remains
    generic for future providers.
    """
    if not account_id:
        return []
    try:
        season = int(current_season or 0)
    except (TypeError, ValueError):
        season = 0

    leagues_by_key = {}
    for saved in list_user_leagues(account_id):
        platform = str(saved.get("platform") or "sleeper").lower()
        league_id = str(saved.get("league_id") or "")
        if not league_id:
            continue
        saved_season = saved.get("season") or season or None
        if platform == "espn" and season and saved_season and int(saved_season) < season:
            saved_season = season
        leagues_by_key[(platform, league_id)] = dict(
            saved, platform=platform, league_id=league_id, season=saved_season,
        )

    for live in enrichments or []:
        platform = str(live.get("platform") or "").lower()
        league_id = str(live.get("league_id") or "")
        if not platform or not league_id:
            continue
        key = (platform, league_id)
        stored = leagues_by_key.get(key)
        if stored is None:
            continue
        # Null/empty provider fields mean "not returned", not "delete the
        # account-owned value". Truthy live values may safely enrich metadata.
        merged = dict(stored)
        merged.update({k: v for k, v in live.items() if v is not None and v != ""})
        merged.update(platform=platform, league_id=league_id)
        leagues_by_key[key] = merged

    return list(leagues_by_key.values())


def _hide_confirmed_deleted_sleeper_leagues(leagues: list[dict], live_leagues: list[dict]) -> list[dict]:
    """Hide Sleeper leagues whose canonical endpoint confirms deletion.

    A league merely missing from a membership response is not enough: the user
    may have left it, the response may be stale, or Sleeper may be degraded. We
    retain the durable association on every request error and only suppress the
    card when Sleeper's direct league lookup successfully returns no league.
    The database row is intentionally preserved for outage safety and audit.
    """
    live_ids = {str(league.get("league_id") or "") for league in live_leagues}
    candidates = [
        league for league in leagues
        if str(league.get("platform") or "").lower() == "sleeper"
        and str(league.get("league_id") or "") not in live_ids
    ]
    if not candidates:
        return leagues

    from dashboard_services.api import sleeper_league_exists
    deleted_ids = set()
    for league in candidates:
        league_id = str(league.get("league_id") or "")
        try:
            if sleeper_league_exists(league_id) is False:
                deleted_ids.add(league_id)
        except Exception:
            # Defensive adapter boundary: any probe failure retains the card.
            continue
    if not deleted_ids:
        return leagues
    return [
        league for league in leagues
        if not (
            str(league.get("platform") or "").lower() == "sleeper"
            and str(league.get("league_id") or "") in deleted_ids
        )
    ]


def resolve_my_leagues(viewer_user_id, account_id, current_season):
    """The canonical "my leagues" set, shared by the My Leagues page
    (/portfolio) and the league switcher (/api/my-leagues) so the two never show
    different leagues.

    Account-backed first, with live Sleeper enrichment:

      * Every row saved to the Google account's ``user_leagues`` is included,
        regardless of platform. The database is the durable source of truth for
        a fresh Google login and remains available during provider outages.
      * Sleeper memberships from every identity linked to the account are
        fetched live to refresh saved metadata. A leftover session
        ``viewer_user_id`` from another platform is never treated as a Sleeper
        user. Discovery alone neither attaches new leagues nor removes saved
        account leagues.
      * For ESPN, Yahoo, and any future provider, saved leagues do not depend on
        a provider-specific browser session being present after Google login.
        ESPN league IDs persist across seasons, so a league linked in a prior
        year is bumped to the current season (it's the same league); Yahoo keys
        are season-specific and kept as stored.

    Returns ``(leagues, season)`` where ``season`` is the season actually used
    (after any prior-season fallback) and each league dict carries at least
    platform / league_id / season / name. Sleeper entries are the raw provider
    dicts (so downstream summary code sees exactly what it did before); ESPN /
    Yahoo entries are the stored account rows.
    """
    season = int(current_season or 0)
    sleeper_ids = []
    if account_id:
        # Google account: only Sleeper identities actually linked to the
        # account. Session viewer_user_id is league-scoped — opening ESPN /
        # Fleaflicker / Yahoo overwrites it with that platform's owner id,
        # which must not be sent to Sleeper as if it were a user.
        sleeper_ids.extend(list_account_platform_ids(account_id, "sleeper"))
    elif viewer_user_id:
        sleeper_ids.append(str(viewer_user_id))
    # Preserve order while avoiding duplicate provider requests when the same
    # identity is listed twice.
    sleeper_ids = list(dict.fromkeys(sleeper_ids))

    from dashboard_services.api import get_sleeper_user_leagues

    def _fetch_sleeper_memberships(fetch_season):
        memberships = []
        for sleeper_id in sleeper_ids:
            try:
                memberships.extend(get_sleeper_user_leagues(sleeper_id, fetch_season) or [])
            except Exception:
                # One stale/unavailable identity must not hide the other linked
                # identities or the account's non-Sleeper leagues.
                continue
        return memberships

    raw = _fetch_sleeper_memberships(season) if season else []
    if not raw and season:
        previous = _fetch_sleeper_memberships(season - 1)
        if previous:
            raw, season = previous, season - 1

    live_sleeper = []
    for lg in raw:
        league_id = str(lg.get("league_id") or "")
        if not league_id:
            continue
        live_sleeper.append(dict(
            lg, platform="sleeper", league_id=league_id,
            season=int(lg.get("season") or season or 0),
        ))

    if account_id:
        # Google account context: durable saved portfolio plus optional provider
        # enrichment. Provider availability cannot remove a saved association.
        account_leagues = resolve_account_leagues(account_id, live_sleeper, season)
        return _hide_confirmed_deleted_sleeper_leagues(account_leagues, live_sleeper), season

    # Provider-only context: show only what that provider session discovered.
    # Never infer or activate a Google account from a matching provider identity.
    by_key = {(m["platform"], m["league_id"]): m for m in live_sleeper}
    return list(by_key.values()), season


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
    from utils.redzone_user import match_viewer_roster, owner_id_variants
    platform = str(platform).lower()
    key = (account_id, platform, str(league_id), int(season))
    with get_conn() as conn:
        membership = conn.execute(
            """SELECT team_id, season FROM user_leagues WHERE account_id=%s AND platform=%s
               AND league_id=%s AND season=%s""", key,
        ).fetchone()
        # ESPN league IDs persist year to year; the portfolio bumps the display
        # season without copying the user_leagues row.
        if not membership and platform == "espn":
            membership = conn.execute(
                """SELECT team_id, season FROM user_leagues WHERE account_id=%s AND platform=%s
                   AND league_id=%s ORDER BY season DESC LIMIT 1""",
                (account_id, platform, str(league_id)),
            ).fetchone()
        # Fleaflicker saves are keyed by season; opening a new year before re-saving
        # should still resolve the stored team from the latest membership row.
        if not membership and platform == "fleaflicker":
            membership = conn.execute(
                """SELECT team_id, season FROM user_leagues WHERE account_id=%s AND platform=%s
                   AND league_id=%s ORDER BY season DESC LIMIT 1""",
                (account_id, platform, str(league_id)),
            ).fetchone()
        if not membership:
            return None
        identity_rows = conn.execute(
            """SELECT platform_user_id,handle FROM account_identities
               WHERE account_id=%s AND platform=%s""", (account_id, platform),
        ).fetchall()

    identities = {str(row["platform_user_id"]): row.get("handle") for row in identity_rows}
    stored_team_id = str(membership.get("team_id") or "")
    stored_season = int(membership.get("season") or season)
    ident_ids = []
    for pid in identities:
        ident_ids.extend(owner_id_variants(pid))
    roster = match_viewer_roster(
        rosters, team_id=stored_team_id, owner_ids=ident_ids,
    )
    if roster is None and platform == "espn":
        creds = get_espn_league_credentials(account_id, league_id, int(season)) or {}
        roster = match_viewer_roster(rosters, owner_ids=list(owner_id_variants(creds.get("swid"))))
    # Fleaflicker stores team ids on rosters, but private-login credentials carry the
    # Fleaflicker owner id — match via metadata.flea_owner_id when team_id was never
    # persisted (common on reconnect / saved-league open paths).
    if roster is None and platform == "fleaflicker":
        from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
        resolved_team_id = None
        try:
            credentials = get_provider_league_credentials(account_id, platform, league_id, season) or {}
        except Exception:
            credentials = {}
        flea_uid = str(credentials.get("flea_user_id") or "").strip()
        if flea_uid:
            resolved_team_id = resolve_fleaflicker_team_id(users, flea_user_id=flea_uid)
        if not resolved_team_id and identities:
            for platform_uid in identities:
                resolved_team_id = resolve_fleaflicker_team_id(users, flea_user_id=platform_uid)
                if resolved_team_id:
                    break
        if resolved_team_id:
            roster = next(
                (r for r in rosters or [] if str(r.get("roster_id") or "") == str(resolved_team_id)),
                None,
            )
    if roster is None:
        return None

    roster_id = str(roster.get("roster_id") or "")
    owner_id = str(roster.get("owner_id") or "")
    user = next((u for u in users or []
                 if str(u.get("user_id") or "") == owner_id), None) or {}
    user_meta, roster_meta = user.get("metadata") or {}, roster.get("metadata") or {}
    username = (
        user.get("username")
        or identities.get(owner_id)
        or next((identities[k] for k in identities if owner_id in owner_id_variants(k)), None)
        or user.get("display_name")
        or owner_id
    )
    team_name = (roster_meta.get("team_name") or user_meta.get("team_name")
                 or user.get("display_name") or username or f"Roster {roster_id}")
    if roster_id and roster_id != stored_team_id:
        with get_conn() as conn:
            conn.execute(
                """UPDATE user_leagues SET team_id=%s WHERE account_id=%s AND platform=%s
                   AND league_id=%s AND season=%s""",
                (roster_id, account_id, platform, str(league_id), stored_season),
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
    leagues = resolve_account_leagues(account_id)  # shared resolver loads list_user_leagues
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


def mark_provider_connection_status(
    account_id: int, provider: str, league_id: str, season: int, status: str,
    error_code: Optional[str] = None,
) -> None:
    """Update provider status through an account-owned league association."""
    if status not in ("connected", "reauth_required", "sync_error", "disconnected"):
        raise ValueError("Invalid provider connection status")
    provider = str(provider or "").strip().lower()
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """UPDATE fantasy_provider_connections c SET status=%s,last_error_code=%s,updated_at=now()
               FROM user_leagues l WHERE l.provider_connection_id=c.id AND l.account_id=%s
               AND l.platform=%s AND l.league_id=%s AND l.season=%s""",
            (status, (error_code or "")[:64] or None, account_id, provider,
             str(league_id), int(season)),
        )
        conn.commit()


def mark_espn_connection_status(
    account_id: int, league_id: str, season: int, status: str,
    error_code: Optional[str] = None,
) -> None:
    mark_provider_connection_status(
        account_id, "espn", league_id, season, status, error_code=error_code,
    )


def replace_provider_credentials(
    account_id: int, provider: str, league_id: str, season: int, credentials: dict,
) -> bool:
    """Replace credentials only through an account-owned league connection."""
    provider = str(provider or "").strip().lower()
    safe = {k: v for k, v in (credentials or {}).items()
            if k.lower() not in {"password", "passwd", "pass"} and v}
    if not safe:
        return False
    encrypted = _encrypt_provider_credentials(safe)
    init_accounts_tables()
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            """UPDATE fantasy_provider_connections c SET encrypted_credentials=%s,
               status='connected',last_error_code=NULL,last_authenticated_at=now(),updated_at=now()
               FROM user_leagues l WHERE l.provider_connection_id=c.id AND l.account_id=%s
               AND l.platform=%s AND l.league_id=%s AND l.season=%s RETURNING c.id""",
            (encrypted, account_id, provider, str(league_id), int(season)),
        ).fetchone()
        conn.commit()
    return bool(row)


def replace_espn_credentials(
    account_id: int, league_id: str, season: int, swid: str, espn_s2: str,
) -> bool:
    """Replace ESPN credentials only through an account-owned league connection."""
    return replace_provider_credentials(
        account_id, "espn", league_id, season, {"swid": swid, "espn_s2": espn_s2},
    )

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
