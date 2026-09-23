"""Shareable public links for Season Wrapped / Weekly Wrapped decks.

A share stores the *rendered* overlay HTML (slides + branding) plus the share
payload, so the public link keeps working even if the league is deleted and no
auth is needed to view it. The stored payload is exactly what the image Share
card already exposes (league name, week, highlights, slide contents) -- no
rosters, no user identity.

Payloads live in Postgres, not in process memory.
"""
from __future__ import annotations

import logging
import secrets

logger = logging.getLogger(__name__)

_TABLES_READY = False

#: Links expire a year after creation; the public route 404s past expiry.
SHARE_TTL_SQL = "INTERVAL '1 year'"


def init_wrapped_shares_table() -> None:
    """Create the wrapped_shares table once per process."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wrapped_shares (
                token        TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,           -- 'season' | 'weekly'
                ns           TEXT NOT NULL DEFAULT 'wrapped',
                overlay_html TEXT NOT NULL,
                share_data   JSONB,
                label        TEXT,                    -- e.g. "Blackedraw — Week 2 Wrapped" (OG title)
                created_at   TIMESTAMPTZ DEFAULT now(),
                expires_at   TIMESTAMPTZ DEFAULT now() + INTERVAL '1 year'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wrapped_shares_expires "
            "ON wrapped_shares (expires_at)"
        )
        conn.commit()
    _TABLES_READY = True


def create_wrapped_share(*, kind: str, ns: str, overlay_html: str,
                         share_data: dict | None, label: str) -> str:
    """Store a share and return its token. Raises on DB errors."""
    from dashboard_services.db import get_conn
    from psycopg.types.json import Json
    init_wrapped_shares_table()
    token = secrets.token_urlsafe(16)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO wrapped_shares (token, kind, ns, overlay_html, share_data, label)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (token, kind, ns, overlay_html, Json(share_data or {}), label),
        )
        conn.commit()
    return token


def get_wrapped_share(token: str) -> dict | None:
    """Fetch a share by token; None when unknown or expired."""
    from dashboard_services.db import get_conn
    init_wrapped_shares_table()
    token = (token or "").strip()
    if not token or len(token) > 128:
        return None
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT token, kind, ns, overlay_html, share_data, label, created_at
            FROM wrapped_shares
            WHERE token = %s AND expires_at > now()
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    return dict(row)
