"""Server-side persistence for the custom draft board (personal ranking overrides).

Stores a small JSON blob of per-player overrides keyed by the owner (account or
Sleeper viewer), the league, and the board format (mode + superflex). See
docs/custom-draft-board.md. The table is created lazily, matching the pattern in
dashboard_services/accounts.py.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_TABLE_READY = False

# A single stored board is a few hundred tiny entries; cap well above that but
# below anything abusive.
_MAX_BLOB = 200_000


def init_draft_board_table() -> None:
    """Create the overrides table once per process (idempotent)."""
    global _TABLE_READY
    if _TABLE_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS draft_board_overrides (
                owner_key   TEXT NOT NULL,
                platform    TEXT NOT NULL,
                league_id   TEXT NOT NULL,
                board_key   TEXT NOT NULL,
                overrides   JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at  TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (owner_key, platform, league_id, board_key)
            )
            """
        )
    _TABLE_READY = True


def get_overrides(owner_key: str, platform: str, league_id: str, board_key: str) -> dict:
    """Return the stored override map, or {} when there is none / on any error."""
    if not owner_key:
        return {}
    try:
        init_draft_board_table()
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT overrides FROM draft_board_overrides "
                    "WHERE owner_key=%s AND platform=%s AND league_id=%s AND board_key=%s",
                    (owner_key, platform or "sleeper", league_id or "", board_key or ""),
                )
                row = cur.fetchone()
        if not row:
            return {}
        val = row["overrides"] if isinstance(row, dict) else row[0]
        if isinstance(val, str):
            val = json.loads(val)
        return val if isinstance(val, dict) else {}
    except Exception:
        logger.warning("[draft-board] get_overrides failed", exc_info=True)
        return {}


def save_overrides(owner_key: str, platform: str, league_id: str, board_key: str, overrides: dict) -> bool:
    """Upsert the override map. An empty map deletes the row. Returns success."""
    if not owner_key:
        return False
    if not isinstance(overrides, dict):
        return False
    try:
        blob = json.dumps(overrides)
    except (TypeError, ValueError):
        return False
    if len(blob) > _MAX_BLOB:
        return False
    try:
        init_draft_board_table()
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            if overrides:
                conn.execute(
                    """
                    INSERT INTO draft_board_overrides
                        (owner_key, platform, league_id, board_key, overrides, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, now())
                    ON CONFLICT (owner_key, platform, league_id, board_key)
                    DO UPDATE SET overrides = EXCLUDED.overrides, updated_at = now()
                    """,
                    (owner_key, platform or "sleeper", league_id or "", board_key or "", blob),
                )
            else:
                conn.execute(
                    "DELETE FROM draft_board_overrides "
                    "WHERE owner_key=%s AND platform=%s AND league_id=%s AND board_key=%s",
                    (owner_key, platform or "sleeper", league_id or "", board_key or ""),
                )
        return True
    except Exception:
        logger.warning("[draft-board] save_overrides failed", exc_info=True)
        return False
