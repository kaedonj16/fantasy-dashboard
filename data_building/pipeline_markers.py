"""Durable one-shot markers for cron backfills.

Render's cron containers get a fresh ephemeral disk on every run, so
dotfile markers (``cache/.foo.done``) never survive to the next run and
"one-time" backfills re-ran daily, burning CPU and churning Postgres
writes. These helpers persist the marker in the shared ``app_state``
table instead, which both the cron container and the web service see.
"""
from __future__ import annotations

from dashboard_services.db import get_conn

_MARKER_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def backfill_done(key: str) -> bool:
    """Return True when the named one-shot backfill already completed."""
    with get_conn() as conn:
        conn.execute(_MARKER_TABLE_DDL)
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = %s", (key,)
        ).fetchone()
        return row is not None


def mark_backfill_done(key: str, value: str = "done") -> None:
    """Record the named one-shot backfill as completed."""
    with get_conn() as conn:
        conn.execute(_MARKER_TABLE_DDL)
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )
        conn.commit()
