"""Durable store for the O-line unit ratings built by the weekly cron.

The Wednesday cron builds these ratings in an ephemeral Render container, so
the ``cache/oline_ratings_s{season}.json`` file it writes never survives to be
read by the web dyno. This table is the durable home: the cron upserts here,
the web app reads here. The flat file remains as a fallback for local dev and
backtests running without a database.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TABLES_READY = False


def init_oline_tables() -> None:
    """Create the oline_ratings table once per process (idempotent)."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oline_ratings (
                season INTEGER PRIMARY KEY,
                through_week INTEGER,
                generated_at TIMESTAMPTZ,
                ratings JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.commit()
    _TABLES_READY = True


def save_oline_ratings(season: int, payload: dict) -> bool:
    """Upsert one season of ratings. Returns False (never raises) on any DB
    failure so a missing/unreachable database degrades to the flat file."""
    try:
        ratings = payload.get("ratings") or {}
        if not ratings:
            logger.warning("[oline_store] refusing to save empty ratings "
                           "for season %s", season)
            return False
        init_oline_tables()
        from dashboard_services.db import get_conn
        from psycopg.types.json import Json
        generated_at = payload.get("generated_at")
        if isinstance(generated_at, str):
            try:
                from datetime import datetime

                generated_at = datetime.fromisoformat(generated_at)
            except ValueError:
                generated_at = None
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO oline_ratings (season, through_week, generated_at,
                                           ratings, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (season) DO UPDATE SET
                    through_week = EXCLUDED.through_week,
                    generated_at = EXCLUDED.generated_at,
                    ratings = EXCLUDED.ratings,
                    updated_at = now()
                """,
                (
                    int(season),
                    payload.get("through_week"),
                    generated_at,
                    Json(ratings),
                ),
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.warning("[oline_store] save failed for season %s: %s",
                       season, exc)
        return False


def load_oline_ratings(season: int) -> dict | None:
    """Return ``{"through_week", "generated_at", "ratings"}`` for a season, or
    None when the database is unreachable or has no row."""
    try:
        init_oline_tables()
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                "SELECT through_week, generated_at, ratings "
                "FROM oline_ratings WHERE season = %s",
                (int(season),),
            ).fetchone()
        if not row:
            return None
        through_week, generated_at, ratings = row
        return {
            "through_week": through_week,
            "generated_at": (generated_at.isoformat()
                             if generated_at is not None else None),
            "ratings": ratings or {},
        }
    except Exception as exc:
        logger.debug("[oline_store] load failed for season %s: %s",
                     season, exc)
        return None


def newest_oline_season() -> int | None:
    """Newest season with stored ratings, or None when unavailable."""
    try:
        init_oline_tables()
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                "SELECT max(season) FROM oline_ratings").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    except Exception as exc:
        logger.debug("[oline_store] newest-season lookup failed: %s", exc)
        return None
