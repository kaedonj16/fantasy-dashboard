"""Durable store for the defense-vs-position matchup ratings.

The Wednesday cron builds these ratings in an ephemeral Render container,
so the ``cache/matchup_ratings_s{season}.json`` file it writes never
survives to be read by the web service - the web dyno kept serving the
committed June 2026 snapshot (a 2025-data window) all season. This table
is the durable home, mirroring dashboard_services.oline_store: the cron
upserts here, the web app reads here. The flat file remains as a fallback
for local dev and backtests running without a database.

Rows are keyed by (season, scoring_profile): the default PPR build and
each deduplicated common scoring profile get their own row, matching the
profiled flat-file layout app._load_matchup_ratings already understands.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TABLES_READY = False


def init_matchup_ratings_tables() -> None:
    """Create the matchup_ratings table once per process (idempotent)."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matchup_ratings (
                season INTEGER NOT NULL,
                scoring_profile TEXT NOT NULL,
                through_week INTEGER,
                generated_at TIMESTAMPTZ,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (season, scoring_profile)
            )
            """
        )
        conn.commit()
    _TABLES_READY = True


def _profile_key(payload: dict, scoring_profile: str | None) -> str:
    return str(scoring_profile or payload.get("scoring_profile") or "standard-ppr")


def save_matchup_ratings(season: int, payload: dict,
                         scoring_profile: str | None = None) -> bool:
    """Upsert one season+profile payload. Returns False (never raises) on
    any DB failure so a missing/unreachable database degrades to the flat
    file - callers that must not lose the build (the Wednesday cron) check
    the return value and fail loudly."""
    try:
        ratings = payload.get("ratings") or {}
        if not ratings:
            logger.warning("[matchup_ratings_store] refusing to save empty "
                           "ratings for season %s", season)
            return False
        init_matchup_ratings_tables()
        from dashboard_services.db import get_conn
        from psycopg.types.json import Json
        generated_at = payload.get("generated_at")
        if isinstance(generated_at, str):
            try:
                from datetime import datetime

                generated_at = datetime.fromisoformat(
                    generated_at.replace("Z", "+00:00"))
            except ValueError:
                generated_at = None
        profile = _profile_key(payload, scoring_profile)
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO matchup_ratings (season, scoring_profile,
                                             through_week, generated_at,
                                             payload, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (season, scoring_profile) DO UPDATE SET
                    through_week = EXCLUDED.through_week,
                    generated_at = EXCLUDED.generated_at,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    int(season),
                    profile,
                    payload.get("through_week"),
                    generated_at,
                    Json(payload),
                ),
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.warning("[matchup_ratings_store] save failed for season %s: %s",
                       season, exc)
        return False


def load_matchup_ratings(season: int,
                         scoring_profile: str | None = None) -> dict | None:
    """Return the stored payload for a season (+profile), or None when the
    database is unreachable or has no such row."""
    try:
        init_matchup_ratings_tables()
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            if scoring_profile:
                row = conn.execute(
                    "SELECT payload FROM matchup_ratings "
                    "WHERE season = %s AND scoring_profile = %s",
                    (int(season), str(scoring_profile)),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT payload FROM matchup_ratings WHERE season = %s "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (int(season),),
                ).fetchone()
        if not row:
            return None
        payload = row[0]
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.debug("[matchup_ratings_store] load failed for season %s: %s",
                     season, exc)
        return None
