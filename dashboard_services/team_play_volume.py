"""Persistent storage and cached reads for the small NFL team pace table."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dashboard_services.db import get_conn
from utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)
READ_TTL_SECONDS = int(os.getenv("TEAM_PLAY_VOLUME_TTL_SECONDS", "1800"))
FRESH_SECONDS = 48 * 60 * 60
_cache: dict[int, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def invalidate_team_play_volume(season: int) -> None:
    """Invalidate this process's season cache after a successful write."""
    with _cache_lock:
        _cache.pop(int(season), None)


def persist_team_play_volume(blob: dict) -> int:
    """Atomically UPSERT a valid snapshot. Empty snapshots never alter the DB."""
    teams = blob.get("teams") or {}
    if not teams:
        logger.warning("team_play_volume empty snapshot; persisted data retained")
        return 0
    season = int(blob["season"])
    generated_at = blob.get("generated_at") or datetime.now(timezone.utc).isoformat()
    avg = blob.get("nfl_avg_plays_faced_pg")
    values = [(
        season, str(team).upper(), row.get("plays_faced_pg"),
        row.get("plays_faced_l4_pg"), row.get("off_plays_pg"), row.get("games"),
        avg, generated_at,
    ) for team, row in teams.items()]
    sql = """
        INSERT INTO team_play_volume
          (season, team, plays_faced_pg, plays_faced_l4_pg, off_plays_pg, games,
           nfl_avg_plays_faced_pg, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (season, team) DO UPDATE SET
          plays_faced_pg = EXCLUDED.plays_faced_pg,
          plays_faced_l4_pg = EXCLUDED.plays_faced_l4_pg,
          off_plays_pg = EXCLUDED.off_plays_pg,
          games = EXCLUDED.games,
          nfl_avg_plays_faced_pg = EXCLUDED.nfl_avg_plays_faced_pg,
          generated_at = EXCLUDED.generated_at
    """
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
    invalidate_team_play_volume(season)
    return len(values)


def _read_postgres(season: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT team, plays_faced_pg, plays_faced_l4_pg, off_plays_pg,
                      games, nfl_avg_plays_faced_pg, generated_at
                 FROM team_play_volume WHERE season = %s ORDER BY team""",
            (int(season),),
        ).fetchall()
    if not rows:
        return {}
    generated = max(row["generated_at"] for row in rows)
    now = datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return {
        "season": int(season),
        "generated_at": generated.isoformat(),
        "fresh": (now - generated).total_seconds() <= FRESH_SECONDS,
        "stale": (now - generated).total_seconds() > FRESH_SECONDS,
        "play_volume_source": "postgres",
        "nfl_avg_plays_faced_pg": rows[0]["nfl_avg_plays_faced_pg"],
        "teams": {row["team"]: {
            "plays_faced_pg": row["plays_faced_pg"],
            "plays_faced_l4_pg": row["plays_faced_l4_pg"],
            "off_plays_pg": row["off_plays_pg"], "games": row["games"],
        } for row in rows},
    }


def _read_local(season: int) -> dict:
    path = Path(CACHE_DIR) / f"team_play_volume_s{int(season)}.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        blob = json.load(handle) or {}
    if blob:
        blob["play_volume_source"] = "local_json"
    return blob


def load_team_play_volume(season: int, *, allow_local: bool | None = None) -> dict:
    """Read all teams once, DB first; use JSON only outside Render production."""
    season = int(season)
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(season)
        if cached and now - cached[0] < READ_TTL_SECONDS:
            return cached[1]
    blob = {}
    try:
        blob = _read_postgres(season)
    except Exception as exc:
        logger.info("team_play_volume postgres unavailable season=%s: %s", season, exc)
    if allow_local is None:
        allow_local = not bool(os.getenv("RENDER"))
    if not blob and allow_local:
        try:
            blob = _read_local(season)
        except Exception:
            logger.debug("team_play_volume local fallback failed", exc_info=True)
    with _cache_lock:
        _cache[season] = (now, blob)
    return blob
