"""Current NFL team affiliation shared across cron and web.

Render runs cron and the web service on separate disks, so updating
``cache/players_index.json`` in cron never reaches the app. This table is the
shared source of truth for *current* team (trades / FA / signings). Historical
per-week teams stay in ``player_week_team``.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Alias map to the players_index / NFL_TEAMS convention (LAR, WAS, JAX).
_TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LA": "LAR",  # Sleeper occasionally uses LA for the Rams
}

_OVERLAY_CACHE: Dict[str, object] = {"ts": 0.0, "data": {}}
_OVERLAY_TTL = 900.0  # 15 minutes — daily cron is the writer


def normalize_nfl_team(team) -> str:
    """Normalize a team abbrev to players_index convention (LAR/WAS/JAX)."""
    s = str(team or "").strip().upper()
    if not s or s in ("FA", "NONE", "NULL"):
        return ""
    return _TEAM_ALIASES.get(s, s)


def init_player_current_team_table(conn=None) -> None:
    """Ensure the player_current_team table exists (cron writer)."""
    sql = """
        CREATE TABLE IF NOT EXISTS player_current_team (
            player_id  TEXT        PRIMARY KEY,
            team       TEXT        NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """
    if conn is not None:
        conn.execute(sql)
        return
    try:
        from dashboard_services.db import get_conn
    except Exception:
        logger.debug("player_current_team init: no DB", exc_info=True)
        return
    try:
        with get_conn() as c:
            c.execute(sql)
    except Exception:
        logger.debug("player_current_team init failed", exc_info=True)


def upsert_current_teams(teams: Dict[str, str]) -> int:
    """Upsert {player_id: team} into player_current_team. Returns rows touched."""
    if not teams:
        return 0
    try:
        from dashboard_services.db import get_conn
    except Exception:
        logger.debug("player_current_team upsert: no DB", exc_info=True)
        return 0

    rows = [
        (str(pid), normalize_nfl_team(team))
        for pid, team in teams.items()
        if pid
    ]
    if not rows:
        return 0

    try:
        with get_conn() as conn:
            init_player_current_team_table(conn)
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO player_current_team (player_id, team, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (player_id) DO UPDATE SET
                        team = EXCLUDED.team,
                        updated_at = NOW()
                    """,
                    rows,
                )
            # Invalidate in-process overlay so this process sees fresh data.
            _OVERLAY_CACHE["ts"] = 0.0
            return len(rows)
    except Exception:
        logger.exception("player_current_team upsert failed")
        return 0


def update_player_values_teams(teams: Dict[str, str]) -> int:
    """Best-effort UPDATE of player_values.team for players that already have a row."""
    if not teams:
        return 0
    try:
        from dashboard_services.db import get_conn
    except Exception:
        return 0
    updated = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for pid, team in teams.items():
                    cur.execute(
                        """
                        UPDATE player_values
                           SET team = %s
                         WHERE player_id = %s
                           AND COALESCE(team, '') IS DISTINCT FROM %s
                        """,
                        (normalize_nfl_team(team), str(pid), normalize_nfl_team(team)),
                    )
                    updated += cur.rowcount or 0
        return updated
    except Exception:
        logger.debug("player_values team update failed", exc_info=True)
        return 0


def load_current_team_overlay(*, force: bool = False) -> Dict[str, str]:
    """{player_id: team} from the shared DB, or {} if unavailable.

    Read-only on the web path (no CREATE) so a restricted DB role still works.
    Cached in-process with a short TTL.
    """
    now = time.time()
    if not force:
        cached = _OVERLAY_CACHE.get("data") or {}
        ts = float(_OVERLAY_CACHE.get("ts") or 0.0)
        if cached is not None and now - ts < _OVERLAY_TTL:
            return cached  # type: ignore[return-value]

    try:
        from dashboard_services.db import get_conn
    except Exception:
        return {}

    out: Dict[str, str] = {}
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id, team FROM player_current_team"
            ).fetchall()
        for r in rows:
            pid = str(r["player_id"] if hasattr(r, "keys") else r[0])
            team = normalize_nfl_team(
                r["team"] if hasattr(r, "keys") else r[1]
            )
            if pid:
                out[pid] = team
    except Exception:
        logger.debug("player_current_team read unavailable", exc_info=True)
        # Keep any previous good cache rather than wiping to {}.
        prev = _OVERLAY_CACHE.get("data")
        if isinstance(prev, dict) and prev:
            return prev
        return {}

    _OVERLAY_CACHE["ts"] = now
    _OVERLAY_CACHE["data"] = out
    return out


def apply_team_overlay(
    index: Optional[Dict],
    overlay: Optional[Dict[str, str]] = None,
    *,
    bye_by_team: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """Return an index with DB team (and optional byeWeek) overlaid.

    Does not mutate the input dict or its nested player dicts. Returns the
    original object when there is nothing to change (shared read-only cache).
    """
    if not isinstance(index, dict) or not index:
        return index
    if overlay is None:
        overlay = load_current_team_overlay()
    if not overlay:
        return index

    merged: Optional[Dict] = None
    for pid, new_team in overlay.items():
        meta = index.get(pid)
        if not isinstance(meta, dict):
            continue
        old_team = str(meta.get("team") or "").strip().upper()
        team = normalize_nfl_team(new_team)
        # Empty overlay team means FA / unsigned — apply it so releases stick.
        if old_team == team:
            # Still refresh byeWeek if team matches but bye is stale/missing.
            if (
                bye_by_team
                and team
                and bye_by_team.get(team) is not None
                and meta.get("byeWeek") != bye_by_team.get(team)
            ):
                if merged is None:
                    merged = dict(index)
                new_meta = dict(meta)
                new_meta["byeWeek"] = bye_by_team[team]
                merged[pid] = new_meta
            continue
        if merged is None:
            merged = dict(index)
        new_meta = dict(meta)
        new_meta["team"] = team
        if bye_by_team and team and bye_by_team.get(team) is not None:
            new_meta["byeWeek"] = bye_by_team[team]
        merged[pid] = new_meta

    return index if merged is None else merged


def clear_overlay_cache() -> None:
    """Test helper / post-write hook."""
    _OVERLAY_CACHE["ts"] = 0.0
    _OVERLAY_CACHE["data"] = {}
