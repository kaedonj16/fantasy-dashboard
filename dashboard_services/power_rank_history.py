"""
Weekly power-ranking snapshots so the standings page can show movement arrows.

Each render of the power rankings upserts the current week's ranks, then loads
the most recent earlier week to compute per-team movement. Failures are
swallowed by the caller — arrows are decorative and must never break the page.
"""
from typing import Dict, Optional

from dashboard_services.db import get_conn

_TABLE_READY = False


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS power_rank_history (
                league_id  TEXT    NOT NULL,
                season     INTEGER NOT NULL,
                week       INTEGER NOT NULL,
                owner_key  TEXT    NOT NULL,
                rank       INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (league_id, season, week, owner_key)
            )
            """
        )
    _TABLE_READY = True


def record_and_movement(
    league_id: str,
    season: int,
    week: int,
    ranks: Dict[str, int],
) -> Dict[str, Optional[int]]:
    """Upsert this week's ranks and return movement vs the previous snapshot.

    Returns {owner_key: delta} where positive = moved up (e.g. +2 means the
    team climbed two spots), negative = dropped, 0 = unchanged, and None =
    no earlier snapshot exists for that team (new entry).
    Returns {} when there is no earlier week to compare against.
    """
    _ensure_table()
    with get_conn() as conn:
        for owner_key, rank in ranks.items():
            conn.execute(
                """
                INSERT INTO power_rank_history (league_id, season, week, owner_key, rank)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (league_id, season, week, owner_key)
                DO UPDATE SET rank = EXCLUDED.rank, created_at = NOW()
                """,
                (str(league_id), int(season), int(week), str(owner_key), int(rank)),
            )

        prev = conn.execute(
            """
            SELECT MAX(week) AS w FROM power_rank_history
            WHERE league_id = %s AND season = %s AND week < %s
            """,
            (str(league_id), int(season), int(week)),
        ).fetchone()
        prev_week = prev["w"] if prev else None
        if prev_week is None:
            return {}

        rows = conn.execute(
            """
            SELECT owner_key, rank FROM power_rank_history
            WHERE league_id = %s AND season = %s AND week = %s
            """,
            (str(league_id), int(season), int(prev_week)),
        ).fetchall()

    prev_ranks = {r["owner_key"]: int(r["rank"]) for r in rows}
    movement: Dict[str, Optional[int]] = {}
    for owner_key, rank in ranks.items():
        old = prev_ranks.get(str(owner_key))
        movement[owner_key] = (old - rank) if old is not None else None
    return movement
