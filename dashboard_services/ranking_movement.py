"""
Daily snapshots of any team ranking (value, power, ...) so ranked lists can show
▲/▼ position-movement arrows year-round — the offseason included, where a trade
reshuffles value and power rankings even though no games are played.

Date-keyed (not week-keyed), mirroring playoff_odds_history's daily snapshot, so
movement works whenever a ranking is viewed. One table serves many rankings via a
``kind`` discriminator ("value", "power", ...). Best-effort throughout: arrows
are decorative and must never break a page.
"""
import time
from typing import Dict, List


def get_conn():
    """Lazy DB handle: importing this module (e.g. under the pure test suite,
    which has no psycopg) must not pull in the driver until a query runs."""
    from dashboard_services.db import get_conn as _get_conn
    return _get_conn()


_TABLE_READY = False
_LAST_WRITE: Dict[tuple, float] = {}
_WRITE_THROTTLE = 600  # seconds — one snapshot write per (league, season, kind) per 10 min


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ranking_movement (
                league_id  TEXT        NOT NULL,
                season     INTEGER     NOT NULL,
                kind       TEXT        NOT NULL,
                snap_date  DATE        NOT NULL,
                roster_id  INTEGER     NOT NULL,
                rank       INTEGER     NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (league_id, season, kind, snap_date, roster_id)
            )
            """
        )
    _TABLE_READY = True


def has_snapshot(league_id: str, season: int, kind: str, snap_date: str) -> bool:
    """True if a snapshot already exists for this (league, season, kind, date).
    Lets the seeder skip the expensive reconstruction once a baseline exists."""
    try:
        _ensure_table()
        with get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM ranking_movement "
                "WHERE league_id = %s AND season = %s AND kind = %s AND snap_date = %s "
                "LIMIT 1",
                (str(league_id), int(season), str(kind), str(snap_date)),
            ).fetchone()
        return bool(row)
    except Exception:
        return False


def record_ranks_for_date(
    league_id: str,
    season: int,
    kind: str,
    ordered_roster_ids: List,
    snap_date: str,
    overwrite: bool = False,
) -> bool:
    """Write a ranking snapshot for a specific date (roster ids in ranked order).

    Used by the daily seeder to lay down today's snapshot and to backfill a
    *reconstructed* earlier-day baseline. With ``overwrite`` False (the default)
    it writes only when no snapshot exists yet for that (league, season, kind,
    date) — so a reconstructed baseline never clobbers a genuine one already
    recorded. Returns True if it wrote. Best-effort."""
    ordered = [str(r) for r in (ordered_roster_ids or []) if r is not None]
    if not ordered:
        return False
    try:
        _ensure_table()
        with get_conn() as conn:
            if not overwrite:
                existing = conn.execute(
                    "SELECT 1 FROM ranking_movement "
                    "WHERE league_id = %s AND season = %s AND kind = %s AND snap_date = %s "
                    "LIMIT 1",
                    (str(league_id), int(season), str(kind), str(snap_date)),
                ).fetchone()
                if existing:
                    return False
            for rid, rk in ((r, i + 1) for i, r in enumerate(ordered)):
                conn.execute(
                    """
                    INSERT INTO ranking_movement
                        (league_id, season, kind, snap_date, roster_id, rank)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (league_id, season, kind, snap_date, roster_id)
                    DO UPDATE SET rank = EXCLUDED.rank, created_at = NOW()
                    """,
                    (str(league_id), int(season), str(kind), str(snap_date), int(rid), int(rk)),
                )
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "ranking_movement: record_ranks_for_date failed", exc_info=True)
        return False


def record_daily_and_movement(
    league_id: str,
    season: int,
    kind: str,
    ordered_roster_ids: List,
    write: bool = True,
) -> Dict[str, int]:
    """Snapshot today's ranking and return each team's rank movement vs the most
    recent earlier daily snapshot of the same ``kind``.

    ``ordered_roster_ids`` are the roster ids in ranked order (index 0 = rank 1).
    Returns {roster_id: prev_rank - cur_rank} where positive = climbed since the
    previous snapshot; {} when there's no earlier snapshot to compare against.
    Writes are throttled to once per (league, season, kind) per 10 min, and only
    taken when ``write`` is True."""
    ordered = [str(r) for r in (ordered_roster_ids or []) if r is not None]
    if not ordered:
        return {}
    try:
        from datetime import datetime, timezone
        _ensure_table()
        today = datetime.now(timezone.utc).date().isoformat()
        cur_rank = {rid: i + 1 for i, rid in enumerate(ordered)}

        throttle_key = (str(league_id), int(season), str(kind))
        do_write = write and (
            time.time() - _LAST_WRITE.get(throttle_key, 0.0) >= _WRITE_THROTTLE
        )

        with get_conn() as conn:
            if do_write:
                for rid, rk in cur_rank.items():
                    conn.execute(
                        """
                        INSERT INTO ranking_movement
                            (league_id, season, kind, snap_date, roster_id, rank)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (league_id, season, kind, snap_date, roster_id)
                        DO UPDATE SET rank = EXCLUDED.rank, created_at = NOW()
                        """,
                        (str(league_id), int(season), str(kind), today, int(rid), int(rk)),
                    )
                _LAST_WRITE[throttle_key] = time.time()

            prev = conn.execute(
                "SELECT MAX(snap_date) AS d FROM ranking_movement "
                "WHERE league_id = %s AND season = %s AND kind = %s AND snap_date < %s",
                (str(league_id), int(season), str(kind), today),
            ).fetchone()
            prev_date = prev and prev["d"]
            if not prev_date:
                return {}

            prev_rows = conn.execute(
                "SELECT roster_id, rank FROM ranking_movement "
                "WHERE league_id = %s AND season = %s AND kind = %s AND snap_date = %s",
                (str(league_id), int(season), str(kind), prev_date),
            ).fetchall()

        prev_rank = {str(r["roster_id"]): int(r["rank"]) for r in (prev_rows or [])}
        return {rid: prev_rank[rid] - cur_rank[rid]
                for rid in cur_rank if rid in prev_rank}
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "ranking_movement: record/movement failed", exc_info=True)
        return {}
