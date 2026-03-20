from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "player_value_history.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_value_history_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS player_value_history (
                as_of_date TEXT NOT NULL,
                player_id   TEXT NOT NULL,
                name        TEXT,
                position    TEXT,
                team        TEXT,
                value       REAL NOT NULL,
                source      TEXT NOT NULL DEFAULT 'model',
                PRIMARY KEY (as_of_date, player_id, source)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_value_history_player_date
            ON player_value_history (player_id, as_of_date)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_player_value_history_date
            ON player_value_history (as_of_date)
            """
        )


def record_model_value_snapshot(
    players: Iterable[dict],
    *,
    as_of: Optional[date] = None,
    source: str = "model",
) -> int:
    """
    Expects rows shaped like:
      {
        "id": "9509",
        "name": "Bijan Robinson",
        "position": "RB",
        "team": "ATL",
        "value": 968.0
      }
    """
    init_value_history_db()

    snapshot_date = (as_of or date.today()).isoformat()
    rows_to_insert: list[tuple] = []

    for p in players or []:
        if not isinstance(p, dict):
            continue

        pid = str(p.get("id") or "").strip()
        if not pid:
            continue

        raw_val = p.get("value", 0)
        try:
            value = float(raw_val or 0.0)
        except (TypeError, ValueError):
            value = 0.0

        rows_to_insert.append(
            (
                snapshot_date,
                pid,
                p.get("name"),
                p.get("position"),
                p.get("team"),
                value,
                source,
            )
        )

    if not rows_to_insert:
        return 0

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO player_value_history
                (as_of_date, player_id, name, position, team, value, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date, player_id, source)
            DO UPDATE SET
                name = excluded.name,
                position = excluded.position,
                team = excluded.team,
                value = excluded.value
            """,
            rows_to_insert,
        )

    return len(rows_to_insert)


def get_latest_snapshot_date(source: str = "model") -> Optional[str]:
    init_value_history_db()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT MAX(as_of_date) AS latest_date
            FROM player_value_history
            WHERE source = ?
            """,
            (source,),
        ).fetchone()
    return row["latest_date"] if row and row["latest_date"] else None


def get_player_value_history(
    player_id: str,
    *,
    days: int = 30,
    source: str = "model",
) -> list[dict]:
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return []

    cutoff = (date.fromisoformat(latest_date) - timedelta(days=max(days, 1) - 1)).isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                as_of_date,
                player_id,
                name,
                position,
                team,
                value,
                source
            FROM player_value_history
            WHERE source = ?
              AND player_id = ?
              AND as_of_date >= ?
            ORDER BY as_of_date ASC
            """,
            (source, str(player_id), cutoff),
        ).fetchall()

    out: list[dict] = []
    prev_val: Optional[float] = None
    for r in rows:
        val = float(r["value"])
        delta = None if prev_val is None else round(val - prev_val, 1)
        out.append(
            {
                "as_of_date": r["as_of_date"],
                "player_id": r["player_id"],
                "name": r["name"],
                "position": r["position"],
                "team": r["team"],
                "value": round(val, 1),
                "delta_from_prev": delta,
                "source": r["source"],
            }
        )
        prev_val = val

    return out


def get_top_movers(
    *,
    days: int = 7,
    limit: int = 15,
    source: str = "model",
) -> dict:
    """
    Compares latest snapshot vs latest snapshot on/before (latest - days).
    """
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return {
            "latest_date": None,
            "comparison_date": None,
            "risers": [],
            "fallers": [],
        }

    latest_dt = date.fromisoformat(latest_date)
    comparison_date = (latest_dt - timedelta(days=max(days, 1))).isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            """
            WITH latest_rows AS (
                SELECT h.*
                FROM player_value_history h
                INNER JOIN (
                    SELECT player_id, MAX(as_of_date) AS as_of_date
                    FROM player_value_history
                    WHERE source = ?
                      AND as_of_date <= ?
                    GROUP BY player_id
                ) x
                  ON x.player_id = h.player_id
                 AND x.as_of_date = h.as_of_date
                WHERE h.source = ?
            ),
            baseline_rows AS (
                SELECT h.*
                FROM player_value_history h
                INNER JOIN (
                    SELECT player_id, MAX(as_of_date) AS as_of_date
                    FROM player_value_history
                    WHERE source = ?
                      AND as_of_date <= ?
                    GROUP BY player_id
                ) x
                  ON x.player_id = h.player_id
                 AND x.as_of_date = h.as_of_date
                WHERE h.source = ?
            )
            SELECT
                l.player_id,
                l.name,
                l.position,
                l.team,
                ROUND(b.value, 1) AS old_value,
                ROUND(l.value, 1) AS new_value,
                ROUND(l.value - b.value, 1) AS delta
            FROM latest_rows l
            INNER JOIN baseline_rows b
                ON b.player_id = l.player_id
            WHERE l.value IS NOT NULL
              AND b.value IS NOT NULL
            ORDER BY delta DESC, l.value DESC
            """,
            (source, latest_date, source, source, comparison_date, source),
        ).fetchall()

    movers = [dict(r) for r in rows]
    risers = movers[:limit]
    fallers = list(reversed(sorted(movers, key=lambda x: (x["delta"], x["new_value"]))))[:limit]

    return {
        "latest_date": latest_date,
        "comparison_date": comparison_date,
        "risers": risers,
        "fallers": fallers,
    }