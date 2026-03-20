from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from dashboard_services.db import get_conn


def init_value_history_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS player_value_history (
                    as_of_date DATE NOT NULL,
                    player_id TEXT NOT NULL,
                    name TEXT,
                    position TEXT,
                    team TEXT,
                    value NUMERIC NOT NULL,
                    source TEXT NOT NULL DEFAULT 'model',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (as_of_date, player_id, source)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_player_date
                ON player_value_history (player_id, as_of_date DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_date
                ON player_value_history (as_of_date DESC)
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
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO player_value_history
                    (as_of_date, player_id, name, position, team, value, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
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
            WHERE source = %s
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
            WHERE source = %s
              AND player_id = %s
              AND as_of_date >= %s
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
    Try requested window first (ex: 7 days).
    If no baseline exists, fall back to 6, then 5, ... down to 1.
    """
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return {
            "latest_date": None,
            "comparison_date": None,
            "requested_days": days,
            "used_days": None,
            "risers": [],
            "fallers": [],
        }

    max_days = max(int(days), 1)

    with get_conn() as conn:
        with conn.cursor() as cur:
            comparison_date = None
            used_days = None

            for candidate_days in range(max_days, 0, -1):
                target_date = latest_date - timedelta(days=candidate_days)

                cur.execute(
                    """
                    SELECT MAX(as_of_date) AS comparison_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date <= %s
                    """,
                    (source, target_date),
                )
                row = cur.fetchone()
                candidate_date = row["comparison_date"] if row else None

                if candidate_date and candidate_date < latest_date:
                    comparison_date = candidate_date
                    used_days = candidate_days
                    break

            if comparison_date is None:
                return {
                    "latest_date": latest_date.isoformat(),
                    "comparison_date": None,
                    "requested_days": max_days,
                    "used_days": None,
                    "risers": [],
                    "fallers": [],
                }

            cur.execute(
                """
                WITH latest_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        name,
                        position,
                        team,
                        value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                ),
                baseline_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                )
                SELECT
                    l.player_id,
                    l.name,
                    l.position,
                    l.team,
                    ROUND(b.value::numeric, 1) AS old_value,
                    ROUND(l.value::numeric, 1) AS new_value,
                    ROUND((l.value - b.value)::numeric, 1) AS delta
                FROM latest_rows l
                JOIN baseline_rows b
                  ON b.player_id = l.player_id
                ORDER BY delta DESC, new_value DESC
                """
                , (source, latest_date, source, comparison_date))

            rows = cur.fetchall()

    movers = [dict(row) for row in rows]
    risers = movers[:limit]
    fallers = sorted(movers, key=lambda x: (x["delta"], x["new_value"]))[:limit]

    return {
        "latest_date": latest_date.isoformat(),
        "comparison_date": comparison_date.isoformat(),
        "requested_days": max_days,
        "used_days": used_days,
        "risers": risers,
        "fallers": fallers,
    }
