from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Iterable

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
                    sf_value NUMERIC,
                    source TEXT NOT NULL DEFAULT 'model',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (as_of_date, player_id, source)
                )
                """
            )
            # Add sf_value column if it doesn't exist (migration)
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'player_value_history'
                        AND column_name = 'sf_value'
                    ) THEN
                        ALTER TABLE player_value_history ADD COLUMN sf_value NUMERIC;
                    END IF;
                END $$;
                """
            )
            # Add league size columns (value_8, value_12, value_14, sf_value_8, sf_value_12, sf_value_14)
            for size in [8, 12, 14]:
                cur.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'player_value_history'
                            AND column_name = 'value_{size}'
                        ) THEN
                            ALTER TABLE player_value_history ADD COLUMN value_{size} NUMERIC;
                        END IF;
                    END $$;
                    """
                )
                cur.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'player_value_history'
                            AND column_name = 'sf_value_{size}'
                        ) THEN
                            ALTER TABLE player_value_history ADD COLUMN sf_value_{size} NUMERIC;
                        END IF;
                    END $$;
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
            # Performance indexes for top movers queries
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_date_value
                ON player_value_history (as_of_date, value DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_date_sf_value
                ON player_value_history (as_of_date, sf_value DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_player_position
                ON player_value_history (player_id, position)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_source_date
                ON player_value_history (source, as_of_date DESC)
                """
            )


def record_model_value_snapshot(
        players: Iterable[dict],
        *,
        as_of: Optional[date] = None,
        source: str = "model",
        ema_alpha: float = 0.70,
        min_change_pct: float = 0.005,
) -> int:
    """
    Write a smoothed daily value snapshot using EMA blending.

    ema_alpha: weight for new value (0.70 = 70% new, 30% previous).
      Softens step-function jumps when the model is retrained.
    min_change_pct: skip writing if ALL value columns changed less than
      this fraction (reduces DB noise from micro-fluctuations).
    Pass ema_alpha=1.0 for an intentional hard reset (no blending).
    """
    init_value_history_db()

    snapshot_date = (as_of or date.today()).isoformat()

    _VALUE_COLS = ["value", "sf_value", "value_8", "value_12", "value_14",
                   "sf_value_8", "sf_value_12", "sf_value_14"]

    player_list = []
    for p in players or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue

        def safe_float(key, default=0.0):
            try:
                return float(p.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        player_list.append({
            "pid": pid,
            "name": p.get("name"),
            "position": p.get("position"),
            "team": p.get("team"),
            "value": safe_float("value"),
            "sf_value": safe_float("sf_value", safe_float("value")),
            "value_8": safe_float("value_8"),
            "value_12": safe_float("value_12"),
            "value_14": safe_float("value_14"),
            "sf_value_8": safe_float("sf_value_8"),
            "sf_value_12": safe_float("sf_value_12"),
            "sf_value_14": safe_float("sf_value_14"),
        })

    if not player_list:
        return 0

    # Batch-fetch the most recent previous values for all players in one query
    all_pids = [row["pid"] for row in player_list]
    prev_rows: dict[str, dict] = {}
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (player_id)
                player_id, value, sf_value,
                value_8, value_12, value_14,
                sf_value_8, sf_value_12, sf_value_14
            FROM player_value_history
            WHERE source = %s
              AND player_id = ANY(%s)
              AND as_of_date < %s
            ORDER BY player_id, as_of_date DESC
            """,
            (source, all_pids, snapshot_date),
        ).fetchall()
        for r in rows:
            prev_rows[r["player_id"]] = {col: (float(r[col]) if r[col] is not None else 0.0) for col in _VALUE_COLS}

    rows_to_insert: list[tuple] = []
    for p in player_list:
        pid = p["pid"]
        prev = prev_rows.get(pid)

        blended = {}
        changed = False
        for col in _VALUE_COLS:
            new_val = p[col]
            if prev is not None and prev.get(col, 0.0) > 0:
                old_val = prev[col]
                b = ema_alpha * new_val + (1.0 - ema_alpha) * old_val
                if abs(b - old_val) / old_val >= min_change_pct:
                    changed = True
                blended[col] = round(b, 2)
            else:
                blended[col] = round(new_val, 2)
                changed = True

        if not changed:
            continue

        rows_to_insert.append((
            snapshot_date, pid, p["name"], p["position"], p["team"],
            blended["value"], blended["sf_value"],
            blended["value_8"], blended["value_12"], blended["value_14"],
            blended["sf_value_8"], blended["sf_value_12"], blended["sf_value_14"],
            source,
        ))

    if not rows_to_insert:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO player_value_history
                    (as_of_date, player_id, name, position, team, value, sf_value,
                     value_8, value_12, value_14, sf_value_8, sf_value_12, sf_value_14, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(as_of_date, player_id, source)
                DO UPDATE SET
                    name = excluded.name,
                    position = excluded.position,
                    team = excluded.team,
                    value = excluded.value,
                    sf_value = excluded.sf_value,
                    value_8 = excluded.value_8,
                    value_12 = excluded.value_12,
                    value_14 = excluded.value_14,
                    sf_value_8 = excluded.sf_value_8,
                    sf_value_12 = excluded.sf_value_12,
                    sf_value_14 = excluded.sf_value_14
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

    # Handle both date objects and strings
    if isinstance(latest_date, date):
        latest_date_obj = latest_date
    else:
        latest_date_obj = date.fromisoformat(str(latest_date))

    cutoff = (latest_date_obj - timedelta(days=max(days, 1) - 1)).isoformat()

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
                "as_of_date": str(r["as_of_date"]),  # ISO string "YYYY-MM-DD"
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
        league_type: str = "1qb",
        league_size: int = 10,
) -> dict:
    """
    Try requested window first (ex: 7 days).
    If no baseline exists, fall back to 6, then 5, ... down to 1.

    Args:
        days: Number of days to look back for comparison
        limit: Max number of risers/fallers to return
        source: Source of values ('model', etc.)
        league_type: "1qb" or "sf" (superflex) to determine which value field to use
        league_size: League size (8, 10, 12, 14) to determine which value field to use
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

            # Determine which value field to use based on league type and size
            if league_size == 10:
                value_field = "sf_value" if league_type == "sf" else "value"
            else:
                value_field = f"sf_value_{league_size}" if league_type == "sf" else f"value_{league_size}"

            # Fallback chain: size-specific -> 10-team -> value
            if league_type == "sf" and league_size != 10:
                value_expr = f"COALESCE(sf_value_{league_size}, sf_value, value)"
            elif league_type == "sf":
                value_expr = "COALESCE(sf_value, value)"
            elif league_size != 10:
                value_expr = f"COALESCE(value_{league_size}, value)"
            else:
                value_expr = "value"

            cur.execute(
                f"""
                WITH latest_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        name,
                        position,
                        team,
                        {value_expr} as value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                ),
                baseline_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        {value_expr} as value,
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

    # Build name map: model table first (covers picks + all players), then players_index
    name_map: dict = {}
    try:
        from utils.utils import load_model_value_table
        for p in (load_model_value_table(apply_calibration=False) or []):
            pid = str(p.get("id") or "")
            nm = p.get("name") or ""
            if pid and nm and nm != "Unknown":
                name_map[pid] = nm
    except Exception:
        pass
    try:
        from utils.utils import load_players_index
        for pid, info in (load_players_index() or {}).items():
            if pid not in name_map:
                nm = (info or {}).get("name") or ""
                if nm:
                    name_map[str(pid)] = nm
    except Exception:
        pass

    movers = []
    for row in rows:
        row_dict = dict(row)
        player_id = str(row_dict["player_id"])
        resolved = name_map.get(player_id)
        if resolved:
            row_dict["name"] = resolved
        elif not row_dict.get("name") or row_dict["name"] == "Unknown":
            row_dict["name"] = f"Player {player_id}"
        movers.append(row_dict)
    
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
