from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional, Iterable

from dashboard_services.db import get_conn

_db_initialized = False
_latest_snapshot_cache: dict = {}  # source -> (date_str, cached_at_ts)
_SNAPSHOT_TTL = 300  # 5 minutes

# Per-player history cache: (player_id, days, source, league_type, league_size) -> (result, cached_at_ts)
_player_history_cache: dict = {}
_PLAYER_HISTORY_TTL = 600  # 10 minutes - history only updates daily

_PICK_ID_RE = re.compile(r'^(\d{4})_(\d+)(?:_(.+))?$')

def _format_pick_id(pid: str) -> str | None:
    """Return a human-readable label for a pick ID, or None if not a pick."""
    m = _PICK_ID_RE.match(pid)
    if not m:
        return None
    year, rd_str, third = m.group(1), m.group(2), m.group(3)
    rd = int(rd_str)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rd, "th")
    if not third:
        return f"{year} {rd}{suffix}"
    bucket = {"early": "Early", "mid": "Mid", "late": "Late"}.get(third.lower())
    if bucket:
        return f"{year} {rd}{suffix} ({bucket})"
    try:
        slot = int(third)
        return f"{year} {rd}.{slot:02d}"
    except ValueError:
        return f"{year} {rd}{suffix} {third.title()}"


def init_value_history_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
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
    _db_initialized = True


def record_model_value_snapshot(
        players: Iterable[dict],
        *,
        as_of: Optional[date] = None,
        source: str = "model",
        ema_alpha: float = 0.35,
        min_change_pct: float = 0.005,
) -> int:
    """
    Write a smoothed daily value snapshot using EMA blending.

    ema_alpha: weight for new value (0.35 = 35% new, 65% previous).
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
    import time
    cached = _latest_snapshot_cache.get(source)
    if cached and time.time() - cached[1] < _SNAPSHOT_TTL:
        return cached[0]
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
    result = row["latest_date"] if row and row["latest_date"] else None
    _latest_snapshot_cache[source] = (result, time.time())
    return result


def get_player_value_history(
        player_id: str,
        *,
        days: int = 30,
        source: str = "model",
        league_type: str = "1qb",
        league_size: int = 10,
) -> list[dict]:
    import time as _time
    _cache_key = (str(player_id), days, source, league_type, league_size)
    _cached = _player_history_cache.get(_cache_key)
    if _cached and _time.time() - _cached[1] < _PLAYER_HISTORY_TTL:
        return _cached[0]

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

    # Size-specific column names for both formats
    if league_size == 8:
        _1qb_col, _sf_col = "value_8", "sf_value_8"
    elif league_size == 12:
        _1qb_col, _sf_col = "value_12", "sf_value_12"
    elif league_size == 14:
        _1qb_col, _sf_col = "value_14", "sf_value_14"
    else:
        _1qb_col, _sf_col = "value", "value_sf"

    # SF column: prefer calibrated value_sf, fall back to raw sf_value, then 1QB value
    _sf_expr = f"COALESCE({_sf_col}, sf_value, {_1qb_col}, value)"

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                as_of_date,
                player_id,
                name,
                position,
                team,
                COALESCE({_1qb_col}, value) AS value_1qb,
                {_sf_expr}                  AS value_sf,
                source
            FROM player_value_history
            WHERE source = %s
              AND player_id = %s
              AND as_of_date >= %s
            ORDER BY as_of_date ASC
            """,
            (source, str(player_id), cutoff),
        ).fetchall()
        _cal_row = conn.execute(
            """SELECT calibrated_value_1qb, calibrated_value_sf
               FROM player_values WHERE player_id = %s""",
            (str(player_id),),
        ).fetchone()

    # Scale historical values to the calibrated scale so the graph matches the
    # current modal value. Use the latest history row's raw value as the
    # denominator so the final graph point scales to exactly the calibrated value.
    _cal_scale_1qb = 1.0
    _cal_scale_sf  = 1.0
    try:
        _last_raw_1qb = float(rows[-1]["value_1qb"]) if rows else 0.0
        _last_raw_sf  = float(rows[-1]["value_sf"])  if rows else 0.0
        if _cal_row:
            if _cal_row["calibrated_value_1qb"] and _last_raw_1qb > 0:
                _cal_scale_1qb = float(_cal_row["calibrated_value_1qb"]) / _last_raw_1qb
            if _cal_row["calibrated_value_sf"] and _last_raw_sf > 0:
                _cal_scale_sf = float(_cal_row["calibrated_value_sf"]) / _last_raw_sf
    except Exception:
        pass

    _use_sf = league_type.lower() == "sf"

    out: list[dict] = []
    prev_val: Optional[float] = None
    for r in rows:
        v1qb = float(r["value_1qb"]) * _cal_scale_1qb
        vsf  = float(r["value_sf"])  * _cal_scale_sf
        # Primary "value" follows league_type for backwards-compat callers
        primary = vsf if _use_sf else v1qb
        delta = None if prev_val is None else round(primary - prev_val, 1)
        out.append(
            {
                "as_of_date":    str(r["as_of_date"]),
                "player_id":     r["player_id"],
                "name":          r["name"],
                "position":      r["position"],
                "team":          r["team"],
                "value":         round(primary, 1),
                "value_1qb":     round(v1qb, 1),
                "value_sf":      round(vsf, 1),
                "delta_from_prev": delta,
                "source":        r["source"],
            }
        )
        prev_val = primary

    _player_history_cache[_cache_key] = (out, _time.time())
    return out


def get_top_movers(
        *,
        days: int = 7,
        limit: int = 15,
        source: str = "model",
        league_type: str = "1qb",
        league_size: int = 10,
        min_baseline_value: int = 0,
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
        min_baseline_value: Percentage (0-100). old_value must be >= this % of new_value.
            E.g. 10 filters out players who went from ~0 to a real value (just-drafted
            rookies), while keeping established players with genuine movement.
            Scale-independent - works regardless of how history rows were written.
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

            best_candidate_date = None
            best_candidate_days = None
            best_player_count = 0

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
                    # Check data coverage for this candidate date
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT player_id) as player_count
                        FROM player_value_history
                        WHERE source = %s AND as_of_date = %s
                        """,
                        (source, candidate_date),
                    )
                    coverage_row = cur.fetchone()
                    player_count = coverage_row["player_count"] if coverage_row else 0
                    
                    # Track the best candidate (highest player count)
                    if player_count > best_player_count:
                        best_candidate_date = candidate_date
                        best_candidate_days = candidate_days
                        best_player_count = player_count
                    
                    # Use this date if it has decent coverage (at least 100 players)
                    if player_count >= 100:
                        comparison_date = candidate_date
                        used_days = candidate_days
                        break
            
            # If no date had 100+ players, use the best available date
            if comparison_date is None and best_candidate_date is not None:
                comparison_date = best_candidate_date
                used_days = best_candidate_days

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

            # Fallback chain: size-specific -> 10-team (value_sf=calibrated, sf_value=raw) -> value
            if league_type == "sf" and league_size != 10:
                value_expr = f"COALESCE(sf_value_{league_size}, value_sf, sf_value, value)"
            elif league_type == "sf":
                value_expr = "COALESCE(value_sf, sf_value, value)"
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
                    ROUND(b.value, 1) AS old_value,
                    ROUND(l.value, 1) AS new_value,
                    ROUND(l.value - b.value, 1) AS delta
                FROM latest_rows l
                JOIN baseline_rows b
                  ON b.player_id = l.player_id
                WHERE l.value IS NOT NULL 
                  AND b.value IS NOT NULL
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
            pick_label = _format_pick_id(player_id)
            row_dict["name"] = pick_label if pick_label else f"Player {player_id}"
        else:
            # Name came from DB but might be a raw pick ID (e.g. "2027_1")
            pick_label = _format_pick_id(str(row_dict["name"]))
            if pick_label:
                row_dict["name"] = pick_label

        # Filter out brand-new players (e.g. just-drafted rookies who went from
        # ~0 to a real value).  Require old_value >= new_value * min_baseline_ratio
        # so that scale differences (0-1 vs 0-1000) don't cause false positives.
        if min_baseline_value > 0:
            old_v = float(row_dict.get("old_value") or 0)
            new_v = float(row_dict.get("new_value") or 0)
            # Use ratio: old must be at least min_baseline_value % of new
            ratio = min_baseline_value / 100.0
            if new_v > 0 and old_v < new_v * ratio:
                continue

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

