from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional, Iterable
import numpy as np

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


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
                    value_sf NUMERIC,
                    value_8 NUMERIC,
                    value_12 NUMERIC,
                    value_14 NUMERIC,
                    sf_value_8 NUMERIC,
                    sf_value_12 NUMERIC,
                    sf_value_14 NUMERIC,
                    source TEXT NOT NULL DEFAULT 'model',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (as_of_date, player_id, source)
                )
                """
            )
            # Add size/type columns to existing tables that predate this schema
            for col, typ in [
                ("value_sf",    "NUMERIC"),
                ("value_8",     "NUMERIC"),
                ("value_12",    "NUMERIC"),
                ("value_14",    "NUMERIC"),
                ("sf_value_8",  "NUMERIC"),
                ("sf_value_12", "NUMERIC"),
                ("sf_value_14", "NUMERIC"),
            ]:
                cur.execute(
                    f"ALTER TABLE player_value_history ADD COLUMN IF NOT EXISTS {col} {typ}"
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
        ema_alpha: float = 0.70,
        min_change_pct: float = 0.005,
) -> int:
    """
    Write a smoothed daily value snapshot using EMA blending.

    ema_alpha: weight for new value (0.70 = 70% new, 30% previous).
      Softens step-function jumps when the model is retrained.
    min_change_pct: skip writing if the blended value changed less than
      this fraction from the previous snapshot (reduces DB noise).
    """
    init_value_history_db()

    snapshot_date = (as_of or date.today()).isoformat()

    player_list = []
    for p in players or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        try:
            value = float(p.get("value") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        player_list.append((pid, p.get("name"), p.get("position"), p.get("team"), value))

    if not player_list:
        return 0

    # Fetch the most recent previous value for every player in one batch query
    all_pids = [row[0] for row in player_list]
    prev_values: dict[str, float] = {}
    with get_conn() as conn:
        # Get the latest value per player before today using a lateral/distinct-on query
        rows = conn.execute(
            """
            SELECT DISTINCT ON (player_id)
                player_id, value
            FROM player_value_history
            WHERE source = %s
              AND player_id = ANY(%s)
              AND as_of_date < %s
            ORDER BY player_id, as_of_date DESC
            """,
            (source, all_pids, snapshot_date),
        ).fetchall()
        for r in rows:
            prev_values[r["player_id"]] = float(r["value"])

    # Build smoothed rows, skipping tiny changes
    rows_to_insert: list[tuple] = []
    for pid, name, position, team, new_val in player_list:
        prev = prev_values.get(pid)
        if prev is not None and prev > 0:
            blended = ema_alpha * new_val + (1.0 - ema_alpha) * prev
            # Skip if change from previous is below threshold
            if abs(blended - prev) / prev < min_change_pct:
                continue
        else:
            blended = new_val

        rows_to_insert.append((snapshot_date, pid, name, position, team, round(blended, 2), source))

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
    return str(row["latest_date"]) if row and row["latest_date"] else None


def get_player_value_history(
        player_id: str,
        *,
        days: int = 30,
        source: str = "model",
        league_type: str = "1qb",
        league_size: int = 10,
) -> list[dict]:
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return []

    cutoff = (date.fromisoformat(latest_date) - timedelta(days=max(days, 1) - 1)).isoformat()
    col = _value_col(league_type, league_size)

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT
                as_of_date,
                player_id,
                name,
                position,
                team,
                COALESCE({col}, value) AS value,
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
        val = min(float(r["value"]), 999.9)  # cap at scale max; filters corrupted snapshots
        if val <= 0:
            continue
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


def load_latest_value_snapshot(source: str = "model") -> list[dict]:
    """
    Load the most recent value snapshot from the database.
    Returns a list of player dicts in the same format as the JSON file.
    Includes computed fields like search_name and pos_rank.
    """
    from utils.utils import normalize_name

    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return []

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                player_id as id,
                name,
                position,
                team,
                value,
                value_sf,
                value_8,
                value_12,
                value_14,
                sf_value_8,
                sf_value_12,
                sf_value_14
            FROM player_value_history
            WHERE source = %s
              AND as_of_date = %s
            ORDER BY value DESC
            LIMIT 600
            """,
            (source, latest_date),
        ).fetchall()

    # Convert to dicts and add computed fields
    players = []
    for row in rows:
        player = dict(row)
        # Add search_name for fuzzy matching
        player["search_name"] = normalize_name(player.get("name", ""))
        players.append(player)

    # Calculate position ranks
    pos_to_indices = {}
    for idx, player in enumerate(players):
        pos = str(player.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        pos_to_indices.setdefault(pos, []).append(idx)

    # Standard position ranks (by value)
    for pos, indices in pos_to_indices.items():
        indices.sort(key=lambda i: float(players[i].get("value") or 0.0), reverse=True)
        rank = 1
        for i in indices:
            players[i]["pos_rank"] = rank
            players[i]["pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    # Superflex position ranks (by sf_value)
    sf_pos_to_indices = {}
    for idx, player in enumerate(players):
        pos = str(player.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        sf_pos_to_indices.setdefault(pos, []).append(idx)

    for pos, indices in sf_pos_to_indices.items():
        indices.sort(key=lambda i: float(players[i].get("sf_value") or 0.0), reverse=True)
        rank = 1
        for i in indices:
            players[i]["sf_pos_rank"] = rank
            players[i]["sf_pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    # Enrich with age from players_index
    try:
        from utils.utils import load_players_index
        from dashboard_services.service import age_from_bday
        players_index = load_players_index() or {}
        for player in players:
            pid = str(player.get("id") or player.get("player_id") or "")
            if pid and pid != "" and pid in players_index:
                # Use age_from_bday function for consistent 1 decimal place precision
                bday_str = players_index[pid].get("bDay")
                player["age"] = age_from_bday(bday_str)
            else:
                player["age"] = None
    except Exception as e:
        logger.info(f"[load_latest_value_snapshot] Failed to enrich ages: {e}")
        for player in players:
            player["age"] = None

    return players


def load_current_values_from_db() -> list[dict]:
    """
    Load current player and pick values from the player_values table (one row per player/pick,
    updated daily by cron_daily).  Falls back gracefully if the table doesn't
    exist yet or the DB is unavailable.
    """
    from utils.utils import normalize_name

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        player_id  AS id,
                        GREATEST(calibrated_value_1qb, value_1qb)            AS value,
                        GREATEST(calibrated_value_sf,  value_sf, value_1qb)  AS sf_value,
                        value_1qb  AS model_value,
                        value_sf   AS model_sf_value,
                        GREATEST(calibrated_value_8,      value_8)      AS value_8,
                        GREATEST(calibrated_value_12,     value_12)     AS value_12,
                        GREATEST(calibrated_value_14,     value_14)     AS value_14,
                        GREATEST(calibrated_sf_value_8,   sf_value_8)   AS sf_value_8,
                        GREATEST(calibrated_sf_value_12,  sf_value_12)  AS sf_value_12,
                        GREATEST(calibrated_sf_value_14,  sf_value_14)  AS sf_value_14,
                        calibration_source,
                        calibration_weight,
                        position,
                        pos_rank,
                        pos_rank_label,
                        sf_pos_rank,
                        sf_pos_rank_label,
                        age,
                        team,
                        years_exp,
                        last_updated,
                        rank_change_7d,
                        pos_rank_change_7d,
                        redraft_value_1qb,
                        redraft_value_sf
                    FROM player_values
                    WHERE value_1qb IS NOT NULL
                      AND value_1qb > 0
                      AND (position IS NULL OR position != 'PICK')
                    ORDER BY value_1qb DESC NULLS LAST
                    """
                )
                rows = cur.fetchall()
    except Exception as e:
        return []

    if not rows:
        return []

    # Load players index for name matching
    from utils.utils import load_players_index
    players_index = load_players_index() or {}

    # Pre-scan picks to build pick-type sets per (year, round).
    # Hierarchy: slot picks (1.01) > bucket picks (Early/Mid/Late) > generic (2027_1)
    _BUCKET_KWORDS = {"early", "mid", "late"}
    _slot_yr_rnd: set = set()    # (yr, rnd) combos that have specific slot picks
    _bucket_yr_rnd: set = set()  # (yr, rnd) combos that have bucket picks
    for row in rows:
        if str(dict(row).get("position") or "").upper() == "PICK":
            _pid = str(dict(row).get("id") or "")
            _pp = _pid.split("_")
            if len(_pp) >= 3:
                _key = (_pp[0], _pp[1])
                if _pp[2].lower() in _BUCKET_KWORDS:
                    _bucket_yr_rnd.add(_key)
                else:
                    try:
                        int(_pp[2])
                        _slot_yr_rnd.add(_key)
                    except ValueError:
                        pass

    assets = []
    for row in rows:
        asset = dict(row)

        # Enforce pick hierarchy: slot > bucket > generic
        if str(asset.get("position") or "").upper() == "PICK":
            _pid = str(asset.get("id") or "")
            _pp = _pid.split("_")
            if len(_pp) >= 2:
                _yr_rnd_key = (_pp[0], _pp[1])
                is_generic = len(_pp) == 2
                is_bucket = len(_pp) >= 3 and _pp[2].lower() in _BUCKET_KWORDS
                if _yr_rnd_key in _slot_yr_rnd:
                    # Slot picks exist → drop bucket and generic
                    if is_generic or is_bucket:
                        continue
                elif _yr_rnd_key in _bucket_yr_rnd:
                    # Bucket picks exist but no slots → drop generic
                    if is_generic:
                        continue

        # Skip players with no value
        value = float(asset.get("value") or 0.0)
        if value <= 0:
            continue
        
                
        # Add name from players index
        player_id = str(asset.get("id"))
        player_data = players_index.get(player_id)
        if player_data:
            asset["name"] = player_data.get("name")
            asset["team"] = asset.get("team") or player_data.get("team")
            
            # Calculate age from birthday if age is not available, using the consistent age_from_bday function
            if not asset.get("age"):
                birthday = player_data.get("bDay")
                if birthday:
                    try:
                        from dashboard_services.service import age_from_bday
                        calculated_age = age_from_bday(birthday)
                        asset["age"] = calculated_age if calculated_age is not None else player_data.get("age")
                    except Exception:
                        asset["age"] = player_data.get("age")
                else:
                    asset["age"] = player_data.get("age")
            else:
                # Ensure existing age is properly rounded to 1 decimal place for consistency
                try:
                    asset["age"] = round(float(asset["age"]), 1)
                except Exception:
                    pass
        
        # Normalise field names expected by the rest of the app
        asset.setdefault("sf_value", asset.get("value") or 0.0)
        
        # Handle search_name differently for picks vs players
        if str(asset.get("position") or "").upper() == "PICK":
            # For picks, use the ID as search name (e.g., "2026_1_01")
            asset["search_name"] = str(asset.get("id") or "")
        else:
            # For players, normalize the actual name
            asset["search_name"] = normalize_name(str(asset.get("name") or asset.get("id") or ""))
        
        # Add league-size variants as the current value (no per-size data in player_values)
        val    = float(asset.get("value")    or 0.0)
        sf_val = float(asset.get("sf_value") or 0.0)
        for sz in (8, 12, 14):
            asset.setdefault(f"value_{sz}",    val)
            asset.setdefault(f"sf_value_{sz}", sf_val)
        
        # Set name for picks if not present
        if str(asset.get("position") or "").upper() == "PICK":
            pick_id = str(asset.get("id") or "")
            # Format pick ID: 2026_1_01 -> 2026 1.01, 2027_1_mid -> 2027 Mid 1st
            if "_" in pick_id:
                parts = pick_id.split("_")
                if len(parts) >= 3:
                    year = parts[0]
                    round_num = parts[1]
                    pick_num = parts[2]

                    # Handle special pick designations
                    _bucket_labels = {"early": "Early", "mid": "Mid", "late": "Late"}
                    _blabel = _bucket_labels.get(pick_num.lower())
                    if _blabel:
                        _rnd_int = int(round_num)
                        _sfx = {1: "st", 2: "nd", 3: "rd"}.get(_rnd_int, "th")
                        asset["name"] = f"{year} {_rnd_int}{_sfx} ({_blabel})"
                    else:
                        # Regular pick number: 01 -> 1.01, 12 -> 1.12
                        try:
                            pick_int = int(pick_num)
                            asset["name"] = f"{year} {round_num}.{pick_num.zfill(2)}"
                        except ValueError:
                            asset["name"] = f"{year} {round_num} {pick_num}"
                else:
                    asset["name"] = pick_id
            else:
                asset["name"] = pick_id
        elif not asset.get("name"):
            asset["name"] = str(asset.get("id") or "")
            
        assets.append(asset)

    # Database already has correct position ranks based on calibrated values
    # No recalculation needed - use the database values directly

    return assets


def load_calibration_overrides() -> dict[str, dict]:
    """
    FC/DP market corrections are now applied directly in get_model_value_table_cached()
    in app.py, where the loaded player values are already available for comparison.
    This function is kept for the load_model_value_table() call path but returns
    empty - the cache path handles corrections without needing an extra DB round-trip.
    """
    return {}


def _value_col(league_type: str = "1qb", league_size: int = 10) -> str:
    """Return the player_value_history column for a given league type/size."""
    sf = league_type.lower() == "sf"
    if sf:
        return {8: "sf_value_8", 12: "sf_value_12", 14: "sf_value_14"}.get(league_size, "value_sf")
    return {8: "value_8", 12: "value_12", 14: "value_14"}.get(league_size, "value")


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
    """
    try:
        init_value_history_db()
        return _get_top_movers_from_db(
            days=days, limit=limit, source=source,
            league_type=league_type, league_size=league_size,
        )
    except RuntimeError as e:
        if "DATABASE_URL is not set" in str(e):
            return _get_top_movers_from_parquet(days=days, limit=limit)
        else:
            raise


def _get_top_movers_from_db(
        days: int, limit: int, source: str,
        league_type: str = "1qb", league_size: int = 10,
) -> dict:
    """Get movers from database table."""
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

            vcol = _value_col(league_type, league_size)
            # Fall back to base 'value' column for rows that predate multi-size storage
            vcol_expr = f"COALESCE({vcol}, value)"

            cur.execute(
                f"""
                WITH latest_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        name,
                        position,
                        team,
                        {vcol_expr} AS value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                ),
                baseline_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        {vcol_expr} AS value,
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
                """,
                (source, latest_date, source, comparison_date))

            rows = cur.fetchall()

    # Build a players_index fallback for rows where the DB name column is missing/Unknown
    players_index_map: dict = {}
    try:
        from utils.utils import load_players_index
        for pid, info in (load_players_index() or {}).items():
            nm = (info or {}).get("name") or ""
            if nm:
                players_index_map[str(pid)] = nm
    except Exception:
        pass

    import re as _re

    def _format_pick_id(pid):
        """Return human-readable pick label for IDs like '2026_1_01' or '2026_1_early'."""
        if not _re.match(r"^\d{4}_\d+_", pid):
            return None
        parts = pid.split("_")
        if len(parts) < 3:
            return None
        year, rnd_str = parts[0], parts[1]
        third = "_".join(parts[2:])
        try:
            rnd = int(rnd_str)
        except ValueError:
            return None
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
        bucket = {"early": "Early", "mid": "Mid", "late": "Late"}.get(third.lower())
        if bucket:
            return f"{year} {rnd}{suffix} ({bucket})"
        try:
            slot = int(third)
            return f"{year} {rnd}.{slot:02d}"
        except ValueError:
            return None

    _pick_re = _re.compile(r"^\d{4}_\d+_")
    movers = []
    for row in rows:
        row_dict = dict(row)
        player_id = str(row_dict["player_id"])
        # Skip picks - they're not players and their value swings are data noise
        if _pick_re.match(player_id):
            continue
        # Trust the name column from the DB; fall back to players_index then a placeholder
        if not row_dict.get("name") or row_dict["name"] == "Unknown":
            fallback = players_index_map.get(player_id)
            row_dict["name"] = fallback if fallback else f"Player {player_id}"
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


def _get_top_movers_from_parquet(days: int, limit: int) -> dict:
    """Get movers from parquet files when database is not available."""
    from pathlib import Path
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Load players index for name lookups
    from utils.utils import load_players_index
    players_index = load_players_index() or {}
    
    # Try to load the most recent parquet file
    parquet_file = Path(f"cache/player_history/player_history_{datetime.now().year}.parquet")
    if not parquet_file.exists():
        # Fallback to the combined file
        parquet_file = Path("cache/player_history/player_history_all.parquet")
    
    if not parquet_file.exists():
        return {
            "latest_date": None,
            "comparison_date": None,
            "requested_days": days,
            "used_days": None,
            "risers": [],
            "fallers": [],
        }
    
    # Parquet files contain only a single snapshot - no historical deltas are available,
    # so we cannot compute real risers/fallers. Return empty lists rather than fabricating
    # movement from random data.
    return {
        "latest_date": None,
        "comparison_date": None,
        "requested_days": days,
        "used_days": None,
        "risers": [],
        "fallers": [],
    }
