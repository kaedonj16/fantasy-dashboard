"""
Save daily player values to database for historical tracking.

This module provides functions to persist player values to PostgreSQL,
enabling historical trend analysis and value change tracking over time.
"""

import math
import os
from datetime import date
from typing import List, Dict, Any


def _safe_int(v):
    """Convert a value to int, returning None for NaN/None/invalid."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def save_daily_values_to_db(value_table: List[Dict[str, Any]], snapshot_date: date = None) -> int:
    """
    Save player values to database for the given date.

    Args:
        value_table: List of player value dictionaries from model
        snapshot_date: Date for this snapshot (defaults to today)

    Returns:
        Number of players saved to database
    """
    # Only import if DATABASE_URL is set
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or any(token in db_url for token in ("USER", "PASSWORD", "HOST")):
        print("[save_player_values] DATABASE_URL not configured, skipping database save")
        return 0

    try:
        from dashboard_services.db import get_conn
    except Exception as e:
        print(f"[save_player_values] Database not available: {e}")
        return 0

    if snapshot_date is None:
        snapshot_date = date.today()

    if not value_table:
        print("[save_player_values] No value table provided, skipping")
        return 0

    saved_count = 0

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for row in value_table:
                    if not isinstance(row, dict):
                        continue

                    player_id = row.get("id")
                    if not player_id:
                        continue

                    # Extract values with defaults
                    def _cap(v): return min(float(v), 999.9) if v is not None else None
                    value_1qb = _cap(row.get("value"))
                    value_sf  = _cap(row.get("sf_value"))
                    redraft_value_1qb = row.get("redraft_value_1qb")
                    redraft_value_sf  = row.get("redraft_value_sf")
                    position = row.get("position") or row.get("pos")
                    pos_rank = row.get("pos_rank")
                    pos_rank_label = row.get("pos_rank_label")
                    age = row.get("age")
                    team = row.get("team")
                    years_exp = row.get("years_exp")
                    rank_change_7d = row.get("rank_change_7d")
                    pos_rank_change_7d = row.get("pos_rank_change_7d")
                    
                    # League-size specific values
                    value_8    = _cap(row.get("value_8"))
                    value_12   = _cap(row.get("value_12"))
                    value_14   = _cap(row.get("value_14"))
                    sf_value_8  = _cap(row.get("sf_value_8"))
                    sf_value_12 = _cap(row.get("sf_value_12"))
                    sf_value_14 = _cap(row.get("sf_value_14"))
                    sf_pos_rank = row.get("sf_pos_rank")
                    sf_pos_rank_label = row.get("sf_pos_rank_label")

                    cur.execute(
                        """
                        INSERT INTO player_values (
                            player_id,
                            last_updated,
                            value_1qb,
                            value_sf,
                            value_8,
                            value_12,
                            value_14,
                            sf_value_8,
                            sf_value_12,
                            sf_value_14,
                            redraft_value_1qb,
                            redraft_value_sf,
                            position,
                            pos_rank,
                            pos_rank_label,
                            sf_pos_rank,
                            sf_pos_rank_label,
                            age,
                            team,
                            years_exp,
                            rank_change_7d,
                            pos_rank_change_7d
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id)
                        DO UPDATE SET
                            last_updated = EXCLUDED.last_updated,
                            value_1qb = EXCLUDED.value_1qb,
                            value_sf = EXCLUDED.value_sf,
                            value_8 = EXCLUDED.value_8,
                            value_12 = EXCLUDED.value_12,
                            value_14 = EXCLUDED.value_14,
                            sf_value_8 = EXCLUDED.sf_value_8,
                            sf_value_12 = EXCLUDED.sf_value_12,
                            sf_value_14 = EXCLUDED.sf_value_14,
                            redraft_value_1qb = COALESCE(EXCLUDED.redraft_value_1qb, player_values.redraft_value_1qb),
                            redraft_value_sf  = COALESCE(EXCLUDED.redraft_value_sf,  player_values.redraft_value_sf),
                            position = EXCLUDED.position,
                            pos_rank = EXCLUDED.pos_rank,
                            pos_rank_label = EXCLUDED.pos_rank_label,
                            sf_pos_rank = EXCLUDED.sf_pos_rank,
                            sf_pos_rank_label = EXCLUDED.sf_pos_rank_label,
                            age = EXCLUDED.age,
                            team = EXCLUDED.team,
                            years_exp = EXCLUDED.years_exp,
                            rank_change_7d = EXCLUDED.rank_change_7d,
                            pos_rank_change_7d = EXCLUDED.pos_rank_change_7d
                        """,
                        (
                            str(player_id),
                            snapshot_date,
                            value_1qb,
                            value_sf,
                            value_8,
                            value_12,
                            value_14,
                            sf_value_8,
                            sf_value_12,
                            sf_value_14,
                            redraft_value_1qb,
                            redraft_value_sf,
                            position,
                            _safe_int(pos_rank),
                            pos_rank_label,
                            _safe_int(sf_pos_rank),
                            sf_pos_rank_label,
                            _safe_int(age),
                            team,
                            _safe_int(years_exp),
                            _safe_int(rank_change_7d),
                            _safe_int(pos_rank_change_7d),
                        ),
                    )
                    saved_count += 1

        print(f"[save_player_values] Successfully saved {saved_count} player values for {snapshot_date}")
        return saved_count

    except Exception as e:
        print(f"[save_player_values] Error saving to database: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    # Test/manual run
    from utils.utils import load_model_value_table

    print("Loading value table...")
    value_table = load_model_value_table()
    print(f"Loaded {len(value_table)} players from value table")

    count = save_daily_values_to_db(value_table)
    print(f"Saved {count} players to database")
