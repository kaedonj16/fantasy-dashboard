"""
Save daily player values to database for historical tracking.

This module provides functions to persist player values to PostgreSQL,
enabling historical trend analysis and value change tracking over time.
"""

from datetime import date
from typing import List, Dict, Any
import os


def save_daily_values_to_db(value_table: List[Dict[str, Any]], snapshot_date: date = None) -> int:
    """
    Save player values to database for the given date.

    Args:
        value_table: List of player value dictionaries from model
        snapshot_date: Date for this snapshot (defaults to today)

    Returns:
        Number of players saved to database

    Example:
        >>> from utils.utils import load_value_table
        >>> value_table = load_value_table()
        >>> count = save_daily_values_to_db(value_table)
        >>> print(f"Saved {count} player values")
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
                    value_1qb = row.get("value")
                    value_sf = row.get("sf_value")
                    position = row.get("position") or row.get("pos")
                    pos_rank = row.get("pos_rank")
                    pos_rank_label = row.get("pos_rank_label")
                    age = row.get("age")
                    team = row.get("team")
                    years_exp = row.get("years_exp")

                    # Insert or update (upsert)
                    cur.execute(
                        """
                        INSERT INTO player_values (
                            player_id,
                            date,
                            value_1qb,
                            value_sf,
                            position,
                            pos_rank,
                            pos_rank_label,
                            age,
                            team,
                            years_exp
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, date)
                        DO UPDATE SET
                            value_1qb = EXCLUDED.value_1qb,
                            value_sf = EXCLUDED.value_sf,
                            position = EXCLUDED.position,
                            pos_rank = EXCLUDED.pos_rank,
                            pos_rank_label = EXCLUDED.pos_rank_label,
                            age = EXCLUDED.age,
                            team = EXCLUDED.team,
                            years_exp = EXCLUDED.years_exp
                        """,
                        (
                            str(player_id),
                            snapshot_date,
                            value_1qb,
                            value_sf,
                            position,
                            pos_rank,
                            pos_rank_label,
                            age,
                            team,
                            years_exp,
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
