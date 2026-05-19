"""
Repair corrupted player_value_history snapshots.

Snapshots written after CORRUPT_AFTER_DATE have inflated values from bad
model retraining runs. This script overwrites those rows with the correct
values from the current player_values table (which was already restored to
the last known-good state).

Usage:
    python data_building/repair_value_history.py
"""
from __future__ import annotations

import logging
from datetime import date

from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Snapshots after this date are corrupted; repair them using current player_values
CORRUPT_AFTER_DATE = date(2026, 5, 15)


def repair_history() -> int:
    with get_conn() as conn:
        # Find which dates need fixing
        bad_dates = conn.execute(
            """
            SELECT DISTINCT as_of_date
            FROM player_value_history
            WHERE source = 'model' AND as_of_date > %s
            ORDER BY as_of_date
            """,
            (CORRUPT_AFTER_DATE,),
        ).fetchall()

        if not bad_dates:
            logger.info("No corrupted snapshots found after %s", CORRUPT_AFTER_DATE)
            return 0

        logger.info(
            "Found %d date(s) to repair: %s",
            len(bad_dates),
            [str(r["as_of_date"]) for r in bad_dates],
        )

        # Overwrite value/value_sf on corrupted rows with current player_values
        result = conn.execute(
            """
            UPDATE player_value_history pvh
            SET
                value    = pv.value_1qb,
                value_sf = COALESCE(pv.value_sf, pv.value_1qb)
            FROM player_values pv
            WHERE pvh.player_id  = pv.player_id
              AND pvh.source     = 'model'
              AND pvh.as_of_date > %s
              AND pv.value_1qb IS NOT NULL
            """,
            (CORRUPT_AFTER_DATE,),
        )
        n = result.rowcount if hasattr(result, "rowcount") else 0
        logger.info("Repaired %d rows in player_value_history", n)
        return n


if __name__ == "__main__":
    repaired = repair_history()
    print(f"\nDone. Repaired {repaired} history rows.")
