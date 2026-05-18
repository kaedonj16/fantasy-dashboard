"""
Run once to NULL out calibrated_value_1qb rows that are impossibly inflated
compared to the model prior (value_1qb). This fixes corrupted DB state written
by bad pipeline runs and lets the app fall back to the correct model values.

Usage:
    python data_building/reset_bad_calibrations.py
"""
from __future__ import annotations

import logging
from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def reset_bad_calibrations() -> int:
    with get_conn() as conn:
        bad_rows = conn.execute(
            """
            SELECT player_id, value_1qb, calibrated_value_1qb
            FROM player_values
            WHERE calibrated_value_1qb IS NOT NULL
              AND (
                calibrated_value_1qb > 999.9
                OR (value_1qb IS NOT NULL AND value_1qb > 0 AND calibrated_value_1qb > value_1qb * 3.0)
                OR (COALESCE(value_1qb, 0) < 10 AND calibrated_value_1qb > 100)
              )
            ORDER BY calibrated_value_1qb DESC
            """
        ).fetchall()

        if not bad_rows:
            logger.info("No bad calibration rows found — DB looks clean.")
            return 0

        logger.info("Found %d bad rows:", len(bad_rows))
        for r in bad_rows[:30]:
            logger.info("  %-14s  model=%-8s  calibrated=%s",
                        r["player_id"], r["value_1qb"], r["calibrated_value_1qb"])

        conn.execute(
            """
            UPDATE player_values
            SET
                calibrated_value_1qb = NULL,
                calibrated_value_sf  = NULL,
                calibration_source   = NULL,
                calibration_weight   = NULL
            WHERE calibrated_value_1qb IS NOT NULL
              AND (
                calibrated_value_1qb > 999.9
                OR (value_1qb IS NOT NULL AND value_1qb > 0 AND calibrated_value_1qb > value_1qb * 3.0)
                OR (COALESCE(value_1qb, 0) < 10 AND calibrated_value_1qb > 100)
              )
            """
        )
        logger.info("Nulled out %d corrupted calibration rows.", len(bad_rows))
        return len(bad_rows)


if __name__ == "__main__":
    n = reset_bad_calibrations()
    print(f"Reset {n} bad calibration rows.")
