"""
Full reset to the last known-good value state.

Steps:
  1. Restore player_values (value_1qb, calibrated cols) from player_value_history
     using the most recent snapshot on or before 2026-05-05.
  2. Clear trade_intel_player_stats for the corrupted season so the next
     analytics run rebuilds market values from scratch with correct model values.

Run this, then run analytics + market_calibration to get correct values.

Usage:
    python data_building/full_reset_to_good_state.py
"""
from __future__ import annotations

import logging
from datetime import date

from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# The last known-good snapshot date
GOOD_DATE = date(2026, 5, 5)


def restore_player_values() -> int:
    """Overwrite player_values from the last good player_value_history snapshot."""
    with get_conn() as conn:
        # Find the closest snapshot on or before GOOD_DATE
        snap = conn.execute(
            """
            SELECT MAX(as_of_date) AS snap_date
            FROM player_value_history
            WHERE as_of_date <= %s AND source = 'model'
            """,
            (GOOD_DATE,),
        ).fetchone()

        if not snap or not snap["snap_date"]:
            logger.error("No player_value_history snapshot found on or before %s", GOOD_DATE)
            return 0

        snap_date = snap["snap_date"]
        logger.info("Restoring player_values from snapshot: %s", snap_date)

        # Restore value_1qb from snapshot, NULL out all calibration cols
        result = conn.execute(
            """
            UPDATE player_values pv
            SET
                value_1qb            = pvh.value,
                value_sf             = COALESCE(pvh.value_sf, pvh.value),
                calibrated_value_1qb = NULL,
                calibrated_value_sf  = NULL,
                calibration_source   = NULL,
                calibration_weight   = NULL
            FROM player_value_history pvh
            WHERE pvh.player_id  = pv.player_id
              AND pvh.as_of_date = %s
              AND pvh.source     = 'model'
            """,
            (snap_date,),
        )
        n = result.rowcount if hasattr(result, "rowcount") else 0
        logger.info("Restored %d player_values rows", n)
        return n


def clear_trade_intel_stats(season: int = 2026) -> int:
    """Delete corrupted trade_intel_player_stats rows for the given season."""
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM trade_intel_player_stats WHERE season = %s",
            (season,),
        )
        n = result.rowcount if hasattr(result, "rowcount") else 0
        logger.info("Cleared %d rows from trade_intel_player_stats (season %d)", n, season)
        return n


if __name__ == "__main__":
    logger.info("=== Step 1: Restore player_values from %s snapshot ===", GOOD_DATE)
    restored = restore_player_values()

    logger.info("=== Step 2: Clear corrupted trade_intel_player_stats ===")
    cleared = clear_trade_intel_stats(season=2026)

    print(f"\nDone. Restored {restored} player rows, cleared {cleared} market stats rows.")
    print("\nNext steps to fully rebuild:")
    print("  1. python -c \"from data_building.trade_intel.analytics import run_analytics; run_analytics()\"")
    print("  2. python -c \"from data_building.trade_intel.market_calibration import run_calibration; run_calibration()\"")
