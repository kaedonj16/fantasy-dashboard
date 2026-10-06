"""
One-off script: clear all WLS calibration columns from player_values.

Run this when calibrated_value_1qb contains bad data so that rankings
fall back to the raw model values (value_1qb) which are correct.
"""
from dotenv import load_dotenv
load_dotenv()

from dashboard_services.db import get_conn

with get_conn() as conn:
    result = conn.execute(
        """
        UPDATE player_values
        SET calibrated_value_1qb = NULL,
            calibrated_value_sf  = NULL,
            calibration_weight   = NULL,
            calibration_source   = NULL
        WHERE calibration_source = 'trade_wls'
        """
    )
    print(f"Cleared calibration for {result.rowcount} players")
    # Confirm the top 5 by raw model value
    rows = conn.execute(
        """
        SELECT player_id, value_1qb, calibrated_value_1qb, position
        FROM player_values
        ORDER BY value_1qb DESC NULLS LAST
        LIMIT 5
        """
    ).fetchall()
    print("Top 5 by raw model value after reset:")
    for r in rows:
        print(f"  {r['player_id']}: value_1qb={r['value_1qb']}, calibrated={r['calibrated_value_1qb']}, pos={r['position']}")
