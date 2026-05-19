"""
Restore player_values table from model_values.json.

Overwrites value_1qb, value_sf, and all size-specific value columns
with the values from model_values.json, then NULLs out all calibrated
columns so the app serves clean model values.

Usage:
    python data_building/restore_values_from_json.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_json() -> list[dict]:
    # Prefer dated file, fall back to plain model_values.json
    candidates = sorted(DATA_DIR.glob("model_values_*.json"), reverse=True)
    path = candidates[0] if candidates else DATA_DIR / "model_values.json"
    logger.info("Loading values from %s", path)
    return json.loads(path.read_text())


def restore() -> int:
    players = load_json()
    if not players:
        logger.error("No data loaded from JSON")
        return 0

    updated = 0
    with get_conn() as conn:
        for p in players:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue

            v1qb = p.get("value") or 0.0
            vsf  = p.get("sf_value") or v1qb

            conn.execute(
                """
                UPDATE player_values SET
                    value_1qb             = %(v1qb)s,
                    value_sf              = %(vsf)s,
                    value_8               = %(v8)s,
                    value_12              = %(v12)s,
                    value_14              = %(v14)s,
                    sf_value_8            = %(sf8)s,
                    sf_value_12           = %(sf12)s,
                    sf_value_14           = %(sf14)s,
                    calibrated_value_1qb  = NULL,
                    calibrated_value_sf   = NULL,
                    calibration_source    = NULL,
                    calibration_weight    = NULL
                WHERE player_id = %(pid)s
                """,
                {
                    "pid":  pid,
                    "v1qb": float(v1qb),
                    "vsf":  float(vsf),
                    "v8":   float(p.get("value_8")   or v1qb),
                    "v12":  float(p.get("value_12")  or v1qb),
                    "v14":  float(p.get("value_14")  or v1qb),
                    "sf8":  float(p.get("sf_value_8")  or vsf),
                    "sf12": float(p.get("sf_value_12") or vsf),
                    "sf14": float(p.get("sf_value_14") or vsf),
                },
            )
            updated += 1

    logger.info("Restored %d player rows from JSON", updated)
    return updated


if __name__ == "__main__":
    n = restore()
    print(f"Restored {n} rows.")
