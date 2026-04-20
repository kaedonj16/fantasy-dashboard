"""
Restore player_values.value_1qb / value_sf from the JSON model file.

Run this when value_1qb has been corrupted by calibrated values being
written back into the model prior column.  After this runs, execute
run_trade_value_model() to rebuild calibrated values from clean priors.

Usage:
    python scripts/restore_model_values.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard_services.db import get_conn


def _latest_model_json() -> Path:
    data_dir = ROOT / "data"
    files = sorted(data_dir.glob("model_values_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No model_values_*.json found in data/")
    return files[0]


def restore_model_values(json_path: Path | None = None) -> dict:
    path = json_path or _latest_model_json()
    print(f"[restore] Loading model values from {path.name} ...")

    with open(path) as f:
        players = json.load(f)

    updated = 0
    skipped = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            # First clear all calibrated values so WLS starts fresh
            cur.execute(
                "UPDATE player_values SET calibrated_value_1qb = NULL, calibrated_value_sf = NULL, "
                "calibration_weight = NULL, calibration_source = NULL"
            )
            print(f"[restore] Cleared calibrated values for all players.")

            for p in players:
                pid     = str(p.get("id") or "")
                v_1qb   = p.get("value")
                v_sf    = p.get("sf_value")
                if not pid or v_1qb is None:
                    skipped += 1
                    continue

                cur.execute(
                    """
                    UPDATE player_values
                    SET value_1qb = %s, value_sf = %s
                    WHERE player_id = %s
                    """,
                    (float(v_1qb), float(v_sf or v_1qb), pid),
                )
                if cur.rowcount:
                    updated += 1
                else:
                    skipped += 1

        conn.commit()

    print(f"[restore] Done — {updated} players restored, {skipped} skipped.")
    return {"updated": updated, "skipped": skipped, "source": path.name}


if __name__ == "__main__":
    result = restore_model_values()
    print(result)
    print("\nNext step: python data_building/trade_intel/trade_value_model.py")
