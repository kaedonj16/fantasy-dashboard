"""
Restore the player_values table from a known-good player_value_history snapshot.

Use this to roll back after a bad model/calibration run (e.g. a day where the
whole pool rescaled and small players roughly doubled). It copies the model
value columns from player_value_history for the chosen date into player_values,
and NULLs the calibrated columns so the app serves the clean model values -
exactly what restore_values_from_json.py does, but sourced from the durable DB
history rather than the (possibly already-overwritten) model_values.json file.

Usage:
    python -m data_building.restore_values_from_history 2026-07-14
    python -m data_building.restore_values_from_history            # newest date < today

Prints a before/after preview for a few players and a total count. Picks are
left untouched (they are not stored in player_value_history).
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# player_value_history column -> player_values column
_COL_MAP = {
    "value": "value_1qb",
    "sf_value": "value_sf",
    "value_8": "value_8",
    "value_12": "value_12",
    "value_14": "value_14",
    "sf_value_8": "sf_value_8",
    "sf_value_12": "sf_value_12",
    "sf_value_14": "sf_value_14",
}
_HIST_COLS = list(_COL_MAP.keys())
_PREVIEW = {"9488": "Jaxon Smith-Njigba", "9493": "Puka Nacua", "8134": "Khalil Shakir"}


def _resolve_date(arg: str | None) -> str:
    if arg:
        return arg
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(as_of_date) AS d FROM player_value_history "
            "WHERE source = 'model' AND as_of_date < %s",
            (date.today().isoformat(),),
        ).fetchone()
    if not row or not row.get("d"):
        raise SystemExit("No prior model snapshot found in player_value_history.")
    return row["d"].isoformat() if hasattr(row["d"], "isoformat") else str(row["d"])


def restore(as_of: str) -> int:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, " + ", ".join(_HIST_COLS)
            + " FROM player_value_history WHERE source = 'model' AND as_of_date = %s",
            (as_of,),
        ).fetchall()
        if not rows:
            raise SystemExit(f"No model snapshot for {as_of}.")

        # Preview a few well-known players (current -> restored).
        cur = {
            r["player_id"]: r
            for r in conn.execute(
                "SELECT player_id, value_1qb, calibrated_value_1qb FROM player_values "
                "WHERE player_id = ANY(%s)",
                (list(_PREVIEW.keys()),),
            ).fetchall()
        }
        by_pid = {str(r["player_id"]): r for r in rows}
        for pid, name in _PREVIEW.items():
            h = by_pid.get(pid)
            c = cur.get(pid)
            if h and c:
                shown = c.get("calibrated_value_1qb") or c.get("value_1qb")
                logger.info("  %-22s now=%s -> restore=%s", name, shown, h["value"])

        updated = 0
        for r in rows:
            pid = str(r["player_id"]).strip()
            if not pid:
                continue
            v1qb = float(r["value"] or 0.0)
            vsf = float(r["sf_value"] or v1qb)
            conn.execute(
                """
                UPDATE player_values SET
                    value_1qb            = %(v1qb)s,
                    value_sf             = %(vsf)s,
                    value_8              = %(v8)s,
                    value_12             = %(v12)s,
                    value_14             = %(v14)s,
                    sf_value_8           = %(sf8)s,
                    sf_value_12          = %(sf12)s,
                    sf_value_14          = %(sf14)s,
                    calibrated_value_1qb = NULL,
                    calibrated_value_sf  = NULL,
                    calibration_source   = NULL,
                    calibration_weight   = NULL
                WHERE player_id = %(pid)s
                """,
                {
                    "pid": pid,
                    "v1qb": v1qb,
                    "vsf": vsf,
                    "v8": float(r["value_8"] or v1qb),
                    "v12": float(r["value_12"] or v1qb),
                    "v14": float(r["value_14"] or v1qb),
                    "sf8": float(r["sf_value_8"] or vsf),
                    "sf12": float(r["sf_value_12"] or vsf),
                    "sf14": float(r["sf_value_14"] or vsf),
                },
            )
            updated += 1
        conn.commit()
    return updated


if __name__ == "__main__":
    target = _resolve_date(sys.argv[1] if len(sys.argv) > 1 else None)
    logger.info("Restoring player_values from the %s model snapshot…", target)
    n = restore(target)
    logger.info("Restored %d player rows from %s. Calibrated columns cleared.", n, target)
    print(f"Restored {n} rows from {target}.")
