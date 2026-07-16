"""
Purge a polluted daily snapshot from player_value_history.

The top-movers widget and the player sparklines read raw snapshots from
player_value_history (source='model'). If one day's snapshot was written with
bad values - e.g. the day the whole pool rescaled and small players roughly
doubled - that snapshot poisons every comparison that lands on it: the movers
board fills with uniform ~100% swings, and sparklines show a step.

restore_values_from_history.py fixes the live player_values table, but NOT the
history table. This tool removes the bad snapshot's model rows so movers and
sparklines fall back to the clean neighbouring days. Snapshots are derived data
rebuilt daily by the cron, so deleting one historical day is safe and reversible
(the cron will not recreate a past day, but nothing depends on its existence).

It is DRY-RUN by default: it prints how doubled the target day looks (median
value vs the nearest earlier snapshot) plus a few well-known players, and does
nothing. Pass --confirm to actually delete.

Usage:
    python -m data_building.purge_history_snapshot 2026-07-15            # preview
    python -m data_building.purge_history_snapshot 2026-07-15 --confirm  # delete
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_PREVIEW = {"9488": "Jaxon Smith-Njigba", "9493": "Puka Nacua", "8134": "Khalil Shakir"}


def _prev_date(conn, as_of: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(as_of_date) AS d FROM player_value_history "
        "WHERE source = 'model' AND as_of_date < %s",
        (as_of,),
    ).fetchone()
    if not row or not row.get("d"):
        return None
    return row["d"].isoformat() if hasattr(row["d"], "isoformat") else str(row["d"])


def preview(as_of: str) -> int:
    """Print how doubled `as_of` looks vs the prior snapshot. Returns row count."""
    with get_conn() as conn:
        n = (conn.execute(
            "SELECT COUNT(*) AS n FROM player_value_history "
            "WHERE source = 'model' AND as_of_date = %s",
            (as_of,),
        ).fetchone() or {}).get("n", 0)
        if not n:
            logger.info("No model rows on %s - nothing to purge.", as_of)
            return 0

        prev = _prev_date(conn, as_of)
        if prev:
            # Median is robust to the handful of top-of-board players whose
            # values are pinned; the pool-wide ~2x shows up clearly in it.
            med_target = conn.execute(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS m "
                "FROM player_value_history WHERE source='model' AND as_of_date=%s "
                "AND value > 0",
                (as_of,),
            ).fetchone()["m"]
            med_prev = conn.execute(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS m "
                "FROM player_value_history WHERE source='model' AND as_of_date=%s "
                "AND value > 0",
                (prev,),
            ).fetchone()["m"]
            ratio = (float(med_target) / float(med_prev)) if med_prev else 0.0
            logger.info(
                "median value %s=%.1f  vs prior %s=%.1f  ->  ratio %.2fx",
                as_of, float(med_target or 0), prev, float(med_prev or 0), ratio,
            )
            if ratio >= 1.6:
                logger.info("  ratio >= 1.6x: this snapshot looks inflated (as expected).")
            else:
                logger.info("  ratio looks normal - double-check you picked the right date.")

            # Per-player before/after for a few well-known ids.
            for pid, name in _PREVIEW.items():
                t = conn.execute(
                    "SELECT value FROM player_value_history WHERE source='model' "
                    "AND as_of_date=%s AND player_id=%s", (as_of, pid),
                ).fetchone()
                p = conn.execute(
                    "SELECT value FROM player_value_history WHERE source='model' "
                    "AND as_of_date=%s AND player_id=%s", (prev, pid),
                ).fetchone()
                if t and p:
                    logger.info("  %-22s %s=%s  prior %s=%s",
                                name, as_of, t["value"], prev, p["value"])
        else:
            logger.info("No earlier snapshot to compare against.")
    return n


def purge(as_of: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM player_value_history "
            "WHERE source = 'model' AND as_of_date = %s",
            (as_of,),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
    return deleted


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    do_delete = "--confirm" in sys.argv
    if not args:
        raise SystemExit("Usage: python -m data_building.purge_history_snapshot "
                         "<YYYY-MM-DD> [--confirm]")
    target = args[0]
    if target >= date.today().isoformat():
        # Guard against nuking today's (post-fix, clean) snapshot by accident.
        logger.warning("Refusing to purge today or a future date (%s).", target)
        raise SystemExit(1)

    logger.info("Inspecting model snapshot for %s ...", target)
    n = preview(target)
    if not n:
        raise SystemExit(0)
    if not do_delete:
        logger.info("DRY RUN: %d rows would be deleted. Re-run with --confirm to purge.", n)
        print(f"[dry-run] {n} rows on {target}. Add --confirm to delete.")
        raise SystemExit(0)

    deleted = purge(target)
    logger.info("Deleted %d model rows for %s. Movers/sparklines will use clean neighbours.",
                deleted, target)
    print(f"Purged {deleted} rows for {target}.")
