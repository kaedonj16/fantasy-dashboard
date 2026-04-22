"""
Retroactively smooth player_value_history after a value model change.

When the formula changes, old snapshots were built with a different scale,
creating artificial spikes in charts. This script re-applies EMA in
chronological order across the entire history so the chart reflects a
smooth progression rather than a step-function jump.

Usage
-----
    # Smooth everything (default alpha=0.55, gentler than the daily 0.70)
    python -m data_building.smooth_value_history

    # Stronger smoothing (lower alpha = more weight on history)
    python -m data_building.smooth_value_history --alpha 0.35

    # Dry run — prints stats without writing to DB
    python -m data_building.smooth_value_history --dry-run

    # Only smooth history from a specific date onwards (e.g. the day you
    # changed the formula)
    python -m data_building.smooth_value_history --from 2025-03-01

Algorithm
---------
For each player, iterate snapshots chronologically:
    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

The first snapshot for a player is kept as-is (no prior to blend with).
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import date

from dashboard_services.db import get_conn
from utils.utils import load_players_index

logger = logging.getLogger(__name__)

DEFAULT_ALPHA   = 0.55   # gentler than the daily 0.70; more history weight
DEFAULT_SOURCE  = "model"


def _load_calibrated_values() -> dict[str, float]:
    """Load COALESCE(calibrated_value_1qb, value_1qb) from player_values."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_id, COALESCE(calibrated_value_1qb, value_1qb) AS value
                FROM player_values
                WHERE COALESCE(calibrated_value_1qb, value_1qb) > 0
                """
            ).fetchall()
        return {r["player_id"]: float(r["value"]) for r in rows}
    except Exception as e:
        logger.warning("[smooth] Could not load calibrated values: %s", e)
        return {}


def smooth_value_history(
    *,
    alpha: float = DEFAULT_ALPHA,
    source: str  = DEFAULT_SOURCE,
    from_date: date | None = None,
    dry_run: bool = False,
    use_calibrated: bool = True,
) -> dict:
    """
    Re-apply EMA to all (or post-from_date) snapshots in player_value_history.

    use_calibrated: if True (default), seed today's entry with the calibrated
      value from player_values before smoothing so history blends toward the
      WLS-adjusted value rather than the raw model value.

    Returns a summary dict with counts.
    """
    logger.info(
        "[smooth] alpha=%.2f  source=%s  from=%s  use_calibrated=%s  dry_run=%s",
        alpha, source, from_date, use_calibrated, dry_run,
    )

    today = date.today()

    # Seed today's history with calibrated values before EMA
    if use_calibrated and not dry_run:
        cal_values = _load_calibrated_values()
        if cal_values:
            today_iso = today.isoformat()
            seeded = 0
            # Load players index to get player names
            players_index = load_players_index() or {}
            with get_conn() as conn:
                for pid, val in cal_values.items():
                    # Get player name from players_index
                    player_info = players_index.get(str(pid)) or {}
                    player_name = player_info.get("name", "Unknown")
                    
                    conn.execute(
                        """
                        INSERT INTO player_value_history
                            (as_of_date, player_id, name, value, source)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (as_of_date, player_id, source)
                        DO UPDATE SET 
                            name = EXCLUDED.name,
                            value = EXCLUDED.value
                        """,
                        (today_iso, pid, player_name, val, source),
                    )
                    seeded += 1
            logger.info("[smooth] Seeded %d calibrated values for %s", seeded, today_iso)

    # Load all rows ordered chronologically so we can walk them in sequence
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT as_of_date, player_id, name, position, team, value
            FROM player_value_history
            WHERE source = %s
            ORDER BY player_id, as_of_date ASC
            """,
            (source,),
        ).fetchall()

    if not rows:
        logger.info("[smooth] No history rows found.")
        return {"rows_read": 0, "rows_updated": 0}

    logger.info("[smooth] Loaded %d history rows across all players.", len(rows))

    # Group by player — walk each player's timeline and apply EMA
    by_player: dict[str, list] = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    updates: list[tuple] = []  # (smoothed_value, as_of_date, player_id)
    from_iso = from_date.isoformat() if from_date else None

    for pid, history in by_player.items():
        prev_smoothed: float | None = None

        for r in history:           # already sorted ASC by the query
            raw_val = float(r["value"])
            d       = r["as_of_date"]

            if prev_smoothed is None:
                # First known snapshot — keep as-is; becomes the EMA seed
                smoothed = raw_val
            else:
                smoothed = alpha * raw_val + (1.0 - alpha) * prev_smoothed

            # Only write back rows that fall on/after from_date (if set)
            if from_iso is None or (d.isoformat() if hasattr(d, "isoformat") else str(d)) >= from_iso:
                smoothed_rounded = round(smoothed, 2)
                if abs(smoothed_rounded - raw_val) >= 0.01:   # skip no-op writes
                    updates.append((smoothed_rounded, d, pid))

            prev_smoothed = smoothed

    logger.info("[smooth] %d rows will be updated.", len(updates))

    if dry_run:
        logger.info("[smooth] Dry run — no changes written.")
        return {"rows_read": len(rows), "rows_updated": 0, "dry_run": True, "would_update": len(updates)}

    # Write back in batches
    BATCH = 500
    written = 0
    with get_conn() as conn:
        for i in range(0, len(updates), BATCH):
            batch = updates[i : i + BATCH]
            for smoothed_val, d, pid in batch:
                conn.execute(
                    """
                    UPDATE player_value_history
                       SET value = %s
                     WHERE as_of_date = %s
                       AND player_id  = %s
                       AND source     = %s
                    """,
                    (smoothed_val, d, pid, source),
                )
            written += len(batch)
            logger.info("[smooth] Written %d / %d", written, len(updates))

    logger.info("[smooth] Done. %d rows smoothed.", written)
    return {"rows_read": len(rows), "rows_updated": written}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Retroactively smooth player_value_history EMA.")
    parser.add_argument("--alpha",   type=float, default=DEFAULT_ALPHA,
                        help="EMA weight for newest value (0–1). Lower = smoother. Default %(default)s")
    parser.add_argument("--source",  default=DEFAULT_SOURCE,
                        help="history source tag. Default '%(default)s'")
    parser.add_argument("--from",    dest="from_date", default=None,
                        help="Only smooth rows on/after this date (YYYY-MM-DD). "
                             "Rows before this date are used as EMA seed only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be changed without writing to DB.")
    parser.add_argument("--no-calibrated", action="store_true",
                        help="Skip seeding today's snapshot with calibrated values.")
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date) if args.from_date else None
    result = smooth_value_history(
        alpha=args.alpha,
        source=args.source,
        from_date=from_date,
        dry_run=args.dry_run,
        use_calibrated=not args.no_calibrated,
    )
    print(result)
