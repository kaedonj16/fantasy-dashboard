"""
Retroactively smooth player_value_history after a value model change.

When the formula changes, old snapshots were built with a different scale,
creating artificial spikes in charts. This script re-applies EMA in
chronological order across the entire history so the chart reflects a
smooth progression rather than a step-function jump.

Usage
-----
    # Smooth everything (default alpha=0.55, gentler than the daily 0.70)
    python3 -m data_building.smooth_value_history

    # Stronger smoothing (lower alpha = more weight on history)
    python -m data_building.smooth_value_history --alpha 0.35

    # Dry run - prints stats without writing to DB
    python -m data_building.smooth_value_history --dry-run

    # Only smooth history from a specific date onwards (e.g. the day you
    # changed the formula)
    python -m data_building.smooth_value_history --from 2025-03-01

Algorithm
---------
For each player, iterate snapshots chronologically:
    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

The first snapshot for a player is kept as-is (no prior to blend with).

Both the 1QB (`value`) and Superflex (`sf_value`) columns are smoothed, each
with its own independent EMA carry. Rows missing an sf_value are left untouched
on the SF side and don't reset the SF carry.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from datetime import date

from dashboard_services.db import get_conn
from utils.utils import load_players_index

logger = logging.getLogger(__name__)

DEFAULT_ALPHA   = 0.55   # gentler than the daily 0.70; more history weight
DEFAULT_SOURCE  = "model"


def _load_calibrated_values() -> dict[str, tuple[float, float | None]]:
    """Load (1QB, SF) calibrated values from player_values, keyed by player_id.

    Returns {player_id: (value_1qb, value_sf_or_None)}. SF is None when the
    player has no SF value recorded (so seeding won't wipe an existing one).
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_id,
                       COALESCE(calibrated_value_1qb, value_1qb) AS v1,
                       COALESCE(calibrated_value_sf,  value_sf)  AS vsf
                FROM player_values
                WHERE COALESCE(calibrated_value_1qb, value_1qb) > 0
                """
            ).fetchall()
        return {
            r["player_id"]: (
                float(r["v1"]),
                float(r["vsf"]) if r["vsf"] is not None else None,
            )
            for r in rows
        }
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
                for pid, (v1, vsf) in cal_values.items():
                    # Get player name from players_index
                    player_info = players_index.get(str(pid)) or {}
                    player_name = player_info.get("name", "Unknown")

                    conn.execute(
                        """
                        INSERT INTO player_value_history
                            (as_of_date, player_id, name, value, sf_value, source)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (as_of_date, player_id, source)
                        DO UPDATE SET
                            name = EXCLUDED.name,
                            value = EXCLUDED.value,
                            sf_value = COALESCE(EXCLUDED.sf_value, player_value_history.sf_value)
                        """,
                        (today_iso, pid, player_name, v1, vsf, source),
                    )
                    seeded += 1
            logger.info("[smooth] Seeded %d calibrated values for %s", seeded, today_iso)

    # Load all rows ordered chronologically so we can walk them in sequence
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT as_of_date, player_id, name, position, team, value, sf_value
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

    # Group by player - walk each player's timeline and apply EMA
    by_player: dict[str, list] = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)

    # Each update: (new_value_or_None, new_sf_value_or_None, as_of_date, player_id).
    # A None column is left unchanged (COALESCE in the UPDATE), so 1QB and SF are
    # smoothed independently and a row is written only if either actually changed.
    updates: list[tuple] = []
    from_iso = from_date.isoformat() if from_date else None

    for pid, history in by_player.items():
        prev_smoothed: float | None = None      # 1QB EMA carry
        prev_smoothed_sf: float | None = None   # SF EMA carry

        for r in history:           # already sorted ASC by the query
            raw_val = float(r["value"])
            raw_sf  = float(r["sf_value"]) if r["sf_value"] is not None else None
            d       = r["as_of_date"]

            if prev_smoothed is None:
                # First known snapshot - keep as-is; becomes the EMA seed
                smoothed = raw_val
            else:
                smoothed = alpha * raw_val + (1.0 - alpha) * prev_smoothed

            # SF tracks its own EMA; rows missing sf_value don't reset the carry.
            smoothed_sf: float | None = None
            if raw_sf is not None:
                if prev_smoothed_sf is None:
                    smoothed_sf = raw_sf
                else:
                    smoothed_sf = alpha * raw_sf + (1.0 - alpha) * prev_smoothed_sf

            # Only write back rows that fall on/after from_date (if set)
            if from_iso is None or (d.isoformat() if hasattr(d, "isoformat") else str(d)) >= from_iso:
                val_update = None
                sf_update = None
                sr = round(smoothed, 2)
                if abs(sr - raw_val) >= 0.01:           # 1QB changed
                    val_update = sr
                if smoothed_sf is not None and raw_sf is not None:
                    ssf = round(smoothed_sf, 2)
                    if abs(ssf - raw_sf) >= 0.01:        # SF changed
                        sf_update = ssf
                if val_update is not None or sf_update is not None:
                    updates.append((val_update, sf_update, d, pid))

            prev_smoothed = smoothed
            if smoothed_sf is not None:
                prev_smoothed_sf = smoothed_sf

    logger.info("[smooth] %d rows will be updated.", len(updates))

    if dry_run:
        logger.info("[smooth] Dry run - no changes written.")
        return {"rows_read": len(rows), "rows_updated": 0, "dry_run": True, "would_update": len(updates)}

    # Write back in batches with connection recovery
    BATCH = 500
    written = 0
    for batch_start in range(0, len(updates), BATCH):
        batch = updates[batch_start : batch_start + BATCH]
        batch_written = 0
        
        # Retry each batch up to 3 times with fresh connections
        for attempt in range(3):
            try:
                with get_conn(autocommit=True) as conn:
                    for smoothed_val, smoothed_sf, d, pid in batch:
                        conn.execute(
                            """
                            UPDATE player_value_history
                               SET value    = COALESCE(%s, value),
                                   sf_value = COALESCE(%s, sf_value)
                             WHERE as_of_date = %s
                               AND player_id  = %s
                               AND source     = %s
                            """,
                            (smoothed_val, smoothed_sf, d, pid, source),
                        )
                    
                    batch_written = len(batch)
                    written += batch_written
                    logger.info("[smooth] Written batch %d-%d (%d rows) - Total: %d / %d", 
                               batch_start, batch_start + len(batch) - 1, batch_written, written, len(updates))
                    break  # Success, exit retry loop
                    
            except Exception as e:
                logger.warning("[smooth] Batch %d-%d failed (attempt %d/3): %s", 
                             batch_start, batch_start + len(batch) - 1, attempt + 1, e)
                if attempt == 2:  # Last attempt failed
                    logger.error("[smooth] Failed to write batch %d-%d after 3 attempts, skipping", 
                                batch_start, batch_start + len(batch) - 1)
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

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
