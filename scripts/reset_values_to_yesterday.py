#!/usr/bin/env python3
"""
Reset player_values table back to yesterday's level using the DB snapshots.

The 1QB normalization scale reset during today's deploys, compressing every
non-anchor player's value by ~4%.  Yesterday's values are stored in
player_value_history; this script copies them into player_values (the live,
shared-across-all-instances table) so the website immediately shows
yesterday's numbers on every instance.

It also overwrites today's player_value_history row with yesterday's values so
the sparkline doesn't show the artificial drop, and seeds pipeline_state so
the next daily build holds the restored level.

Usage:
    python scripts/reset_values_to_yesterday.py            # dry run
    python scripts/reset_values_to_yesterday.py --commit   # apply
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn

_SCALE_STATE_KEY = "value_scale_1qb"
_ALL_VALUE_COLS  = ["value", "sf_value", "value_8", "value_12", "value_14",
                    "sf_value_8", "sf_value_12", "sf_value_14"]
_1QB_COLS        = ["value", "value_8", "value_12", "value_14"]
_RATIO_BAND      = (0.80, 1.25)  # filter genuine per-player outliers


def _two_snapshots(conn) -> tuple:
    rows = conn.execute(
        """
        SELECT DISTINCT as_of_date
        FROM player_value_history
        WHERE source = 'model'
        ORDER BY as_of_date DESC
        LIMIT 2
        """
    ).fetchall()
    dates = [r["as_of_date"] for r in rows]
    if len(dates) < 2:
        return (dates[0] if dates else None, None)
    return dates[0], dates[1]   # today, yesterday


def _snapshot(conn, d) -> dict:
    """Return {player_id: {col: float}} for a given snapshot date."""
    rows = conn.execute(
        """
        SELECT player_id, value, sf_value,
               value_8, value_12, value_14,
               sf_value_8, sf_value_12, sf_value_14
        FROM player_value_history
        WHERE source = 'model' AND as_of_date = %s
        """,
        (d,),
    ).fetchall()
    out = {}
    for r in rows:
        pid = str(r["player_id"])
        out[pid] = {c: (float(r[c]) if r[c] is not None else 0.0) for c in _ALL_VALUE_COLS}
    return out


def _persisted_scale(conn) -> float:
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_state (
                key TEXT PRIMARY KEY,
                value DOUBLE PRECISION,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        row = conn.execute(
            "SELECT value FROM pipeline_state WHERE key = %s", (_SCALE_STATE_KEY,)
        ).fetchone()
        if row and row.get("value"):
            return float(row["value"])
    except Exception:
        pass
    return 0.0


def _write_scale(conn, scale: float) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_state (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE
            SET value = excluded.value, updated_at = NOW()
        """,
        (_SCALE_STATE_KEY, round(float(scale), 6)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="Apply changes (default is a dry run)")
    args = ap.parse_args()

    with get_conn() as conn:
        today, yday = _two_snapshots(conn)
        if yday is None:
            print("ERROR: need at least two snapshots in player_value_history.", file=sys.stderr)
            return 1

        today_snap = _snapshot(conn, today)
        yday_snap  = _snapshot(conn, yday)
        cur_scale  = _persisted_scale(conn)

    print(f"Today's snapshot:    {today}  ({len(today_snap)} players)")
    print(f"Restoring to:        {yday}  ({len(yday_snap)} players)")
    print(f"Persisted scale now: {cur_scale or 'MISSING'}")

    # Measure the uniform 1QB drop from the DB snapshots (today vs yesterday)
    ratios = []
    for pid, yd in yday_snap.items():
        td = today_snap.get(pid)
        if not td or td["value"] <= 0 or yd["value"] <= 0:
            continue
        r = yd["value"] / td["value"]
        if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
            ratios.append(r)

    if len(ratios) < 25:
        print(f"ERROR: only {len(ratios)} comparable players — can't infer scale factor.", file=sys.stderr)
        return 1

    ratio = statistics.median(ratios)
    print(f"Comparable players:  {len(ratios)}")
    print(f"Median ratio (yday/today): {ratio:.4f}  ({(ratio-1)*100:+.2f}%)")

    if abs(ratio - 1.0) < 0.005:
        print("Today's and yesterday's values are within 0.5% — nothing to restore.")
        return 0

    # Preview top players: yday DB value vs today DB value vs what we'll restore
    sorted_players = sorted(
        [(pid, yday_snap[pid]["value"]) for pid in yday_snap if yday_snap[pid]["value"] > 0],
        key=lambda x: x[1], reverse=True
    )[:8]
    print(f"\n  Top players  (today DB -> restoring to yesterday DB)")
    for pid, _ in sorted_players:
        tv = today_snap.get(pid, {}).get("value", 0)
        yv = yday_snap[pid]["value"]
        print(f"    pid={pid:8}  today={tv:6.1f}  yesterday={yv:6.1f}")

    # Derive the scale to seed: yesterday's scale ≈ today's scale * ratio
    new_scale = (cur_scale * ratio) if cur_scale else 0.0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        if not cur_scale:
            print("NOTE: pipeline_state has no scale yet. The --commit restore will")
            print("still fix player_values and today's history rows immediately, but")
            print("the next build will re-drop unless you run --commit again after")
            print("the next daily build has run (which will populate pipeline_state).")
        else:
            print(f"Will seed pipeline_state scale: {new_scale:.6f}")
        return 0

    # --- APPLY ---
    with get_conn() as conn:
        # 1. Overwrite today's player_value_history rows with yesterday's values
        n_hist = 0
        for pid, yd in yday_snap.items():
            conn.execute(
                """
                INSERT INTO player_value_history
                    (as_of_date, player_id, value, sf_value,
                     value_8, value_12, value_14,
                     sf_value_8, sf_value_12, sf_value_14, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'model')
                ON CONFLICT (as_of_date, player_id, source) DO UPDATE SET
                    value        = excluded.value,
                    sf_value     = excluded.sf_value,
                    value_8      = excluded.value_8,
                    value_12     = excluded.value_12,
                    value_14     = excluded.value_14,
                    sf_value_8   = excluded.sf_value_8,
                    sf_value_12  = excluded.sf_value_12,
                    sf_value_14  = excluded.sf_value_14
                """,
                (today, pid,
                 yd["value"], yd["sf_value"],
                 yd["value_8"], yd["value_12"], yd["value_14"],
                 yd["sf_value_8"], yd["sf_value_12"], yd["sf_value_14"]),
            )
            n_hist += 1
        print(f"Overwrote {n_hist} player_value_history rows for {today}.")

        # 2. Update player_values table (live headline source, shared across instances)
        n_pv = 0
        for pid, yd in yday_snap.items():
            v1qb = yd["value"]
            vsf  = yd["sf_value"] or v1qb
            if v1qb <= 0:
                continue
            conn.execute(
                """
                UPDATE player_values SET
                    value_1qb            = %s,
                    value_sf             = %s,
                    value_8              = %s,
                    value_12             = %s,
                    value_14             = %s,
                    sf_value_8           = %s,
                    sf_value_12          = %s,
                    sf_value_14          = %s,
                    calibrated_value_1qb = NULL,
                    calibrated_value_sf  = NULL,
                    calibration_source   = NULL,
                    calibration_weight   = NULL
                WHERE player_id = %s
                """,
                (v1qb, vsf,
                 yd["value_8"] or v1qb, yd["value_12"] or v1qb, yd["value_14"] or v1qb,
                 yd["sf_value_8"] or vsf, yd["sf_value_12"] or vsf, yd["sf_value_14"] or vsf,
                 pid),
            )
            n_pv += 1
        print(f"Updated {n_pv} rows in player_values (live headline source).")

        # 3. Seed the persisted scale so next build holds restored level
        if new_scale:
            _write_scale(conn, new_scale)
            print(f"Seeded pipeline_state.{_SCALE_STATE_KEY} = {new_scale:.6f}")
        else:
            print("Skipping scale seed (no persisted scale to base it on).")
            print("Run again after the next daily build to lock the scale.")

    print("\nDone. The website should show yesterday's values immediately (no restart needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
