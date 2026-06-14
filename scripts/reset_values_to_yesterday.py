#!/usr/bin/env python3
"""
Restore player_values to a previous day's exact values from player_value_history.

A deploy re-anchored the normalization scale and dropped the headline value of
every player. Rather than estimate a scale factor, this copies each player's
exact values from a chosen snapshot date straight into player_values (the live,
shared-across-instances table the website reads). JSN's 880.8 simply becomes the
911.4 stored for June 13.

Usage:
    python scripts/reset_values_to_yesterday.py                       # list snapshots + dry run
    python scripts/reset_values_to_yesterday.py --date 2026-06-13     # preview the copy
    python scripts/reset_values_to_yesterday.py --date 2026-06-13 --commit
"""
import argparse
import os
import sys
from datetime import date as _date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn

_SCALE_STATE_KEY = "value_scale_1qb"

# (history column, player_values column). Only copied when the snapshot value > 0.
_COL_PAIRS = [
    ("value",       "value_1qb"),
    ("sf_value",    "value_sf"),
    ("value_8",     "value_8"),
    ("value_12",    "value_12"),
    ("value_14",    "value_14"),
    ("sf_value_8",  "sf_value_8"),
    ("sf_value_12", "sf_value_12"),
    ("sf_value_14", "sf_value_14"),
]
_HIST_COLS = [h for h, _ in _COL_PAIRS]


def _list_snapshots(conn) -> list:
    rows = conn.execute(
        """
        SELECT as_of_date, COUNT(DISTINCT player_id) AS n
        FROM player_value_history
        WHERE source = 'model'
        GROUP BY as_of_date
        ORDER BY as_of_date DESC
        LIMIT 14
        """
    ).fetchall()
    return [(r["as_of_date"], r["n"]) for r in rows]


def _snapshot(conn, d) -> dict:
    rows = conn.execute(
        f"""
        SELECT player_id, {", ".join(_HIST_COLS)}
        FROM player_value_history
        WHERE source = 'model' AND as_of_date = %s
        """,
        (d,),
    ).fetchall()
    out = {}
    for r in rows:
        pid = str(r["player_id"])
        out[pid] = {c: (float(r[c]) if r[c] is not None else 0.0) for c in _HIST_COLS}
    return out


def _live_1qb(conn) -> dict:
    """Current displayed 1QB value per player (COALESCE matches the modal)."""
    rows = conn.execute(
        """
        SELECT player_id, COALESCE(calibrated_value_1qb, value_1qb) AS v
        FROM player_values
        WHERE value_1qb IS NOT NULL
        """
    ).fetchall()
    return {str(r["player_id"]): float(r["v"]) for r in rows}


def _write_scale(conn, scale: float) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_state (
            key TEXT PRIMARY KEY,
            value DOUBLE PRECISION,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
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
    ap.add_argument("--date", help="Snapshot date to restore from (YYYY-MM-DD). Defaults to most recent.")
    ap.add_argument("--commit", action="store_true", help="Apply (default is a dry run).")
    args = ap.parse_args()

    with get_conn() as conn:
        snapshots = _list_snapshots(conn)
        live_1qb  = _live_1qb(conn)

    if not snapshots:
        print("ERROR: no snapshots in player_value_history.", file=sys.stderr)
        return 1

    print("Available snapshots (newest first):")
    for d, n in snapshots:
        print(f"  {d}  ({n} players)")
    print()

    target_date = args.date or str(snapshots[0][0])
    if target_date not in {str(d) for d, _ in snapshots}:
        print(f"ERROR: no snapshot for {target_date}. Pick a date from the list.", file=sys.stderr)
        return 1

    with get_conn() as conn:
        target_snap = _snapshot(conn, target_date)

    print(f"Restoring from:      {target_date}  ({len(target_snap)} players)")
    print(f"Live player_values:  {len(live_1qb)} players")

    # How many players will actually change, and by how much
    changed = []
    for pid, td in target_snap.items():
        tv = td["value"]
        lv = live_1qb.get(pid)
        if tv > 0 and lv is not None and abs(tv - lv) >= 0.1:
            changed.append((pid, lv, tv))
    changed.sort(key=lambda x: x[2], reverse=True)

    print(f"Players whose 1QB value will change: {len(changed)}\n")
    print("  Top changes  (live 1QB -> restored 1QB)")
    for pid, lv, tv in changed[:12]:
        print(f"    pid={pid:8}  {lv:6.1f}  ->  {tv:6.1f}   ({tv-lv:+.1f})")

    if not changed:
        print("\nNothing to change — live values already match the snapshot.")
        return 0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        print("NOTE: pipeline_state scale is not touched here; after the next daily build")
        print("      runs you may need to re-check that values hold.")
        return 0

    # --- APPLY: copy each snapshot value straight into player_values ---
    with get_conn() as conn:
        n_pv = 0
        for pid, td in target_snap.items():
            sets, params = [], []
            for hist_col, pv_col in _COL_PAIRS:
                v = td.get(hist_col, 0.0)
                if v > 0:
                    sets.append(f"{pv_col} = %s")
                    params.append(round(v, 1))
            if not sets:
                continue
            sets += [
                "calibrated_value_1qb = NULL",
                "calibrated_value_sf  = NULL",
                "calibration_source   = NULL",
                "calibration_weight   = NULL",
            ]
            params.append(pid)
            cur = conn.execute(
                f"UPDATE player_values SET {', '.join(sets)} WHERE player_id = %s",
                tuple(params),
            )
            n_pv += 1
        print(f"Updated {n_pv} rows in player_values.")

        # Overwrite today's player_value_history rows with the restored values
        # so the sparkline doesn't show the artificial drop.
        today = str(_date.today())
        n_hist = 0
        for pid, td in target_snap.items():
            if td.get("value", 0) <= 0:
                continue
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
                 td["value"], td["sf_value"],
                 td["value_8"], td["value_12"], td["value_14"],
                 td["sf_value_8"], td["sf_value_12"], td["sf_value_14"]),
            )
            n_hist += 1
        print(f"Overwrote {n_hist} player_value_history rows for {today} (fixes sparkline).")

    print("\nDone. The website should show the restored values immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
