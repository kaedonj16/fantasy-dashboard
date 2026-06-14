#!/usr/bin/env python3
"""
Restore player_values after a uniform normalization-scale drop.

Today's deploy(s) re-anchored the 1QB/SF normalization scale, compressing the
headline value of every player by a single uniform factor (~3-4%).  The drop is
multiplicative and column-specific (it hit `value`/`value_12` but, for some
players, not `value_8`/`value_14`), and it affects EVERY live player — including
ones that aren't in a given day's player_value_history snapshot.

So instead of copying a snapshot (which would only touch the players present
that day), this script:

  1. Picks a healthy pre-drop snapshot date from player_value_history.
  2. For each value column, measures the median ratio (snapshot / live) across
     the players present in BOTH — this is the scale factor for that column.
     Columns that didn't drop come out at ~1.0 (a no-op).
  3. Multiplies EVERY live player_values row by its column's ratio (capped at
     999.9), nulling stale calibration. This restores the full pool, not just
     the snapshot subset, and leaves untouched columns untouched.

Usage:
    python scripts/reset_values_to_yesterday.py                       # list snapshots + dry run
    python scripts/reset_values_to_yesterday.py --date 2026-06-13     # pick the pre-drop date
    python scripts/reset_values_to_yesterday.py --date 2026-06-13 --commit
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn

_SCALE_STATE_KEY = "value_scale_1qb"
_RATIO_BAND      = (0.80, 1.25)  # ignore genuine per-player moves when measuring scale

# (history column, player_values base column).  Order matters for display.
_COL_PAIRS = [
    ("value",       "value_1qb"),
    ("value_8",     "value_8"),
    ("value_12",    "value_12"),
    ("value_14",    "value_14"),
    ("sf_value",    "value_sf"),
    ("sf_value_8",  "sf_value_8"),
    ("sf_value_12", "sf_value_12"),
    ("sf_value_14", "sf_value_14"),
]
_HIST_COLS = [h for h, _ in _COL_PAIRS]
_PV_COLS   = [p for _, p in _COL_PAIRS]


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


def _live_values(conn) -> dict:
    """Current displayed values per player (COALESCE matches the modal)."""
    rows = conn.execute(
        """
        SELECT player_id,
               COALESCE(calibrated_value_1qb, value_1qb)           AS value_1qb,
               COALESCE(calibrated_value_sf,  value_sf, value_1qb) AS value_sf,
               value_8, value_12, value_14,
               sf_value_8, sf_value_12, sf_value_14
        FROM player_values
        WHERE value_1qb IS NOT NULL AND value_1qb > 0
        """
    ).fetchall()
    out = {}
    for r in rows:
        pid = str(r["player_id"])
        out[pid] = {c: (float(r[c]) if r[c] is not None else 0.0) for c in _PV_COLS}
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


def _column_ratios(target_snap: dict, live: dict) -> dict:
    """Median (snapshot / live) per column, over players present in both."""
    ratios = {}
    for hist_col, pv_col in _COL_PAIRS:
        samples = []
        for pid, td in target_snap.items():
            lv = live.get(pid)
            if not lv:
                continue
            t, l = td.get(hist_col, 0.0), lv.get(pv_col, 0.0)
            if t > 0 and l > 0:
                r = t / l
                if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
                    samples.append(r)
        ratios[pv_col] = (statistics.median(samples), len(samples)) if samples else (1.0, 0)
    return ratios


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Pre-drop snapshot date (YYYY-MM-DD). Defaults to most recent.")
    ap.add_argument("--commit", action="store_true", help="Apply (default is a dry run).")
    args = ap.parse_args()

    with get_conn() as conn:
        snapshots = _list_snapshots(conn)
        live      = _live_values(conn)
        cur_scale = _persisted_scale(conn)

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

    print(f"Target snapshot:     {target_date}  ({len(target_snap)} players)")
    print(f"Live player_values:  {len(live)} players")
    print(f"Persisted scale now: {cur_scale or 'MISSING'}\n")

    ratios = _column_ratios(target_snap, live)
    print("Per-column scale factor (snapshot / live):")
    any_drop = False
    for _, pv_col in _COL_PAIRS:
        r, n = ratios[pv_col]
        flag = "  <- restore" if abs(r - 1.0) >= 0.005 else ""
        if abs(r - 1.0) >= 0.005:
            any_drop = True
        print(f"  {pv_col:14} x{r:.4f}  ({(r-1)*100:+.2f}%, n={n}){flag}")

    if not any_drop:
        print("\nAll columns within 0.5% — nothing to restore.")
        return 0

    # Preview a few well-known players
    sample = sorted(
        [(pid, live[pid]["value_1qb"]) for pid in live],
        key=lambda x: x[1], reverse=True
    )[:8]
    print(f"\n  Top players  (live 1QB -> restored 1QB)")
    for pid, lv1qb in sample:
        new1qb = min(lv1qb * ratios["value_1qb"][0], 999.9)
        print(f"    pid={pid:8}  live={lv1qb:6.1f}  ->  {new1qb:6.1f}")

    new_scale = (cur_scale * ratios["value_1qb"][0]) if cur_scale else 0.0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        if not args.date:
            print(f"TIP: {target_date} is the latest snapshot; if its ratios are ~0% it is")
            print("     already post-drop. Pass --date with an earlier pre-drop date above.")
        if not cur_scale:
            print("NOTE: pipeline_state has no scale yet — values restore now but the next")
            print("      build may re-drop. Re-run after the next daily build to lock it in.")
        else:
            print(f"Will seed pipeline_state scale: {new_scale:.6f}")
        return 0

    # --- APPLY: multiply every live row by its column ratio ---
    with get_conn() as conn:
        n_pv = 0
        for pid, lv in live.items():
            sets, params = [], []
            for _, pv_col in _COL_PAIRS:
                r, _n = ratios[pv_col]
                if abs(r - 1.0) < 0.005:
                    continue  # column didn't drop; leave it alone
                cur_v = lv.get(pv_col, 0.0)
                if cur_v <= 0:
                    continue
                sets.append(f"{pv_col} = %s")
                params.append(round(min(cur_v * r, 999.9), 1))
            if not sets:
                continue
            sets += [
                "calibrated_value_1qb = NULL",
                "calibrated_value_sf  = NULL",
                "calibration_source   = NULL",
                "calibration_weight   = NULL",
            ]
            params.append(pid)
            conn.execute(
                f"UPDATE player_values SET {', '.join(sets)} WHERE player_id = %s",
                tuple(params),
            )
            n_pv += 1
        print(f"Updated {n_pv} rows in player_values.")

        if new_scale:
            _write_scale(conn, new_scale)
            print(f"Seeded pipeline_state.{_SCALE_STATE_KEY} = {new_scale:.6f}")
        else:
            print("Skipping scale seed (no persisted scale to base it on).")
            print("Re-run after the next daily build to lock the scale.")

    print("\nDone. The website should show restored values immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
