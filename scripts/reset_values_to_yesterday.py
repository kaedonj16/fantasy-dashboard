#!/usr/bin/env python3
"""
Reset player_values table back to yesterday's level using the DB snapshots.

The 1QB normalization scale reset during today's deploys, compressing every
non-anchor player's value by ~4%.  Yesterday's values are stored in
player_value_history; this script copies them into player_values (the live,
shared-across-all-instances table) so the website immediately shows
yesterday's numbers on every instance.

It seeds pipeline_state so the next daily build holds the restored level.

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
_RATIO_BAND      = (0.80, 1.25)  # filter genuine per-player outliers


def _latest_snapshot(conn) -> tuple:
    row = conn.execute(
        """
        SELECT DISTINCT as_of_date
        FROM player_value_history
        WHERE source = 'model'
        ORDER BY as_of_date DESC
        LIMIT 1
        """
    ).fetchone()
    return row["as_of_date"] if row else None


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


def _current_player_values(conn) -> dict:
    """Return {player_id: {v1qb, vsf}} using the same COALESCEs the modal uses."""
    rows = conn.execute(
        """
        SELECT player_id,
               COALESCE(calibrated_value_1qb, value_1qb)           AS v1qb,
               COALESCE(calibrated_value_sf,  value_sf, value_1qb) AS vsf
        FROM player_values
        WHERE value_1qb IS NOT NULL AND value_1qb > 0
        """
    ).fetchall()
    return {str(r["player_id"]): {"v1qb": float(r["v1qb"]), "vsf": float(r["vsf"])} for r in rows}


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
        yday = _latest_snapshot(conn)
        if yday is None:
            print("ERROR: no snapshots found in player_value_history.", file=sys.stderr)
            return 1

        yday_snap = _snapshot(conn, yday)
        cur_pv    = _current_player_values(conn)
        cur_scale = _persisted_scale(conn)

    print(f"Restoring to:        {yday}  ({len(yday_snap)} players in history)")
    print(f"Current player_values: {len(cur_pv)} players with value_1qb > 0")
    print(f"Persisted scale now: {cur_scale or 'MISSING'}")

    # Measure the drop: compare live player_values against yesterday's history for
    # both 1QB and SF values. Detects the real gap shown on the website.
    ratios_1qb, ratios_sf = [], []
    for pid, yd in yday_snap.items():
        cur = cur_pv.get(pid)
        if not cur:
            continue
        if cur["v1qb"] > 0 and yd["value"] > 0:
            r = yd["value"] / cur["v1qb"]
            if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
                ratios_1qb.append(r)
        if cur["vsf"] > 0 and yd["sf_value"] > 0:
            r = yd["sf_value"] / cur["vsf"]
            if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
                ratios_sf.append(r)

    if len(ratios_1qb) < 25 and len(ratios_sf) < 25:
        print(f"ERROR: only {len(ratios_1qb)} 1QB / {len(ratios_sf)} SF comparable players — can't infer scale factor.", file=sys.stderr)
        return 1

    ratio_1qb = statistics.median(ratios_1qb) if ratios_1qb else 1.0
    ratio_sf  = statistics.median(ratios_sf)  if ratios_sf  else 1.0
    print(f"Comparable players:  {len(ratios_1qb)} (1QB)  {len(ratios_sf)} (SF)")
    print(f"Median ratio 1QB (history/live): {ratio_1qb:.4f}  ({(ratio_1qb-1)*100:+.2f}%)")
    print(f"Median ratio SF  (history/live): {ratio_sf:.4f}  ({(ratio_sf-1)*100:+.2f}%)")

    if abs(ratio_1qb - 1.0) < 0.005 and abs(ratio_sf - 1.0) < 0.005:
        print("Live player_values are within 0.5% of yesterday's history — nothing to restore.")
        return 0

    # Use 1QB ratio as the primary scale signal (SF scale is independent)
    ratio = ratio_1qb if abs(ratio_1qb - 1.0) >= 0.005 else ratio_sf

    # Preview top players: live value vs what we'll restore
    sorted_players = sorted(
        [(pid, yday_snap[pid]["value"]) for pid in yday_snap if yday_snap[pid]["value"] > 0],
        key=lambda x: x[1], reverse=True
    )[:8]
    print(f"\n  Top players  (live 1QB / SF -> restoring to yesterday history)")
    for pid, _ in sorted_players:
        cur = cur_pv.get(pid, {})
        yv   = yday_snap[pid]["value"]
        yvsf = yday_snap[pid]["sf_value"]
        print(f"    pid={pid:8}  live={cur.get('v1qb',0):6.1f}/{cur.get('vsf',0):6.1f}  yesterday={yv:6.1f}/{yvsf:6.1f}")

    # Derive the scale to seed: yesterday's scale ≈ today's scale * ratio
    new_scale = (cur_scale * ratio) if cur_scale else 0.0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        if not cur_scale:
            print("NOTE: pipeline_state has no scale yet. Values will still be restored")
            print("immediately, but the next build may re-drop. Run again after the next")
            print("daily build to lock in the corrected scale.")
        else:
            print(f"Will seed pipeline_state scale: {new_scale:.6f}")
        return 0

    # --- APPLY ---
    with get_conn() as conn:
        # 1. Update player_values table (live headline source, shared across instances)
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

        # 2. Seed the persisted scale so next build holds restored level
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
