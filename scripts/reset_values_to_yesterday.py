#!/usr/bin/env python3
"""
Reset player_values table back to a historical snapshot from player_value_history.

Compares live player_values (what the website shows) against a chosen snapshot
date and restores values for players whose live value is lower than historical.

Usage:
    python scripts/reset_values_to_yesterday.py            # list snapshots + dry run vs most recent
    python scripts/reset_values_to_yesterday.py --date 2026-06-12   # target a specific date
    python scripts/reset_values_to_yesterday.py --date 2026-06-12 --commit   # apply
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


def _list_snapshots(conn) -> list:
    """Return [(as_of_date, player_count)] ordered newest first."""
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
    ap.add_argument("--date", help="Snapshot date to restore to (YYYY-MM-DD). "
                                   "Defaults to the most recent snapshot.")
    ap.add_argument("--commit", action="store_true",
                    help="Apply changes (default is a dry run)")
    args = ap.parse_args()

    with get_conn() as conn:
        snapshots  = _list_snapshots(conn)
        cur_pv     = _current_player_values(conn)
        cur_scale  = _persisted_scale(conn)

    if not snapshots:
        print("ERROR: no snapshots found in player_value_history.", file=sys.stderr)
        return 1

    print("Available snapshots (newest first):")
    for d, n in snapshots:
        print(f"  {d}  ({n} players)")
    print()

    target_date = args.date or str(snapshots[0][0])

    # Validate date
    valid_dates = {str(d) for d, _ in snapshots}
    if target_date not in valid_dates:
        print(f"ERROR: no snapshot for {target_date}. Choose a date from the list above.", file=sys.stderr)
        return 1

    with get_conn() as conn:
        target_snap = _snapshot(conn, target_date)

    print(f"Target snapshot:     {target_date}  ({len(target_snap)} players)")
    print(f"Current player_values: {len(cur_pv)} players")
    print(f"Persisted scale now: {cur_scale or 'MISSING'}")

    # Compare live player_values against the target snapshot for 1QB and SF.
    ratios_1qb, ratios_sf = [], []
    for pid, td in target_snap.items():
        cur = cur_pv.get(pid)
        if not cur:
            continue
        if cur["v1qb"] > 0 and td["value"] > 0:
            r = td["value"] / cur["v1qb"]
            if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
                ratios_1qb.append(r)
        if cur["vsf"] > 0 and td["sf_value"] > 0:
            r = td["sf_value"] / cur["vsf"]
            if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
                ratios_sf.append(r)

    if len(ratios_1qb) < 10 and len(ratios_sf) < 10:
        print(f"ERROR: only {len(ratios_1qb)} 1QB / {len(ratios_sf)} SF comparable players — "
              "try a different --date.", file=sys.stderr)
        return 1

    ratio_1qb = statistics.median(ratios_1qb) if ratios_1qb else 1.0
    ratio_sf  = statistics.median(ratios_sf)  if ratios_sf  else 1.0
    print(f"Comparable players:  {len(ratios_1qb)} (1QB)  {len(ratios_sf)} (SF)")
    print(f"Median ratio 1QB (target/live): {ratio_1qb:.4f}  ({(ratio_1qb-1)*100:+.2f}%)")
    print(f"Median ratio SF  (target/live): {ratio_sf:.4f}  ({(ratio_sf-1)*100:+.2f}%)")

    if abs(ratio_1qb - 1.0) < 0.005 and abs(ratio_sf - 1.0) < 0.005:
        print("Live player_values are within 0.5% of the target snapshot — nothing to restore.")
        return 0

    # Preview top players
    sorted_players = sorted(
        [(pid, target_snap[pid]["value"]) for pid in target_snap if target_snap[pid]["value"] > 0],
        key=lambda x: x[1], reverse=True
    )[:8]
    print(f"\n  Top players  (live 1QB/SF  ->  target 1QB/SF)")
    for pid, _ in sorted_players:
        cur  = cur_pv.get(pid, {})
        tv   = target_snap[pid]["value"]
        tvsf = target_snap[pid]["sf_value"]
        print(f"    pid={pid:8}  live={cur.get('v1qb',0):6.1f}/{cur.get('vsf',0):6.1f}"
              f"  target={tv:6.1f}/{tvsf:6.1f}")

    # Use 1QB ratio as primary scale signal; fall back to SF if 1QB didn't change
    ratio = ratio_1qb if abs(ratio_1qb - 1.0) >= 0.005 else ratio_sf
    new_scale = (cur_scale * ratio) if cur_scale else 0.0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        if not args.date:
            print(f"TIP: if {target_date} is already a post-drop snapshot, pass --date with an")
            print("     earlier date from the list above (e.g. the one with the most players).")
        if not cur_scale:
            print("NOTE: pipeline_state has no scale yet — values will still be restored but")
            print("      the next build may re-drop. Run again after the next daily build.")
        else:
            print(f"Will seed pipeline_state scale: {new_scale:.6f}")
        return 0

    # --- APPLY ---
    with get_conn() as conn:
        n_pv = 0
        for pid, td in target_snap.items():
            v1qb = td["value"]
            vsf  = td["sf_value"] or v1qb
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
                 td["value_8"] or v1qb, td["value_12"] or v1qb, td["value_14"] or v1qb,
                 td["sf_value_8"] or vsf, td["sf_value_12"] or vsf, td["sf_value_14"] or vsf,
                 pid),
            )
            n_pv += 1
        print(f"Updated {n_pv} rows in player_values.")

        if new_scale:
            _write_scale(conn, new_scale)
            print(f"Seeded pipeline_state.{_SCALE_STATE_KEY} = {new_scale:.6f}")
        else:
            print("Skipping scale seed (no persisted scale to base it on).")

    print("\nDone. The website should show restored values immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
