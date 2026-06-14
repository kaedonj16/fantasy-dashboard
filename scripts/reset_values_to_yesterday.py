#!/usr/bin/env python3
"""
Reset displayed player values back to yesterday's level.

Today's daily build re-anchored the 1QB normalization scale (top non-QB ->
999.9) without the smoothed scale, which compressed every non-anchor player's
1QB value down by a single uniform factor. Yesterday's values are still stored
in the player_value_history table, so this script:

  1. Reads yesterday's snapshot and today's currently-served values
     (data/model_values.json) and measures the uniform 1QB drop as a median
     ratio (robust to genuine per-player moves).
  2. Multiplies the served 1QB values back up by that ratio (capped at 999.9),
     restoring the headline numbers to yesterday's level immediately.
  3. Seeds pipeline_state.value_scale_1qb to yesterday's scale so the NEXT
     daily build keeps them there instead of re-dropping. (Requires a build to
     have run since the scale-persistence fix deployed, so pipeline_state holds
     today's scale; if it doesn't, the JSON restore still happens and a warning
     is printed.)

Only the 1QB value columns (value, value_8/12/14) are touched — the SF scale is
independent and was not affected by the bug.

IMPORTANT: run this on the WEB service shell, where the served
data/model_values.json lives (the cron service has its own copy).

Usage:
    python scripts/reset_values_to_yesterday.py            # dry run, prints plan
    python scripts/reset_values_to_yesterday.py --commit   # apply the changes
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from utils.paths import DATA_DIR
from dashboard_services.db import get_conn

_SCALE_STATE_KEY = "value_scale_1qb"
_ONE_QB_KEYS = ["value", "value_8", "value_12", "value_14"]
# Only ratios within this band count toward the uniform-drop median, so a player
# who genuinely cratered/spiked doesn't skew the measured scale factor.
_RATIO_BAND = (0.90, 1.25)


def _model_values_path() -> Path:
    return DATA_DIR / "model_values.json"


def _latest_two_dates(conn) -> tuple:
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
    return (dates[0], dates[1])


def _yesterday_values(conn, yday) -> dict:
    rows = conn.execute(
        """
        SELECT player_id, value
        FROM player_value_history
        WHERE source = 'model' AND as_of_date = %s
        """,
        (yday,),
    ).fetchall()
    return {str(r["player_id"]): float(r["value"]) for r in rows if r["value"] is not None}


def _read_persisted_scale(conn) -> float:
    try:
        row = conn.execute(
            "SELECT value FROM pipeline_state WHERE key = %s",
            (_SCALE_STATE_KEY,),
        ).fetchone()
        if row and row.get("value"):
            return float(row["value"])
    except Exception:
        pass
    return 0.0


def _write_persisted_scale(conn, scale: float) -> None:
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
    ap.add_argument("--commit", action="store_true",
                    help="Apply changes (default is a dry run)")
    args = ap.parse_args()

    json_path = _model_values_path()
    if not json_path.exists():
        print(f"ERROR: {json_path} not found — run on the web service shell.", file=sys.stderr)
        return 1

    assets = json.loads(json_path.read_text(encoding="utf-8"))
    json_vals = {
        str(a.get("id")): float(a.get("value") or 0)
        for a in assets
        if a.get("id") and (a.get("value") or 0) > 0
    }

    with get_conn() as conn:
        today, yday = _latest_two_dates(conn)
        if yday is None:
            print("ERROR: need at least two daily snapshots in player_value_history.", file=sys.stderr)
            return 1
        yvals = _yesterday_values(conn, yday)
        today_scale = _read_persisted_scale(conn)

    print(f"Today's snapshot date:     {today}")
    print(f"Restoring to (yesterday):  {yday}")
    print(f"Players in yesterday snap: {len(yvals)}")
    print(f"Players in served JSON:    {len(json_vals)}")
    print(f"Persisted scale (today):   {today_scale or 'MISSING'}")

    ratios = []
    for pid, yv in yvals.items():
        jv = json_vals.get(pid)
        if not jv or jv <= 0:
            continue
        r = yv / jv
        if _RATIO_BAND[0] <= r <= _RATIO_BAND[1]:
            ratios.append(r)

    if len(ratios) < 25:
        print(f"ERROR: only {len(ratios)} comparable players in band — refusing "
              f"to infer a scale factor from too little data.", file=sys.stderr)
        return 1

    ratio = statistics.median(ratios)
    pct = (ratio - 1.0) * 100.0
    print(f"Comparable players in band: {len(ratios)}")
    print(f"Median restore ratio:       {ratio:.4f}  ({pct:+.2f}% to 1QB values)")

    if abs(pct) < 0.25:
        print("Drop is under 0.25% — nothing meaningful to restore. Exiting.")
        return 0

    # Preview a few well-known anchors
    preview = sorted(assets, key=lambda a: float(a.get("value") or 0), reverse=True)[:8]
    print("\n  Sample (1QB value):  now -> restored")
    for a in preview:
        old = float(a.get("value") or 0)
        new = round(min(old * ratio, 999.9), 1)
        print(f"    {str(a.get('name'))[:22]:22}  {old:6.1f} -> {new:6.1f}")

    new_scale = (today_scale * ratio) if today_scale else 0.0

    if not args.commit:
        print("\nDRY RUN — re-run with --commit to apply.")
        if not today_scale:
            print("WARNING: pipeline_state has no scale yet (the persistence fix "
                  "build hasn't run). The JSON restore will apply, but the scale "
                  "won't be re-seeded, so the next build may re-drop values. "
                  "Deploy the fix, let one build run, then re-run this.")
        return 0

    # Apply: scale up the 1QB value columns in the served JSON.
    for a in assets:
        for k in _ONE_QB_KEYS:
            v = a.get(k)
            if v is not None and float(v) > 0:
                a[k] = round(min(float(v) * ratio, 999.9), 1)
    json_path.write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRewrote {json_path} ({len(assets)} assets) at the restored scale.")

    if new_scale:
        with get_conn() as conn:
            _write_persisted_scale(conn, new_scale)
        print(f"Seeded pipeline_state.{_SCALE_STATE_KEY} = {new_scale:.6f} "
              f"(yesterday's scale) so future builds hold this level.")
    else:
        print("WARNING: no persisted scale to base the re-seed on — JSON restored, "
              "but run again after the next build to lock the scale.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
