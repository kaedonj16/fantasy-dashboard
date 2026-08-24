#!/usr/bin/env python3
"""Preview the dynasty Superflex board under the "SF derived like 1QB" change.

Read-only. The LIVE baseline is read straight from player_values (the exact
COALESCE the rankings page uses), so the "SF live" column matches what you see on
the site today. The PROPOSED column is a fresh model solve with the ratio override
OFF (SF derived by the same guarded WLS solve as 1QB) and a tighter band.

Important: the live calibrated value is clamped to ±2%/day of its previous value
(`_clamp_to_prev`), so it's a slow accumulation, NOT a fresh solve — that's why a
from-scratch recompute never matches the live board. The proposed column is the
*steady-state target*: the live board would drift toward it at ≤2%/day, not jump
there overnight.

Run on Render (needs DB access):
    python scripts/preview_sf_values.py                       # proposed band ±25%
    python scripts/preview_sf_values.py --max-lift 1.20 --min-lift 0.80
    python scripts/preview_sf_values.py --top 40 --also-fresh-current

Nothing is written; production is unchanged until we flip the defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _live_values():
    """id -> {name, position, v1qb_live, sf_live} from player_values (the exact
    COALESCE the rankings page reads via load_current_values_from_db)."""
    from dashboard_services.player_value_history import load_current_values_from_db
    out = {}
    for r in load_current_values_from_db() or []:
        pid = str(r.get("id"))
        out[pid] = {
            "name": r.get("name") or pid,
            "position": str(r.get("position") or "").upper(),
            "v1qb_live": float(r.get("value") or 0),      # COALESCE(calibrated_value_1qb, value_1qb)
            "sf_live":   float(r.get("sf_value") or 0),   # COALESCE(calibrated_value_sf, value_sf, value_1qb)
        }
    return out


def _rows(result):
    return {str(r["player_id"]): r for r in (result.get("rows") or [])}


def main():
    ap = argparse.ArgumentParser(description="Preview SF board (dry-run, no writes).")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--size", type=int, default=10, help="league size (8/10/12/14)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-lift", type=float, default=1.25,
                    help="proposed upper band as a fraction of prior (default 1.25)")
    ap.add_argument("--min-lift", type=float, default=0.75,
                    help="proposed lower band as a fraction of prior (default 0.75)")
    ap.add_argument("--also-fresh-current", action="store_true",
                    help="also show a fresh recompute of the CURRENT method (unclamped) "
                         "so you can see the clamp-trajectory gap vs live")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    from data_building.trade_intel.trade_value_model import run_trade_value_model

    print("[preview] reading LIVE values from player_values (matches the rankings page)…")
    live = _live_values()

    print(f"[preview] computing PROPOSED board (SF like 1QB, band "
          f"{args.min_lift:.2f}–{args.max_lift:.2f}, fresh solve, dry-run)…")
    prop = _rows(run_trade_value_model(
        season=args.season, league_type=2, league_size=args.size,
        use_market_faithful=False, min_lift=args.min_lift, max_lift=args.max_lift,
        dry_run=True))

    fresh_cur = {}
    if args.also_fresh_current:
        print("[preview] computing CURRENT method fresh (unclamped, dry-run)…")
        fresh_cur = _rows(run_trade_value_model(
            season=args.season, league_type=2, league_size=args.size,
            use_market_faithful=True, dry_run=True))

    def nm(pid):
        return (live.get(pid, {}).get("name") or pid)[:22]

    def pos(pid):
        return live.get(pid, {}).get("position") or ""

    # Rank by proposed SF so we see the new board order. Drop pick buckets
    # (keys like "2026_1_01" / "pick_…") so the table is players only.
    import re as _re
    def _is_pick(pid: str) -> bool:
        return bool(_re.match(r"^(pick_|\d{4}_)", str(pid)))
    ranked = sorted((r for r in prop.values() if not _is_pick(r.get("player_id"))),
                    key=lambda r: float(r.get("calibrated_value_sf") or 0), reverse=True)

    cols = f"{'#':>3}  {'PLAYER':22} {'POS':3} {'1QB live':>9} {'SF live':>8} {'SF prop':>8}"
    if args.also_fresh_current:
        cols += f" {'SF cur*':>8}"
    cols += f" {'Δ live%':>8}"
    print("\n=== Dynasty SF: live vs proposed steady-state (sorted by proposed SF) ===")
    print(cols)
    print("-" * len(cols))
    for i, r in enumerate(ranked[: args.top], start=1):
        pid = str(r["player_id"])
        sf_prop = float(r.get("calibrated_value_sf") or 0)
        v1qb_live = live.get(pid, {}).get("v1qb_live", 0.0)
        sf_live = live.get(pid, {}).get("sf_live", 0.0)
        d = ((sf_prop - sf_live) / sf_live * 100.0) if sf_live else 0.0
        line = (f"{i:>3}  {nm(pid):22} {pos(pid):3} {v1qb_live:9.1f} {sf_live:8.1f} "
                f"{sf_prop:8.1f}")
        if args.also_fresh_current:
            line += f" {float((fresh_cur.get(pid) or {}).get('calibrated_value_sf') or 0):8.1f}"
        line += f" {d:+8.1f}"
        print(line)

    def top5_live():
        rows = sorted(live.items(), key=lambda kv: kv[1]["sf_live"], reverse=True)[:5]
        return [(v["name"], v["position"], round(v["sf_live"], 1)) for _pid, v in rows]

    def top5_prop():
        return [(nm(r["player_id"]), pos(r["player_id"]),
                 round(float(r.get("calibrated_value_sf") or 0), 1)) for r in ranked[:5]]

    print("\nTop-5 SF — LIVE     :", top5_live())
    print("Top-5 SF — PROPOSED :", top5_prop())
    if args.also_fresh_current:
        print("* SF cur = current method recomputed fresh (unclamped); differs from live by the "
              "±2%/day clamp trajectory.")
    print("\nProposed is the steady-state target; the live board drifts toward it at ≤2%/day. "
          "Nothing was written (dry-run).")


if __name__ == "__main__":
    main()
