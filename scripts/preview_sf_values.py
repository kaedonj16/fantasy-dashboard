#!/usr/bin/env python3
"""Preview the dynasty Superflex board under the "SF derived like 1QB" change.

Read-only: runs the trade value model in dry-run (no DB writes) twice — once with
the CURRENT behavior (non-QB SF = market-faithful ratio, ±40% band) and once with
the PROPOSED behavior (SF derived by the same guarded WLS solve as 1QB, tighter
band) — and prints a side-by-side board so we can SEE the values before changing
anything in production.

Run on Render (needs DB access):
    python scripts/preview_sf_values.py                # default proposed band ±25%
    python scripts/preview_sf_values.py --max-lift 1.20 --min-lift 0.80
    python scripts/preview_sf_values.py --top 40 --size 10

Nothing is written; production behavior is unchanged until we flip the defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _names() -> dict:
    """pid -> (name, position) from the players index; falls back to {} on any issue."""
    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
        out = {}
        for pid, info in idx.items():
            out[str(pid)] = (
                info.get("name") or info.get("full_name") or str(pid),
                str(info.get("pos") or info.get("position") or "").upper(),
            )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[preview] name lookup unavailable ({type(exc).__name__}); showing ids")
        return {}


def _index(result: dict) -> dict:
    """pid -> row from a dry-run result."""
    return {str(r["player_id"]): r for r in (result.get("rows") or [])}


def main():
    ap = argparse.ArgumentParser(description="Preview SF board (dry-run, no writes).")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--size", type=int, default=10, help="league size (8/10/12/14)")
    ap.add_argument("--top", type=int, default=30, help="rows to show")
    ap.add_argument("--max-lift", type=float, default=1.25,
                    help="proposed upper band as a fraction of prior (default 1.25)")
    ap.add_argument("--min-lift", type=float, default=0.75,
                    help="proposed lower band as a fraction of prior (default 0.75)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    from data_building.trade_intel.trade_value_model import run_trade_value_model

    print(f"[preview] dynasty size={args.size} — computing CURRENT board (dry-run)…")
    cur = run_trade_value_model(season=args.season, league_type=2, league_size=args.size,
                                use_market_faithful=True, dry_run=True)
    print(f"[preview] computing PROPOSED board (SF like 1QB, band "
          f"{args.min_lift:.2f}–{args.max_lift:.2f}, dry-run)…")
    prop = run_trade_value_model(season=args.season, league_type=2, league_size=args.size,
                                 use_market_faithful=False,
                                 min_lift=args.min_lift, max_lift=args.max_lift,
                                 dry_run=True)

    names = _names()
    cur_by = _index(cur)
    prop_by = _index(prop)
    priors = prop.get("priors") or cur.get("priors") or {}

    def nm(pid):
        return names.get(str(pid), (str(pid), priors.get(str(pid), {}).get("position") or ""))[0]

    def pos(pid):
        return (priors.get(str(pid), {}).get("position")
                or names.get(str(pid), ("", ""))[1] or "")

    # Rank by the PROPOSED SF value so we see the new board order.
    ranked = sorted(prop_by.values(),
                    key=lambda r: float(r.get("calibrated_value_sf") or 0), reverse=True)

    hdr = f"{'#':>3}  {'PLAYER':22} {'POS':3} {'1QB':>8} {'SF now':>8} {'SF new':>8} {'SF prior':>9} {'Δ%':>7}"
    print("\n=== Dynasty SF: current vs proposed (sorted by proposed SF) ===")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranked[: args.top], start=1):
        pid = str(r["player_id"])
        sf_new = float(r.get("calibrated_value_sf") or 0)
        v1qb = float(r.get("calibrated_value_1qb") or 0)
        sf_now = float((cur_by.get(pid) or {}).get("calibrated_value_sf") or 0)
        sf_prior = float((priors.get(pid) or {}).get("value_sf") or 0)
        d = ((sf_new - sf_now) / sf_now * 100.0) if sf_now else 0.0
        print(f"{i:>3}  {nm(pid)[:22]:22} {pos(pid):3} {v1qb:8.1f} {sf_now:8.1f} "
              f"{sf_new:8.1f} {sf_prior:9.1f} {d:+7.1f}")

    # Quick "does a QB top the SF board now?" check for both configs.
    def top_pos(by):
        rows = sorted(by.values(), key=lambda r: float(r.get("calibrated_value_sf") or 0), reverse=True)
        return [(nm(r["player_id"]), pos(r["player_id"]),
                 round(float(r.get("calibrated_value_sf") or 0), 1)) for r in rows[:5]]

    print("\nTop-5 SF — CURRENT :", top_pos(cur_by))
    print("Top-5 SF — PROPOSED:", top_pos(prop_by))
    print(f"\ntrades_used current={cur.get('trades_used')} proposed={prop.get('trades_used')} "
          f"(nothing written; dry-run)")


if __name__ == "__main__":
    main()
