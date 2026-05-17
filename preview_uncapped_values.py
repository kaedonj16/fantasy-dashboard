"""
Preview: current calibrated values vs raw market values from real trades.

Shows what players are actually trading for in the market (weighted_market_value)
compared to what the system currently has them valued at.  No model blending.
Does NOT write anything to the DB.

Run: python preview_uncapped_values.py [--top N] [--pos QB|RB|WR|TE] [--capped-only]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/user/fantasy-dashboard")

from dashboard_services.db import get_conn

MIN_TRADES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--pos", default=None, help="Filter by position (QB/RB/WR/TE)")
    parser.add_argument("--capped-only", action="store_true", help="Only show players currently at 999.9")
    args = parser.parse_args()

    from utils.utils import load_players_index
    players_index = load_players_index() or {}
    name_map: dict[str, str] = {
        str(pid): (meta.get("name") or str(pid))
        for pid, meta in players_index.items()
    }

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                pv.player_id,
                pv.position,
                pv.calibrated_value_1qb AS cur_1qb,
                pv.calibrated_value_sf  AS cur_sf,
                ti.weighted_market_value_1qb AS mkt_1qb,
                ti.weighted_market_value_sf  AS mkt_sf,
                ti.trade_count,
                ti.market_trend_1qb          AS trend_1qb,
                ti.market_trend_sf           AS trend_sf
            FROM player_values pv
            JOIN trade_intel_player_stats ti
              ON ti.player_id = pv.player_id
             AND ti.season = (
                 SELECT MAX(season) FROM trade_intel_player_stats
                 WHERE trade_count >= %s
             )
            WHERE pv.calibrated_value_1qb IS NOT NULL
              AND ti.trade_count >= %s
              AND ti.weighted_market_value_1qb IS NOT NULL
        """, (MIN_TRADES, MIN_TRADES)).fetchall()

    # Normalize market values to 0-999.9 — same logic as market_calibration.py
    raw_max_1qb = max((float(r["mkt_1qb"]) for r in rows if r["mkt_1qb"]), default=999.9)
    raw_max_sf  = max((float(r["mkt_sf"])  for r in rows if r["mkt_sf"]),  default=999.9)
    scale_1qb = 999.9 / raw_max_1qb
    scale_sf  = 999.9 / raw_max_sf

    results = []
    for row in rows:
        pos = (row["position"] or "").upper()
        if args.pos and pos != args.pos.upper():
            continue

        cur_1qb = float(row["cur_1qb"] or 0)
        cur_sf  = float(row["cur_sf"]  or 0)
        mkt_1qb = round(float(row["mkt_1qb"] or 0) * scale_1qb, 1)
        mkt_sf  = round(float(row["mkt_sf"]  or row["mkt_1qb"] or 0) * scale_sf, 1)

        if args.capped_only and cur_1qb < 999.0 and cur_sf < 999.0:
            continue

        delta_1qb = round(mkt_1qb - cur_1qb, 1)
        delta_sf  = round(mkt_sf  - cur_sf,  1)

        results.append({
            "pid":       row["player_id"],
            "name":      name_map.get(str(row["player_id"]), str(row["player_id"])),
            "pos":       pos,
            "cur_1qb":   cur_1qb,
            "cur_sf":    cur_sf,
            "mkt_1qb":   mkt_1qb,
            "mkt_sf":    mkt_sf,
            "delta_1qb": delta_1qb,
            "delta_sf":  delta_sf,
            "trades":    int(row["trade_count"] or 0),
            "trend_1qb": float(row["trend_1qb"] or 0),
            "trend_sf":  float(row["trend_sf"]  or 0),
        })

    results.sort(key=lambda x: x["mkt_1qb"], reverse=True)
    top = results[:args.top]

    W = 26
    print(
        f"\n{'Player':<{W}} {'Pos':<4}"
        f" {'Cur 1QB':>8} {'Mkt 1QB':>8} {'Δ 1QB':>7}"
        f"  {'Cur SF':>8} {'Mkt SF':>8} {'Δ SF':>7}"
        f"  {'Trades':>6}  {'Trend':>6}"
    )
    print("-" * 110)

    for r in top:
        d1 = f"+{r['delta_1qb']:.0f}" if r['delta_1qb'] >= 0 else f"{r['delta_1qb']:.0f}"
        ds = f"+{r['delta_sf']:.0f}"  if r['delta_sf']  >= 0 else f"{r['delta_sf']:.0f}"
        flag = " *" if r["cur_1qb"] >= 999.0 or r["cur_sf"] >= 999.0 else "  "
        trend = f"{r['trend_1qb']:+.0f}" if r["trend_1qb"] else "    —"
        print(
            f"{r['name'][:W-2]+flag:<{W}} {r['pos']:<4}"
            f" {r['cur_1qb']:>8.1f} {r['mkt_1qb']:>8.1f} {d1:>7}"
            f"  {r['cur_sf']:>8.1f} {r['mkt_sf']:>8.1f} {ds:>7}"
            f"  {r['trades']:>6}  {trend:>6}"
        )

    capped_1qb = sum(1 for r in results if r["cur_1qb"] >= 999.0)
    capped_sf  = sum(1 for r in results if r["cur_sf"]  >= 999.0)
    above_cur  = sum(1 for r in results if r["delta_1qb"] > 0)
    below_cur  = sum(1 for r in results if r["delta_1qb"] < 0)

    print(f"\nTotal players with trade data: {len(results)}")
    print(f"Currently capped at 999.9 — 1QB: {capped_1qb}  SF: {capped_sf}")
    print(f"Market > current value: {above_cur}   Market < current value: {below_cur}")
    print(f"\n* = currently at or above 999.9 (capped)\n")


if __name__ == "__main__":
    main()
