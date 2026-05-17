"""
Preview: calibrated values before vs after removing the 999.9 cap.

Shows what the CURRENT stored values are, what they WOULD be if the cap
was removed and calibration re-ran right now (using the same blend logic),
and the delta.  Does NOT write anything to the DB.

Run: python preview_uncapped_values.py [--top N] [--pos QB|RB|WR|TE]
"""
from __future__ import annotations

import argparse
import math
import sys

sys.path.insert(0, "/home/user/fantasy-dashboard")

from dashboard_services.db import get_conn

MAX_BLEND = 0.65
MIN_TRADES = 5
ROOKIE_DIRECT_THRESHOLD = 15
TREND_BOOST_THRESHOLD = 40
TREND_BOOST_AMOUNT = 0.10
STALENESS_PENALTY_THRESHOLD = 0.15


def _blend_weight(trade_count, trade_count_14d, trend_1qb):
    base = min(MAX_BLEND, math.sqrt(trade_count / 50) * MAX_BLEND)
    if abs(trend_1qb) >= TREND_BOOST_THRESHOLD:
        base = min(MAX_BLEND, base + TREND_BOOST_AMOUNT)
    recency = trade_count_14d / trade_count if trade_count else 0
    if recency < STALENESS_PENALTY_THRESHOLD and trade_count >= 20:
        base *= 0.6
    return round(base, 3)


def simulate_uncapped(model_1qb, model_sf, market, years_exp):
    is_rookie = years_exp is None or int(years_exp) == 0
    trade_count = (market or {}).get("trade_count", 0)
    has_direct = trade_count >= MIN_TRADES
    rookie_ok = not is_rookie or trade_count >= ROOKIE_DIRECT_THRESHOLD

    if has_direct and rookie_ok:
        weight = _blend_weight(
            market["trade_count"],
            market["trade_count_14d"],
            market["trend_1qb"],
        )
        mkt_1qb = market["market_1qb"]
        mkt_sf = market["market_sf"]
        trend = market["trend_1qb"]
        if abs(trend) >= TREND_BOOST_THRESHOLD:
            mkt_1qb += trend * 0.5
            mkt_sf += market["trend_sf"] * 0.5

        cal_1qb = round(model_1qb * (1 - weight) + mkt_1qb * weight, 2)
        cal_sf  = round(model_sf  * (1 - weight) + mkt_sf  * weight, 2)
        return max(0, cal_1qb), max(0, cal_sf), weight, "direct"

    return round(model_1qb, 2), round(model_sf, 2), 0.0, "model_only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=40, help="Number of players to show (default 40)")
    parser.add_argument("--pos", default=None, help="Filter by position (QB/RB/WR/TE)")
    parser.add_argument("--capped-only", action="store_true", help="Only show players currently capped at 999.9")
    args = parser.parse_args()

    with get_conn() as conn:
        # Load players with current stored values
        rows = conn.execute("""
            SELECT
                pv.player_id,
                pv.player_name,
                pv.position,
                pv.years_exp,
                pv.value_1qb        AS model_1qb,
                pv.value_sf         AS model_sf,
                pv.calibrated_value_1qb AS cur_1qb,
                pv.calibrated_value_sf  AS cur_sf,
                pv.calibration_source
            FROM player_values pv
            WHERE pv.calibrated_value_1qb IS NOT NULL
              AND pv.value_1qb IS NOT NULL
        """).fetchall()

        # Load market data
        market_rows = conn.execute("""
            SELECT
                player_id,
                trade_count,
                trade_count_14d,
                weighted_market_value_1qb,
                weighted_market_value_sf,
                market_trend_1qb,
                market_trend_sf
            FROM trade_intel_player_stats
            WHERE season = (
                SELECT MAX(season) FROM trade_intel_player_stats
                WHERE trade_count >= %s
            )
            AND trade_count >= %s
        """, (MIN_TRADES, MIN_TRADES)).fetchall()

    market_map = {}
    for r in market_rows:
        wm = r["weighted_market_value_1qb"]
        if not wm:
            continue
        market_map[r["player_id"]] = {
            "market_1qb":      float(wm),
            "market_sf":       float(r["weighted_market_value_sf"] or wm),
            "trade_count":     int(r["trade_count"] or 0),
            "trade_count_14d": int(r["trade_count_14d"] or 0),
            "trend_1qb":       float(r["market_trend_1qb"] or 0),
            "trend_sf":        float(r["market_trend_sf"] or 0),
        }

    results = []
    for row in rows:
        pos = (row["position"] or "").upper()
        if args.pos and pos != args.pos.upper():
            continue

        m1qb = float(row["model_1qb"] or 0)
        msf  = float(row["model_sf"] or m1qb)
        cur_1qb = float(row["cur_1qb"] or 0)
        cur_sf  = float(row["cur_sf"] or 0)
        market = market_map.get(row["player_id"])

        new_1qb, new_sf, weight, source = simulate_uncapped(m1qb, msf, market, row["years_exp"])

        delta_1qb = round(new_1qb - cur_1qb, 1)
        delta_sf  = round(new_sf  - cur_sf,  1)

        if args.capped_only and cur_1qb < 999.0 and cur_sf < 999.0:
            continue

        results.append({
            "name":     row["player_name"] or row["player_id"],
            "pos":      pos,
            "cur_1qb":  cur_1qb,
            "cur_sf":   cur_sf,
            "new_1qb":  new_1qb,
            "new_sf":   new_sf,
            "delta_1qb": delta_1qb,
            "delta_sf":  delta_sf,
            "weight":   weight,
            "source":   source,
            "mkt_1qb":  market["market_1qb"] if market else None,
            "mkt_sf":   market["market_sf"]  if market else None,
        })

    # Sort by new_1qb descending (shows biggest values first)
    results.sort(key=lambda x: x["new_1qb"], reverse=True)
    top = results[:args.top]

    # Header
    W = 26
    print(f"\n{'Player':<{W}} {'Pos':<4} {'Cur 1QB':>8} {'New 1QB':>8} {'Δ 1QB':>8}  {'Cur SF':>8} {'New SF':>8} {'Δ SF':>8}  {'Mkt 1QB':>8}  {'Wt':>5}  Source")
    print("-" * 125)

    for r in top:
        delta_str_1qb = f"+{r['delta_1qb']:.0f}" if r['delta_1qb'] >= 0 else f"{r['delta_1qb']:.0f}"
        delta_str_sf  = f"+{r['delta_sf']:.0f}"  if r['delta_sf']  >= 0 else f"{r['delta_sf']:.0f}"
        mkt = f"{r['mkt_1qb']:.0f}" if r["mkt_1qb"] else "  n/a"
        # Highlight rows where current value is capped at 999.9
        flag = " *" if r["cur_1qb"] >= 999.0 or r["cur_sf"] >= 999.0 else "  "
        print(
            f"{r['name'][:W-2]+flag:<{W}} {r['pos']:<4}"
            f" {r['cur_1qb']:>8.1f} {r['new_1qb']:>8.1f} {delta_str_1qb:>8}"
            f"  {r['cur_sf']:>8.1f} {r['new_sf']:>8.1f} {delta_str_sf:>8}"
            f"  {mkt:>8}  {r['weight']:>5.3f}  {r['source']}"
        )

    # Summary counts
    capped_1qb = sum(1 for r in results if r["cur_1qb"] >= 999.0)
    capped_sf  = sum(1 for r in results if r["cur_sf"]  >= 999.0)
    would_grow = sum(1 for r in results if r["delta_1qb"] > 5)
    print(f"\nTotal players: {len(results)}")
    print(f"Currently capped at 999.9 — 1QB: {capped_1qb}  SF: {capped_sf}")
    print(f"Would gain >5 pts in 1QB after uncap: {would_grow}")
    print(f"\n* = currently at or above 999.9 (capped)\n")


if __name__ == "__main__":
    main()
