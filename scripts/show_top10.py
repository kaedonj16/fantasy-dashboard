#!/usr/bin/env python3
"""
Print the model's current top 10 per position from the DB.

Usage:
    python scripts/show_top10.py
    python scripts/show_top10.py --year 2026
    python scripts/show_top10.py --pos WR RB
    python scripts/show_top10.py --n 15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn

POSITIONS  = ["WR", "RB", "TE", "QB"]
TIER_LABEL = {1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5"}


def show_top10(year: int, positions: list[str], top_n: int = 10) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    rp.position,
                    rr.position_rank,
                    rp.name,
                    rp.school,
                    rp.age,
                    rr.prospect_score,
                    rr.tier,
                    rr.production_score,
                    rr.efficiency_score,
                    rr.athleticism_score,
                    rr.age_score,
                    rr.breakout_profile_score,
                    rr.projected_draft_capital_score,
                    rr.rookie_value,
                    rmc.projected_pick_low,
                    rmc.projected_pick_high,
                    rmc.projected_round,
                    rmc.num_mocks_used
                FROM   rookie_rankings rr
                JOIN   rookie_prospects rp ON rp.player_id = rr.player_id
                LEFT   JOIN rookie_mock_draft_consensus rmc ON rmc.player_id = rr.player_id
                WHERE  rr.draft_class_year = %s
                  AND  rp.position = ANY(%s)
                ORDER  BY rp.position, rr.position_rank
                """,
                (year, positions),
            )
            rows = cur.fetchall()

    by_pos: dict[str, list] = {p: [] for p in positions}
    for row in rows:
        pos = row["position"]
        if pos in by_pos:
            by_pos[pos].append(row)

    print(f"\n{'═' * 92}")
    print(f"  {year} Rookie Draft Class — Top {top_n} per Position  (dynasty 1QB PPR)")
    print(f"{'═' * 92}")

    for pos in positions:
        players = by_pos[pos][:top_n]
        if not players:
            print(f"\n  {pos}: no data in DB for {year}")
            continue

        print(f"\n  ── {pos} {'─' * 72}")
        print(
            f"  {'#':>2}  {'Player':<24} {'School':<18} {'Age':>4}  "
            f"{'Score':>5}  {'Pick Range':<12}  {'Dyn$':>6}  "
            f"{'Prd':>3} {'Eff':>3} {'Ath':>3} {'Age':>3} {'Brk':>3}"
        )
        print(f"  {'─' * 88}")

        for p in players:
            name   = (p["name"]   or "")[:24]
            school = (p["school"] or "")[:18]
            age    = f"{p['age']:.1f}" if p.get("age") else "  —"
            score  = p["prospect_score"] or 0
            tier   = TIER_LABEL.get(p.get("tier") or 0, "")
            dval   = p.get("rookie_value") or 0

            lo    = p.get("projected_pick_low")
            hi    = p.get("projected_pick_high")
            rd    = p.get("projected_round")
            if lo and hi:
                pick_str = f"#{lo}–#{hi}"
            elif rd:
                pick_str = f"Rd {rd}"
            else:
                pick_str = "—"

            prod = p.get("production_score") or 0
            eff  = p.get("efficiency_score") or 0
            ath  = p.get("athleticism_score") or 0
            age_s = p.get("age_score") or 0
            brk  = p.get("breakout_profile_score") or 0

            print(
                f"  {p['position_rank']:>2}. {name:<24} {school:<18} {age:>4}  "
                f"{score:>5.1f}  {pick_str:<12}  {dval:>6.1f}  "
                f"{prod:>3.0f} {eff:>3.0f} {ath:>3.0f} {age_s:>3.0f} {brk:>3.0f}"
            )

    print(
        f"\n  Score = prospect_score (0–100)  |  Dyn$ = dynasty value  |  "
        f"Prd/Eff/Ath/Age/Brk = component scores\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--pos",  nargs="+", default=POSITIONS,
                        help="Positions to show (default: WR RB TE QB)")
    parser.add_argument("--n",   type=int, default=10,
                        help="Players per position (default: 10)")
    args = parser.parse_args()
    show_top10(args.year, [p.upper() for p in args.pos], args.n)
