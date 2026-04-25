#!/usr/bin/env python3
"""
Write actual NFL draft pick results into rookie_prospects.

Only updates actual_pick, actual_round, actual_nfl_team, draft_confirmed.
Does NOT re-score or touch rankings.

Usage:
    python scripts/seed_actual_picks.py --from-json picks.json
    python scripts/seed_actual_picks.py --from-json picks.json --dry-run
    python scripts/seed_actual_picks.py --from-json picks.json --year 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", metavar="FILE", required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.from_json) as f:
        picks = json.load(f)

    skill_pos = {"QB", "RB", "WR", "TE"}
    picks = [p for p in picks if (p.get("position") or "").upper() in skill_pos]
    print(f"Loaded {len(picks)} skill-position picks")

    if args.dry_run:
        for p in sorted(picks, key=lambda x: x["pick"]):
            print(f"  Pick {p['pick']:3d} Rd{p['round']} {p['position']:2s}  {p['player_name']}  → {p['nfl_team']}")
        return

    from dashboard_services.db import get_conn
    from data_building.rookie_pipeline.pipeline import upsert_actual_draft_picks

    with get_conn() as conn:
        n = upsert_actual_draft_picks(picks, args.year, conn)
        conn.commit()

    print(f"Updated {n} prospects with actual pick data.")


if __name__ == "__main__":
    main()
