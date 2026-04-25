#!/usr/bin/env python3
"""
Fetch actual NFL draft pick results and re-score all prospects.

Pulls real pick data from nflverse (or a local JSON file), stores them in
rookie_prospects, rebuilds the consensus map, and re-scores the class so
draft_capital_score reflects the actual pick instead of mock projections.

Usage:
    python scripts/seed_actual_picks.py                    # auto-fetch nflverse
    python scripts/seed_actual_picks.py --year 2026
    python scripts/seed_actual_picks.py --from-json picks.json
    python scripts/seed_actual_picks.py --dry-run

picks.json format (list of objects):
    [
      {"player_name": "Cam Ward",       "position": "QB", "pick": 1,  "round": 1, "nfl_team": "TEN"},
      {"player_name": "Travis Hunter",  "position": "WR", "pick": 2,  "round": 1, "nfl_team": "JAX"},
      ...
    ]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn
from data_building.rookie_pipeline.pipeline import (
    fetch_nflverse_draft_picks,
    upsert_actual_draft_picks,
    build_consensus_from_db_entries,
    upsert_mock_consensus,
    load_prospects_from_db,
    get_active_rookie_class,
)
from data_building.rookie_pipeline.ml_model import score_all_prospects_ml as score_all_prospects
from data_building.rookie_pipeline.rookie_db_storage import save_rankings_to_db


def _fetch_picks(draft_year: int, from_json: str | None) -> list:
    if from_json:
        print(f"[seed_picks] Loading picks from {from_json}")
        with open(from_json) as f:
            picks = json.load(f)
        print(f"[seed_picks] Loaded {len(picks)} picks from file")
        return picks

    print(f"[seed_picks] Fetching {draft_year} picks from nflverse…")
    picks = fetch_nflverse_draft_picks(draft_year)
    if not picks:
        print("[seed_picks] nflverse returned no data — draft CSV may not be updated yet.")
        print("  Tip: provide picks manually with --from-json picks.json")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed actual NFL draft picks and re-score prospects.")
    parser.add_argument("--year", type=int, default=None, help="Draft year (default: active class)")
    parser.add_argument("--from-json", metavar="FILE", help="Load picks from a local JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing to DB")
    args = parser.parse_args()

    draft_year = args.year or get_active_rookie_class()
    print(f"\n[seed_picks] ── {draft_year} draft ──────────────────────────────────────────")

    picks = _fetch_picks(draft_year, args.from_json)
    if not picks:
        sys.exit(1)

    # Filter to skill positions only
    skill_pos = {"QB", "RB", "WR", "TE"}
    skill_picks = [p for p in picks if (p.get("position") or "").upper() in skill_pos]
    print(f"[seed_picks] {len(skill_picks)} skill-position picks (of {len(picks)} total)")

    if args.dry_run:
        for p in sorted(skill_picks, key=lambda x: x["pick"]):
            print(f"  Pick {p['pick']:3d} Rd{p['round']} {p['position']:2s}  {p['player_name']}  → {p['nfl_team']}")
        print("\n[seed_picks] Dry run — no DB changes.")
        return

    # 1. Store actual picks in rookie_prospects
    print("\n[seed_picks] ── Stage 1: Store actual picks ──")
    with get_conn() as conn:
        n_updated = upsert_actual_draft_picks(skill_picks, draft_year, conn)
        conn.commit()
    print(f"[seed_picks] Updated {n_updated} prospects with actual pick data")

    # 2. Rebuild consensus map (actual picks overlay mock projections)
    print("\n[seed_picks] ── Stage 2: Rebuild consensus map ──")
    with get_conn() as conn:
        consensus_map = build_consensus_from_db_entries(draft_year, conn)
    n_actual = sum(1 for v in consensus_map.values() if v.get("is_actual_pick"))
    print(f"[seed_picks] Consensus: {len(consensus_map)} players, {n_actual} with confirmed picks")

    with get_conn() as conn:
        upsert_mock_consensus(consensus_map, draft_year, conn)
        conn.commit()

    # 3. Re-score prospects
    print("\n[seed_picks] ── Stage 3: Re-score prospects ──")
    with get_conn() as conn:
        prospects = load_prospects_from_db(draft_year, conn)
    print(f"[seed_picks] Loaded {len(prospects)} prospects from DB")

    scored = score_all_prospects(prospects, consensus_map, skip_sagarin=True)
    print(f"[seed_picks] Scored {len(scored)} prospects")

    # 4. Save updated rankings
    print("\n[seed_picks] ── Stage 4: Save rankings ──")
    with get_conn() as conn:
        save_rankings_to_db(scored, draft_year, conn)
        conn.commit()

    # 5. Bust the in-memory cache so the API reflects new scores
    try:
        from dashboard_services.rookie_api import _cache as _rookie_cache
        _rookie_cache.pop(draft_year, None)
        print(f"[seed_picks] Cleared in-memory cache for {draft_year}")
    except Exception:
        pass

    print(f"\n[seed_picks] Done — {n_updated} prospects updated with actual picks, {len(scored)} re-scored.")


if __name__ == "__main__":
    main()
