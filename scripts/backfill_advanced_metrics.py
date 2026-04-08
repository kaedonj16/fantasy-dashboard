#!/usr/bin/env python3
"""
Backfill advanced player metrics for historical NFL seasons.

Fetches full-season stats from Sleeper's API (weeks 1-18), calculates
efficiency metrics for all active skill-position players, and saves them
to the player_advanced_metrics table tagged with the correct season year.

Safe to re-run — saves use ON CONFLICT DO UPDATE (upsert).

Usage:
    python scripts/backfill_advanced_metrics.py              # 2022, 2023, 2024, 2025
    python scripts/backfill_advanced_metrics.py 2025         # one season only
    python scripts/backfill_advanced_metrics.py 2023 2024    # multiple seasons
"""

import sys

from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
from data_building.external_data.sleeper_usage import build_usage_map_for_season
from utils.utils import load_players_index


def backfill_season(season: int, players_index: dict) -> int:
    """
    Calculate and save advanced metrics for a full historical season.

    Uses the end-of-regular-season date ({season+1}-01-10) as the snapshot
    date so seasons never collide on as_of_date.

    Returns the number of player rows saved.
    """
    print(f"  Fetching usage data for all 18 weeks (Sleeper API + cache)...")
    usage_map = build_usage_map_for_season(season, weeks=range(1, 19))
    print(f"  Usage map built: {len(usage_map)} players found")

    metrics_list = []
    skipped = 0
    failed = 0

    for pid, usage in usage_map.items():
        if usage.get("games", 0) == 0:
            skipped += 1
            continue

        # Look up position from players index (keys may be str or int)
        meta = players_index.get(pid) or players_index.get(str(pid)) or {}
        pos = meta.get("pos") or meta.get("position")

        if pos not in ("QB", "RB", "WR", "TE"):
            skipped += 1
            continue

        try:
            metrics_list.append(calculate_player_metrics(str(pid), usage, pos))
        except Exception as e:
            print(f"  [warn] player {pid}: {e}")
            failed += 1

    if not metrics_list:
        print(f"  No metrics to save (skipped={skipped}, failed={failed})")
        return 0

    # Use early January of the following year as the representative snapshot date.
    # This places the snapshot firmly within the season's date range
    # (Sep Y – Jan Y+1) and ensures each season has a unique as_of_date.
    as_of_date = f"{season + 1}-01-10"
    save_metrics_snapshot(metrics_list, as_of_date, season=season)
    print(f"  Saved {len(metrics_list)} players (skipped={skipped}, failed={failed}, date={as_of_date})")
    return len(metrics_list)


def main():
    if len(sys.argv) > 1:
        try:
            seasons = [int(s) for s in sys.argv[1:]]
        except ValueError:
            print("Usage: python scripts/backfill_advanced_metrics.py [season ...]\n"
                  "Example: python scripts/backfill_advanced_metrics.py 2023 2024 2025")
            sys.exit(1)
    else:
        seasons = [2022, 2023, 2024, 2025]

    print(f"Backfilling advanced metrics for seasons: {seasons}")
    print("=" * 60)

    players_index = load_players_index() or {}
    if not players_index:
        print("[error] Could not load players index. Aborting.")
        sys.exit(1)
    print(f"Loaded players index: {len(players_index)} entries\n")

    total_saved = 0
    for season in seasons:
        print(f"=== Season {season} ===")
        try:
            saved = backfill_season(season, players_index)
            total_saved += saved
        except Exception as e:
            import traceback
            print(f"  [error] Season {season} failed: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Done. Total rows saved/updated: {total_saved}")
    print("\nVerify with:")
    print("  python -c \"from dashboard_services.db import get_conn; "
          "[print(dict(r)) for r in get_conn().__enter__().execute("
          "'SELECT season, COUNT(*) cnt FROM player_advanced_metrics "
          "GROUP BY season ORDER BY season DESC').fetchall()]\"")


if __name__ == "__main__":
    main()
