#!/usr/bin/env python3
"""
Populate NFL schedules for 2023 and 2024 seasons.
Fetches schedule data from Tank01 API and caches it locally.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.api import get_nfl_games_for_week_raw
from utils.utils import get_week_schedule_cached, get_or_refresh_schedule_path


def populate_schedules_for_season(season: int, max_week: int = 18, delay: float = 2.0):
    """
    Fetch and cache schedules for all weeks in a season.

    Args:
        season: The NFL season year (e.g., 2023, 2024)
        max_week: Maximum week number to fetch (default 18 for regular season)
        delay: Delay in seconds between API requests to avoid rate limiting
    """
    print(f"\n🏈 Populating schedules for {season} season...")

    fetched_count = 0
    cached_count = 0

    for week in range(1, max_week + 1):
        try:
            # Check if already cached
            cache_path = get_or_refresh_schedule_path(season, week)

            if cache_path is not None:
                print(f"  Week {week}... ✓ (cached)")
                cached_count += 1
                continue

            print(f"  Week {week}... ", end="", flush=True)

            # This will fetch and cache if not already cached
            schedule = get_week_schedule_cached(
                season=season,
                week=week,
                fetch_fn=get_nfl_games_for_week_raw,
                season_type="reg"
            )

            game_count = len(schedule) if schedule else 0
            print(f"✓ ({game_count} games)")
            fetched_count += 1

            # Add delay to avoid rate limiting (except for last week)
            if week < max_week and delay > 0:
                time.sleep(delay)

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print(f"✗ API key issue (401 Unauthorized)")
            elif "429" in error_msg:
                print(f"✗ Rate limited (429) - waiting {delay * 3}s...")
                time.sleep(delay * 3)  # Wait longer on rate limit
            else:
                print(f"✗ Error: {error_msg[:50]}")
            # Continue with next week even if one fails
            continue

    print(f"✅ Completed {season} season ({fetched_count} fetched, {cached_count} cached)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("NFL Schedule Population Script")
    print("=" * 60)

    # Populate 2023 season (18 weeks)
    populate_schedules_for_season(2023, max_week=18)

    # Populate 2024 season (18 weeks)
    populate_schedules_for_season(2024, max_week=18)

    print("\n" + "=" * 60)
    print("✅ Schedule population complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
