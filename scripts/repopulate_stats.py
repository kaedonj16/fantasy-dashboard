#!/usr/bin/env python3
"""
Repopulate missing Sleeper stats for 2025 weeks 16-18.

This script fetches stats from Sleeper's API and saves them to the cache.
"""

from data_building.external_data.sleeper_bulk_stats import fetch_week_stats

def main():
    print("Repopulating missing 2025 stats...")
    print("=" * 50)

    # Fetch missing weeks for 2025
    for week in [16, 17, 18]:
        print(f"\nFetching 2025 week {week}...")
        try:
            stats = fetch_week_stats(2025, week)
            player_count = len(stats) if isinstance(stats, dict) else 0
            print(f"  ✓ Fetched {player_count} players")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Done! Stats files updated.")
    print("\nYou can now refresh player modals to see weeks 16-18 stats.")

if __name__ == "__main__":
    main()
