#!/usr/bin/env python3
"""
Populate rookie prospect data and rankings.

This script runs the rookie pipeline to:
1. Load prospect data (from CFBD API if key is set, otherwise seed data)
2. Build mock draft consensus
3. Score all prospects
4. Translate scores to dynasty values
5. Write everything to the database

Usage:
    python scripts/populate_rookie_data.py              # Populate active class (2026)
    python scripts/populate_rookie_data.py --year 2025  # Populate specific year
    python scripts/populate_rookie_data.py --all        # Populate all years (2025, 2026)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Populate rookie prospect data")
    parser.add_argument(
        "--year",
        type=int,
        help="Draft class year to populate (default: active class)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Populate all available years (2025, 2026)"
    )
    args = parser.parse_args()

    print("🏈 Rookie Data Population Script")
    print("=" * 60)

    from data_building.rookie_pipeline.pipeline import (
        run_rookie_pipeline,
        get_active_rookie_class
    )
    from data_building.rookie_pipeline.rookie_evaluation_pipeline import (
        run_rookie_evaluation_pipeline,
    )

    if args.all:
        years = [2025, 2026]
        print(f"📅 Populating all years: {', '.join(map(str, years))}")
    elif args.year:
        years = [args.year]
        print(f"📅 Populating {args.year} draft class")
    else:
        active_year = get_active_rookie_class()
        years = [active_year]
        print(f"📅 Populating active class: {active_year}")

    print()

    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing {year} Draft Class")
        print(f"{'='*60}\n")

        try:
            print(f"  Step 1/2: Running evaluation pipeline (computes + saves eval metrics)...")
            eval_result = run_rookie_evaluation_pipeline(year)

            print(f"  Step 2/2: Running main pipeline (reads eval metrics from DB for scoring)...")
            result = run_rookie_pipeline(year)

            print(f"✅ Success! {year} draft class populated:")
            print(
                "   • Rookie evaluation: "
                f"{eval_result.get('profile_count', 0)} profiles, "
                f"db_metrics_rows={eval_result.get('db_metrics_rows', 0)}, "
                f"db_profiles_rows={eval_result.get('db_profiles_rows', 0)}"
            )
            print(f"   • {len(result.get('prospects', []))} prospects scored")
            print(f"   • {len(result.get('values', {}))} values calculated")

            if result.get('consensus'):
                print(f"   • {len(result['consensus'])} mock draft consensus entries")

        except Exception as exc:
            print(f"❌ Error processing {year}: {exc}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'='*60}")
    print("🎉 Rookie data population complete!")
    print(f"{'='*60}")

    # Show how to access the data
    print("\nℹ️  Next steps:")
    print("   • View rookies page: /sleeper/2026/<league_id>/rookies")
    print("   • API endpoint: /api/rookies/rankings?year=2026")
    print("   • Refresh anytime: POST /api/rookies/refresh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
