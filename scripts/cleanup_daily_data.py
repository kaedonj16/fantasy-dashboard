#!/usr/bin/env python3
"""
Script to clean up previous day's breakout scores and projected opportunities.
This ensures fresh data is calculated each day.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup_season_data(season=None):
    """
    Clean up all breakout-related data for a specific season.
    
    Args:
        season: Season year (defaults to current year)
    """
    if season is None:
        season = datetime.now().year

    print(f"🧹 Cleaning up breakout data for season {season}...")

    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            # Delete breakout scores
            deleted_scores = conn.execute("""
                DELETE FROM breakout_opportunity_scores 
                WHERE season = %s
            """, (season,)).rowcount

            # Delete projected opportunities
            deleted_projections = conn.execute("""
                DELETE FROM projected_opportunity 
                WHERE season = %s
            """, (season,)).rowcount

            # Delete roster changes
            deleted_changes = conn.execute("""
                DELETE FROM roster_changes 
                WHERE season = %s
            """, (season,)).rowcount

            # Delete vacated opportunity
            deleted_vacated = conn.execute("""
                DELETE FROM vacated_opportunity 
                WHERE season = %s
            """, (season,)).rowcount

            conn.commit()

            print(f"✅ Cleanup completed:")
            print(f"   - {deleted_scores} breakout scores deleted")
            print(f"   - {deleted_projections} projected opportunities deleted")
            print(f"   - {deleted_changes} roster changes deleted")
            print(f"   - {deleted_vacated} vacated opportunities deleted")
            print(f"   - Total: {deleted_scores + deleted_projections + deleted_changes + deleted_vacated} records")

            return deleted_scores + deleted_projections + deleted_changes + deleted_vacated

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return 0


def cleanup_all_data():
    """Clean up all breakout data across all seasons."""
    print("🧹 Cleaning up ALL breakout data (all seasons)...")

    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            # Delete all data from all tables
            tables = [
                'breakout_opportunity_scores',
                'projected_opportunity',
                'roster_changes',
                'vacated_opportunity'
            ]

            total_deleted = 0
            for table in tables:
                deleted = conn.execute(f"DELETE FROM {table}").rowcount
                total_deleted += deleted
                print(f"   - {deleted} records deleted from {table}")

            conn.commit()
            print(f"✅ All data cleanup completed: {total_deleted} total records deleted")
            return total_deleted

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean up breakout data')
    parser.add_argument('--season', type=int, help='Season year (defaults to current year)')
    parser.add_argument('--all', action='store_true', help='Clean up all seasons')

    args = parser.parse_args()

    if args.all:
        cleanup_all_data()
    else:
        cleanup_season_data(args.season)

    print("\n🚀 Ready for fresh data calculation!")


if __name__ == "__main__":
    main()
