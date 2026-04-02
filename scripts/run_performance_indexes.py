#!/usr/bin/env python3
"""
Script to create performance indexes for the breakout opportunity database.
Run this once to improve query performance for the UI and API.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("Creating performance indexes for breakout opportunity database...")

    try:
        from data_building.offseason_opportunity import init_offseason_opportunity_db
        from dashboard_services.db import get_conn

        # Initialize database (this will create indexes)
        init_offseason_opportunity_db()

        # Verify indexes were created
        with get_conn() as conn:
            # Check if key indexes exist
            indexes = conn.execute("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename IN ('projected_opportunity', 'breakout_opportunity_scores', 'roster_changes', 'vacated_opportunity')
                AND indexname LIKE 'idx_%'
                ORDER BY tablename, indexname
            """).fetchall()

            print(f"\n✅ Created {len(indexes)} performance indexes:")
            for idx in indexes:
                print(f"   - {idx['indexname']} on {idx['tablename']}")

        print("\n🚀 Database indexes created successfully!")
        print("UI and API queries should now be significantly faster.")

    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
