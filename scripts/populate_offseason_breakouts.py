#!/usr/bin/env python3
"""
Quick script to populate offseason breakout data for production deployment.
Run this after deploying to ensure breakout candidates appear on the website.

Usage:
    export DATABASE_URL="postgresql://user@host:5432/dbname"
    python3 scripts/populate_offseason_breakouts.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Check DATABASE_URL is set
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable not set")
        print("Set it with: export DATABASE_URL=\"postgresql://user@host:5432/dbname\"")
        sys.exit(1)

    print("\n" + "="*60)
    print("POPULATING OFFSEASON BREAKOUT DATA")
    print("="*60 + "\n")

    try:
        from data_building.offseason_opportunity import init_offseason_opportunity_db
        from data_building.populate_roster_changes import populate_offseason_data
        from datetime import datetime

        # Initialize database tables
        print("Step 1: Initializing database tables...")
        init_offseason_opportunity_db()
        print("✓ Tables created/verified\n")

        # Get current season
        season = datetime.now().year
        if datetime.now().month < 3:  # Before March, use previous year
            season = season

        print(f"Step 2: Populating data for {season} season...")
        populate_offseason_data(season)

        # Verify data was populated
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            roster_changes = conn.execute('SELECT COUNT(*) as count FROM roster_changes').fetchone()['count']
            projected = conn.execute('SELECT COUNT(*) as count FROM projected_opportunity').fetchone()['count']

        print("\n" + "="*60)
        print("✓ DEPLOYMENT COMPLETE")
        print("="*60)
        print(f"\nResults:")
        print(f"  - Roster Changes: {roster_changes}")
        print(f"  - Breakout Candidates: {projected}")
        print(f"\nThe breakouts tab should now show {projected} candidates!")
        print()

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
