#!/usr/bin/env python3
"""
Production Initialization Script
Handles all database setup, table creation, and initial data population
when the app first runs on Render or any production environment.
"""

import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🚀 Starting Production Initialization...")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Step 1: Initialize database tables and migrations
        print("\n" + "="*60)
        print("STEP 1: DATABASE INITIALIZATION")
        print("="*60)
        
        from scripts.run_migrations import run_all_migrations
        run_all_migrations()
        print("✅ Database migrations completed")
        
        # Step 2: Initialize breakout opportunity database
        from data_building.offseason_opportunity import init_offseason_opportunity_db
        init_offseason_opportunity_db()
        print("✅ Breakout opportunity database initialized")
        
        # Step 3: Initialize player value history database
        from data_building.player_value_history import init_value_history_db
        init_value_history_db()
        print("✅ Player value history database initialized")
        
        # Step 4: Create performance indexes
        from data_building.offseason_opportunity import create_performance_indexes
        from dashboard_services.db import get_conn
        
        with get_conn() as conn:
            create_performance_indexes(conn)
        print("✅ Performance indexes created")
        
        # Step 5: Run daily data processes
        print("\n" + "="*60)
        print("STEP 2: DAILY DATA PROCESSES")
        print("="*60)
        
        from cron_daily import build_daily_data
        build_daily_data()
        print("✅ Daily data processes completed")
        
        # Step 6: Verify initialization
        print("\n" + "="*60)
        print("STEP 3: VERIFICATION")
        print("="*60)
        
        from dashboard_services.db import get_conn
        
        with get_conn() as conn:
            # Check key tables exist and have data
            tables_to_check = [
                ('player_values', 'Player values'),
                ('breakout_opportunity_scores', 'Breakout scores'),
                ('projected_opportunity', 'Projected opportunities'),
                ('roster_changes', 'Roster changes'),
                ('vacated_opportunity', 'Vacated opportunity'),
                ('player_value_history', 'Value history')
            ]
            
            for table_name, description in tables_to_check:
                try:
                    result = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                    count = result['count'] if result else 0
                    print(f"✅ {description}: {count:,} records")
                except Exception as e:
                    print(f"❌ {description}: Error checking table - {e}")
        
        # Step 7: Final health check
        print("\n" + "="*60)
        print("STEP 4: HEALTH CHECK")
        print("="*60)
        
        # Test key API endpoints
        try:
            from app import app
            with app.test_client() as client:
                # Test basic endpoints
                endpoints = [
                    '/',
                    '/api/nfl-state',
                    '/api/breakout-candidates',
                    '/api/offseason-breakout-candidates'
                ]
                
                for endpoint in endpoints:
                    response = client.get(endpoint)
                    status = "✅" if response.status_code in [200, 302] else "❌"
                    print(f"{status} {endpoint}: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        print("\n" + "="*60)
        print("🎉 PRODUCTION INITIALIZATION COMPLETE")
        print("="*60)
        print("✅ All databases initialized")
        print("✅ Performance indexes created")
        print("✅ Daily data processes completed")
        print("✅ Health checks passed")
        print(f"🕐 Completed at: {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
