#!/usr/bin/env python3
"""
Script to verify PostgreSQL database contents.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def main():
    """Check what's actually in the PostgreSQL database."""
    print("Verifying PostgreSQL rookie_prospect_source_data table contents...")
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Check total records
            cursor.execute("SELECT COUNT(*) FROM rookie_prospect_source_data")
            total_count = cursor.fetchone()[0]
            print(f"Total records in table: {total_count}")
            
            # Check records with advanced metrics (non-null values)
            advanced_columns = [
                'yards_after_catch', 'avg_depth_of_target', 'contested_catch_rate',
                'avoided_tackles', 'drop_rate', 'slot_rate', 'wide_rate', 'inline_rate',
                'pass_block_rate', 'grades_offense', 'grades_pass_block',
                'explosive_runs_10_plus', 'breakaway_percentage', 'elusive_rating',
                'pff_rushing_grade', 'pff_passing_grade', 'big_time_throw_rate',
                'adjusted_completion_rate', 'pressure_to_sack_rate', 'nfl_passer_rating'
            ]
            
            print("\nRecords with non-null data:")
            for col in advanced_columns:
                cursor.execute(f"SELECT COUNT(*) FROM rookie_prospect_source_data WHERE {col} IS NOT NULL AND {col} != 0")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"  {col}: {count} records")
                else:
                    print(f"  {col}: 0 records")
            
            # Show some sample records
            cursor.execute("""
                SELECT player_id, season, yards_after_catch, contested_catch_rate, grades_offense 
                FROM rookie_prospect_source_data 
                WHERE yards_after_catch IS NOT NULL 
                LIMIT 3
            """)
            
            samples = cursor.fetchall()
            if samples:
                print(f"\nSample records with data:")
                for record in samples:
                    print(f"  {record['player_id']} ({record['season']}): yards_after_catch={record['yards_after_catch']}, contested_catch_rate={record['contested_catch_rate']}, grades_offense={record['grades_offense']}")
            else:
                print("\nNo records found with yards_after_catch data")
            
            # Check recent updates
            cursor.execute("""
                SELECT player_id, season, fetched_at 
                FROM rookie_prospect_source_data 
                ORDER BY fetched_at DESC 
                LIMIT 3
            """)
            
            recent = cursor.fetchall()
            if recent:
                print(f"\nMost recently updated records:")
                for record in recent:
                    print(f"  {record['player_id']} ({record['season']}): updated at {record['fetched_at']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
