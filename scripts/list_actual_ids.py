#!/usr/bin/env python3
"""
Script to list actual player IDs in the database.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def main():
    """List actual player IDs in the database."""
    print("Listing actual player IDs in database...")
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Get sample of actual player IDs
            cursor.execute("""
                SELECT DISTINCT player_id, season, COUNT(*) as count
                FROM rookie_prospect_source_data 
                WHERE player_id LIKE 'ROOKIE_%'
                GROUP BY player_id, season
                ORDER BY season, player_id
                LIMIT 20
            """)
            
            players = cursor.fetchall()
            
            if players:
                print("Sample player IDs in database:")
                for player in players:
                    print(f"  {player['player_id']} ({player['season']}) - {player['count']} records")
                
                # Show patterns
                print("\nID patterns found:")
                patterns = {}
                for player in players:
                    if 'ROOKIE_2026_' in player['player_id']:
                        patterns['ROOKIE_2026_*'] = patterns.get('ROOKIE_2026_*', 0) + 1
                    elif 'ROOKIE_2025_' in player['player_id']:
                        patterns['ROOKIE_2025_*'] = patterns.get('ROOKIE_2025_*', 0) + 1
                    elif 'ROOKIE_2024_' in player['player_id']:
                        patterns['ROOKIE_2024_*'] = patterns.get('ROOKIE_2024_*', 0) + 1
                    else:
                        patterns['Other'] = patterns.get('Other', 0) + 1
                
                for pattern, count in patterns.items():
                    print(f"  {pattern}: {count} players")
                
            else:
                print("No player IDs found with ROOKIE_ prefix")
            
            # Check total count
            cursor.execute("SELECT COUNT(*) FROM rookie_prospect_source_data WHERE player_id LIKE 'ROOKIE_%'")
            total = cursor.fetchone()[0]
            print(f"\nTotal ROOKIE_* records: {total}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
