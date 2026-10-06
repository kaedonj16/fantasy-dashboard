#!/usr/bin/env python3
"""
Debug script to see what's happening during the update process.
"""

import csv
import os
import sys
import re
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def normalize_player_name(player_name: str) -> str:
    """Normalize player name for matching."""
    name_slug = re.sub(r'[^A-Z0-9]', '_', player_name.upper())
    name_slug = re.sub(r'_+', '_', name_slug)
    return name_slug.strip('_')


def debug_single_player():
    """Debug a single player update process."""
    print("Debugging single player update...")
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Test with a simple CSV row
            test_data = {
                'player': 'Caden Veltkamp',
                'grades_offense': '73.2',
                'grades_pass': '73.6',
                'big_time_throws': '23',
                'completion_percent': '67.4',
                'pressure_to_sack_rate': '15.3',
                'qb_rating': '88.9'
            }
            
            player_name = test_data['player']
            normalized_name = normalize_player_name(player_name)
            player_id_2026 = f"ROOKIE_2026_{normalized_name}"
            
            print(f"Testing player: {player_name}")
            print(f"Generated ID: {player_id_2026}")
            
            # Check if player exists
            cursor.execute(
                "SELECT player_id, season FROM rookie_prospect_source_data WHERE player_id = %s LIMIT 1",
                (player_id_2026,)
            )
            result = cursor.fetchone()
            
            if result:
                print(f"Found player: {result['player_id']} in season {result['season']}")
                
                # Try to update with explicit values
                update_data = {
                    'grades_offense': float(test_data['grades_offense']),
                    'pff_passing_grade': float(test_data['grades_pass']),
                    'big_time_throw_rate': float(test_data['big_time_throws']),
                    'adjusted_completion_rate': float(test_data['completion_percent']),
                    'pressure_to_sack_rate': float(test_data['pressure_to_sack_rate']),
                    'nfl_passer_rating': float(test_data['qb_rating'])
                }
                
                print(f"Update data: {update_data}")
                
                # Build UPDATE query
                set_clauses = [f"{col} = %s" for col in update_data.keys()]
                values = list(update_data.values()) + [player_id_2026, result['season'], 'pff_college']
                
                query = f"""
                    UPDATE rookie_prospect_source_data 
                    SET {', '.join(set_clauses)}
                    WHERE player_id = %s AND season = %s AND source = %s
                """
                
                print(f"Query: {query}")
                print(f"Values: {values}")
                
                cursor.execute(query, values)
                rows_affected = cursor.rowcount
                
                print(f"Rows affected by UPDATE: {rows_affected}")
                
                # Check if data was actually updated
                cursor.execute(
                    "SELECT grades_offense, pff_passing_grade FROM rookie_prospect_source_data WHERE player_id = %s AND season = %s",
                    (player_id_2026, result['season'])
                )
                updated_record = cursor.fetchone()
                
                print(f"After update: grades_offense={updated_record['grades_offense']}, pff_passing_grade={updated_record['pff_passing_grade']}")
                
                conn.commit()
                
            else:
                print("Player not found in database")
                
                # Try other patterns
                patterns = [
                    f"ROOKIE_2025_{normalized_name}",
                    f"ROOKIE_2024_{normalized_name}",
                    f"ROOKIE_{normalized_name}"
                ]
                
                for pattern in patterns:
                    cursor.execute(
                        "SELECT player_id, season FROM rookie_prospect_source_data WHERE player_id = %s LIMIT 1",
                        (pattern,)
                    )
                    result = cursor.fetchone()
                    if result:
                        print(f"Found player with pattern {pattern}: {result['player_id']} in season {result['season']}")
                        break
                else:
                    print("No matching player found with any pattern")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_player()
