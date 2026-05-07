#!/usr/bin/env python3
"""
Script to update player ages in the database with precise decimal values.
This fixes the issue where ages were stored as whole numbers (24.0, 22.0, 23.0)
instead of precise decimal ages (24.1, 22.3, 23.7).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard_services.db import get_conn
from dashboard_services.service import age_from_bday
from utils.utils import load_players_index

def update_player_ages():
    """Update all player ages in the database with precise decimal values."""
    print("Starting precise age update...")
    
    # Load players index for birthday data
    players_index = load_players_index()
    if not players_index:
        print("ERROR: Could not load players index")
        return False
    
    print(f"Loaded {len(players_index)} players from index")
    
    # Connect to database
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Get all players with their IDs
                cur.execute("""
                    SELECT player_id, age, position 
                    FROM player_values 
                    WHERE player_id NOT LIKE '%_%'  -- Exclude picks (which have format like '2026_1_01')
                    AND position IS NOT NULL
                """)
                players = cur.fetchall()
                print(f"Found {len(players)} players in database")
                
                updated_count = 0
                error_count = 0
                
                for player in players:
                    player_id = str(player['player_id'])
                    current_age = player['age']
                    position = player['position']
                    
                    # Get player data from index
                    player_data = players_index.get(player_id)
                    if not player_data:
                        print(f"WARNING: No data found for player ID {player_id}")
                        continue
                    
                    # Calculate precise age from birthday
                    birthday = player_data.get('bDay')
                    if not birthday:
                        print(f"WARNING: No birthday found for {player_data.get('name', player_id)}")
                        continue
                    
                    precise_age = age_from_bday(birthday)
                    if precise_age is None:
                        print(f"WARNING: Could not calculate age for {player_data.get('name', player_id)}")
                        continue
                    
                    # Only update if age has changed significantly
                    if current_age is None or abs(float(current_age) - precise_age) > 0.05:
                        try:
                            cur.execute("""
                                UPDATE player_values 
                                SET age = ? 
                                WHERE player_id = ?
                            """, (precise_age, player_id))
                            updated_count += 1
                            
                            if updated_count % 100 == 0:
                                print(f"Updated {updated_count} players...")
                                
                        except Exception as e:
                            print(f"ERROR updating {player_data.get('name', player_id)}: {e}")
                            error_count += 1
                
                # Commit the transaction
                conn.commit()
                print(f"Successfully updated {updated_count} player ages")
                print(f"Encountered {error_count} errors")
                
                return updated_count > 0
                
    except Exception as e:
        print(f"Database error: {e}")
        return False

if __name__ == "__main__":
    success = update_player_ages()
    if success:
        print("Age update completed successfully!")
    else:
        print("Age update failed!")
        sys.exit(1)
