#!/usr/bin/env python3
"""
Script to update existing players with the correct source value.
"""

import csv
import os
import sys
import re
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def normalize_name_for_matching(name: str) -> str:
    """Normalize name for database matching."""
    return re.sub(r'[^A-Z]', '', name.upper())


def validate_and_convert_value(value: str, column_name: str) -> any:
    """Validate and convert value to appropriate type with bounds checking."""
    try:
        if column_name in ['avoided_tackles', 'explosive_runs_10_plus']:
            int_val = int(float(value))
            return int_val
        elif column_name.endswith('_rate') or column_name.endswith('_percentage'):
            float_val = float(value)
            if abs(float_val) >= 100:
                return 99.999 if float_val > 0 else -99.999
            return float_val
        elif column_name.endswith('_rating'):
            float_val = float(value)
            if abs(float_val) >= 1000:
                return 999.99 if float_val > 0 else -999.99
            return float_val
        elif column_name.startswith('grades_'):
            float_val = float(value)
            if abs(float_val) >= 10000:
                return 9999.9 if float_val > 0 else -9999.9
            return float_val
        else:
            return float(value)
    except (ValueError, TypeError):
        return None


def update_with_correct_source(csv_file: str, column_mapping: dict):
    """Update players using the correct source value ('cfbd')."""
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Get all existing players from database
            cursor.execute("""
                SELECT player_id, season 
                FROM rookie_prospect_source_data 
                WHERE player_id LIKE 'ROOKIE_%'
            """)
            
            existing_players = cursor.fetchall()
            player_lookup = {}
            for player in existing_players:
                if '_' in player['player_id']:
                    name_part = player['player_id'].split('_', 2)[-1]
                    normalized_name = normalize_name_for_matching(name_part.replace('_', ' '))
                    player_lookup[normalized_name] = (player['player_id'], player['season'])
            
            print(f"Found {len(existing_players)} existing players in database")
            
            # Read CSV and update matches
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                total_updated = 0
                total_skipped = 0
                total_not_found = 0
                
                for row in reader:
                    try:
                        player_name = row['player']
                        csv_normalized = normalize_name_for_matching(player_name)
                        
                        # Check if player exists in database
                        if csv_normalized not in player_lookup:
                            total_not_found += 1
                            continue
                        
                        matching_player_id, matching_season = player_lookup[csv_normalized]
                        
                        # Prepare and validate update data
                        update_data = {}
                        for csv_col, db_col in column_mapping.items():
                            value = row.get(csv_col)
                            if value and value.strip():
                                validated_value = validate_and_convert_value(value, db_col)
                                if validated_value is not None:
                                    update_data[db_col] = validated_value
                        
                        if not update_data:
                            total_skipped += 1
                            continue
                        
                        # Update the record with CORRECT source value
                        set_clauses = [f"{col} = %s" for col in update_data.keys()]
                        values = list(update_data.values()) + [matching_player_id, matching_season, 'cfbd']  # Use 'cfbd' instead of 'pff_college'
                        
                        query = f"""
                            UPDATE rookie_prospect_source_data 
                            SET {', '.join(set_clauses)}
                            WHERE player_id = %s AND season = %s AND source = %s
                        """
                        
                        cursor.execute(query, values)
                        
                        if cursor.rowcount > 0:
                            total_updated += 1
                            if total_updated % 10 == 0:
                                print(f"Updated {total_updated} records so far...")
                        else:
                            total_skipped += 1
                    
                    except Exception as e:
                        print(f"Error processing {row.get('player', 'Unknown')}: {e}")
                        total_skipped += 1
                        continue
                
                conn.commit()
                
                print(f"\nSummary for {csv_file}:")
                print(f"  Total Updated: {total_updated}")
                print(f"  Total Skipped: {total_skipped}")
                print(f"  Total Not Found in DB: {total_not_found}")
                
                return total_updated
                
    except Exception as e:
        print(f"Database error: {e}")
        return 0


def main():
    """Main execution function."""
    print("Updating existing players with correct source value ('cfbd')...")
    
    # Define CSV files and their column mappings
    csv_configs = [
        {
            'file': 'data/receiving_summary.csv',
            'mapping': {
                'yards_after_catch': 'yards_after_catch',
                'yards_after_catch_per_reception': 'yards_after_catch_per_reception', 
                'avg_depth_of_target': 'avg_depth_of_target',
                'contested_catch_rate': 'contested_catch_rate',
                'avoided_tackles': 'avoided_tackles',
                'drop_rate': 'drop_rate',
                'slot_rate': 'slot_rate',
                'wide_rate': 'wide_rate',
                'inline_rate': 'inline_rate',
                'pass_block_rate': 'pass_block_rate',
                'grades_offense': 'grades_offense',
                'grades_pass_block': 'grades_pass_block',
                'explosive_runs_10_plus': 'explosive_runs_10_plus',
                'breakaway_percentage': 'breakaway_percentage',
                'elusive_rating': 'elusive_rating',
                'pff_rushing_grade': 'pff_rushing_grade'
            }
        },
        {
            'file': 'data/rushing_summary.csv',
            'mapping': {
                'explosive': 'explosive_runs_10_plus',
                'breakaway_percent': 'breakaway_percentage',
                'elusive_rating': 'elusive_rating',
                'grades_offense': 'grades_offense',
                'grades_run': 'pff_rushing_grade'
            }
        },
        {
            'file': 'data/passing_summary.csv',
            'mapping': {
                'grades_offense': 'grades_offense',
                'grades_pass': 'pff_passing_grade',
                'big_time_throws': 'big_time_throw_rate',
                'completion_percent': 'adjusted_completion_rate',
                'pressure_to_sack_rate': 'pressure_to_sack_rate',
                'qb_rating': 'nfl_passer_rating'
            }
        }
    ]
    
    total_updated = 0
    
    for config in csv_configs:
        csv_file = config['file']
        column_mapping = config['mapping']
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            continue
        
        print(f"\nProcessing {csv_file}...")
        updated = update_with_correct_source(csv_file, column_mapping)
        total_updated += updated
    
    print(f"\nFinal Summary:")
    print(f"  Total records updated: {total_updated}")


if __name__ == "__main__":
    main()
