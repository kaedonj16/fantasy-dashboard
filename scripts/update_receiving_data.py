#!/usr/bin/env python3
"""
Script to update rookie_prospect_source_data table with receiving summary data.
Processes data/receiving_summary.csv and maps it to database columns.
"""

import csv
import os
import sys
import re
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def format_player_id(player_name: str) -> str:
    """Format player name to ROOKIE_2025_NAME_SLUG format."""
    # Convert to uppercase and replace spaces/special chars with underscores
    name_slug = re.sub(r'[^A-Z0-9]', '_', player_name.upper())
    # Remove consecutive underscores
    name_slug = re.sub(r'_+', '_', name_slug)
    # Remove leading/trailing underscores
    name_slug = name_slug.strip('_')
    
    return f"ROOKIE_2025_{name_slug}"


def map_csv_to_db_columns():
    """Map CSV column names to database column names."""
    return {
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


def update_receiving_data(csv_file_path: str, season: int = 2026, source: str = 'pff_college'):
    """Update rookie_prospect_source_data with receiving summary data."""
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return False
    
    column_mapping = map_csv_to_db_columns()
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Read CSV data
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                updated_count = 0
                skipped_count = 0
                
                for row in reader:
                    try:
                        # Format player ID
                        player_name = row['player']
                        player_id = format_player_id(player_name)
                        
                        # Check if player exists in rookie_prospects
                        cursor.execute(
                            "SELECT player_id FROM rookie_prospects WHERE player_id = %s",
                            (player_id,)
                        )
                        
                        if not cursor.fetchone():
                            print(f"Skipping {player_name} - player_id {player_id} not found in rookie_prospects")
                            skipped_count += 1
                            continue
                        
                        # Prepare update data
                        update_data = {}
                        for csv_col, db_col in column_mapping.items():
                            value = row.get(csv_col)
                            if value and value.strip():  # Skip empty values
                                # Convert to appropriate type
                                if db_col in ['avoided_tackles', 'explosive_runs_10_plus']:
                                    update_data[db_col] = int(value) if value else None
                                elif db_col.endswith('_rate') or db_col.endswith('_percentage') or db_col.endswith('_rating') or db_col in ['grades_offense', 'grades_pass_block', 'pff_rushing_grade']:
                                    update_data[db_col] = float(value) if value else None
                                else:
                                    update_data[db_col] = float(value) if value else None
                        
                        if not update_data:
                            print(f"Skipping {player_name} - no valid data")
                            skipped_count += 1
                            continue
                        
                        # Build UPDATE query
                        set_clauses = [f"{col} = %s" for col in update_data.keys()]
                        values = list(update_data.values()) + [player_id, season, source]
                        
                        query = f"""
                            UPDATE rookie_prospect_source_data 
                            SET {', '.join(set_clauses)}
                            WHERE player_id = %s AND season = %s AND source = %s
                        """
                        
                        cursor.execute(query, values)
                        
                        # If no rows were updated, insert new row
                        if cursor.rowcount == 0:
                            # Insert new row with basic required fields plus the new metrics
                            insert_cols = ['player_id', 'season', 'source'] + list(update_data.keys())
                            insert_values = [player_id, season, source] + list(update_data.values())
                            placeholders = ', '.join(['%s'] * len(insert_cols))
                            
                            insert_query = f"""
                                INSERT INTO rookie_prospect_source_data ({', '.join(insert_cols)})
                                VALUES ({placeholders})
                            """
                            
                            cursor.execute(insert_query, insert_values)
                        
                        updated_count += 1
                        print(f"Updated {player_name} ({player_id})")
                        
                    except Exception as e:
                        continue
                
                print(f"\nSummary:")
                print(f"  Updated/Inserted: {updated_count} records")
                print(f"  Skipped: {skipped_count} records")
                return True
                
    except Exception as e:
        print(f"Database error: {e}")
        return False


def main():
    """Main execution function."""
    csv_file = "data/receiving_summary.csv"
    
    print("Updating rookie_prospect_source_data with receiving summary data...")
    print(f"Source file: {csv_file}")
    
    success = update_receiving_data(csv_file)
    
    if success:
        print("✓ Update completed successfully")
    else:
        print("✗ Update failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
