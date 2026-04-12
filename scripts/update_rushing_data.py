#!/usr/bin/env python3
"""
Script to update rookie_prospect_source_data table with rushing summary data.
Processes data/rushing_summary.csv and maps only specific columns to database.
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


def map_rushing_csv_to_db_columns():
    """Map rushing CSV column names to database column names for specific columns only."""
    return {
        'explosive': 'explosive_runs_10_plus',  # Map 'explosive' to explosive_runs_10_plus
        'breakaway_percent': 'breakaway_percentage',
        'elusive_rating': 'elusive_rating',
        'grades_offense': 'grades_offense',
        'grades_run': 'pff_rushing_grade'  # Map 'grades_run' to pff_rushing_grade
    }


def update_rushing_data(csv_file_path: str, season: int = 2026, source: str = 'pff_college'):
    """Update rookie_prospect_source_data with rushing summary data for specific columns only."""
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return False
    
    column_mapping = map_rushing_csv_to_db_columns()
    
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
                        
                        # Prepare update data - only process specific columns
                        update_data = {}
                        for csv_col, db_col in column_mapping.items():
                            value = row.get(csv_col)
                            if value and value.strip():  # Skip empty values
                                # Convert to appropriate type
                                if db_col == 'explosive_runs_10_plus':
                                    update_data[db_col] = int(value) if value else None
                                elif db_col.endswith('_percentage') or db_col.endswith('_rating') or db_col.startswith('grades_'):
                                    update_data[db_col] = float(value) if value else None
                                else:
                                    update_data[db_col] = float(value) if value else None
                        
                        if not update_data:
                            print(f"Skipping {player_name} - no valid data for specified columns")
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
                        
                        # If no rows were updated, insert new row with basic required fields plus new metrics
                        if cursor.rowcount == 0:
                            # Insert new row with only the specific columns we want
                            insert_cols = ['player_id', 'season', 'source'] + list(update_data.keys())
                            insert_values = [player_id, season, source] + list(update_data.values())
                            placeholders = ', '.join(['%s'] * len(insert_cols))
                            
                            insert_query = f"""
                                INSERT INTO rookie_prospect_source_data ({', '.join(insert_cols)})
                                VALUES ({placeholders})
                            """
                            
                            cursor.execute(insert_query, insert_values)
                        
                        updated_count += 1
                        print(f"Updated {player_name} ({player_id}) with rushing data")
                        
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
    csv_file = "data/rushing_summary.csv"
    
    print("Updating rookie_prospect_source_data with rushing summary data (specific columns only)...")
    print(f"Source file: {csv_file}")
    print(f"Target columns: explosive_runs_10_plus, breakaway_percentage, elusive_rating, grades_offense, pff_rushing_grade")
    
    success = update_rushing_data(csv_file)
    
    if success:
        print("✓ Rushing data update completed successfully")
    else:
        print("✗ Rushing data update failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
