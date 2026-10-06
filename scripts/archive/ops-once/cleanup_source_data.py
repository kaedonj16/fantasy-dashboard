#!/usr/bin/env python3
"""
Script to delete all rookie_prospect_source_data records where source is not 'cfbd'.
This cleans up the database to keep only cfbd source data.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def cleanup_source_data(season=None, dry_run=True):
    """
    Delete records where source is not 'cfbd'.
    
    Args:
        season (int, optional): If provided, only delete for specific season
        dry_run (bool): If True, show what would be deleted without actually deleting
    """
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # First, show what will be deleted (much faster query)
            if season:
                cursor.execute("""
                    SELECT source, COUNT(*) as count
                    FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd' 
                    AND season = %s
                    GROUP BY source
                    ORDER BY count DESC
                """, (season,))
                season_text = f" for season {season}"
            else:
                cursor.execute("""
                    SELECT source, COUNT(*) as count
                    FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd'
                    GROUP BY source
                    ORDER BY count DESC
                """)
                season_text = ""
            
            results = cursor.fetchall()
            
            if not results:
                print("No records to delete - all records have source 'cfbd'")
                return True
            
            print(f"Records to be deleted{season_text}:")
            total_to_delete = 0
            for row in results:
                print(f"  {row['source']}: {row['count']} records")
                total_to_delete += row['count']
            
            print(f"\nTotal records to delete: {total_to_delete}")
            
            if dry_run:
                print("\nDRY RUN - No records were actually deleted")
                print("Run with dry_run=False to actually delete the records")
                return True
            
            # Confirm before deleting
            response = input(f"\nAre you sure you want to delete {total_to_delete} records? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled")
                return True
            
            # Delete the records
            if season:
                cursor.execute("""
                    DELETE FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd' 
                    AND season = %s
                """, (season,))
            else:
                cursor.execute("""
                    DELETE FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd'
                """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            print(f"Successfully deleted {deleted_count} records")
            
            # Show remaining records (faster query)
            cursor.execute("""
                SELECT source, COUNT(*) as count
                FROM rookie_prospect_source_data 
                GROUP BY source
                ORDER BY count DESC
            """)
            
            results = cursor.fetchall()
            print("\nRemaining records:")
            for row in results:
                print(f"  {row['source']}: {row['count']} records")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up rookie_prospect_source_data by removing non-cfbd records')
    parser.add_argument('--season', type=int, help='Only delete for specific season (e.g., 2025)')
    parser.add_argument('--execute', action='store_true', help='Actually delete records (default is dry run)')
    
    args = parser.parse_args()
    
    print("Cleaning up rookie_prospect_source_data...")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    if args.season:
        print(f"Season: {args.season}")
    
    success = cleanup_source_data(season=args.season, dry_run=not args.execute)
    
    if success:
        print("Cleanup completed successfully")
    else:
        print("Cleanup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
