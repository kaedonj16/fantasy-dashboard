#!/usr/bin/env python3
"""
Fast script to delete all rookie_prospect_source_data records where source is not 'cfbd'.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn


def main():
    """Fast cleanup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up rookie_prospect_source_data by removing non-cfbd records')
    parser.add_argument('--season', type=int, help='Only delete for specific season (e.g., 2025)')
    parser.add_argument('--dry-run', action='store_true', help='Show what will be deleted without actually deleting')
    
    args = parser.parse_args()
    
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            # Quick count of what will be deleted
            if args.season:
                cursor.execute("""
                    SELECT source, COUNT(*) as count
                    FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd' 
                    AND season = %s
                    GROUP BY source
                """, (args.season,))
                season_text = f" for season {args.season}"
            else:
                cursor.execute("""
                    SELECT source, COUNT(*) as count
                    FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd'
                    GROUP BY source
                """)
                season_text = ""
            
            results = cursor.fetchall()
            
            if not results:
                print("No records to delete - all records have source 'cfbd'")
                return
            
            print(f"Records to be deleted{season_text}:")
            total_to_delete = 0
            for row in results:
                print(f"  {row['source']}: {row['count']} records")
                total_to_delete += row['count']
            
            print(f"\nTotal records to delete: {total_to_delete}")
            
            if args.dry_run:
                print("\nDRY RUN - No records were actually deleted")
                print("Run without --dry-run to actually delete the records")
                return
            
            # Quick confirmation
            response = input(f"\nDelete {total_to_delete} records? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled")
                return
            
            # Delete the records
            print("Deleting...")
            if args.season:
                cursor.execute("""
                    DELETE FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd' 
                    AND season = %s
                """, (args.season,))
            else:
                cursor.execute("""
                    DELETE FROM rookie_prospect_source_data 
                    WHERE source != 'cfbd'
                """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            print(f"Deleted {deleted_count} records")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
