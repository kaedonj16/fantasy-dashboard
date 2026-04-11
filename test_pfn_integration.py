#!/usr/bin/env python3
"""
Test script to verify PFN scraper integration works with the pipeline.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_building'))

from dashboard_services.db import get_conn
from data_building.rookie_pipeline.pipeline import upsert_mock_entries

def main():
    print("Testing PFN scraper integration...")
    
    # Test with 2026 draft year
    draft_year = 2026
    
    with get_conn() as conn:
        print(f"Upserting mock entries for {draft_year}...")
        saved = upsert_mock_entries(draft_year, conn)
        print(f"Successfully saved {saved} mock entries")
        
        # Check what was saved
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_name, COUNT(*) as count,
                       AVG(projected_pick) as avg_pick,
                       STRING_AGG(DISTINCT position, ', ') as positions
                FROM rookie_mock_draft_entries 
                WHERE draft_class_year = %s
                GROUP BY source_name
                ORDER BY source_name
            """, (draft_year,))
            
            results = cur.fetchall()
            print(f"\nMock entries by source for {draft_year}:")
            for row in results:
                source, count, avg_pick, positions = row
                print(f"  {source}: {count} entries, avg_pick={avg_pick:.1f}, positions={positions}")

if __name__ == "__main__":
    main()
