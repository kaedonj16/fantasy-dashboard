#!/usr/bin/env python3
"""
Verification script for PFN scraper integration.
Tests the scraper independently to verify it produces the correct data format.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_building'))

from data_building.rookie_pipeline.pfn_scraper import scrape_pfn_mock_consensus

def main():
    print("Verifying PFN scraper produces correct data format...")
    
    # Test with 2026 draft year
    draft_year = 2026
    
    results = scrape_pfn_mock_consensus(draft_year)
    
    print(f"\nPFN Scraper Results: {len(results)} entries")
    
    # Verify expected format
    expected_positions = ['QB', 'WR', 'RB', 'TE']
    found_positions = set()
    
    for entry in results:
        # Check required fields
        required_fields = ['player_name', 'position', 'school', 'projected_pick', 
                          'projected_round', 'mock_date', 'source_name', 'source_url', 'analyst_name']
        
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            print(f"ERROR: Missing fields {missing_fields} in entry: {entry}")
            return False
        
        # Check position is one of expected
        position = entry['position']
        if position not in expected_positions:
            print(f"WARNING: Unexpected position '{position}' found")
        
        found_positions.add(position)
        
        # Check format
        print(f"  {position}: Pick {entry['projected_pick']} (Round {entry['projected_round']}) - {entry['source_name']}")
    
    # Verify we found all expected positions
    missing_positions = set(expected_positions) - found_positions
    if missing_positions:
        print(f"WARNING: Missing positions: {missing_positions}")
    
    print(f"\n✓ PFN scraper integration verified successfully!")
    print(f"✓ Found {len(found_positions)}/{len(expected_positions)} expected positions")
    print(f"✓ All entries have required fields in correct format")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
