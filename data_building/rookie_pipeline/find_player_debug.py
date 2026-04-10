#!/usr/bin/env python3
"""
Comprehensive debug script to find a player in CFBD data.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_building.rookie_pipeline.ingestion import CFBD_KEY, _cfbd_get
from utils.utils import normalize_name


def find_player_comprehensive(player_name: str, draft_year: int):
    """Comprehensive search for a player across multiple approaches."""
    if not CFBD_KEY:
        print("No CFBD_API_KEY set")
        return
    
    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    normalized = normalize_name(player_name)
    
    print(f"Comprehensive search for '{player_name}' (normalized: '{normalized}')")
    print(f"Years: {years}")
    print("=" * 60)
    
    # Generate search variations
    variations = [
        player_name.lower(),
        normalized,
        player_name.replace('.', '').lower(),
        player_name.split()[-1].lower(),  # Last name only
    ]
    
    # Add first name variations
    if ' ' in player_name:
        first_name = player_name.split()[0].replace('.', '').lower()
        variations.extend([first_name, first_name + ' concepcion'])
    
    variations = list(set(variations))
    print(f"Search variations: {variations}")
    print()
    
    for year in years:
        print(f"\n=== YEAR {year} ===")
        try:
            # Get all player data for this year
            data = _cfbd_get("/stats/player/season", {"year": year, "seasonType": "regular"}) or []
            print(f"Total players in CFBD for {year}: {len(data)}")
            
            # Search for any partial matches
            matches = []
            for row in data:
                name = row.get("player", "").lower()
                position = row.get("position", "").upper()
                team = row.get("team", "")
                
                # Check for any variation match
                for variation in variations:
                    if variation in name or name in variation:
                        matches.append({
                            'name': row.get("player"),
                            'position': position,
                            'team': team,
                            'matched_variation': variation,
                            'stats': {k: v for k, v in row.items() if k not in ['player', 'position', 'team'] and v not in [None, '', 0]}
                        })
                        break
            
            if matches:
                print(f"Found {len(matches)} potential matches:")
                for i, match in enumerate(matches):
                    print(f"  {i+1}. {match['name']} ({match['position']}, {match['team']})")
                    print(f"     Matched on: '{match['matched_variation']}'")
                    if match['stats']:
                        print(f"     Stats: {list(match['stats'].keys())[:5]}")  # Show first 5 stat keys
                        # Show some actual values
                        sample_stats = {k: v for k, v in list(match['stats'].items())[:3]}
                        print(f"     Sample values: {sample_stats}")
                    else:
                        print("     No non-zero stats found")
                    print()
            else:
                print("No matches found")
                
                # Show some similar names for debugging
                all_names = [row.get("player", "") for row in data if row.get("player")]
                concepcion_similar = [name for name in all_names if "concepcion" in name.lower()]
                kevin_similar = [name for name in all_names if "kevin" in name.lower()]
                
                if concepcion_similar:
                    print(f"Names with 'concepcion': {concepcion_similar[:3]}")
                if kevin_similar:
                    print(f"Names with 'kevin': {kevin_similar[:3]}")
                
                # Show some sample player names to understand the data format
                print(f"Sample player names: {all_names[:5]}")
                    
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_player_debug.py \"Player Name\" [draft_year]")
        print("Example: python find_player_debug.py \"K.C. Concepcion\" 2026")
        sys.exit(1)
    
    player_name = sys.argv[1]
    draft_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    
    find_player_comprehensive(player_name, draft_year)


if __name__ == "__main__":
    main()
