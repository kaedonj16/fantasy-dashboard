#!/usr/bin/env python3
"""
Script to fetch CFBD college stats for a single player.

Usage:
    python fetch_single_player.py "Travis Hunter" 2025
    python fetch_single_player.py "Player Name" [draft_year]
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_building.rookie_pipeline.single_player_fetch import update_single_player_stats
from dashboard_services.db import get_conn


def main():
    
    player_name = sys.argv[1]
    draft_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    
    print(f"Fetching CFBD stats for: {player_name}")
    print(f"Draft year: {draft_year}")
    print("-" * 50)
    
    try:
        with get_conn() as conn:
            n_saved = update_single_player_stats(player_name, draft_year, conn)
            print(f"\n✅ Successfully saved {n_saved} season records for {player_name}")
            
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
