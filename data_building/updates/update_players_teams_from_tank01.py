#!/usr/bin/env python3
"""
Update all players' team data from Tank01 API.
This will refresh team assignments for both players_index and relevant_players_index.
"""

import json
from pathlib import Path
import requests

from utils.utils import TANK01_API_HOST, TANK01_API_KEY, path_players_index, path_relevant_index

def update_players_teams_from_tank01():
    """Update team data for all existing players from Tank01."""
    
    # Load existing players index
    players_index_path = Path(path_players_index())
    with players_index_path.open("r", encoding="utf-8") as f:
        players_index = json.load(f)
    
    print(f"Loaded {len(players_index)} existing players")
    
    # Fetch fresh data from Tank01
    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }
    
    print("📡 Fetching Tank01 player list...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json().get("body", [])
    updated_count = 0
    
    # Create mapping of sleeper_id -> team from Tank01
    tank01_teams = {}
    for p in data:
        # Try multiple Sleeper ID field names
        sleeper_id = (
            p.get("sleeperBotID")
            or p.get("sleeperId") 
            or p.get("sleeper_id")
            or p.get("sleeperid")
        )
        
        if not sleeper_id:
            continue
            
        sleeper_id = str(sleeper_id)
        team = p.get("team", p.get("proTeam", ""))
        # Normalize WSH to WAS for consistency
        if team == "WSH":
            team = "WAS"
        tank01_teams[sleeper_id] = team
    
    # Update players_index with fresh team data
    for sleeper_id, player_data in players_index.items():
        if sleeper_id in tank01_teams:
            old_team = player_data.get("team", "")
            new_team = tank01_teams[sleeper_id]
            
            if old_team != new_team:
                player_data["team"] = new_team
                updated_count += 1
                print(f"Updated {player_data.get('name', 'Unknown')}: {old_team} → {new_team}")
    
    # Save updated players_index
    with players_index_path.open("w", encoding="utf-8") as f:
        json.dump(players_index, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Updated team data for {updated_count} players in players_index")
    
    # Also update relevant_players_index if it exists
    relevant_path = Path(path_relevant_index())
    if relevant_path.exists():
        with relevant_path.open("r", encoding="utf-8") as f:
            relevant_index = json.load(f)
        
        relevant_updated = 0
        for sleeper_id, player_data in relevant_index.items():
            if sleeper_id in tank01_teams:
                old_team = player_data.get("team", "")
                new_team = tank01_teams[sleeper_id]
                
                if old_team != new_team:
                    player_data["team"] = new_team
                    relevant_updated += 1
        
        with relevant_path.open("w", encoding="utf-8") as f:
            json.dump(relevant_index, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Updated team data for {relevant_updated} players in relevant_players_index")
    else:
        print("⚠️  relevant_players_index not found, skipping")
    
    return updated_count

if __name__ == "__main__":
    update_players_teams_from_tank01()
