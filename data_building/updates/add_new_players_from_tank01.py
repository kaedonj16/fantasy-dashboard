#!/usr/bin/env python3
"""
Add new players from Tank01 to existing players_index without overwriting.
"""

import json
from pathlib import Path
import requests

from utils.utils import TANK01_API_HOST, TANK01_API_KEY, path_players_index

def add_new_players_from_tank01():
    """Add only new players from Tank01 to existing players_index."""
    
    # Load existing players index
    players_index_path = Path(path_players_index())
    with players_index_path.open("r", encoding="utf-8") as f:
        existing_index = json.load(f)
    
    print(f"Loaded {len(existing_index)} existing players")
    
    # Fetch from Tank01
    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }
    
    print("📡 Fetching Tank01 player list...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json().get("body", [])
    new_players = {}
    updated_players = []
    
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
        
        # Skip if already exists
        if sleeper_id in existing_index:
            continue
            
        # Add new player with all available fields
        import datetime as _dt_mod
        _current_nfl_season = _dt_mod.datetime.now().year

        tank_id = str(p.get("playerID") or p.get("playerId") or p.get("id") or "")
        name = p.get("espnName", p.get("fullName", p.get("name", "")))
        team = p.get("team", p.get("proTeam", ""))
        # Normalize WSH to WAS for consistency
        if team == "WSH":
            team = "WAS"
        bDay = p.get("bDay")
        position = p.get("position", p.get("pos", ""))
        espn_id = tank_id  # Tank01 playerID is often the same as ESPN ID

        entry = {
            "name": name or "",
            "team": team or "",
            "tankId": tank_id,
            "bDay": bDay,
            "espnID": espn_id,
            "pos": position,
        }

        # Derive draft_year from exp field (exp=1 means 1st NFL season)
        raw_exp = p.get("exp") or p.get("espnYrsPro")
        if raw_exp is not None:
            try:
                exp_int = int(raw_exp)
                if exp_int > 0:
                    entry["exp"] = exp_int
                    entry["draft_year"] = _current_nfl_season - exp_int + 1
            except (TypeError, ValueError):
                pass

        new_players[sleeper_id] = entry
        updated_players.append(f"{name} ({team})")
    
    # Merge new players with existing
    if new_players:
        existing_index.update(new_players)
        
        # Save merged index
        with players_index_path.open("w", encoding="utf-8") as f:
            json.dump(existing_index, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Added {len(new_players)} new players:")
        for player in updated_players[:10]:  # Show first 10
            print(f"   • {player}")
        if len(updated_players) > 10:
            print(f"   ... and {len(updated_players) - 10} more")
    else:
        print("✅ No new players found")
    
    return len(new_players)

if __name__ == "__main__":
    add_new_players_from_tank01()
