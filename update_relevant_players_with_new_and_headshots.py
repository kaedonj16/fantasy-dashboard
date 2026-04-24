#!/usr/bin/env python3
"""
Add new players from players_index.json to players_index_relevant.json
and fetch their ESPN headshots.
"""

import json
from pathlib import Path
import requests

from utils.utils import TANK01_API_HOST, TANK01_API_KEY, load_usage_table
from dashboard_services.players import get_league_rostered_player_ids
from dashboard_services.platform_api import get_rosters

# Fantasy position whitelist
POS_WHITELIST = {"QB", "RB", "WR", "TE"}

def load_players_index():
    """Load the full players index."""
    with open("cache/players_index.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_relevant_players_index():
    """Load the relevant players index."""
    with open("cache/players_index_relevant.json", "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_tank01_players():
    """Fetch all players from Tank01 API for headshot data."""
    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }
    
    print("📡 Fetching Tank01 player data for headshots...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    
    data = resp.json().get("body", [])
    
    # Create mapping of tankId to headshot URL
    headshot_map = {}
    for player in data:
        tank_id = str(player.get("playerID") or player.get("playerId") or player.get("id") or "")
        headshot = player.get("espnHeadshot")
        if tank_id and headshot:
            headshot_map[tank_id] = headshot
    
    print(f"📸 Found {len(headshot_map)} players with headshots")
    return headshot_map

def is_rookie_or_new_player(meta: dict) -> bool:
    """Check if player is likely a rookie or new player based on available data."""
    # Check if player has a recent birth year (2020-2006 for typical rookies)
    bday = meta.get("bDay", "")
    if bday:
        try:
            birth_year = int(bday.split("/")[-1])  # MM/DD/YYYY format
            # Players born 2000-2006 are likely rookies/young players
            if 2000 <= birth_year <= 2006:
                return True
        except (ValueError, IndexError):
            pass
    
    return False

def is_fantasy_relevant(pid: str, meta: dict, usage: dict, rostered_pids: set) -> bool:
    """Check if a player meets fantasy relevance criteria."""
    pos = meta.get("pos") or meta.get("position")
    if pos not in POS_WHITELIST:
        return False

    # always keep rostered players
    if pid in rostered_pids:
        return True

    # Special handling for rookies/new players - include them if they're skill positions
    if is_rookie_or_new_player(meta):
        return True

    if not usage:
        return False

    # usage + production thresholds
    games = float(usage.get("games") or 0.0)
    ppr_ppg = float(usage.get("ppr_ppg") or 0.0)
    avg_snaps = float(usage.get("avg_off_snaps") or 0.0)
    avg_tgt = float(usage.get("avg_targets") or 0.0)
    avg_car = float(usage.get("avg_carries") or 0.0)

    if games >= 3:
        return True
    if ppr_ppg >= 6.0:
        return True
    if avg_snaps >= 20:
        return True
    if (avg_tgt + avg_car) >= 3:
        return True

    return False

def update_relevant_players_with_new_and_headshots(league_id: str = "default"):
    """Add new players and update headshots in relevant players index."""
    
    # Load both indexes
    full_index = load_players_index()
    relevant_index = load_relevant_players_index()
    
    print(f"📊 Full players index: {len(full_index)} players")
    print(f"📊 Relevant players index: {len(relevant_index)} players")
    
    # Load usage data and roster info
    usage_table = load_usage_table()
    
    # Fetch rosters for the league
    try:
        rosters = get_rosters(league_id)
        rostered_by_team = get_league_rostered_player_ids(league_id, rosters)
        rostered_pids = {str(pid) for pids in rostered_by_team.values() for pid in pids}
        print(f"📋 Found {len(rostered_pids)} rostered players")
    except Exception as e:
        print(f"⚠️ Could not fetch rosters: {e}")
        print("🔄 Proceeding without roster data - only rookies and players with usage will be added")
        rostered_pids = set()
    
    # Normalize usage table
    if isinstance(usage_table, dict):
        usage_table = {str(pid): (u or {}) for pid, u in usage_table.items()}
    elif isinstance(usage_table, list):
        normalized = {}
        for obj in usage_table:
            pid = obj.get("id")
            if pid is not None:
                normalized[str(pid)] = obj.get("usage") or {}
        usage_table = normalized
    
    # Find new players that meet relevance criteria
    new_players = {}
    for player_id, player_data in full_index.items():
        if player_id not in relevant_index:
            usage = usage_table.get(player_id, {})
            if is_fantasy_relevant(player_id, player_data, usage, rostered_pids):
                new_players[player_id] = player_data
    
    print(f"🆕 Found {len(new_players)} fantasy-relevant new players to add")
    
    # Fetch headshot data
    headshot_map = fetch_tank01_players()
    
    # Add new players with headshots
    added_count = 0
    updated_headshot_count = 0
    
    for player_id, player_data in new_players.items():
        # Add the new player with usage data (empty for rookies)
        usage = usage_table.get(player_id, {})
        merged = dict(player_data)
        merged["usage"] = usage  # Will be empty dict for rookies
        
        # Add headshot if available
        tank_id = str(player_data.get("tankId", ""))
        if tank_id and tank_id in headshot_map:
            merged["espnHeadshot"] = headshot_map[tank_id]
            added_count += 1
            print(f"  ➕ Added {player_data.get('name', 'Unknown')} ({player_data.get('pos')}) with headshot")
        else:
            print(f"  ➕ Added {player_data.get('name', 'Unknown')} ({player_data.get('pos')}) (no headshot)")
        
        relevant_index[player_id] = merged
    
    # Update headshots for existing players
    for player_id, player_data in relevant_index.items():
        if "espnHeadshot" not in player_data:  # Only update if missing headshot
            tank_id = str(player_data.get("tankId", ""))
            if tank_id and tank_id in headshot_map:
                player_data["espnHeadshot"] = headshot_map[tank_id]
                updated_headshot_count += 1
                print(f"  📸 Updated headshot for {player_data.get('name', 'Unknown')}")
    
    # Save updated relevant index
    with open("cache/players_index_relevant.json", "w", encoding="utf-8") as f:
        json.dump(relevant_index, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Summary:")
    print(f"   • Added {len(new_players)} new fantasy-relevant players")
    print(f"   • Added headshots for {added_count} new players")
    print(f"   • Updated headshots for {updated_headshot_count} existing players")
    print(f"   • Total relevant players: {len(relevant_index)}")
    
    return len(new_players), added_count, updated_headshot_count

if __name__ == "__main__":
    # You'll need to provide your actual league ID
    update_relevant_players_with_new_and_headshots("your_league_id_here")
