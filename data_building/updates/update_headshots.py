#!/usr/bin/env python3
import json
import requests
import time
from pathlib import Path

def fetch_nfl_players():
    """Fetch NFL player data from the API"""
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLPlayerList"
    headers = {
        'Content-Type': 'application/json',
        'x-rapidapi-host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com',
        'x-rapidapi-key': 'a31667ff00msh6d542faa96aa36bp1513aajsn612c819feca4'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def update_players_with_headshots():
    """Update players_index_relevant.json with ESPN headshot URLs"""
    # Load current players data
    players_file = Path("/cache/players_index_relevant.json")
    
    with open(players_file, 'r') as f:
        players_data = json.load(f)
    
    # Fetch new player data
    print("Fetching NFL player data...")
    api_data = fetch_nfl_players()
    
    if not api_data or 'body' not in api_data:
        print("Failed to fetch API data")
        return
    
    # Create mapping of espnID to headshot URL
    headshot_map = {}
    for player in api_data['body']:
        if 'espnID' in player and 'espnHeadshot' in player:
            headshot_map[player['espnID']] = player['espnHeadshot']
    
    print(f"Found {len(headshot_map)} players with headshots")
    
    # Update players data with headshots
    updated_count = 0
    for player_id, player_info in players_data.items():
        if 'tankId' in player_info and player_info['tankId'] in headshot_map:
            player_info['espnHeadshot'] = headshot_map[player_info['tankId']]
            updated_count += 1
    
    print(f"Updated {updated_count} players with headshots")
    
    # Save updated data
    with open(players_file, 'w') as f:
        json.dump(players_data, f, indent=2)
    
    print("Updated players_index_relevant.json")

if __name__ == "__main__":
    update_players_with_headshots()
