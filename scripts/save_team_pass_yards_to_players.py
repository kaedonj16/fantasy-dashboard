#!/usr/bin/env python3
"""
Save team net passing yards to player records based on CFBD team stats.

This script fetches team stats from the CFBD API and updates each player's
team_pass_yards field with the net passing yards from their college team.
"""

import sys
import os
import time
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services.db import get_conn

# CFBD API configuration
CFBD_BASE = "https://api.collegefootballdata.com"
CFBD_KEY = os.environ.get("CFBD_API_KEY")
_CFBD_THROTTLE_S = 4.0   # 600 req/hr = 1 per 6s; 4s sleep + ~0.5s latency

def _cfbd_get(path: str, params: Dict[str, Any] = None, retries: int = 5) -> Optional[Any]:
    """Fetch data from CFBD API with throttling and retries."""
    if not CFBD_KEY:
        print("[migration] ERROR: CFBD_API_KEY environment variable not set")
        return None
    
    import requests
    
    url = f"{CFBD_BASE}{path}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {CFBD_KEY}"}
    
    for attempt in range(retries):
        try:
            time.sleep(_CFBD_THROTTLE_S)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[migration] CFBD API error after {retries} attempts: {exc}")
                return None
            print(f"[migration] CFBD API error (attempt {attempt + 1}/{retries}): {exc}")
            time.sleep(2 ** attempt)
    
    return None

def normalize_team_name(team_name: str) -> str:
    """Normalize team name to match CFBD API format."""
    if not team_name:
        return team_name
    
    # Common team name mappings
    team_mappings = {
        # State universities
        'Alabama': 'Alabama',
        'Arizona': 'Arizona',
        'Arizona State': 'Arizona State',
        'Arkansas': 'Arkansas',
        'Auburn': 'Auburn',
        'California': 'California',
        'Colorado': 'Colorado',
        'Connecticut': 'UConn',
        'Duke': 'Duke',
        'Florida': 'Florida',
        'Florida State': 'Florida State',
        'Georgia': 'Georgia',
        'Georgia Tech': 'Georgia Tech',
        'Illinois': 'Illinois',
        'Indiana': 'Indiana',
        'Iowa': 'Iowa',
        'Iowa State': 'Iowa State',
        'Kansas': 'Kansas',
        'Kansas State': 'Kansas State',
        'Kentucky': 'Kentucky',
        'LSU': 'LSU',
        'Louisville': 'Louisville',
        'Maryland': 'Maryland',
        'Michigan': 'Michigan',
        'Michigan State': 'Michigan State',
        'Minnesota': 'Minnesota',
        'Mississippi': 'Ole Miss',
        'Mississippi State': 'Mississippi State',
        'Missouri': 'Missouri',
        'Nebraska': 'Nebraska',
        'Nevada': 'Nevada',
        'North Carolina': 'North Carolina',
        'North Carolina State': 'NC State',
        'Northwestern': 'Northwestern',
        'Notre Dame': 'Notre Dame',
        'Ohio State': 'Ohio State',
        'Oklahoma': 'Oklahoma',
        'Oklahoma State': 'Oklahoma State',
        'Oregon': 'Oregon',
        'Oregon State': 'Oregon State',
        'Penn State': 'Penn State',
        'Pittsburgh': 'Pittsburgh',
        'Purdue': 'Purdue',
        'Rutgers': 'Rutgers',
        'South Carolina': 'South Carolina',
        'Stanford': 'Stanford',
        'Syracuse': 'Syracuse',
        'TCU': 'TCU',
        'Texas': 'Texas',
        'Texas A&M': 'Texas A&M',
        'Texas Tech': 'Texas Tech',
        'UCLA': 'UCLA',
        'USC': 'USC',
        'Utah': 'Utah',
        'Virginia': 'Virginia',
        'Virginia Tech': 'Virginia Tech',
        'Washington': 'Washington',
        'Washington State': 'Washington State',
        'West Virginia': 'West Virginia',
        'Wisconsin': 'Wisconsin',
        'BYU': 'BYU',
        'Cincinnati': 'Cincinnati',
        'Houston': 'Houston',
        'UCF': 'UCF',
        'Memphis': 'Memphis',
        'SMU': 'SMU',
        'Tulane': 'Tulane',
        'Tulsa': 'Tulsa',
        'Temple': 'Temple',
        'USF': 'USF',
        'UCF': 'UCF',
        'East Carolina': 'East Carolina',
        'North Texas': 'North Texas',
        'Rice': 'Rice',
        'UTSA': 'UTSA',
        'UAB': 'UAB',
        'Louisiana Tech': 'Louisiana Tech',
        'Marshall': 'Marshall',
        'Middle Tennessee': 'Middle Tennessee',
        'Southern Miss': 'Southern Miss',
        'UAB': 'UAB',
        'UTEP': 'UTEP',
        'Western Kentucky': 'Western Kentucky',
        'Army': 'Army',
        'Navy': 'Navy',
        'Air Force': 'Air Force',
        'New Mexico': 'New Mexico',
        'New Mexico State': 'New Mexico State',
        'UTEP': 'UTEP',
        'Wyoming': 'Wyoming',
        'Colorado State': 'Colorado State',
        'Boise State': 'Boise State',
        'Fresno State': 'Fresno State',
        'Hawaii': 'Hawai\'i',
        'Nevada': 'Nevada',
        'San Diego State': 'San Diego State',
        'San Jose State': 'San Jose State',
        'UNLV': 'UNLV',
        'Utah State': 'Utah State',
        'Appalachian State': 'Appalachian State',
        'Arkansas State': 'Arkansas State',
        'Coastal Carolina': 'Coastal Carolina',
        'Georgia Southern': 'Georgia Southern',
        'Georgia State': 'Georgia State',
        'James Madison': 'James Madison',
        'Liberty': 'Liberty',
        'Louisiana': 'Louisiana',
        'Louisiana-Lafayette': 'Louisiana',
        'Louisiana-Monroe': 'Louisiana Monroe',
        'Marshall': 'Marshall',
        'Middle Tennessee': 'Middle Tennessee',
        'Old Dominion': 'Old Dominion',
        'South Alabama': 'South Alabama',
        'Texas State': 'Texas State',
        'Troy': 'Troy',
        'UL Monroe': 'Louisiana Monroe',
        'UTSA': 'UTSA',
        'Western Kentucky': 'Western Kentucky',
        'Ball State': 'Ball State',
        'Bowling Green': 'Bowling Green',
        'Buffalo': 'Buffalo',
        'Central Michigan': 'Central Michigan',
        'Eastern Michigan': 'Eastern Michigan',
        'Kent State': 'Kent State',
        'Miami (OH)': 'Miami (OH)',
        'Northern Illinois': 'Northern Illinois',
        'Ohio': 'Ohio',
        'Toledo': 'Toledo',
        'Western Michigan': 'Western Michigan',
        'Akron': 'Akron',
        'Charlotte': 'Charlotte',
        'Florida Atlantic': 'Florida Atlantic',
        'FIU': 'FIU',
        'Florida International': 'FIU',
        'North Texas': 'North Texas',
        'Rice': 'Rice',
        'Southern Miss': 'Southern Miss',
        'UAB': 'UAB',
        'UTEP': 'UTEP',
        'UTSA': 'UTSA',
        'Charlotte': 'Charlotte',
        'Florida Atlantic': 'Florida Atlantic',
        'FIU': 'FIU',
        'North Texas': 'North Texas',
        'Rice': 'Rice',
        'Southern Miss': 'Southern Miss',
        'UAB': 'UAB',
        'UTEP': 'UTEP',
        'UTSA': 'UTSA',
        'Boston College': 'Boston College',
        'Clemson': 'Clemson',
        'Duke': 'Duke',
        'Florida State': 'Florida State',
        'Georgia Tech': 'Georgia Tech',
        'Louisville': 'Louisville',
        'Miami': 'Miami',
        'North Carolina': 'North Carolina',
        'NC State': 'NC State',
        'Syracuse': 'Syracuse',
        'Virginia': 'Virginia',
        'Virginia Tech': 'Virginia Tech',
        'Wake Forest': 'Wake Forest',
        'Baylor': 'Baylor',
        'Iowa State': 'Iowa State',
        'Kansas': 'Kansas',
        'Kansas State': 'Kansas State',
        'Oklahoma': 'Oklahoma',
        'Oklahoma State': 'Oklahoma State',
        'TCU': 'TCU',
        'Texas': 'Texas',
        'Texas Tech': 'Texas Tech',
        'West Virginia': 'West Virginia',
        'Arizona': 'Arizona',
        'Arizona State': 'Arizona State',
        'California': 'California',
        'Colorado': 'Colorado',
        'Oregon': 'Oregon',
        'Oregon State': 'Oregon State',
        'Stanford': 'Stanford',
        'UCLA': 'UCLA',
        'USC': 'USC',
        'Utah': 'Utah',
        'Washington': 'Washington',
        'Washington State': 'Washington State',
        'Alabama': 'Alabama',
        'Arkansas': 'Arkansas',
        'Auburn': 'Auburn',
        'Florida': 'Florida',
        'Georgia': 'Georgia',
        'Kentucky': 'Kentucky',
        'LSU': 'LSU',
        'Mississippi State': 'Mississippi State',
        'Missouri': 'Missouri',
        'Ole Miss': 'Ole Miss',
        'South Carolina': 'South Carolina',
        'Tennessee': 'Tennessee',
        'Texas A&M': 'Texas A&M',
        'Vanderbilt': 'Vanderbilt',
        'Illinois': 'Illinois',
        'Indiana': 'Indiana',
        'Iowa': 'Iowa',
        'Michigan': 'Michigan',
        'Michigan State': 'Michigan State',
        'Minnesota': 'Minnesota',
        'Nebraska': 'Nebraska',
        'Northwestern': 'Northwestern',
        'Ohio State': 'Ohio State',
        'Penn State': 'Penn State',
        'Purdue': 'Purdue',
        'Rutgers': 'Rutgers',
        'Wisconsin': 'Wisconsin',
        # FCS teams
        'North Dakota State': 'North Dakota State',
        'South Dakota State': 'South Dakota State',
        'Montana': 'Montana',
        'Montana State': 'Montana State',
        'Eastern Washington': 'Eastern Washington',
        'Idaho': 'Idaho',
        'Idaho State': 'Idaho State',
        'Northern Arizona': 'Northern Arizona',
        'Northern Colorado': 'Northern Colorado',
        'Portland State': 'Portland State',
        'Sacramento State': 'Sacramento State',
        'Southern Utah': 'Southern Utah',
        'UC Davis': 'UC Davis',
        'Weber State': 'Weber State',
        'William & Mary': 'William & Mary',
        'Richmond': 'Richmond',
        'James Madison': 'James Madison',
        'Delaware': 'Delaware',
        'Maine': 'Maine',
        'New Hampshire': 'New Hampshire',
        'Rhode Island': 'Rhode Island',
        'Vermont': 'Vermont',
        'Brown': 'Brown',
        'Cornell': 'Cornell',
        'Dartmouth': 'Dartmouth',
        'Harvard': 'Harvard',
        'Penn': 'Penn',
        'Princeton': 'Princeton',
        'Yale': 'Yale',
        'Colgate': 'Colgate',
        'Fordham': 'Fordham',
        'Georgetown': 'Georgetown',
        'Holy Cross': 'Holy Cross',
        'Lafayette': 'Lafayette',
        'Lehigh': 'Lehigh',
        'Bucknell': 'Bucknell',
        'Davidson': 'Davidson',
        'Elon': 'Elon',
        'Furman': 'Furman',
        'Samford': 'Samford',
        'The Citadel': 'The Citadel',
        'VMI': 'VMI',
        'Wofford': 'Wofford',
        'Charleston Southern': 'Charleston Southern',
        'Gardner-Webb': 'Gardner-Webb',
        'Presbyterian': 'Presbyterian',
        'Western Carolina': 'Western Carolina',
        'Wofford': 'Wofford',
        'The Citadel': 'The Citadel',
        'VMI': 'VMI',
        'Charleston Southern': 'Charleston Southern',
        'Gardner-Webb': 'Gardner-Webb',
        'Presbyterian': 'Presbyterian',
        'Western Carolina': 'Western Carolina',
        'Youngstown State': 'Youngstown State',
        'Illinois State': 'Illinois State',
        'Indiana State': 'Indiana State',
        'Missouri State': 'Missouri State',
        'North Dakota': 'North Dakota',
        'Northern Iowa': 'Northern Iowa',
        'South Dakota': 'South Dakota',
        'Southern Illinois': 'Southern Illinois',
        'Western Illinois': 'Western Illinois',
        'Abilene Christian': 'Abilene Christian',
        'Central Arkansas': 'Central Arkansas',
        'Houston Baptist': 'Houston Christian',
        'Incarnate Word': 'Incarnate Word',
        'Lamar': 'Lamar',
        'McNeese State': 'McNeese',
        'Nicholls State': 'Nicholls',
        'Northwestern State': 'Northwestern State',
        'Sam Houston': 'Sam Houston',
        'Southeastern Louisiana': 'SE Louisiana',
        'Stephen F. Austin': 'Stephen F. Austin',
        'Texas A&M-Commerce': 'Texas A&M-Commerce',
        'Texas Southern': 'Texas Southern',
        'Alcorn State': 'Alcorn State',
        'Alabama A&M': 'Alabama A&M',
        'Alabama State': 'Alabama State',
        'Arkansas-Pine Bluff': 'Arkansas-Pine Bluff',
        'Bethune-Cookman': 'Bethune-Cookman',
        'Florida A&M': 'Florida A&M',
        'Grambling State': 'Grambling',
        'Jackson State': 'Jackson State',
        'Mississippi Valley State': 'Mississippi Valley State',
        'Prairie View A&M': 'Prairie View A&M',
        'Southern': 'Southern',
        'Texas Southern': 'Texas Southern',
    }
    
    # Try exact match first
    if team_name in team_mappings:
        return team_mappings[team_name]
    
    # Try to find partial match
    for db_name, api_name in team_mappings.items():
        if team_name.lower() == db_name.lower():
            return api_name
    
    # Return original if no mapping found
    return team_name

def save_team_pass_yards_to_players():
    """Save team net passing yards to player records from CFBD API."""
    
    print("[migration] Processing team_pass_yards for rookie prospects...")
    
    try:
        with get_conn() as conn:
            # Get all unique player-season combinations with their team names
            players = conn.execute("""
                SELECT DISTINCT player_id, season, team
                FROM rookie_prospect_source_data
                WHERE team IS NOT NULL 
                AND season IS NOT NULL
            """).fetchall()
            
            print(f"[migration] Found {len(players)} player-season combinations")
            
            # Get unique seasons to fetch team stats once per season
            seasons = set(player['season'] for player in players)
            print(f"[migration] Fetching team stats for seasons: {sorted(seasons)}")
            
            # Fetch team stats for each season
            team_stats_by_season = {}
            for season in seasons:
                print(f"[migration] Fetching team stats for {season}")
                data = _cfbd_get("/stats/season", {"year": season, "seasonType": "regular"})
                if not data:
                    print(f"[migration] No team stats data for {season}")
                    team_stats_by_season[season] = {}
                    continue
                
                # Build lookup by team name from stat records
                teams = {}
                for stat_record in data:
                    team_name = stat_record.get("team")
                    stat_name = stat_record.get("statName")
                    stat_value = stat_record.get("statValue")
                    
                    if team_name and stat_name == "netPassingYards":
                        if team_name not in teams:
                            teams[team_name] = {}
                        teams[team_name]["netPassingYards"] = stat_value
                
                team_stats_by_season[season] = teams
                print(f"[migration] Loaded team stats for {season}: {len(teams)} teams")
            
            # Update player records with team passing yards
            updated_count = 0
            for player in players:
                player_id = player['player_id']
                season = player['season']
                team_name = player['team']
                
                # Get team stats for this season
                season_team_stats = team_stats_by_season.get(season, {})
                normalized_team_name = normalize_team_name(team_name)
                team_stat = season_team_stats.get(normalized_team_name)
                
                if team_stat and team_stat.get("netPassingYards") is not None:
                    team_pass_yards = int(team_stat["netPassingYards"] or 0)
                    
                    # Update all records for this player-season combination
                    conn.execute("""
                        UPDATE rookie_prospect_source_data 
                        SET team_pass_yards = %s
                        WHERE player_id = %s 
                        AND season = %s
                        AND team = %s
                    """, (team_pass_yards, player_id, season, team_name))
                    
                    updated_count += 1
                    print(f"[migration] Updated team_pass_yards for player {player_id}, season {season}, team {team_name}: {team_pass_yards}")
                else:
                    print(f"[migration] No team stats found for {team_name} in {season}")
            
            print(f"[migration] Successfully updated team_pass_yards for {updated_count} player-season records")
            return True
            
    except Exception as e:
        print(f"[migration] ERROR: Failed to save team_pass_yards - {e}")
        return False


if __name__ == "__main__":
    success = save_team_pass_yards_to_players()
    if success:
        print("[migration] Team passing yards save completed successfully")
        sys.exit(0)
    else:
        print("[migration] Team passing yards save failed")
        sys.exit(1)
