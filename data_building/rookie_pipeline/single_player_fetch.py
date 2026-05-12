"""
Utility to fetch CFBD college stats for a single player.
Useful for updating specific players without re-fetching the entire draft class.
"""
from typing import Dict, List, Optional

from utils.utils import normalize_name
from .ingestion import (
    CFBD_KEY, _cfbd_get, _safe_int, _safe, _build_cfbd_season,
    fetch_cfbd_games_played
)


# Name mapping: {cfbd_name: desired_display_name}
CFBD_NAME_MAPPINGS = {
    "kevin concepcion": "K.C. Concepcion",
    # Add more mappings as needed
    # "cfbd_name": "Display Name"
}


def get_cfbd_search_names(player_name: str) -> list[str]:
    """
    Get list of CFBD names to search for, including mappings and variations.
    """
    normalized = normalize_name(player_name)
    search_names = [normalized]
    
    # Check if this player has a known CFBD name mapping
    for cfbd_name, display_name in CFBD_NAME_MAPPINGS.items():
        if normalize_name(display_name) == normalized:
            search_names.append(cfbd_name)
            break
    
    return search_names


def fetch_cfbd_stats_for_player(
    player_name: str,
    draft_year: int,
    fetch_games_played: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Fetch college stats from CFBD for a single player.
    Returns {player_name_lower: [season_dict, ...]} sorted oldest→newest.
    
    Args:
        player_name: Name of the player to fetch stats for
        draft_year: Draft year (will fetch for draft_year-1, draft_year-2, draft_year-3)
        fetch_games_played: If True, fetch exact games-played counts
    """
    if not CFBD_KEY:
        print("[cfbd] No CFBD_API_KEY set - cannot fetch stats")
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    # Get all possible CFBD names to search for
    search_names = get_cfbd_search_names(player_name)

    try:
        # Team season totals for market share / dominator calculation
        team_stats: Dict[int, Dict] = {}
        for yr in years:
            data = _cfbd_get("/stats/season", {"year": yr, "seasonType": "regular"})
            if not data:
                print(f"[cfbd] No team stats data for {yr}")
                team_stats[yr] = {}
                continue
            try:
                teams: Dict[str, Dict] = {}
                for row in data:
                    t = row.get("team", "")
                    teams.setdefault(t, {})[row.get("statName", "")] = _safe(row.get("statValue"), 0)
                for t, s in teams.items():
                    pa = s.get("passAttempts", 0) or 0
                    ra = s.get("rushingAttempts", 0) or 0
                    total = pa + ra
                    s["pass_rate"] = round(pa / total, 3) if total > 0 else 0.5
                team_stats[yr] = teams
            except Exception as exc:
                print(f"[cfbd] ERROR processing team stats for {yr} - {type(exc).__name__}: {exc}")
                team_stats[yr] = {}

        # Games-played lookup for this specific player
        games_played_map: Dict[str, Dict[int, int]] = {}
        if fetch_games_played:
            try:
                # Fetch games played for just this player
                games_played_map = fetch_cfbd_games_played(draft_year)
                if player_name.lower() in games_played_map:
                    print(f"[cfbd] Games played resolved for '{player_name}'")
                else:
                    games_played_map = {}
            except Exception as exc:
                print(f"[cfbd] WARNING: games-played fetch failed ({exc}), will default to None")
                games_played_map = {}
        else:
            print("[cfbd] Skipping games-played lookup")

        # Player season stats for this specific player
        result: Dict[str, List[Dict]] = {}
        seasons = []
        
        for yr in years:
            try:
                # Try different conferences to find the player
                conferences = ["sec", "acc", "big-ten", "big-12", "pac-12", "sun-belt", "cusa", "mac", "indep"]
                data = []
                
                for conference in conferences:
                    params = {"year": yr, "seasonType": "regular", "conference": conference}
                    test_data = _cfbd_get("/stats/player/season", params) or []
                    if test_data:
                        # Check if our player is in this conference data
                        for search_name in search_names:
                            for row in test_data:
                                n = (row.get("player") or "").lower()
                                if n == search_name.lower():
                                    data = test_data
                                    break
                            if data:
                                break
                    if data:
                        break
                
                if not data:
                    # Fallback to no conference filter
                    data = _cfbd_get("/stats/player/season", {"year": yr, "seasonType": "regular"}) or []

                # Find rows matching our player (try all search names)
                player_rows = []
                matched_name = None
                for row in data:
                    n = (row.get("player") or "").lower()
                    if n in [search_name.lower() for search_name in search_names]:
                        position = (row.get("position") or "").upper()
                        if position in {"QB", "WR", "RB", "TE"}:
                            player_rows.append(row)
                            if not matched_name:
                                matched_name = row.get("player")
                
                if player_rows:
                    gp = (games_played_map.get(search_names[0].lower()) or {}).get(yr)
                    for i, row in enumerate(player_rows[:2]):  # Show first 2 rows
                        print(f"[cfbd] DEBUG:   Row {i+1}: {dict(list(row.items())[:5])}")  # Show first 5 fields
                    
                    season = _build_cfbd_season(player_rows, team_stats.get(yr, {}), yr, gp)

                    if season:
                        seasons.append(season)
                else:
                    print(f"[cfbd] No stats found for '{player_name}' in {yr}")
                    
            except Exception as exc:
                print(f"[cfbd] ERROR loading player stats for {yr} - {type(exc).__name__}: {exc}")

        if seasons:
            seasons.sort(key=lambda s: s["season"])
            result[search_names[0].lower()] = seasons
        else:
            print(f"[cfbd] COMPLETE: No stats found for '{player_name}'")

        return result
        
    except Exception as exc:
        print(f"[cfbd] FAILED: Unexpected error fetching stats for '{player_name}' - {type(exc).__name__}: {exc}")
        return {}


def update_single_player_stats(player_name: str, draft_year: int, conn) -> int:
    """
    Fetch and update CFBD stats for a single player in the database.
    Returns the number of season records updated.
    """
    print(f"[pipeline] Updating CFBD stats for single player: '{player_name}'")
    
    # First, find existing prospect in the database
    expected_player_id = f"ROOKIE_{draft_year}_{normalize_name(player_name).upper().replace(' ', '_')}"
    
    with conn.cursor() as cur:
        # Try to find existing prospect by player_id or name
        cur.execute(
            """
            SELECT player_id, name FROM rookie_prospects 
            WHERE (player_id = %s OR name = %s) AND draft_class_year = %s
            """,
            (expected_player_id, player_name, draft_year)
        )
        existing_prospect = cur.fetchone()
        
        if not existing_prospect:
            print(f"[pipeline] No existing prospect found for '{player_name}' (player_id: {expected_player_id})")
            return 0
        
        actual_player_id = existing_prospect["player_id"]
        actual_name = existing_prospect["name"]
        print(f"[pipeline] Found existing prospect: {actual_player_id} ('{actual_name}')")
    
    # Fetch stats for this player
    cfbd_stats = fetch_cfbd_stats_for_player(player_name, draft_year)
    
    if not cfbd_stats:
        print(f"[pipeline] No CFBD stats found for '{player_name}'")
        return 0
    
    # Get the search names to find the correct CFBD key
    search_names = get_cfbd_search_names(player_name)
    cfbd_key = search_names[0].lower()  # Use the first search name as key
    
    # Re-map the CFBD stats to use the correct key for upsert
    if cfbd_key not in cfbd_stats and len(cfbd_stats) == 1:
        # If stats exist but under a different key, remap them
        actual_key = list(cfbd_stats.keys())[0]
        cfbd_stats[cfbd_key] = cfbd_stats.pop(actual_key)
        print(f"[pipeline] Remapped CFBD stats from '{actual_key}' to '{cfbd_key}'")
    
    # Create prospect dict with the existing player_id
    prospect = {
        "player_id": actual_player_id,
        "name": actual_name,
        "draft_class_year": draft_year
    }
    
    # Debug: Show what we're about to save
    print(f"[pipeline] DEBUG: CFBD stats keys: {list(cfbd_stats.keys())}")
    for key, seasons in cfbd_stats.items():
        print(f"[pipeline] DEBUG: Key '{key}' has {len(seasons)} seasons:")
        for i, season in enumerate(seasons):
            print(f"[pipeline] DEBUG:   Season {i+1}: {season.get('season')} - yards: {season.get('receiving_yards')}, targets: {season.get('targets')}")
    
    # Update existing records in database
    from .pipeline import upsert_prospect_source_data
    n_saved = upsert_prospect_source_data([prospect], cfbd_stats, draft_year, conn)
    print(f"[pipeline] Updated {n_saved} season records for '{actual_name}' ({actual_player_id})")
    
    return n_saved
