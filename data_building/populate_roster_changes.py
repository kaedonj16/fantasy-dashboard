"""
Populate roster changes by detecting team changes between seasons.

Analyzes players_index to find players who changed teams,
then enriches with usage stats from previous season.
"""

import json
import os
from datetime import date
from typing import Dict, List, Optional

from data_building.offseason_opportunity import track_roster_change, calculate_vacated_opportunity, \
    project_opportunity_redistribution
from utils.utils import load_players_index, DATA_DIR


def load_usage_table_for_season(season: int) -> List[Dict]:
    """
    Load usage table for a specific season.
    Tries cache/player_history first, then falls back to data directory.
    """
    # Try cache/player_history first (historical data)
    cache_path = os.path.join("cache", "player_history", f"usage_rows_{season}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                print(f"[load_usage] Loaded {len(data)} players from {cache_path}")
                return data
        except Exception as e:
            print(f"[load_usage] Error reading {cache_path}: {e}")

    # Try recent dates from that season in data directory
    potential_dates = [
        f"{season}-12-31",
        f"{season}-12-30",
        f"{season}-12-29",
        f"{season + 1}-01-01",
        f"{season + 1}-01-02"
    ]

    for date_str in potential_dates:
        path = os.path.join(DATA_DIR, f"usage_table_{date_str}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[load_usage] Error reading {path}: {e}")
                continue

    # Search for any file matching the season
    try:
        for filename in os.listdir(DATA_DIR):
            if filename.startswith(f"usage_table_{season}-") and filename.endswith(".json"):
                path = os.path.join(DATA_DIR, filename)
                with open(path, 'r') as f:
                    return json.load(f)
    except Exception as e:
        print(f"[load_usage] Error searching directory: {e}")

    print(f"[load_usage] WARNING: No usage table found for season {season}")
    return []


def detect_roster_changes_between_seasons(
        current_season: int,
        compare_to_season: Optional[int] = None
) -> List[Dict]:
    """
    Detect roster changes by comparing current player teams to previous season.

    Args:
        current_season: Current season year
        compare_to_season: Season to compare against (defaults to current_season - 1)

    Returns:
        List of detected roster changes
    """
    if compare_to_season is None:
        compare_to_season = current_season - 1

    print(f"[populate_roster_changes] Comparing {compare_to_season} → {current_season}")

    # Load current players index
    current_players = load_players_index() or {}

    # Load previous season usage to get team info
    prev_usage_table = load_usage_table_for_season(compare_to_season) or []

    # Build prev season team lookup
    prev_season_teams = {}
    prev_season_usage = {}
    prev_season_positions = {}

    for player in prev_usage_table:
        # Handle both 'player_id' and 'id' keys
        pid = str(player.get("player_id") or player.get("id", ""))
        team = player.get("team")
        usage = player.get("usage", {})
        position = player.get("position")

        if pid and team:
            prev_season_teams[pid] = team
            prev_season_usage[pid] = usage
            if position:
                prev_season_positions[pid] = position

    # Detect changes
    changes = []

    for pid, current_player in current_players.items():
        current_team = current_player.get("team")
        prev_team = prev_season_teams.get(pid)

        # Skip if no team data
        if not current_team or not prev_team:
            continue

        # Skip if no change
        if current_team == prev_team:
            continue

        # Detect change type
        change_type = "free_agent"  # Default
        if current_team == "FA":
            change_type = "free_agent"
        elif prev_team == "FA":
            change_type = "signing"
        else:
            change_type = "trade"  # Could be trade or FA, hard to distinguish

        # Get usage stats from previous season
        usage = prev_season_usage.get(pid, {})

        # Extract stats - handle both old and new field names
        # New format uses totals and averages
        games = usage.get("games", 1)
        targets = usage.get("targets") or usage.get("total_targets") or (usage.get("avg_targets", 0) * games) or 0
        carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0

        # Snap share: avg_off_snap_pct is already a decimal (0-1), not a percentage (0-100)
        snap_share = usage.get("avg_off_snap_pct") or 0

        # Opportunity share: calculate from usage data if not available
        from data_building.offseason_opportunity import calculate_opportunity_share_from_usage
        opp_share = usage.get("opportunity_share", 0)
        if opp_share == 0:
            opp_share = calculate_opportunity_share_from_usage(usage)

        # Convert to integers
        targets = int(targets)
        carries = int(carries)

        # Only track meaningful departures (had actual usage)
        # Lower threshold to capture more players who could vacate opportunity
        if targets < 5 and carries < 5:
            continue

        # Use position from current player (key is 'pos', not 'position') or previous season data
        position = current_player.get("pos") or prev_season_positions.get(pid, "")

        changes.append({
            "player_id": pid,
            "player_name": current_player.get("name", "Unknown"),
            "position": position,
            "old_team": prev_team,
            "new_team": current_team if current_team != "FA" else None,
            "change_type": change_type,
            "season": current_season,
            "stats": {
                "targets": targets,
                "carries": carries,
                "snap_share": snap_share,  # Already a decimal (0-1), don't divide by 100
                "opportunity_share": opp_share,
                "team_target_pct": None,  # Would need team totals to calculate
                "team_carry_pct": None
            }
        })

    print(f"[populate_roster_changes] Found {len(changes)} roster changes")

    return changes


def populate_offseason_data(season: int):
    """
    Full pipeline: detect roster changes, calculate vacated opportunity, project redistributions.

    Args:
        season: Season to populate offseason data for
    """
    print(f"\n{'=' * 60}")
    print(f"POPULATING OFFSEASON DATA FOR {season}")
    print(f"{'=' * 60}\n")

    # Step 1: Detect roster changes
    print("STEP 1: Detecting roster changes...")
    changes = detect_roster_changes_between_seasons(season)

    # Step 2: Track roster changes in database
    print("\nSTEP 2: Saving roster changes to database...")
    for change in changes:
        track_roster_change(
            player_id=change["player_id"],
            player_name=change["player_name"],
            position=change["position"],
            old_team=change["old_team"],
            new_team=change["new_team"],
            change_type=change["change_type"],
            change_date=date(season, 3, 1),  # Approximate offseason date
            season=season,
            last_season_stats=change["stats"]
        )

    # Step 3: Calculate vacated opportunity
    print("\nSTEP 3: Calculating vacated opportunity by team/position...")
    calculate_vacated_opportunity(season)

    # Step 4: Project opportunity redistribution
    print("\nSTEP 4: Projecting opportunity redistribution (top 600 players only)...")
    project_opportunity_redistribution(season, top_n_players=600)

    print(f"\n{'=' * 60}")
    print(f"✓ OFFSEASON DATA POPULATION COMPLETE")
    print(f"{'=' * 60}\n")


def manual_add_roster_change(
        player_name: str,
        old_team: str,
        new_team: Optional[str],
        change_type: str,
        season: int,
        change_date: Optional[date] = None
):
    """
    Manually add a roster change (for high-profile moves not auto-detected).

    Example:
        manual_add_roster_change(
            player_name="Mike Evans",
            old_team="TB",
            new_team="DAL",
            change_type="free_agent",
            season=2025
        )

    Args:
        player_name: Player name to search for
        old_team: Team they left
        new_team: Team they joined (None for retirement)
        change_type: 'free_agent', 'trade', 'retirement', 'cut'
        season: Season year
        change_date: Date of change (defaults to March 1 of season)
    """
    from utils.utils import load_players_index

    # Find player by name
    players_index = load_players_index() or {}
    player_id = None
    player_obj = None

    for pid, player in players_index.items():
        if player.get("name", "").lower() == player_name.lower():
            player_id = pid
            player_obj = player
            break

    if not player_id:
        print(f"[manual_add] ERROR: Could not find player '{player_name}'")
        return

    # Get previous season usage
    prev_season = season - 1
    usage_table = load_usage_table_for_season(prev_season) or []

    usage_stats = None
    for p in usage_table:
        if str(p.get("player_id")) == player_id:
            usage = p.get("usage", {})

            # Snap share: avg_off_snap_pct is already a decimal (0-1), not a percentage (0-100)
            snap_share = usage.get("avg_off_snap_pct") or 0

            # Opportunity share: calculate from usage data if not available
            from data_building.offseason_opportunity import calculate_opportunity_share_from_usage
            opp_share = usage.get("opportunity_share", 0)
            if opp_share == 0:
                opp_share = calculate_opportunity_share_from_usage(usage)

            # Calculate total targets and carries
            games = usage.get("games", 1)
            targets = usage.get("targets") or usage.get("total_targets") or (usage.get("avg_targets", 0) * games) or 0
            carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0

            usage_stats = {
                "targets": int(targets),
                "carries": int(carries),
                "snap_share": snap_share,  # Already a decimal (0-1)
                "opportunity_share": opp_share
            }
            break

    if not usage_stats:
        print(f"[manual_add] WARNING: No usage stats found for {player_name} in {prev_season}")
        usage_stats = {}

    # Track the change
    track_roster_change(
        player_id=player_id,
        player_name=player_name,
        position=player_obj.get("pos", ""),
        old_team=old_team,
        new_team=new_team,
        change_type=change_type,
        change_date=change_date or date(season, 3, 1),
        season=season,
        last_season_stats=usage_stats
    )

    print(f"[manual_add] ✓ Added {player_name}: {old_team} → {new_team or 'FA'} ({change_type})")
    print(f"[manual_add]   Stats: {usage_stats.get('targets', 0)} targets, {usage_stats.get('carries', 0)} carries")


def populate_draft_picks(season: int, draft_data: Optional[List[Dict]] = None):
    """
    Import NFL draft picks as roster changes.

    After the NFL draft, this function adds draft picks to the roster_changes table
    with draft_metadata (round, pick, college). This allows drafted rookies to:
    1. Create competition_added_penalty for existing players
    2. Get boosted player_readiness_score from draft capital

    Args:
        season: Season year
        draft_data: Optional list of draft pick dictionaries.
                   If None, will attempt to fetch from Sleeper API.
                   Each dict should have:
                   - player_id
                   - player_name
                   - position
                   - team
                   - round
                   - pick (overall pick number)
                   - college (optional)
                   - draft_date (optional)

    Example:
        draft_data = [
            {
                'player_id': '11625',
                'player_name': 'Jalen McMillan',
                'position': 'WR',
                'team': 'TB',
                'round': 3,
                'pick': 89,
                'college': 'Washington'
            }
        ]
        populate_draft_picks(2025, draft_data)
    """
    from utils.utils import load_players_index

    print(f"\n{'=' * 60}")
    print(f"POPULATING DRAFT PICKS FOR {season}")
    print(f"{'=' * 60}\n")

    if draft_data is None:
        # TODO: Fetch from Sleeper API or other source
        # For now, this would need to be called manually with draft data
        print("[populate_draft_picks] WARNING: No draft data provided.")
        print("[populate_draft_picks] Please call with draft_data parameter containing draft pick info.")
        return

    players_index = load_players_index() or {}
    draft_count = 0

    for pick in draft_data:
        player_id = pick.get('player_id')
        player_name = pick.get('player_name')
        position = pick.get('position')
        team = pick.get('team')
        round_num = pick.get('round')
        pick_num = pick.get('pick')
        college = pick.get('college')
        draft_date = pick.get('draft_date', date(season, 4, 26))  # Default to late April

        if not all([player_id, player_name, position, team, round_num, pick_num]):
            print(f"[populate_draft_picks] WARNING: Skipping incomplete draft pick: {pick}")
            continue

        # Create draft metadata
        draft_metadata = {
            'round': round_num,
            'pick': pick_num,
            'overall_pick': pick_num,
            'college': college
        }

        # Track as roster change
        track_roster_change(
            player_id=player_id,
            player_name=player_name,
            position=position,
            old_team=None,  # Rookies have no old team
            new_team=team,
            change_type='draft',
            change_date=draft_date,
            season=season,
            last_season_stats={},  # Rookies have no previous season stats
            draft_metadata=draft_metadata
        )

        draft_count += 1
        print(f"[populate_draft_picks] ✓ Added {player_name} to {team} (Round {round_num}, Pick {pick_num})")

    print(f"\n{'=' * 60}")
    print(f"✓ DRAFT PICKS POPULATION COMPLETE: {draft_count} picks added")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python populate_roster_changes.py <season>")
        print("Example: python populate_roster_changes.py 2025")
        sys.exit(1)

    season = int(sys.argv[1])

    # Initialize database first
    from data_building.offseason_opportunity import init_offseason_opportunity_db

    init_offseason_opportunity_db()

    # Populate offseason data
    populate_offseason_data(season)
