"""
Reworked modular breakout workflow.

This module implements the new 4-step workflow:
1. Detect and store roster changes
2. Calculate and store vacated opportunity from DB
3. Calculate and store breakout scores from DB
4. Calculate and store projections from DB

Each step is independent, testable, and builds on stored database results.
"""

import json
from datetime import date
from typing import Dict, List, Tuple, Any

from dashboard_services.db import get_conn
from dashboard_services.service import age_from_bday
from data_building.breakout_engine import BreakoutEngine
from data_building.breakout_engine.calculate_breakouts_with_real_data import load_season_aware_usage_data
from data_building.offseason_opportunity import track_roster_change
from data_building.populate_roster_changes import detect_roster_changes_between_seasons, load_usage_table_for_season
from utils.utils import load_players_index


def detect_and_store_roster_changes(season: int) -> int:
    """
    Step 1: Detect roster changes and store to database.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of roster changes stored
    """
    print(f"[workflow] 🔍 Step 1: Detecting roster changes for {season}")

    # Detect changes between seasons
    changes = detect_roster_changes_between_seasons(season)

    if not changes:
        print(f"[workflow] No roster changes detected for {season}")
        return 0

    # Store changes to database
    stored_count = 0
    for change in changes:
        try:
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
            stored_count += 1
        except Exception as e:
            print(f"[workflow] Error storing roster change for {change.get('player_name', 'unknown')}: {e}")

    print(f"[workflow] 📝 Stored {stored_count} roster changes to database")
    return stored_count


def calculate_and_store_vacated_opportunity(season: int) -> int:
    """
    Step 2: Calculate vacated opportunity from stored roster changes.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of team/position combinations with vacated opportunity
    """
    print(f"[workflow] 🧹 Step 2: Calculating vacated opportunity from database")

    with get_conn() as conn:
        # Get all departures by team/position
        departures = conn.execute("""
            SELECT old_team, position,
                   SUM(last_season_targets) as total_targets_vacated,
                   SUM(last_season_carries) as total_carries_vacated,
                   SUM(last_season_snap_share) as total_snap_share_vacated,
                   SUM(last_season_opportunity_share) as total_opportunity_share_vacated,
                   JSON_AGG(json_build_object(
                       'player_id', player_id,
                       'player_name', player_name,
                       'change_type', change_type,
                       'targets', last_season_targets,
                       'carries', last_season_carries,
                       'snap_share', last_season_snap_share,
                       'opportunity_share', last_season_opportunity_share
                   )) as departed_players
            FROM roster_changes 
            WHERE season = %s 
              AND change_type IN ('trade', 'free_agent', 'retirement', 'cut')
              AND old_team IS NOT NULL
              AND old_team != ''
            GROUP BY old_team, position
            HAVING SUM(last_season_targets) > 0 OR SUM(last_season_carries) > 0
        """, (season,)).fetchall()

        if not departures:
            print(f"[workflow] No vacated opportunity found for {season}")
            return 0

        # Store vacated opportunity
        stored_count = 0
        for departure in departures:
            try:
                conn.execute("""
                    INSERT INTO vacated_opportunity (
                        team, position, season,
                        total_targets_vacated, total_carries_vacated,
                        total_snap_share_vacated, total_opportunity_share_vacated,
                        departed_players, calculated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (team, position, season)
                    DO UPDATE SET
                        total_targets_vacated = EXCLUDED.total_targets_vacated,
                        total_carries_vacated = EXCLUDED.total_carries_vacated,
                        total_snap_share_vacated = EXCLUDED.total_snap_share_vacated,
                        total_opportunity_share_vacated = EXCLUDED.total_opportunity_share_vacated,
                        departed_players = EXCLUDED.departed_players,
                        calculated_at = NOW()
                """, (
                    departure['old_team'], departure['position'], season,
                    departure['total_targets_vacated'], departure['total_carries_vacated'],
                    departure['total_snap_share_vacated'], departure['total_opportunity_share_vacated'],
                    json.dumps(departure['departed_players'])
                ))
                stored_count += 1
            except Exception as e:
                print(
                    f"[workflow] Error storing vacated opportunity for {departure.get('old_team', 'unknown')} {departure.get('position', 'unknown')}: {e}")

        conn.commit()
        print(f"[workflow] 🧹 Stored vacated opportunity for {stored_count} team/position combinations")
        return stored_count


def calculate_and_store_breakout_scores(season: int, week: int, nfl_state: dict) -> int:
    """
    Step 3: Calculate breakout scores using vacated opportunity from database.
    
    Args:
        season: Season year to analyze
        week: Current week (for season-aware data loading)
        
    Returns:
        Number of breakout scores calculated and stored
    """
    print(f"[workflow] 🎯 Step 3: Calculating breakout scores from database")

    # Initialize breakout engine
    engine = BreakoutEngine(season=season, as_of_date=date.today())
    season_type = str(nfl_state.get("season_type", "off"))

    # Load players and usage data (same as before)
    players_index = load_players_index() or {}
    from data_building.breakout_engine.calculate_breakouts_with_real_data import apply_candidate_filter, \
        build_usage_maps

    # Load season-aware usage data
    usage_table = load_season_aware_usage_data(season, week, season_type)
    usage_by_id, age_by_id = build_usage_maps(usage_table)

    # Build all players list
    all_players = []
    for player_id, player_data in players_index.items():
        pos = player_data.get('pos')
        team = player_data.get('team')

        if pos in ["QB", "RB", "WR", "TE"] and team:
            age = age_from_bday(player_data.get("bDay"))

            if age is not None and age < 26:
                years_exp = max(0, int(age - 21.5))

                all_players.append({
                    "player_id": player_id,
                    "player_name": player_data.get("name", "Unknown"),
                    "team": team,
                    "position": pos,
                    "age": age,
                    "years_exp": years_exp,
                })

    # Apply candidate filters
    filtered_candidates, filter_summary = apply_candidate_filter(all_players, usage_by_id)
    print(f"[workflow] Candidate filtering: {filter_summary}")

    # Calculate breakout scores
    candidates = engine.calculate_breakout_scores(filtered_candidates, min_score=0)

    # Store to database
    saved_count = engine.save_scores(candidates)
    high_score_count = sum(1 for c in candidates if getattr(c, 'breakout_opportunity_score', 0) >= 40)

    print(f"[workflow] 🎯 Stored {saved_count} breakout scores ({high_score_count} high-score candidates)")
    return saved_count


def calculate_and_store_projections(season: int) -> int:
    """
    Step 4: Calculate projections using breakout scores and vacated opportunity.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of projections calculated and stored
    """
    print(f"[workflow] 📈 Step 4: Calculating projections from database")

    # Load data from database
    with get_conn() as conn:
        # Get vacated opportunity
        vacated_data = conn.execute("""
            SELECT team, position, total_targets_vacated, total_carries_vacated,
                   total_snap_share_vacated, total_opportunity_share_vacated,
                   departed_players
            FROM vacated_opportunity 
            WHERE season = %s
        """, (season,)).fetchall()

        # Get breakout scores
        breakout_data = conn.execute("""
            SELECT player_id, breakout_opportunity_score, team, position
            FROM breakout_opportunity_scores 
            WHERE season = %s
        """, (season,)).fetchall()

    if not vacated_data:
        print(f"[workflow] No vacated opportunity found for projections")
        return 0

    # Build lookup tables
    vacated_by_team_pos = {(v['team'], v['position']): v for v in vacated_data}
    breakout_by_player = {b['player_id']: b for b in breakout_data}

    # Load player data for projections

    prev_season = season - 1
    usage_table = load_usage_table_for_season(prev_season) or []
    usage_by_player = {str(p.get('player_id') or p.get('id', '')): p for p in usage_table}

    # Calculate projections for high-score breakout candidates
    projections = []
    for breakout in breakout_data:
        if breakout['breakout_opportunity_score'] < 30:  # Only significant opportunities
            continue

        player_id = breakout['player_id']
        team = breakout['team']
        position = breakout['position']

        # Get player's previous usage
        player_usage = usage_by_player.get(player_id, {})
        usage = player_usage.get('usage', {})
        games = usage.get('games', 1) or 1

        prev_targets = int(
            usage.get('targets') or usage.get('total_targets') or (usage.get('avg_targets', 0) * games) or 0)
        prev_carries = int(usage.get('carries') or (usage.get('avg_carries', 0) * games) or 0)

        # Snap share: avg_off_snap_pct is already a decimal (0-1), not a percentage (0-100)
        prev_snap_share = usage.get('avg_off_snap_pct') or 0

        # Opportunity share: calculate from usage data
        from data_building.offseason_opportunity import calculate_opportunity_share_from_usage
        prev_opp_share = usage.get('opportunity_share', 0)
        if prev_opp_share == 0:
            prev_opp_share = calculate_opportunity_share_from_usage(usage)

        # Calculate opportunity increases
        vacated = vacated_by_team_pos.get((team, position), {})
        targets_vacated = float(vacated.get('total_targets_vacated', 0))
        carries_vacated = float(vacated.get('total_carries_vacated', 0))

        # Simple projection: give them a share of vacated opportunity based on breakout score
        breakout_score = float(breakout['breakout_opportunity_score'])
        opportunity_share = min(breakout_score / 100, 1.0)  # Cap at 100%

        target_increase = int(targets_vacated * opportunity_share * 0.5)  # Conservative estimate
        carry_increase = int(carries_vacated * opportunity_share * 0.5)

        projected_targets = prev_targets + target_increase
        projected_carries = prev_carries + carry_increase
        projected_snap_share = min(prev_snap_share + (opportunity_share * 0.1), 1.0)

        # Get player name
        player_name = "Unknown"
        for p in usage_table:
            if str(p.get('player_id') or p.get('id', '')) == player_id:
                player_name = p.get('player_name') or p.get('name', 'Unknown')
                break

        projections.append({
            "player_id": player_id,
            "player_name": player_name,
            "season": season,
            "team": team,
            "position": position,
            "prev_season_targets": prev_targets,
            "prev_season_carries": prev_carries,
            "prev_season_snap_share": prev_snap_share,
            "prev_season_opportunity_share": prev_opp_share,  # FIXED: Use calculated value
            "projected_targets": projected_targets,
            "projected_carries": projected_carries,
            "projected_snap_share": projected_snap_share,
            "projected_opportunity_share": opportunity_share,
            "target_increase": target_increase,
            "carry_increase": carry_increase,
            "snap_share_increase": projected_snap_share - prev_snap_share,
            "opportunity_share_increase": opportunity_share,
            "breakout_score": breakout_score,
            "projection_factors": json.dumps({
                "method": "from_breakout_scores",
                "breakout_score": breakout_score,
                "vacated_targets": targets_vacated,
                "vacated_carries": carries_vacated,
                "opportunity_share": opportunity_share
            })
        })

    # Store projections
    if projections:
        with get_conn() as conn:
            for proj in projections:
                conn.execute("""
                    INSERT INTO projected_opportunity (
                        player_id, player_name, season, team, position,
                        prev_season_targets, prev_season_carries,
                        prev_season_snap_share, prev_season_opportunity_share,
                        projected_targets, projected_carries, projected_snap_share,
                        projected_opportunity_share,
                        target_increase, carry_increase, snap_share_increase,
                        opportunity_share_increase,
                        breakout_score, projection_factors
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (player_id, season)
                    DO UPDATE SET
                        player_name = EXCLUDED.player_name,
                        projected_targets = EXCLUDED.projected_targets,
                        projected_carries = EXCLUDED.projected_carries,
                        projected_snap_share = EXCLUDED.projected_snap_share,
                        projected_opportunity_share = EXCLUDED.projected_opportunity_share,
                        target_increase = EXCLUDED.target_increase,
                        carry_increase = EXCLUDED.carry_increase,
                        snap_share_increase = EXCLUDED.snap_share_increase,
                        opportunity_share_increase = EXCLUDED.opportunity_share_increase,
                        breakout_score = EXCLUDED.breakout_score,
                        projection_factors = EXCLUDED.projection_factors,
                        calculated_at = NOW()
                """, (
                    proj["player_id"], proj.get("player_name"), proj["season"], proj["team"], proj["position"],
                    proj["prev_season_targets"], proj["prev_season_carries"],
                    proj["prev_season_snap_share"], proj["prev_season_opportunity_share"],
                    proj["projected_targets"], proj["projected_carries"],
                    proj["projected_snap_share"], proj["projected_opportunity_share"],
                    proj["target_increase"], proj["carry_increase"],
                    proj["snap_share_increase"], proj["opportunity_share_increase"],
                    proj["breakout_score"], proj["projection_factors"]
                ))
            conn.commit()

    print(f"[workflow] 📈 Stored {len(projections)} opportunity projections")
    return len(projections)


def run_modular_breakout_workflow(season: int, week: int, state: dict) -> bool:
    """
    Run the complete modular breakout workflow.
    
    Args:
        season: Season year to analyze
        week: Current week
        
    Returns:
        True if workflow completed successfully, False otherwise
    """
    print(f"[workflow] 🚀 Starting modular breakout workflow for season={season}, week={week}")

    try:
        # Step 1: Detect and store roster changes
        changes_count = detect_and_store_roster_changes(season)

        # Step 2: Calculate and store vacated opportunity
        vacated_count = calculate_and_store_vacated_opportunity(season)

        # Step 3: Calculate and store breakout scores
        scores_count = calculate_and_store_breakout_scores(season, week, state)

        # Step 4: Calculate and store projections
        proj_count = calculate_and_store_projections(season)

        print(f"[workflow] ✅ Workflow completed:")
        print(f"[workflow]   - {changes_count} roster changes")
        print(f"[workflow]   - {vacated_count} vacated opportunity groups")
        print(f"[workflow]   - {scores_count} breakout scores")
        print(f"[workflow]   - {proj_count} projections")

        return True

    except Exception as e:
        print(f"[workflow] ❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
