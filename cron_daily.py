from datetime import date

from data_building.breakout_engine.calculate_breakouts_with_real_data import apply_candidate_filter, build_usage_maps
from dashboard_services.api import get_nfl_state
from dashboard_services.service import age_from_bday
from data_building.build_daily_value_table import build_daily_data, build_daily_market_pulse
from data_building.external_data.player_history import usage_rows_json_path_for_season
from utils.utils import read_json


def build_daily_advanced_metrics(season: int, week: int):
    """
    Calculate and save advanced efficiency metrics for all players.

    This runs after build_daily_data() to ensure we have fresh usage data.
    In offseason (when current usage is empty), uses most recent available data.
    """
    from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
    from utils.utils import load_usage_table
    from dashboard_services.api import get_nfl_state

    print(f"[cron] Calculating advanced metrics for season={season}, week={week}")

    try:
        # Load latest usage data
        usage_table = load_usage_table()
        if not usage_table:
            print("[cron] No usage table found, skipping advanced metrics")
            return

        # Check if we're in offseason (no one has games played)
        nfl_state = get_nfl_state() or {}
        season_type = str(nfl_state.get("season_type", "")).lower().strip()
        is_offseason = season_type == "off"

        players_with_games = sum(1 for p in usage_table if p.get("usage", {}).get("games", 0) > 0)

        if players_with_games == 0 and is_offseason:
            print(f"[cron] Offseason detected - no current usage data available")
            print(f"[cron] Advanced metrics will use last available data when season starts")
            return

        # Calculate metrics for each player
        metrics_list = []
        for player in usage_table:
            player_id = player.get("id")
            position = player.get("position")
            usage = player.get("usage", {})

            if not player_id or not position:
                continue

            # Skip players with no usage data (but don't fail)
            if not usage or usage.get("games", 0) == 0:
                continue

            try:
                metrics = calculate_player_metrics(player_id, usage, position)
                metrics_list.append(metrics)
            except Exception as e:
                print(f"[cron] Failed to calculate metrics for player {player_id}: {e}")

        # Save all metrics to database
        if metrics_list:
            today = date.today().isoformat()
            save_metrics_snapshot(metrics_list, today)
            print(f"[cron] Saved {len(metrics_list)} player metrics")
        else:
            print("[cron] No metrics calculated (no players with usage data)")

    except Exception as e:
        print(f"[cron] Advanced metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()


def build_daily_breakout_candidates(season: int, week: int):
    """
    Calculate breakout candidates using the unified breakout engine.

    This runs daily to:
    1. Track roster changes (trades, signings, cuts, retirements)
    2. Calculate vacated opportunity from departures
    3. Calculate unified breakout scores for all fantasy-relevant players
    4. Save scores to database for API consumption

    Runs during:
    - Offseason and preseason (projections based on roster moves)
    - First 4 weeks of regular season (combo of projections + early actual data)
    - After week 4, actual usage data is reliable enough without projections
    """
    from data_building.offseason_opportunity import (
        init_offseason_opportunity_db,
        calculate_vacated_opportunity
    )
    from data_building.breakout_engine import BreakoutEngine
    from utils.utils import load_players_index

    # Check if we should run breakout calculations
    nfl_state = get_nfl_state() or {}
    season_type = str(nfl_state.get("season_type", "")).lower().strip()

    # Run during offseason, preseason, or early regular season (weeks 1-4)
    should_run = (
        season_type in ["off", "pre"] or
        (season_type == "regular" and week <= 9)
    )

    if not should_run:
        print(f"[cron] Skipping breakout calculations - season_type={season_type}, week={week} (only runs during offseason/preseason/weeks 1-9)")
        return

    print(f"[cron] Calculating breakout candidates for season={season}, week={week}")

    try:
        # Step 1: Initialize database tables (safe to call multiple times)
        print(f"[cron] Initializing database tables...")
        init_offseason_opportunity_db()

        # Step 2: Track roster changes and calculate vacated opportunity
        print(f"[cron] Tracking roster changes and calculating vacated opportunity...")
        calculate_vacated_opportunity(season)

        # Step 3: Calculate unified breakout scores using the new engine
        print(f"[cron] Calculating unified breakout scores...")
        engine = BreakoutEngine(season=season, as_of_date=date.today())

        # Load all fantasy-relevant players with season-aware age data
        import json
        import os

        players_index = load_players_index() or {}

        # Load season-aware usage data
        def load_season_aware_usage(season, week, season_type):
            """Load appropriate usage data based on season phase."""
            season_type = season_type.lower().strip()
            is_offseason = season_type in ['off', 'pre']
            is_early_season = season_type == 'regular' and week <= 7

            # Offseason: use last season
            if is_offseason:
                last_season_file = f"cache/player_history/usage_rows_{season - 1}.json"
                if os.path.exists(last_season_file):
                    print(f"[cron] Loading {season - 1} usage data (offseason)")
                    return read_json(usage_rows_json_path_for_season(season - 1))
                return []

            # Early season: blend last + current
            elif is_early_season:
                print(f"[cron] Blending {season - 1} + {season} usage data (early season week {week})")
                from utils.utils import load_usage_table

                current_usage = load_usage_table() or []
                last_season_file = f"cache/player_history/usage_rows_{season - 1}.json"

                if os.path.exists(last_season_file):
                    with open(last_season_file, 'r') as f:
                        last_season_usage = read_json(usage_rows_json_path_for_season(season - 1))
                else:
                    last_season_usage = []

                # Merge: baseline from last season, overlay current season
                merged_by_id = {}
                for player in last_season_usage:
                    player_id = str(player.get('id'))
                    if player_id:
                        merged_by_id[player_id] = player

                for player in current_usage:
                    player_id = str(player.get('id'))
                    games = player.get('usage', {}).get('games', 0)
                    if player_id and games > 0:
                        merged_by_id[player_id] = player

                return list(merged_by_id.values())

            # Mid/late season: current only
            else:
                print(f"[cron] Loading {season} current usage data (mid/late season week {week})")
                from utils.utils import load_usage_table
                return load_usage_table() or []

        usage_table = load_season_aware_usage(season, week, season_type)
        usage_by_id, age_by_id = build_usage_maps(usage_table)

        # OFFSEASON OPPORTUNITY CALCULATIONS
        if season_type.lower() in ['off', 'pre']:
            print(f"[cron] 🧹 Cleaning up previous day's data for fresh calculations...")
            
            # Clean up previous day's data for fresh calculations
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                # Delete previous day's breakout scores
                deleted_scores = conn.execute("""
                    DELETE FROM breakout_opportunity_scores 
                    WHERE season = %s
                """, (season,)).rowcount
                
                # Delete previous day's projected opportunities
                deleted_projections = conn.execute("""
                    DELETE FROM projected_opportunity 
                    WHERE season = %s
                """, (season,)).rowcount
                
                # Delete previous day's roster changes (will be recalculated)
                deleted_changes = conn.execute("""
                    DELETE FROM roster_changes 
                    WHERE season = %s
                """, (season,)).rowcount
                
                # Delete previous day's vacated opportunity (will be recalculated)
                deleted_vacated = conn.execute("""
                    DELETE FROM vacated_opportunity 
                    WHERE season = %s
                """, (season,)).rowcount
                
                conn.commit()
                total_cleaned = deleted_scores + deleted_projections + deleted_changes + deleted_vacated
                print(f"[cron] ✅ Cleaned up {total_cleaned} previous records: {deleted_scores} scores, {deleted_projections} projections, {deleted_changes} roster changes, {deleted_vacated} vacated opportunities")
            
            print(f"[cron] 🔄 Running fresh offseason opportunity calculations...")
            
            # Initialize database tables
            from data_building.offseason_opportunity import init_offseason_opportunity_db
            init_offseason_opportunity_db()
            
            # Detect roster changes between seasons
            from data_building.populate_roster_changes import detect_roster_changes_between_seasons
            print(f"[cron] Detecting roster changes for {season}...")
            changes = detect_roster_changes_between_seasons(season)
            print(f"[cron] Detected {len(changes)} roster changes")
            
            # Save roster changes to database
            if changes:
                from data_building.offseason_opportunity import track_roster_change
                print(f"[cron] Saving roster changes to database...")
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
                print(f"[cron] ✓ Saved {len(changes)} roster changes to database")
            else:
                print(f"[cron] No roster changes to save")
            
            # Calculate vacated opportunity
            from data_building.offseason_opportunity import calculate_vacated_opportunity
            print(f"[cron] Calculating vacated opportunity...")
            calculate_vacated_opportunity(season)
            
            # Project opportunity redistribution and calculate increases
            from data_building.offseason_opportunity import project_opportunity_redistribution
            print(f"[cron] Projecting opportunity redistribution...")
            project_opportunity_redistribution(season, top_n_players=600)
            print(f"[cron] ✓ Offseason opportunity calculations completed")
        else:
            print(f"[cron] Skipping offseason calculations (in-season)")
            
            # For in-season, still clean up previous day's unified breakout scores for fresh data
            print(f"[cron] 🧹 Cleaning up previous day's unified breakout scores...")
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                deleted_scores = conn.execute("""
                    DELETE FROM breakout_opportunity_scores 
                    WHERE season = %s
                """, (season,)).rowcount
                conn.commit()
                print(f"[cron] ✅ Cleaned up {deleted_scores} previous unified breakout scores")

        # Build age lookup from usage table
        age_by_id = {}
        all_players = []
        for player_id, player_data in players_index.items():
            pos = player_data.get('pos')
            team = player_data.get('team')

            if pos in ['QB', 'RB', 'WR', 'TE'] and team:
                # Get age from usage table, fallback to players_index
                age = age_from_bday(player_data.get('bDay'))
                if age:
                    age_by_id[player_id] = age
                all_players.append({
                    'player_id': player_id,
                    'player_name': player_data.get('name', 'Unknown'),
                    'team': team,
                    'position': pos,
                    'age': age,
                    'years_exp': player_data.get('years_exp', 0)
                })

        print(f"[cron] Loaded {len(all_players)} fantasy-relevant players")
        players_with_age = sum(1 for p in all_players if p.get('age'))
        print(f"[cron] Players with age data: {players_with_age}/{len(all_players)}")

        # Calculate scores for all players (min_score=0 to save all)
        filtered_candidates, filter_summary = apply_candidate_filter(all_players, usage_by_id)
        print(filter_summary)

        candidates = engine.calculate_breakout_scores(filtered_candidates, min_score=0)
        # Step 4: Save scores to database
        saved_count = engine.save_scores(candidates)
        print(f"[cron] ✓ Saved {saved_count} breakout scores to database")

        # Show top candidates
        candidates.sort(key=lambda x: x.breakout_opportunity_score, reverse=True)
        print(f"[cron] Top 5 breakout candidates:")
        for i, c in enumerate(candidates[:5], 1):
            print(f"[cron]   {i}. {c.player_name} ({c.position}, {c.team}) - Score: {c.breakout_opportunity_score:.1f}")

        # Show offseason opportunity candidates (if offseason)
        if season_type.lower() in ['off', 'pre']:
            from data_building.offseason_opportunity import get_offseason_breakout_candidates_legacy
            print(f"[cron] Top 5 offseason opportunity candidates:")
            offseason_candidates = get_offseason_breakout_candidates_legacy(season, min_score=30, top_n_players=5)
            for i, c in enumerate(offseason_candidates[:5], 1):
                targets_inc = c.get('target_increase', 0)
                carries_inc = c.get('carry_increase', 0)
                inc_text = []
                if targets_inc > 0:
                    inc_text.append(f"+{targets_inc} targets")
                if carries_inc > 0:
                    inc_text.append(f"+{carries_inc} carries")
                inc_str = f" ({', '.join(inc_text)})" if inc_text else ""
                print(f"[cron]   {i}. {c['player_name']} ({c['position']}, {c['team']}) - Score: {c['breakout_score']:.1f}{inc_str}")

        print(f"[cron] Breakout candidates calculated successfully")

    except Exception as e:
        print(f"[cron] Breakout candidates calculation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] Running daily for season={season}, week={week}")

    # run your existing function
    build_daily_data(season, week)

    # Save player values to database for historical tracking
    try:
        from data_building.save_player_values import save_daily_values_to_db
        from utils.utils import load_model_value_table

        value_table = load_model_value_table()
        if value_table:
            count = save_daily_values_to_db(value_table)
            print(f"[daily] Saved {count} player values to database")
        else:
            print("[daily] No value table available, skipping database save")
    except Exception as e:
        print(f"[daily] player values save skipped: {e}")

    # Calculate advanced metrics from usage data
    try:
        build_daily_advanced_metrics(season, week)
    except Exception as e:
        print(f"[daily] advanced metrics skipped: {e}")

    # Build market pulse
    try:
        build_daily_market_pulse()
    except Exception as e:
        print(f"[daily] market pulse skipped: {e}")

    # Calculate offseason breakout candidates
    try:
        build_daily_breakout_candidates(season, week)
    except Exception as e:
        print(f"[daily] breakout candidates skipped: {e}")


if __name__ == "__main__":
    main()
