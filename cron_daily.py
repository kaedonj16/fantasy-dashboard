from datetime import date

from data_building.breakout_engine.calculate_breakouts_with_real_data import apply_candidate_filter, build_usage_maps
from dashboard_services.api import get_nfl_state
from dashboard_services.service import age_from_bday
from data_building.build_daily_value_table import build_daily_data, build_daily_market_pulse
from data_building.external_data.player_history import usage_rows_json_path_for_season
from utils.utils import read_json
from data_building.breakout_workflow import run_modular_breakout_workflow


def build_daily_advanced_metrics(season: int, week: int):
    """
    Calculate and save advanced efficiency metrics for all players.

    This runs after build_daily_data() to ensure we have fresh usage data.
    In offseason (when current usage is empty), uses most recent available data.
    """
    from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
    from utils.utils import load_usage_table
    from dashboard_services.api import get_nfl_state

    try:
        # Load latest usage data
        usage_table = load_usage_table()
        if not usage_table:
            print("[cron] ⚠️  No usage table found, skipping advanced metrics")
            return

        # Check if we're in offseason (no one has games played)
        nfl_state = get_nfl_state() or {}
        season_type = str(nfl_state.get("season_type", "")).lower().strip()
        is_offseason = season_type == "off"

        players_with_games = sum(1 for p in usage_table if p.get("usage", {}).get("games", 0) > 0)

        if players_with_games == 0 and is_offseason:
            print(f"[cron] 📅 Offseason detected - {len(usage_table)} players loaded, will use last available data when season starts")
            return

        # Calculate metrics for each player
        metrics_list = []
        failed_count = 0
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
                failed_count += 1

        # Save all metrics to database
        if metrics_list:
            today = date.today().isoformat()
            save_metrics_snapshot(metrics_list, today)
            print(f"[cron] 📊 Advanced metrics: {len(metrics_list)} players processed, {failed_count} failed")
        else:
            print(f"[cron] 📊 Advanced metrics: No metrics calculated (no players with usage data)")

    except Exception as e:
        print(f"[cron] ❌ Advanced metrics failed: {e}")
        import traceback
        traceback.print_exc()


def build_daily_breakout_candidates(season: int, week: int, nfl_state: dict):
    """
    Calculate breakout candidates using new modular workflow.
    
    This runs daily to execute 4-step modular workflow:
    1. Detect and store roster changes
    2. Calculate and store vacated opportunity from DB
    3. Calculate and store breakout scores from DB
    4. Calculate and store projections from DB
    
    Runs during:
    - Offseason and preseason (projections based on roster moves)
    - First 9 weeks of regular season (combo of projections + early actual data)
    """
    from data_building.breakout_workflow import run_modular_breakout_workflow
    
    # Check if we should run breakout calculations
    season_type = str(nfl_state.get("season_type", "")).lower().strip()

    # Run during offseason, preseason, or early regular season (weeks 1-9)
    should_run = (
        season_type in ["off", "pre"] or
        (season_type == "regular" and week <= 9)
    )

    if not should_run:
        print(f"[cron] ⏸️  Breakout calculations skipped - season_type={season_type}, week={week} (only runs offseason/preseason/weeks 1-9)")
        return

    print(f"[cron] 🚀 Starting modular breakout workflow for season={season}, week={week}")
    
    # Clean up previous day's data for fresh calculations
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        deleted_scores = conn.execute("""
            DELETE FROM breakout_opportunity_scores 
            WHERE season = %s AND as_of_date = CURRENT_DATE
        """, (season,)).rowcount
        
        deleted_projections = conn.execute("""
            DELETE FROM projected_opportunity 
            WHERE season = %s AND calculated_at::date = CURRENT_DATE
        """, (season,)).rowcount
        
        deleted_changes = conn.execute("""
            DELETE FROM roster_changes 
            WHERE season = %s AND created_at::date = CURRENT_DATE
        """, (season,)).rowcount
        
        deleted_vacated = conn.execute("""
            DELETE FROM vacated_opportunity 
            WHERE season = %s AND calculated_at::date = CURRENT_DATE
        """, (season,)).rowcount
        
        conn.commit()
        total_cleaned = deleted_scores + deleted_projections + deleted_changes + deleted_vacated
        if total_cleaned > 0:
            print(f"[cron] 🧹 Cleaned {total_cleaned} previous records: {deleted_scores} scores, {deleted_projections} projections, {deleted_changes} changes, {deleted_vacated} vacated")
    
    # Run the modular workflow
    success = run_modular_breakout_workflow(season, week, nfl_state)
    
    if success:
        print(f"[cron] ✅ Modular breakout workflow completed successfully")
        
        # Show top candidates from database
        with get_conn() as conn:
            top_candidates = conn.execute("""
                SELECT player_name, position, team, breakout_opportunity_score
                FROM breakout_opportunity_scores 
                WHERE season = %s AND as_of_date = CURRENT_DATE
                ORDER BY breakout_opportunity_score DESC
                LIMIT 5
            """, (season,)).fetchall()
            
            if top_candidates:
                print(f"[cron] 🏆 Top 5 breakout candidates:")
                for i, c in enumerate(top_candidates, 1):
                    print(f"[cron]   {i}. {c['player_name']} ({c['position']}, {c['team']}) - Score: {c['breakout_opportunity_score']:.1f}")
            
            # Show top projections (if offseason)
            if season_type.lower() in ['off', 'pre']:
                top_projections = conn.execute("""
                    SELECT player_name, position, team, target_increase, carry_increase, breakout_score
                    FROM projected_opportunity 
                    WHERE season = %s AND calculated_at::date = CURRENT_DATE
                    ORDER BY breakout_score DESC
                    LIMIT 5
                """, (season,)).fetchall()
                
                if top_projections:
                    print(f"[cron] 🌟 Top 5 opportunity projections:")
                    for i, p in enumerate(top_projections, 1):
                        inc_text = []
                        if p.get('target_increase', 0) > 0:
                            inc_text.append(f"+{p['target_increase']} targets")
                        if p.get('carry_increase', 0) > 0:
                            inc_text.append(f"+{p['carry_increase']} carries")
                        inc_str = f" ({', '.join(inc_text)})" if inc_text else ""
                        print(f"[cron]   {i}. {p['player_name']} ({p['position']}, {p['team']}) - Score: {p['breakout_score']:.1f}{inc_str}")
    else:
        print(f"[cron] ❌ Modular breakout workflow failed")


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] 🌅 Daily run starting - Season {season}, Week {week}")

    # Step 1: Build usage table and vendor values
    print(f"[cron] 📊 Step 1: Building usage table and vendor values")
    build_daily_data(season, week)

    # Step 2: Calculate advanced metrics from usage data
    print(f"[cron] 📊 Step 2: Calculating advanced metrics")
    build_daily_advanced_metrics(season, week)

    # Step 3: Build model values (uses advanced metrics)
    print(f"[cron] 📊 Step 3: Building model values")
    from data_building.build_daily_value_table import build_daily_model_values
    build_daily_model_values()

    # Step 4: Save to database
    print(f"[cron] 📊 Step 4: Saving to database")
    from data_building.save_player_values import save_daily_values_to_db
    from utils.utils import load_model_value_table

    value_table = load_model_value_table()
    if not value_table:
        raise RuntimeError("No value table available after build_daily_model_values")

    value_count = save_daily_values_to_db(value_table)
    print(f"[cron] 💰 Saved {value_count} player values")

    # Run remaining calculations
    build_daily_market_pulse()
    build_daily_breakout_candidates(season, week, state)

    print(f"[cron] ✅ Daily run completed - Season {season}, Week {week}")


if __name__ == "__main__":
    main()
