from datetime import date

from dashboard_services.api import get_nfl_state
from data_building.build_daily_value_table import build_daily_data, build_daily_market_pulse


def build_daily_advanced_metrics():
    """
    Calculate and save advanced efficiency metrics for all players.
    """
    from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
    from utils.utils import load_usage_table
    from dashboard_services.api import get_nfl_state

    try:
        usage_table = load_usage_table()
        if not usage_table:
            print("[cron] No usage table found, skipping advanced metrics")
            return

        nfl_state = get_nfl_state() or {}
        season_type = str(nfl_state.get("season_type", "")).lower().strip()
        is_offseason = season_type == "off"

        players_with_games = sum(1 for p in usage_table if p.get("usage", {}).get("games", 0) > 0)

        if players_with_games == 0 and is_offseason:
            print("[cron] Offseason detected, skipping advanced metrics")
            return

        metrics_list = []
        failed_count = 0
        for player in usage_table:
            player_id = player.get("id")
            position = player.get("position")
            usage = player.get("usage", {})

            if not player_id or not position or not usage or usage.get("games", 0) == 0:
                continue

            try:
                metrics = calculate_player_metrics(player_id, usage, position)
                metrics_list.append(metrics)
            except Exception:
                failed_count += 1

        if metrics_list:
            today = date.today().isoformat()
            save_metrics_snapshot(metrics_list, today)
            print(f"[cron] Advanced metrics: {len(metrics_list)} processed, {failed_count} failed")
        else:
            print("[cron] No advanced metrics calculated")

    except Exception as e:
        print(f"[cron] Advanced metrics failed: {e}")
        import traceback
        traceback.print_exc()


def build_daily_breakout_candidates(season: int, week: int, nfl_state: dict):
    """
    Calculate breakout candidates using new modular workflow with smart data management.
    """
    from data_building.breakout_workflow import run_modular_breakout_workflow
    from data_building.breakout_data_manager import BreakoutDataManager
    
    # Initialize data manager
    data_manager = BreakoutDataManager()

    season_type = str(nfl_state.get("season_type", "")).lower().strip()
    should_run = (
            season_type in ["off", "pre"] or
            (season_type == "regular" and week <= 9)
    )

    if not should_run:
        print(f"[cron] Breakout calculations skipped - season_type={season_type}, week={week}")
        return

    # Check if data needs refreshing
    needs_refresh = data_manager.needs_refresh()
    should_refresh_for_changes, refresh_reason = data_manager.should_refresh_for_changes()
    
    if not needs_refresh and not should_refresh_for_changes:
        print(f"[cron] Breakout data fresh, skipping refresh")
        return

    print(f"[cron] Starting breakout calculations for season={season}, week={week}")
    if should_refresh_for_changes:
        print(f"[cron] Reason: {refresh_reason}")

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
            print(f"[cron] Cleaned {total_cleaned} previous records")

    # Run the modular workflow
    success = run_modular_breakout_workflow(season, week, nfl_state)

    if success:
        print(f"[cron] Breakout workflow completed successfully")
        
        # Show freshness report
        freshness_report = data_manager.get_data_freshness_report()
        print(f"[cron] Data freshness: Scores {freshness_report['scores']['days_old']} days old, Projections {freshness_report['projections']['days_old']} days old")
        
        # Show any auto-calculations that were performed
        if freshness_report.get('auto_calculations'):
            print(f"[cron] Auto-calculations performed:")
            for calc in freshness_report['auto_calculations']:
                print(f"  - {calc}")
    else:
        print(f"[cron] Breakout workflow failed, attempting force refresh...")
        
        # Force refresh as fallback
        force_results = data_manager.force_refresh_all_data()
        print(f"[cron] Force refresh results:")
        for data_type, result in force_results.items():
            print(f"  - {data_type}: {result}")


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] Daily run starting - Season {season}, Week {week}")

    try:
        build_daily_data(season, week)
        build_daily_advanced_metrics()

        from data_building.build_daily_value_table import build_daily_model_values
        build_daily_model_values()

        from data_building.save_player_values import save_daily_values_to_db
        from utils.utils import load_model_value_table

        value_table = load_model_value_table()
        if not value_table:
            raise RuntimeError("No value table available after build_daily_model_values")

        value_count = save_daily_values_to_db(value_table)
        print(f"[cron] Saved {value_count} player values")

        build_daily_market_pulse()
        build_daily_breakout_candidates(season, week, state)

        print(f"[cron] Daily run completed - Season {season}, Week {week}")

    except Exception as e:
        print(f"[cron] Daily run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
