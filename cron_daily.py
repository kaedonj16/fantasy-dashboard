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
    Calculate breakout candidates using the upgraded BreakoutEngine.

    This uses the new multi-component scoring system with:
    - Opportunity opened signals
    - Competition removed/added tracking
    - Player readiness scoring
    - Team environment analysis
    - Role trajectory (in-season)
    """
    from datetime import date as date_module
    from data_building.breakout_engine.calculate_breakouts_with_real_data import main as calculate_breakouts

    season_type = str(nfl_state.get("season_type", "")).lower().strip()

    # Determine if we should run based on season phase
    # Run year-round but more frequently during key periods
    should_run = True

    # Skip during playoffs (Jan 1 - Mar 14) unless explicitly needed
    today = date_module.today()
    if today.month == 1 or today.month == 2 or (today.month == 3 and today.day < 15):
        print(f"[cron] Breakout calculations skipped - playoff/early offseason period")
        return

    print(f"[cron] Starting breakout calculations for season={season}, week={week}, type={season_type}")

    try:
        # Run the new BreakoutEngine
        result = calculate_breakouts()

        print(f"[cron] Breakout scoring completed:")
        print(f"  - Season: {result.get('season', season)}")
        print(f"  - Phase: {result.get('phase', 'unknown')}")
        print(f"  - Players analyzed: {result.get('players_loaded', 0)}")
        print(f"  - Raw candidates: {result.get('raw_candidates', 0)}")
        print(f"  - Filtered candidates: {result.get('filtered_candidates', 0)}")
        print(f"  - Scores saved: {result.get('saved_count', 0)}")

        # Show filter summary if available
        filter_summary = result.get('filter_summary', {})
        if filter_summary:
            print(f"  - Filter breakdown:")
            print(f"    • Excluded stars: {filter_summary.get('excluded_star', 0)}")
            print(f"    • Excluded true dust: {filter_summary.get('excluded_true_dust', 0)}")
            print(f"    • Excluded age: {filter_summary.get('excluded_age', 0)}")
            print(f"    • Ideal breakout band: {filter_summary.get('ideal_breakout_band', 0)}")
            print(f"    • Viable small role: {filter_summary.get('viable_small_role', 0)}")
            print(f"    • Longshot: {filter_summary.get('longshot', 0)}")

    except Exception as e:
        print(f"[cron] Breakout calculations failed: {e}")
        import traceback
        traceback.print_exc()


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

        # build_daily_market_pulse()
        build_daily_breakout_candidates(season, week, state)

        print(f"[cron] Daily run completed - Season {season}, Week {week}")

    except Exception as e:
        print(f"[cron] Daily run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
