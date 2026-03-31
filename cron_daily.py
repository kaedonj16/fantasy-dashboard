from datetime import date

from dashboard_services.api import get_nfl_state
from data_building.build_daily_value_table import build_daily_data, build_daily_market_pulse


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
    Calculate offseason breakout candidates based on roster changes and vacated opportunity.

    This creates database tables on first run, then calculates:
    - Vacated opportunity from roster changes
    - Projected opportunity redistribution to remaining players
    - Offseason breakout scores

    Runs during:
    - Offseason and preseason (projections based on roster moves)
    - First 4 weeks of regular season (combo of projections + early actual data)
    - After week 4, actual usage data is reliable enough without projections
    """
    from data_building.offseason_opportunity import (
        init_offseason_opportunity_db,
        calculate_vacated_opportunity,
        project_opportunity_redistribution
    )

    # Check if we should run breakout calculations
    nfl_state = get_nfl_state() or {}
    season_type = str(nfl_state.get("season_type", "")).lower().strip()

    # Run during offseason, preseason, or early regular season (weeks 1-4)
    should_run = (
        season_type in ["off", "pre"] or
        (season_type == "regular" and week <= 4)
    )

    if not should_run:
        print(f"[cron] Skipping breakout calculations - season_type={season_type}, week={week} (only runs during offseason/preseason/weeks 1-4)")
        return

    print(f"[cron] Calculating offseason breakout candidates for season={season}, week={week}")

    try:
        # Initialize database tables (safe to call multiple times)
        init_offseason_opportunity_db()

        # Calculate vacated opportunity from roster changes
        calculate_vacated_opportunity(season)

        # Project opportunity redistribution and identify breakout candidates
        project_opportunity_redistribution(season, top_n_players=600)

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
