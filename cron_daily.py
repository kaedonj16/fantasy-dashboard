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


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] Running daily for season={season}, week={week}")

    # run your existing function
    build_daily_data(season, week)

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


if __name__ == "__main__":
    main()
