from datetime import date

from dashboard_services.api import get_nfl_state
from data_building.build_daily_value_table import build_daily_data


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] Running daily for season={season}, week={week}")

    # run your existing function
    build_daily_data(season, week)
    try:
        build_daily_market_pulse()
    except Exception as e:
        print(f"[daily] market pulse skipped: {e}")


if __name__ == "__main__":
    main()
