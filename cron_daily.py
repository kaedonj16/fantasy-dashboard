from dashboard_services.data_building.daily import build_daily_data
from datetime import date

from dashboard_services.api import get_nfl_state


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))

    print(f"[cron] Running daily for season={season}, week={week}")

    # run your existing function
    build_daily_data(season, week)

if __name__ == "__main__":
    main()
