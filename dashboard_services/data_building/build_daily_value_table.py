from pathlib import Path

from dashboard_services.api import get_nfl_state
from dashboard_services.data_building.external_values_scraper import scrape_all_vendor_values, \
    load_fantasycalc_api_values, load_dynastyprocess_values
from dashboard_services.data_building.sleeper_usage import write_usage_table_snapshot
from dashboard_services.data_building.team_enrichment import enrich_all_team_info, \
    enrich_teams_index_with_rushing
from dashboard_services.data_building.value_exports import export_engine_values
from dashboard_services.data_building.value_model_training import rewrite_value_table_with_model
from dashboard_services.utils import path_teams_index, load_usage_table, load_teams_index, load_model_value_table, \
    load_engine_table, get_live_game_ids_for_today, load_week_schedule, build_and_save_week_stats_for_league


def build_daily_data(season: int, week: int):
    live_game_ids = get_live_game_ids_for_today(load_week_schedule(season, week))
    build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)

    if load_fantasycalc_api_values() is None or load_dynastyprocess_values() is None:
        scrape_all_vendor_values()

    if load_usage_table() is None or load_engine_table() is None:
        write_usage_table_snapshot(2025, weeks=range(1, 19))
        enrich_all_team_info(season)
        enrich_teams_index_with_rushing(Path(path_teams_index()))
        export_engine_values()

    if load_model_value_table() is None:
        rewrite_value_table_with_model()


if __name__ == "__main__":
    # adjust this however you determine current season/week
    current = get_nfl_state()
    CURRENT_SEASON = current.get("season")
    CURRENT_WEEK = current.get("week")

    build_daily_data(CURRENT_SEASON, CURRENT_WEEK)