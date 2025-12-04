from pathlib import Path

from dashboard_services.data_building.external_values_scraper import scrape_all_vendor_values, \
    load_fantasycalc_api_values, load_dynastyprocess_values
from dashboard_services.data_building.sleeper_usage import write_usage_table_snapshot
from dashboard_services.data_building.team_enrichment import enrich_all_team_info, \
    enrich_teams_index_with_rushing
from dashboard_services.data_building.value_exports import export_engine_values
from dashboard_services.data_building.value_model_training import train_trade_value_model, \
    rewrite_value_table_with_model
from dashboard_services.utils import path_teams_index, load_usage_table, load_teams_index, load_model_value_table


def build_daily_data(season: int):
    if load_fantasycalc_api_values() is None and load_dynastyprocess_values() is None:
        scrape_all_vendor_values()
    if load_usage_table() is None:
        write_usage_table_snapshot(2025, weeks=range(1, 19))
    export_engine_values()
    if load_teams_index() is None:
        enrich_all_team_info(season)
        enrich_teams_index_with_rushing(Path(path_teams_index()))
    if load_model_value_table() is None:
        bundle = train_trade_value_model()
        rewrite_value_table_with_model()
