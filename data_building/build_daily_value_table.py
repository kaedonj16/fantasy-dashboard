from __future__ import annotations

from pathlib import Path

from dashboard_services.api import get_nfl_state
from dashboard_services.player_value_history import record_model_value_snapshot
from dashboard_services.utils import (
    path_teams_index,
    load_teams_index,
    load_model_value_table,
    get_live_game_ids_for_today,
    load_week_schedule,
    build_and_save_week_stats_for_league,
)
from data_building.external_values_scraper import scrape_all_vendor_values
from data_building.sleeper_usage import write_usage_table_snapshot
from data_building.team_enrichment import (
    enrich_all_team_info,
    enrich_teams_index_with_rushing,
)
from data_building.value_exports import export_engine_values
from data_building.value_model_training import rewrite_value_table_with_model


def build_daily_data(season: int, week: int):
    print(f"[daily] build_daily_data season={season} week={week}")

    live_game_ids = get_live_game_ids_for_today(load_week_schedule(season, week))
    build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)

    print("[daily] scraping vendor values...")
    scrape_all_vendor_values()

    print("[daily] writing usage snapshot...")
    write_usage_table_snapshot(season, weeks=range(1, 19))

    print("[daily] enriching team info...")
    enrich_all_team_info(season)
    enrich_teams_index_with_rushing(Path(path_teams_index()))

    print("[daily] exporting engine values...")
    export_engine_values()

    print("[daily] rebuilding model values...")
    rewrite_value_table_with_model()

    model_value_table = load_model_value_table() or []
    inserted = record_model_value_snapshot(model_value_table)
    print(f"[daily] stored value-history snapshot rows={inserted}")


if __name__ == "__main__":
    current = get_nfl_state() or {}
    current_season = int(current.get("season"))
    current_week = int(current.get("week"))
    build_daily_data(current_season, current_week)
