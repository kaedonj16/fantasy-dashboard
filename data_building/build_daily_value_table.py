from __future__ import annotations

from pathlib import Path

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.api import get_nfl_state
from data_building.player_value_history import record_model_value_snapshot
from utils.utils import (
    path_teams_index,
    load_teams_index,
    load_model_value_table,
    get_live_game_ids_for_today,
    load_week_schedule,
    build_and_save_week_stats_for_league,
)
from data_building.external_data.external_values_scraper import scrape_all_vendor_values
from data_building.external_data.sleeper_usage import write_usage_table_snapshot
from data_building.external_data.team_enrichment import (
    enrich_all_team_info,
    enrich_teams_index_with_rushing,
)
from data_building.value_exports import export_engine_values
from data_building.value_model_training import rewrite_value_table_with_model


def build_daily_data(season: int, week: int):
    nfl_state = get_nfl_state() or {}
    season_type = (nfl_state.get("season_type") or "").lower()
    offseason_mode = season_type == "off"

    if not offseason_mode and week >= 1:
        live_game_ids = get_live_game_ids_for_today(load_week_schedule(season, week))
        build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)

    if load_fantasycalc_api_values() is None or load_dynastyprocess_values() is None:
        scrape_all_vendor_values()

    if load_usage_table() is None or load_engine_table() is None:
        write_usage_table_snapshot(season, weeks=range(1, 19))
        enrich_all_team_info(season)
        enrich_teams_index_with_rushing(Path(path_teams_index()))
        export_engine_values()

    if load_model_value_table() is None:
        rewrite_value_table_with_model()
        model_value_table = load_model_value_table() or []
        inserted = record_model_value_snapshot(model_value_table)
        print(f"[daily] stored value-history snapshot rows={inserted}")


def build_daily_market_pulse():
    value_table = load_model_value_table() or []
    top_assets = sorted(
        [
            {
                "name": p.get("name"),
                "position": p.get("position"),
                "team": p.get("team"),
                "value": _safe_float(p.get("value")),
            }
            for p in value_table
            if isinstance(p, dict) and str(p.get("position") or "").upper() in {"QB", "RB", "WR", "TE"}
        ],
        key=lambda x: x["value"],
        reverse=True,
    )[:15]

    payload = {"top_assets": top_assets}
    cache_key = build_ai_cache_key("daily_market_pulse", payload, "v1")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    html = "<div class='ai-copy'><p><strong>Daily market pulse:</strong> Elite value remains concentrated at the top of the board. Monitor shifting tiers around your weakest position group before forcing trades.</p></div>"
    save_cached_ai_text(cache_key, html)
    return html


if __name__ == "__main__":
    current = get_nfl_state() or {}
    current_season = int(current.get("season"))
    current_week = int(current.get("week"))
    build_daily_data(current_season, current_week)