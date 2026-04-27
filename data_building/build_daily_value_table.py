from __future__ import annotations

from pathlib import Path

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.api import get_nfl_state
from data_building.external_data.sleeper_usage import write_usage_table_snapshot
from data_building.external_data.team_enrichment import (
    enrich_all_team_info,
    enrich_teams_index_with_rushing,
)
from data_building.player_value_history import record_model_value_snapshot
from data_building.value_exports import export_engine_values
from data_building.value_model_training import rewrite_value_table_with_model
from utils.utils import (
    path_teams_index,
    load_teams_index,
    load_model_value_table,
    get_live_game_ids_for_today,
    load_week_schedule,
    build_and_save_week_stats_for_league, )


def _safe_float(value):
    """Safely convert a value to float, returning 0.0 for None or invalid values."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_daily_data(season: int, week: int):
    """
    Build usage table and vendor values.

    NOTE: This does NOT build model values - that should be done AFTER
    advanced metrics are calculated (see build_daily_model_values).
    """
    from data_building.external_data.external_values_scraper import (
        scrape_all_vendor_values,
        load_fantasycalc_api_values,
        load_dynastyprocess_values,
    )
    nfl_state = get_nfl_state() or {}
    season_type = (nfl_state.get("season_type") or "").lower()
    offseason_mode = season_type == "off"

    if not offseason_mode and week >= 1:
        live_game_ids = get_live_game_ids_for_today(load_week_schedule(season, week))
        build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)

    if load_fantasycalc_api_values() is None or load_dynastyprocess_values() is None:
        scrape_all_vendor_values()

    # Only fetch weeks 1 through current week (or max 18)
    # In offseason, fetch last season's full data
    weeks_to_fetch = range(1, min(week + 1, 19)) if not offseason_mode and week >= 1 else range(1, 19)

    print(f"[build_daily_data] Refreshing usage table for weeks {list(weeks_to_fetch)}")

    # Always refresh usage table daily (for up-to-date stats)
    write_usage_table_snapshot(season, weeks=weeks_to_fetch)
    enrich_all_team_info(season)
    enrich_teams_index_with_rushing(Path(path_teams_index()))

    # Rookie evaluation snapshots (prospect-focused, draft-class aware)
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.rookie_evaluation_pipeline import run_rookie_evaluation_pipeline

        draft_class_year = get_active_rookie_class()
        rookie_result = run_rookie_evaluation_pipeline(draft_class_year)
        db_res = rookie_result.get("db_result") or {}
        bridge = db_res.get("db_bridge_rows") or {}
        print(
            "[build_daily_data] Rookie evaluation updated "
            f"class={rookie_result.get('draft_class_year')} "
            f"profiles={rookie_result.get('profile_count')} "
            f"bridge_updated={bridge.get('updated', 0)} "
            f"bridge_inserted={bridge.get('inserted', 0)}"
        )
    except Exception as exc:
        print(f"[build_daily_data] Rookie evaluation pipeline failed: {exc}")

    export_engine_values()


def build_daily_model_values():
    """
    Build model values using ML model.

    MUST be called AFTER build_daily_advanced_metrics() so that advanced
    metrics are available for the model to use.
    """
    print(f"[build_daily_data] Building model values with advanced metrics")
    rewrite_value_table_with_model()
    model_value_table = load_model_value_table(apply_calibration=False) or []
    record_model_value_snapshot(model_value_table)


def record_calibrated_history_snapshot() -> int:
    """
    Write today's player_value_history snapshot using COALESCE(calibrated, raw)
    values from player_values — the same values shown on the rankings page.

    Call this AFTER run_trade_value_model() has written calibrated_value_1qb.
    Uses ON CONFLICT UPDATE so it overwrites any raw-model entry already written
    today by build_daily_model_values().
    """
    from datetime import date
    from dashboard_services.db import get_conn
    from utils.utils import load_players_index, load_model_value_table

    today = date.today().isoformat()

    # Build id -> name map: model table first (has picks + all players), then players_index fallback
    name_map: dict = {}
    try:
        mv = load_model_value_table(apply_calibration=False) or []
        for p in mv:
            pid = str(p.get("id") or "")
            nm = p.get("name") or ""
            if pid and nm and nm != "Unknown":
                name_map[pid] = nm
    except Exception:
        pass
    # Fill gaps with players_index
    try:
        players_index = load_players_index() or {}
        for pid, info in players_index.items():
            if pid not in name_map:
                nm = (info or {}).get("name") or ""
                if nm:
                    name_map[str(pid)] = nm
    except Exception:
        pass

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                player_id,
                position,
                team,
                COALESCE(calibrated_value_1qb, value_1qb) AS value
            FROM player_values
            WHERE COALESCE(calibrated_value_1qb, value_1qb) > 0
            """
        ).fetchall()

    if not rows:
        print("[record_calibrated_history_snapshot] No rows in player_values — skipping")
        return 0

    written = 0
    with get_conn() as conn:
        for r in rows:
            player_name = name_map.get(str(r["player_id"])) or f"Player {r['player_id']}"
            
            conn.execute(
                """
                INSERT INTO player_value_history
                    (as_of_date, player_id, name, position, team, value, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'model')
                ON CONFLICT (as_of_date, player_id, source)
                DO UPDATE SET
                    name     = EXCLUDED.name,
                    value    = EXCLUDED.value,
                    position = EXCLUDED.position,
                    team     = EXCLUDED.team
                """,
                (today, r["player_id"], player_name, r["position"], r["team"], float(r["value"])),
            )
            written += 1

    print(f"[record_calibrated_history_snapshot] Wrote {written} calibrated values to player_value_history")
    return written


def build_daily_market_pulse_for_league_type(league_type: str = "1qb"):
    """
    Build market pulse for a specific league type.

    Args:
        league_type: "1qb" or "sf" (superflex)

    Returns:
        HTML string with market pulse for the specified league type
    """
    value_table = load_model_value_table() or []

    if league_type == "sf":
        # Build top assets for Superflex
        top_assets = sorted(
            [
                {
                    "name": p.get("name"),
                    "position": p.get("position"),
                    "team": p.get("team"),
                    "value": _safe_float(p.get("sf_value")),
                }
                for p in value_table
                if isinstance(p, dict) and str(p.get("position") or "").upper() in {"QB", "RB", "WR", "TE"}
            ],
            key=lambda x: x["value"],
            reverse=True,
        )[:15]

        html = "<div class='ai-copy'><p><strong>Daily market pulse (Superflex):</strong> Elite QBs dominate the top of the market. Build around a QB1 or acquire multiple QB2s to remain competitive.</p></div>"
        payload = {"top_assets": top_assets, "league_type": "superflex"}
        cache_key = build_ai_cache_key("daily_market_pulse_sf", payload, "v1")
    else:
        # Build top assets for 1QB
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

        html = "<div class='ai-copy'><p><strong>Daily market pulse (1QB):</strong> Elite value remains concentrated at the top of the board. Monitor shifting tiers around your weakest position group before forcing trades.</p></div>"
        payload = {"top_assets": top_assets, "league_type": "1qb"}
        cache_key = build_ai_cache_key("daily_market_pulse_1qb", payload, "v1")

    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    save_cached_ai_text(cache_key, html)
    return html


def build_daily_market_pulse():
    """
    Build market pulse for both 1QB and Superflex league types.
    This is called by the daily cron job to cache both versions.
    """
    # Build and cache 1QB market pulse
    build_daily_market_pulse_for_league_type("1qb")

    # Build and cache Superflex market pulse
    build_daily_market_pulse_for_league_type("sf")

    # Return 1QB for backwards compatibility
    return build_daily_market_pulse_for_league_type("1qb")


if __name__ == "__main__":
    current = get_nfl_state() or {}
    current_season = int(current.get("season"))
    current_week = int(current.get("week"))
    build_daily_data(current_season, current_week)
