from datetime import date, datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        current_season = int(nfl_state.get("season") or datetime.now().year)

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
            save_metrics_snapshot(metrics_list, today, season=current_season)
            print(f"[cron] Advanced metrics: {len(metrics_list)} processed, {failed_count} failed")
        else:
            print("[cron] No advanced metrics calculated")

    except Exception as e:
        print(f"[cron] Advanced metrics failed: {e}")
        import traceback
        traceback.print_exc()


def build_weekly_rookie_data(state: dict) -> None:
    """
    Run the full rookie pipeline (eval metrics + scoring) once a week during
    the offseason.

    Fires only when:
      - Today is Sunday
      - NFL season_type is "off" or "pre" (skipped during reg/post season)
    """
    from datetime import date as _date
    today = _date.today()

    if today.weekday() != 6:  # 0=Mon … 6=Sun
        return

    season_type = str(state.get("season_type", "")).lower().strip()
    if season_type in ("reg", "post"):
        print(f"[cron] Rookie weekly run skipped — season_type={season_type!r}")
        return

    from data_building.rookie_pipeline.pipeline import run_rookie_pipeline, get_active_rookie_class
    from data_building.rookie_pipeline.rookie_evaluation_pipeline import run_rookie_evaluation_pipeline

    try:
        year = get_active_rookie_class()
        print(f"[cron] Weekly rookie refresh — {year} draft class (season_type={season_type!r})")

        eval_result = run_rookie_evaluation_pipeline(year)
        print(
            f"[cron] Eval pipeline: {eval_result.get('profile_count', 0)} profiles, "
            f"db_metrics_rows={eval_result.get('db_metrics_rows', 0)}"
        )

        result = run_rookie_pipeline(year)
        print(
            f"[cron] Scoring pipeline: {len(result.get('prospects', []))} prospects, "
            f"{len(result.get('scores', {}))} scored, {len(result.get('values', {}))} values"
        )
    except Exception as e:
        print(f"[cron] Weekly rookie refresh failed: {e}")
        import traceback
        traceback.print_exc()


def backfill_historical_advanced_metrics():
    """
    Backfill advanced metrics for seasons 2022-2025.
    Safe to re-run — uses upsert logic.
    """
    from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
    from data_building.external_data.sleeper_usage import build_usage_map_for_season
    from utils.utils import load_players_index

    seasons = [2022, 2023, 2024, 2025]
    print(f"[cron] Backfilling advanced metrics for seasons: {seasons}")

    players_index = load_players_index() or {}
    if not players_index:
        print("[cron] Could not load players index, skipping backfill")
        return

    for season in seasons:
        try:
            print(f"[cron] Backfill season {season}...")
            usage_map = build_usage_map_for_season(season, weeks=range(1, 19))
            metrics_list = []
            skipped = 0
            failed = 0

            for pid, usage in usage_map.items():
                if usage.get("games", 0) == 0:
                    skipped += 1
                    continue
                meta = players_index.get(pid) or players_index.get(str(pid)) or {}
                pos = meta.get("pos") or meta.get("position")
                if pos not in ("QB", "RB", "WR", "TE"):
                    skipped += 1
                    continue
                try:
                    metrics_list.append(calculate_player_metrics(str(pid), usage, pos))
                except Exception as e:
                    print(f"[cron]   [warn] player {pid}: {e}")
                    failed += 1

            if metrics_list:
                as_of_date = f"{season + 1}-01-10"
                save_metrics_snapshot(metrics_list, as_of_date, season=season)
                print(f"[cron]   Season {season}: saved {len(metrics_list)} players (skipped={skipped}, failed={failed})")
            else:
                print(f"[cron]   Season {season}: no metrics to save (skipped={skipped}, failed={failed})")

        except Exception as e:
            print(f"[cron] Backfill season {season} failed: {e}")
            import traceback
            traceback.print_exc()

    print("[cron] Historical advanced metrics backfill complete")


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
        from data_building.update_player_values_with_rankings import update_player_values_with_rankings
        from utils.utils import load_model_value_table
        from utils.email_notifications import send_cron_failure_notification, send_database_save_notification

        value_table = load_model_value_table()
        if not value_table:
            raise RuntimeError("No value table available after build_daily_model_values")

        # Expected count based on typical roster size
        expected_count = len(value_table)
        value_count = update_player_values_with_rankings()
        print(f"[cron] Saved {value_count} player values")

        # Check if save count is unexpectedly low
        if value_count < expected_count * 0.8:  # Less than 80% of expected
            send_database_save_notification(value_count, expected_count)

        # build_daily_market_pulse()
        build_daily_breakout_candidates(season, week, state)
        build_weekly_rookie_data(state)

        print(f"[cron] Daily run completed - Season {season}, Week {week}")

    except Exception as e:
        print(f"[cron] Daily run failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Send email notification
        try:
            from utils.email_notifications import send_cron_failure_notification
            send_cron_failure_notification(e, {
                'season': season,
                'week': week,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            print("[cron] Email notifications not available")
        except Exception as email_error:
            print(f"[cron] Failed to send email notification: {email_error}")


if __name__ == "__main__":
    main()
