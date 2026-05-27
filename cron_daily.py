from datetime import date, datetime
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from dashboard_services.api import get_nfl_state
from utils.paths import DATA_DIR, CACHE_DIR


# ---------------------------------------------------------------------------
# Cleanup: delete old date-stamped files left from before the undated migration
# ---------------------------------------------------------------------------

import re
_DATED_PATTERN = re.compile(r'_\d{4}-\d{2}-\d{2}')


def cleanup_dated_files() -> int:
    """Delete any files whose names contain a YYYY-MM-DD date stamp."""
    removed = 0
    search_dirs = [DATA_DIR]
    try:
        search_dirs.append(CACHE_DIR)
    except Exception:
        pass
    for base in search_dirs:
        if not Path(base).exists():
            continue
        for p in Path(base).rglob("*"):
            if p.is_file() and _DATED_PATTERN.search(p.stem):
                try:
                    p.unlink()
                    print(f"[cron] Removed dated file: {p}")
                    removed += 1
                except Exception as e:
                    print(f"[cron] Could not remove {p}: {e}")
    return removed


# ---------------------------------------------------------------------------
# Freshness guards - skip expensive steps that already ran today
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _file_fresh_today(path: Path) -> bool:
    """True if the file exists and was last modified today."""
    if not path.exists():
        return False
    return date.fromtimestamp(path.stat().st_mtime) == _today()


def _model_values_fresh() -> bool:
    mv = DATA_DIR / "model_values.json"
    if not _file_fresh_today(mv):
        return False
    # If any vendor CSV is newer than model_values.json, the model needs a rebuild
    # even if it was already written today (it was built from stale data).
    mv_mtime = mv.stat().st_mtime
    for _csv in ("fantasycalc_api_values.csv", "dynastyprocess_values.csv"):
        _p = DATA_DIR / _csv
        if _p.exists() and _p.stat().st_mtime > mv_mtime:
            return False
    return True


def _vendor_values_fresh() -> bool:
    return (
        _file_fresh_today(DATA_DIR / "fantasycalc_api_values.csv") and
        _file_fresh_today(DATA_DIR / "dynastyprocess_values.csv") and
        _file_fresh_today(DATA_DIR / "engine_values.csv")
    )


def _usage_table_fresh() -> bool:
    return _file_fresh_today(DATA_DIR / "usage_table.json")


def _model_mtime() -> float:
    """Return mtime of model_values.json, or 0 if missing."""
    mv = DATA_DIR / "model_values.json"
    return mv.stat().st_mtime if mv.exists() else 0.0


def _player_values_fresh() -> bool:
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_updated) AS t FROM player_values"
            ).fetchone()
        if row and row["t"]:
            t = row["t"]
            db_date = t.date() if hasattr(t, "date") else t
            if db_date != _today():
                return False
            # Also stale if model_values.json was rewritten after the last DB save.
            # Convert the DB timestamp to a Unix timestamp for comparison.
            import calendar
            db_ts = calendar.timegm(t.timetuple()) if hasattr(t, "timetuple") else 0
            if _model_mtime() > db_ts:
                return False
            return True
    except Exception:
        pass
    return False


def _trade_intel_fresh() -> bool:
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_crawled_at) AS t FROM trade_intel_leagues "
                "WHERE last_crawled_at IS NOT NULL"
            ).fetchone()
        if row and row["t"]:
            t = row["t"]
            return (t.date() if hasattr(t, "date") else t) == _today()
    except Exception:
        pass
    return False


def _wls_fresh() -> bool:
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_updated) AS t FROM player_values "
                "WHERE calibration_source = 'trade_wls'"
            ).fetchone()
        if row and row["t"]:
            t = row["t"]
            wls_date = t.date() if hasattr(t, "date") else t
            if wls_date != _today():
                return False
            # Also stale if model_values.json was rebuilt after the last WLS run
            # (the new model would make a different prior for WLS).
            import calendar
            wls_ts = calendar.timegm(t.timetuple()) if hasattr(t, "timetuple") else 0
            if _model_mtime() > wls_ts:
                return False
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Subprocess runner - each step gets a fresh process so memory fully releases
# ---------------------------------------------------------------------------

def _run_step(code: str, step_name: str, timeout: int = 3600) -> bool:
    """
    Run Python code in a fresh interpreter subprocess.
    stdout/stderr flow through. Returns True on success.
    Memory from the subprocess is fully released when it exits.
    """
    print(f"[cron] -> {step_name}")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            print(f"[cron] {step_name} exited with code {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[cron] {step_name} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[cron] {step_name} failed to launch: {e}")
        return False


def main():
    state = get_nfl_state() or {}
    season = int(state.get("season"))
    week = int(state.get("week"))
    season_type = str(state.get("season_type", "")).lower().strip()
    today_weekday = date.today().weekday()  # 6 = Sunday

    print(f"[cron] Daily run starting - Season {season}, Week {week}")

    n_cleaned = cleanup_dated_files()
    if n_cleaned:
        print(f"[cron] Cleaned up {n_cleaned} old dated file(s)")

    # ------------------------------------------------------------------ #
    # Step 1: Vendor data + usage table                                   #
    # ------------------------------------------------------------------ #
    if _vendor_values_fresh() and _usage_table_fresh():
        print("[cron] Vendor + usage data already fresh today, skipping")
    else:
        _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.build_daily_value_table import build_daily_data
build_daily_data({season!r}, {week!r})
""", "build_daily_data")

    # ------------------------------------------------------------------ #
    # Step 2: Advanced metrics                                            #
    # ------------------------------------------------------------------ #
    _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, date
from dashboard_services.api import get_nfl_state
from data_building.advanced_metrics import calculate_player_metrics, save_metrics_snapshot
from utils.utils import load_usage_table

nfl_state = get_nfl_state() or {{}}
season_type = str(nfl_state.get("season_type", "")).lower().strip()
is_offseason = season_type == "off"
current_season = int(nfl_state.get("season") or datetime.now().year)

usage_table = load_usage_table()
if not usage_table:
    print("[cron] No usage table found, skipping advanced metrics")
else:
    players_with_games = sum(1 for p in usage_table if p.get("usage", {{}}).get("games", 0) > 0)
    if players_with_games == 0 and is_offseason:
        print("[cron] Offseason detected, skipping advanced metrics")
    else:
        metrics_list = []
        failed_count = 0
        for player in usage_table:
            player_id = player.get("id")
            position = player.get("position")
            usage = player.get("usage", {{}})
            if not player_id or not position or not usage or usage.get("games", 0) == 0:
                continue
            try:
                metrics_list.append(calculate_player_metrics(player_id, usage, position))
            except Exception:
                failed_count += 1
        if metrics_list:
            save_metrics_snapshot(metrics_list, date.today().isoformat(), season=current_season)
            print(f"[cron] Advanced metrics: {{len(metrics_list)}} processed, {{failed_count}} failed")
        else:
            print("[cron] No advanced metrics calculated")
""", "build_daily_advanced_metrics")

    # ------------------------------------------------------------------ #
    # Step 3: Model values                                                #
    # ------------------------------------------------------------------ #
    if _model_values_fresh():
        print("[cron] Model values already built today, skipping")
    else:
        _run_step("""
from dotenv import load_dotenv; load_dotenv()
from data_building.build_daily_value_table import build_daily_model_values
build_daily_model_values()
""", "build_daily_model_values")

    # ------------------------------------------------------------------ #
    # Step 4: Save player values to DB                                   #
    # ------------------------------------------------------------------ #
    if _player_values_fresh():
        print("[cron] Player values already saved to DB today, skipping")
    else:
        _run_step("""
from dotenv import load_dotenv; load_dotenv()
from data_building.update_player_values_with_rankings import update_player_values_with_rankings
n = update_player_values_with_rankings()
print(f"[cron] Saved {n} player values")
""", "update_player_values_with_rankings")

    # ------------------------------------------------------------------ #
    # Step 5: Breakout candidates                                        #
    # ------------------------------------------------------------------ #
    _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from datetime import date
today = date.today()
if today.month in (1, 2) or (today.month == 3 and today.day < 15):
    print("[cron] Breakout skipped - playoff/early offseason period")
else:
    from data_building.breakout_engine.calculate_breakouts_with_real_data import main as run_breakouts
    result = run_breakouts()
    print(f"[cron] Breakout: {{result}}")
""", "build_daily_breakout_scores")

    # ------------------------------------------------------------------ #
    # Step 6: Weekly rookie data (Sundays only, off/pre season)          #
    # ------------------------------------------------------------------ #
    ROOKIE_PIPELINE_PAUSED = True
    if not ROOKIE_PIPELINE_PAUSED and today_weekday == 6 and season_type not in ("reg", "post"):
        _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.rookie_pipeline.pipeline import run_rookie_pipeline, get_active_rookie_class
from data_building.rookie_pipeline.rookie_evaluation_pipeline import run_rookie_evaluation_pipeline
year = get_active_rookie_class()
print(f"[cron] Weekly rookie refresh - {{year}} draft class")
eval_result = run_rookie_evaluation_pipeline(year)
print(f"[cron] Eval: {{eval_result.get('profile_count', 0)}} profiles")
result = run_rookie_pipeline(year)
print(f"[cron] Scoring: {{len(result.get('prospects', []))}} prospects")
""", "build_weekly_rookie_data")
    else:
        reason = "paused" if ROOKIE_PIPELINE_PAUSED else f"weekday={today_weekday}, season_type={season_type!r}"
        print(f"[cron] Rookie weekly run skipped - {reason}")

    # ------------------------------------------------------------------ #
    # Step 7: Trade intel discovery + crawl + analytics                  #
    # Split into three subprocesses so each step's memory is fully       #
    # released before the next one starts.                                #
    # ------------------------------------------------------------------ #
    if _trade_intel_fresh():
        print("[cron] Trade intel already crawled today, skipping discovery + crawl")
    else:
        _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.trade_intel.league_discovery import run_discovery, backfill_superflex
backfilled = backfill_superflex(batch_size=500)
if backfilled:
    print(f"[cron] Backfilled is_superflex for {{backfilled}} leagues")
discovered = run_discovery(target=200)
print(f"[cron] Trade intel: discovered {{discovered}} new leagues")
""", "trade_intel_discovery")

        _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.trade_intel.trade_crawler import run_crawl
crawl_result = run_crawl(batch_size=100, workers=3)
print(f"[cron] Trade intel: {{crawl_result}}")
""", "trade_intel_crawl")

        _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.trade_intel.analytics import run_analytics
analytics_result = run_analytics(season={season!r})
print(f"[cron] Trade intel analytics: {{analytics_result}}")
""", "trade_intel_analytics")

    # ------------------------------------------------------------------ #
    # Step 8: Draft ADP crawl                                            #
    # ------------------------------------------------------------------ #
    _run_step("""
from dotenv import load_dotenv; load_dotenv()
from data_building.trade_intel.draft_adp_crawler import run_draft_adp_crawl
result = run_draft_adp_crawl(batch_size=150, workers=2, crawl_mode="both", recrawl_days=30)
print(f"[cron] Draft ADP: {result}")
""", "draft_adp_crawl")

    # ------------------------------------------------------------------ #
    # Step 9: WLS calibration - one subprocess per combo so numpy        #
    # matrices and trade data are fully released between runs.            #
    # ------------------------------------------------------------------ #
    if _wls_fresh():
        print("[cron] WLS calibration already ran today, skipping")
    else:
        for _lt, _lt_name, _sz in [
            (2, "dynasty", 10),
            (2, "dynasty", 12),
            (1, "redraft", 10),
            (1, "redraft", 12),
        ]:
            _run_step(f"""
from dotenv import load_dotenv; load_dotenv()
from data_building.trade_intel.trade_value_model import run_trade_value_model
try:
    res = run_trade_value_model(season={season!r}, league_type={_lt}, league_size={_sz})
    print(f"[cron] WLS {_lt_name} {_sz}-team: {{res}}")
except Exception as e:
    print(f"[cron] WLS {_lt_name} {_sz}-team failed: {{e}}")
""", f"wls_{_lt_name}_{_sz}team")

    # ------------------------------------------------------------------ #
    # Step 10: Calibrated history snapshot                               #
    # ------------------------------------------------------------------ #
    _run_step("""
from dotenv import load_dotenv; load_dotenv()
from data_building.build_daily_value_table import record_calibrated_history_snapshot
n = record_calibrated_history_snapshot()
print(f"[cron] Calibrated history snapshot: {n} players")
""", "record_calibrated_history_snapshot")

    print(f"[cron] Daily run completed - Season {season}, Week {week}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[cron] Daily run failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            from utils.email_notifications import send_cron_failure_notification
            from dashboard_services.api import get_nfl_state
            state = get_nfl_state() or {}
            send_cron_failure_notification(e, {
                'season': state.get("season"),
                'week': state.get("week"),
                'timestamp': datetime.now().isoformat()
            })
        except Exception:
            pass
