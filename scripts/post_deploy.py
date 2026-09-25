#!/usr/bin/env python3
"""
Post-deploy migrations and global ADP refresh.

Spawned as a background process by startup.py on every Render deployment.
Breakout rebuilding intentionally belongs to the daily cron, not web startup.
"""

import os
import sys
import time
from datetime import datetime

# `python scripts/post_deploy.py` (how startup.py spawns this) puts scripts/ on
# sys.path, not the repo root, so `import dashboard_services` / `data_building`
# / `from scripts.run_migrations` all raise ModuleNotFoundError. Add the repo
# root explicitly, matching the other scripts in this directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _get_season() -> int:
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state() or {}
        return int(state.get("season", datetime.now().year))
    except Exception:
        return datetime.now().year


def _refresh_global_adp(season: int) -> None:
    """Populate the tokenless global ADP snapshots (Yahoo/ESPN/MFL) on THIS
    container's disk.

    Render runs the web service and the cron jobs in separate containers with
    separate disks, and the ADP resolver reads snapshots from local disk. So a
    fresh web deploy starts with no snapshots until a cron writes to *its* disk
    (which the web container never sees) — which is why the feeds had to be
    refreshed by hand after each deploy. Running it here, in the background
    post-deploy process, makes every deploy self-populate. Isolated and
    best-effort: each provider is isolated inside refresh_global_adp_sources and
    an empty fetch keeps any last-good snapshot."""
    try:
        from dashboard_services.adp_service import refresh_global_adp_sources
        summary = refresh_global_adp_sources(season)
        print(f"[post-deploy] Global ADP refresh: {summary}")
    except Exception as e:
        print(f"[post-deploy] Global ADP refresh failed: {e}")
        import traceback
        traceback.print_exc()


def _build_usage_table_snapshot(season: int) -> None:
    """Build the daily usage table on THIS container's disk.

    data/usage_table.json is gitignored and Render gives the web container an
    ephemeral disk, so every deploy wipes it. The daily cron rebuilds the
    table on its own container, which the web container never sees (separate
    disks), so without this step depth charts fall back to the stale usage
    embedded in the committed relevant-players index (e.g. 2025 numbers
    served as the current season, N/A for carry/touch shares). Completed
    weeks are immutable cache hits; only a few Sleeper week files are
    fetched. Best-effort: never raises.
    """
    try:
        from dashboard_services.api import get_nfl_state
        from data_building.external_data.sleeper_usage import (
            write_usage_table_snapshot,
        )
        state = get_nfl_state() or {}
        week = int(state.get("week") or 0)
        season_type = str(state.get("season_type") or "").lower()
        if season_type == "off" or week < 1:
            weeks = range(1, 19)
        else:
            weeks = range(1, min(week + 1, 19))
        path = write_usage_table_snapshot(season, weeks=weeks)
        print(f"[post-deploy] Usage table snapshot: {path}")
    except Exception as e:
        print(f"[post-deploy] Usage table snapshot failed: {e}")
        import traceback
        traceback.print_exc()


def _warm_team_rankings_cache(port: int, season: int) -> None:
    """Warm the NFL team rankings cache so the first page click is fast.

    The rankings compute downloads several weekly stat files on a cold
    container and can take 60s+ (near gunicorn's 120s worker timeout), which
    made the Teams page sit on "Loading team data." after a deploy until
    someone refreshed. One localhost hit here populates the on-disk cache
    that all gunicorn workers share. Best-effort: never raises.
    """
    url = f"http://127.0.0.1:{int(port)}/api/nfl-team-rankings?season={int(season)}"
    try:
        import requests

        resp = requests.get(url, timeout=110)
        print(f"[post-deploy] Team rankings warmup: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[post-deploy] Team rankings warmup failed: {e}")


def main():
    from dashboard_services.memory_diagnostics import format_memory_snapshot
    _load_dotenv()
    print(f"[post-deploy] Starting at {datetime.now().isoformat()}")
    print(format_memory_snapshot("post-deploy begin"))
    # Allow 5 seconds for gunicorn to start before hammering the DB.
    time.sleep(5)

    target_season = _get_season()
    web_port = int(os.environ.get("PORT", 5000))

    # Always run migrations first — all SQL uses IF NOT EXISTS so it's safe
    # to run on every deploy even if nothing changed.
    print("[post-deploy] Running DB migrations...")
    print(format_memory_snapshot("before migrations"))
    try:
        from scripts.run_migrations import run_migrations
        run_migrations()
    except Exception as e:
        print(f"[post-deploy] Migrations failed: {e}")
        import traceback
        traceback.print_exc()
    print(format_memory_snapshot("after migrations"))

    # Populate this web container's ADP snapshots so the source columns / modal
    # work right after a deploy without a manual fetch. Independent of the
    # daily cron, so it runs every deploy without loading the breakout model.
    print(format_memory_snapshot("before global ADP refresh"))
    _refresh_global_adp(target_season)
    print(format_memory_snapshot("after global ADP refresh"))

    # Build the usage table on this container's disk so depth charts serve
    # current-season numbers immediately instead of the stale embedded
    # fallback. Independent of the daily cron (separate disk).
    print(format_memory_snapshot("before usage table snapshot"))
    _build_usage_table_snapshot(target_season)
    print(format_memory_snapshot("after usage table snapshot"))

    # Warm the team rankings disk cache so the first Teams page visit after
    # a deploy does not pay the cold compute. Runs last; everything above is
    # independent of it.
    _warm_team_rankings_cache(web_port, target_season)


if __name__ == "__main__":
    main()
