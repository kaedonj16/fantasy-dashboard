#!/usr/bin/env python3
"""
Post-deploy breakout score rebuild.

Spawned as a background process by startup.py on every Render deployment.
Checks whether the current season's breakout scores include the projections
block in component_details, and rebuilds via build_historical_scores if not.
"""

import os
import sys
import time
from datetime import datetime

# Allow 5 seconds for gunicorn to start before hammering the DB
time.sleep(5)

from dotenv import load_dotenv
load_dotenv()

print(f"[post-deploy] Starting at {datetime.now().isoformat()}")


def _get_season() -> int:
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state() or {}
        return int(state.get("season", datetime.now().year))
    except Exception:
        return datetime.now().year


def _needs_rebuild(target_season: int) -> bool:
    """True if no rows for target_season with today's as_of_date have projections set."""
    from datetime import date
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM breakout_opportunity_scores
                    WHERE season = %s
                      AND as_of_date = %s
                      AND component_details->'projections' IS NOT NULL
                    """,
                    [target_season, date.today()],
                )
                row = cur.fetchone()
                return (row[0] if row else 0) == 0
    except Exception as e:
        print(f"[post-deploy] DB check failed: {e}")
        return True


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


def main():
    target_season = _get_season()
    stats_season = target_season - 1

    # Always run migrations first — all SQL uses IF NOT EXISTS so it's safe
    # to run on every deploy even if nothing changed.
    print("[post-deploy] Running DB migrations...")
    try:
        from scripts.run_migrations import run_migrations
        run_migrations()
    except Exception as e:
        print(f"[post-deploy] Migrations failed: {e}")
        import traceback
        traceback.print_exc()

    # Populate this web container's ADP snapshots so the source columns / modal
    # work right after a deploy without a manual fetch. Independent of the
    # breakout rebuild below (which may early-return), so it runs every deploy.
    _refresh_global_adp(target_season)

    if not _needs_rebuild(target_season):
        print(
            f"[post-deploy] Breakout scores for {target_season} already contain "
            "projections data — skipping rebuild"
        )
        return

    print(
        f"[post-deploy] Breakout scores for {target_season} are missing projections "
        f"— rebuilding from {stats_season} stats..."
    )
    try:
        from datetime import date
        from data_building.breakout_engine.build_historical_scores import run
        run(seasons=[stats_season], min_score=55.0, as_of_date_override=date.today())
        print(f"[post-deploy] Rebuild complete at {datetime.now().isoformat()}")
    except Exception as e:
        print(f"[post-deploy] Rebuild failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
