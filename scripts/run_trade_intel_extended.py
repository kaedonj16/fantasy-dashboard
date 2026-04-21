"""
Extended trade-intel run: discover leagues then crawl in repeated batches
over several hours with configurable pacing.

Usage
-----
    # Default: discover 1000 leagues, crawl in batches of 300 every 20 min for 4 hours
    python scripts/run_trade_intel_extended.py

    # Crawl-only (skip discovery), 500 per batch, every 15 min for 3 hours
    python scripts/run_trade_intel_extended.py --no-discovery --crawl-batch 500 --interval 15 --hours 3

    # Discover up to 2000 leagues, then crawl
    python scripts/run_trade_intel_extended.py --discover-target 2000

Arguments
---------
    --discover-target N   Max new leagues to discover (default 1000). 0 = skip.
    --crawl-batch N       Leagues per crawl batch (default 300).
    --interval N          Minutes between crawl batches (default 20).
    --hours N             Total hours to run (default 4).
    --no-discovery        Skip discovery, go straight to crawl.
    --analytics           Run analytics + WLS after all crawl batches complete.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now()


def main():
    parser = argparse.ArgumentParser(description="Extended trade-intel discovery + crawl.")
    parser.add_argument("--discover-target", type=int, default=1000,
                        help="Max new leagues to discover. 0 = skip discovery. Default 1000.")
    parser.add_argument("--crawl-batch",     type=int, default=300,
                        help="Leagues per crawl batch. Default 300.")
    parser.add_argument("--interval",        type=int, default=20,
                        help="Minutes between crawl batches. Default 20.")
    parser.add_argument("--hours",           type=float, default=4.0,
                        help="Total hours to run. Default 4.")
    parser.add_argument("--no-discovery",    action="store_true",
                        help="Skip discovery step.")
    parser.add_argument("--analytics",       action="store_true",
                        help="Run analytics + WLS after all crawl batches complete.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel.league_discovery import run_discovery
    from data_building.trade_intel.trade_crawler import run_crawl

    deadline = _now() + timedelta(hours=args.hours)
    logger.info("Starting extended trade-intel run. Deadline: %s", deadline.strftime("%H:%M:%S"))

    # ── Discovery ──────────────────────────────────────────────────────────
    if not args.no_discovery and args.discover_target > 0:
        logger.info("Discovery: targeting %d new leagues...", args.discover_target)
        discovered = run_discovery(target=args.discover_target)
        logger.info("Discovery complete: %d new leagues added.", discovered)
    else:
        logger.info("Discovery skipped.")

    # ── Crawl loop ─────────────────────────────────────────────────────────
    batch_num = 0
    total_trades = 0
    total_leagues_with_trades = 0

    while _now() < deadline:
        batch_num += 1
        logger.info(
            "Crawl batch %d | batch_size=%d | time remaining=%.1f min",
            batch_num,
            args.crawl_batch,
            (_now() - deadline).total_seconds() / -60,
        )

        result = run_crawl(batch_size=args.crawl_batch)
        trades   = result.get("trades", 0) or result.get("total_trades", 0)
        leagues  = result.get("leagues", 0) or result.get("total_leagues", 0)
        total_trades += trades
        total_leagues_with_trades += leagues

        logger.info(
            "Batch %d done: %d trades from %d leagues (cumulative: %d trades, %d leagues)",
            batch_num, trades, leagues, total_trades, total_leagues_with_trades,
        )

        if _now() >= deadline:
            break

        next_run = _now() + timedelta(minutes=args.interval)
        if next_run >= deadline:
            break

        sleep_secs = (next_run - _now()).total_seconds()
        logger.info("Sleeping %.0f seconds until next batch...", sleep_secs)
        time.sleep(max(sleep_secs, 0))

    logger.info(
        "Crawl complete. %d batches | %d total trades | %d leagues with trades",
        batch_num, total_trades, total_leagues_with_trades,
    )

    # ── Optional analytics + WLS ───────────────────────────────────────────
    if args.analytics:
        from dashboard_services.api import get_nfl_state
        from data_building.trade_intel.analytics import run_analytics
        from data_building.trade_intel.trade_value_model import run_trade_value_model
        from data_building.build_daily_value_table import record_calibrated_history_snapshot

        state  = get_nfl_state() or {}
        season = int(state.get("season") or 2026)

        logger.info("Running analytics for season %d...", season)
        analytics_result = run_analytics(season=season)
        logger.info("Analytics: %s", analytics_result)

        logger.info("Running WLS trade value model...")
        wls_result = run_trade_value_model(season=season)
        logger.info("WLS: %s", wls_result)

        cal_n = record_calibrated_history_snapshot()
        logger.info("Calibrated history snapshot: %d players", cal_n)


if __name__ == "__main__":
    main()
