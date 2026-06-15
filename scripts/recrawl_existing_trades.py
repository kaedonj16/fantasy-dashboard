"""
Recurring re-crawl of already-known leagues to keep the trade feed fresh.

The daily cron's trade crawl runs in crawl_mode="new", which only ever visits
leagues that have never been crawled. Once a league is crawled once it is never
revisited, so new trades happening in the thousands of known leagues are missed.

This script is the missing piece: a single-shot re-crawl of EXISTING leagues
(crawl_mode="existing"), designed to run as its own Render cron every few hours.
It does one crawl batch and exits — it does not loop/sleep like
run_trade_intel_extended.py, because the cron scheduler handles cadence.

WLS value calibration is intentionally left to the daily cron; this job only
keeps fresh trades flowing into the DB (and refreshes analytics so they surface
in the trade UI immediately).

Usage
-----
    python scripts/recrawl_existing_trades.py                  # batch 250, 3 workers
    python scripts/recrawl_existing_trades.py --analytics      # also refresh analytics
    python scripts/recrawl_existing_trades.py --batch-size 400 --recrawl-days 2
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-shot re-crawl of existing leagues.")
    parser.add_argument("--batch-size",   type=int, default=250,
                        help="Leagues to re-crawl this run. Default 250.")
    parser.add_argument("--workers",       type=int, default=3,
                        help="Concurrent crawl workers. Kept low to avoid DB "
                             "connection drops. Default 3.")
    parser.add_argument("--recrawl-days",  type=int, default=3,
                        help="Only re-crawl leagues not crawled in the last X days. "
                             "Default 3 (full pool cycles every ~3 days).")
    parser.add_argument("--analytics",     action="store_true",
                        help="Refresh trade analytics after crawling.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel.trade_crawler import run_crawl

    logger.info(
        "Re-crawl: batch_size=%d workers=%d recrawl_days=%d",
        args.batch_size, args.workers, args.recrawl_days,
    )
    result = run_crawl(
        batch_size=args.batch_size,
        workers=args.workers,
        crawl_mode="existing",
        recrawl_days=args.recrawl_days,
    )
    logger.info(
        "Re-crawl done: %d trades from %d leagues",
        result.get("new_trades", 0), result.get("leagues_crawled", 0),
    )

    if args.analytics:
        try:
            from dashboard_services.api import get_nfl_state
            from data_building.trade_intel.analytics import run_analytics

            state  = get_nfl_state() or {}
            season = int(state.get("season") or 2026)
            logger.info("Refreshing analytics for season %d...", season)
            analytics_result = run_analytics(season=season)
            logger.info("Analytics: %s", analytics_result)
        except Exception as e:
            logger.warning("Analytics refresh failed (non-fatal): %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
