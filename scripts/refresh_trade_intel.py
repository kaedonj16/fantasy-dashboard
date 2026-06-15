"""
Recurring trade-intel refresh: GROW the league pool (discover new leagues) AND
keep it FRESH (re-crawl known leagues for new trades) — in a single shot.

The daily cron only discovers a small batch and crawls crawl_mode="new", so it
never revisits known leagues and the pool barely expands. This job runs as its
own Render cron every few hours and does three things:

    1. Discovery  — BFS-expand from known leagues to find NEW ones (expansion).
    2. Crawl      — crawl_mode="both": crawls freshly-discovered leagues AND
                    re-crawls existing ones not seen in the last recrawl window.
    3. Analytics  — refresh trade aggregates so new trades surface in the UI.

It is single-shot (no sleep/loop) — the cron scheduler handles cadence, unlike
the long-running run_trade_intel_extended.py. WLS value calibration stays in the
daily cron.

Usage
-----
    python scripts/refresh_trade_intel.py                       # discover 500 + crawl 2000 (both)
    python scripts/refresh_trade_intel.py --no-discovery        # skip expansion, crawl only
    python scripts/refresh_trade_intel.py --batch-size 3000 --discover-target 1000
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-shot trade-intel discovery + crawl.")
    parser.add_argument("--discover-target", type=int, default=500,
                        help="Max NEW leagues to discover this run (expansion). 0 = skip. Default 500.")
    parser.add_argument("--no-discovery",    action="store_true",
                        help="Skip discovery entirely (crawl only).")
    parser.add_argument("--batch-size",      type=int, default=2000,
                        help="Leagues to crawl this run. Default 2000.")
    parser.add_argument("--workers",         type=int, default=4,
                        help="Concurrent crawl workers. Kept moderate to avoid DB "
                             "connection drops (8 caused drops in manual runs). Default 4.")
    parser.add_argument("--crawl-mode",      choices=["new", "existing", "both"], default="both",
                        help="'new' (uncrawled), 'existing' (re-crawl), 'both' (mixed). Default both.")
    parser.add_argument("--recrawl-days",    type=int, default=2,
                        help="For existing/both: only re-crawl leagues not crawled in the last X days. "
                             "Default 2.")
    parser.add_argument("--analytics",       action="store_true",
                        help="Refresh trade analytics after crawling.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel.trade_crawler import run_crawl

    # ── 1. Discovery (expansion) ───────────────────────────────────────────
    if not args.no_discovery and args.discover_target > 0:
        try:
            from data_building.trade_intel.league_discovery import run_discovery
            logger.info("Discovery: targeting %d new leagues...", args.discover_target)
            discovered = run_discovery(target=args.discover_target)
            logger.info("Discovery: %d new leagues added.", discovered)
        except Exception as e:
            logger.warning("Discovery failed (non-fatal): %s", e)
    else:
        logger.info("Discovery skipped.")

    # ── 2. Crawl (new + existing) ──────────────────────────────────────────
    logger.info(
        "Crawl: batch_size=%d workers=%d mode=%s recrawl_days=%d",
        args.batch_size, args.workers, args.crawl_mode, args.recrawl_days,
    )
    result = run_crawl(
        batch_size=args.batch_size,
        workers=args.workers,
        crawl_mode=args.crawl_mode,
        recrawl_days=args.recrawl_days,
    )
    logger.info(
        "Crawl done: %d trades from %d leagues",
        result.get("new_trades", 0), result.get("leagues_crawled", 0),
    )

    # ── 3. Analytics ───────────────────────────────────────────────────────
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
