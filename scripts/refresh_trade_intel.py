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

The three stages run in **separate subprocesses**. They share no data, and
CPython does not reliably return a stage's peak RSS to the OS after gc.collect()
— stacking discovery + crawl + analytics in one 512Mi process is what OOM'd
the starter cron. Each child is a fresh interpreter, matching cron_daily.py.

Usage
-----
    python scripts/refresh_trade_intel.py                       # discover 500 + crawl 1000 (both)
    python scripts/refresh_trade_intel.py --no-discovery        # skip expansion, crawl only
    python scripts/refresh_trade_intel.py --batch-size 3000 --discover-target 1000
    python scripts/refresh_trade_intel.py --stage crawl         # one stage (used by the orchestrator)
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

# Running `python scripts/refresh_trade_intel.py` puts scripts/ on sys.path, not
# the project root, so `import data_building` fails with ModuleNotFoundError. Add
# the repo root (parent of scripts/) explicitly so the package imports resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _run_discovery(target: int) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    from data_building.trade_intel.league_discovery import run_discovery

    logger.info("Discovery: targeting %d new leagues...", target)
    discovered = run_discovery(target=target)
    logger.info("Discovery: %d new leagues added.", discovered)
    return 0


def _run_crawl(batch_size: int, workers: int, crawl_mode: str, recrawl_days: int) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    from data_building.trade_intel.trade_crawler import run_crawl

    logger.info(
        "Crawl: batch_size=%d workers=%d mode=%s recrawl_days=%d",
        batch_size, workers, crawl_mode, recrawl_days,
    )
    result = run_crawl(
        batch_size=batch_size,
        workers=workers,
        crawl_mode=crawl_mode,
        recrawl_days=recrawl_days,
    )
    logger.info(
        "Crawl done: %d trades from %d leagues",
        result.get("new_trades", 0), result.get("leagues_crawled", 0),
    )
    return 0


def _run_analytics() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    # Let analytics pick the season that actually has trades (Sleeper's
    # current season is often the upcoming one during the offseason). Avoid
    # dashboard_services.api here — that import pulls Flask into the 512Mi box.
    from data_building.trade_intel.analytics import run_analytics

    logger.info("Refreshing analytics...")
    analytics_result = run_analytics()
    logger.info("Analytics: %s", analytics_result)
    return 0


def _spawn_stage(stage: str, extra: list[str]) -> int:
    """Run one stage in a fresh interpreter so its RSS is fully released."""
    cmd = [sys.executable, os.path.abspath(__file__), "--stage", stage, *extra]
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    logger.info("Starting %s stage: %s", stage, " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-shot trade-intel discovery + crawl.")
    parser.add_argument("--discover-target", type=int, default=500,
                        help="Max NEW leagues to discover this run (expansion). 0 = skip. Default 500.")
    parser.add_argument("--no-discovery",    action="store_true",
                        help="Skip discovery entirely (crawl only).")
    parser.add_argument("--batch-size",      type=int, default=1000,
                        help="Leagues to crawl this run. Default 1000.")
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
    parser.add_argument("--stage", choices=["all", "discovery", "crawl", "analytics"],
                        default="all",
                        help="Internal: run one stage in-process. Default 'all' orchestrates "
                             "each stage as a subprocess.")
    args = parser.parse_args()

    if args.stage == "discovery":
        if args.no_discovery or args.discover_target <= 0:
            logger.info("Discovery skipped.")
            return 0
        try:
            return _run_discovery(args.discover_target)
        except Exception as e:
            logger.warning("Discovery failed (non-fatal): %s", e)
            return 0

    if args.stage == "crawl":
        return _run_crawl(args.batch_size, args.workers, args.crawl_mode, args.recrawl_days)

    if args.stage == "analytics":
        try:
            return _run_analytics()
        except Exception as e:
            logger.warning("Analytics refresh failed (non-fatal): %s", e)
            return 0

    # ── Orchestrator: one subprocess per stage so RSS resets between them ──
    if not args.no_discovery and args.discover_target > 0:
        rc = _spawn_stage("discovery", ["--discover-target", str(args.discover_target)])
        if rc != 0:
            logger.warning("Discovery subprocess exited %s (continuing to crawl)", rc)
    else:
        logger.info("Discovery skipped.")

    rc = _spawn_stage("crawl", [
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--crawl-mode", args.crawl_mode,
        "--recrawl-days", str(args.recrawl_days),
    ])
    if rc != 0:
        logger.error("Crawl subprocess exited %s", rc)
        return rc

    if args.analytics:
        rc = _spawn_stage("analytics", [])
        if rc != 0:
            logger.warning("Analytics subprocess exited %s (non-fatal)", rc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
