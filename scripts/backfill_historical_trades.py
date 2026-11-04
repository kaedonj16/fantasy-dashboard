"""
Backfill prior-season trades into the trade database.

Why this exists
---------------
Discovery walks Sleeper previous_league_id chains, so prior-season
ancestor leagues sit in trade_intel_leagues - but the recurring recrawl
historically fetched them only up to the *current* season's week and
ranked never-crawled leagues behind every trade-active league, so
seasons before the current one collected (almost) no trades at all.

This job selects prior-season leagues whose crawl is incomplete
(never crawled, or last_crawled_week < 18) and crawls each one over
its full season: weeks 0..18 (week 0 is Sleeper's offseason bucket).
Inserts are idempotent (ON CONFLICT transaction_id DO NOTHING), and a
league is marked at week 18 only after its crawl ran, so re-running
the job resumes where it left off and steady-state runs find nothing
to do.

Runs as its own Render cron (trade-history-backfill) and by hand:

    python scripts/backfill_historical_trades.py                 # all prior seasons
    python scripts/backfill_historical_trades.py --seasons 2025,2024
    python scripts/backfill_historical_trades.py --full          # re-scan weeks 0..18 even if marked complete
    python scripts/backfill_historical_trades.py --dry-run       # report only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Running `python scripts/...` puts scripts/ on sys.path, not the repo
# root, so `import data_building` fails. Add the root explicitly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _historical_leagues(seasons: list[int], batch_size: int,
                        full: bool = False) -> list[dict]:
    """Prior-season leagues whose full-season crawl is incomplete.

    With ``full``, every league of the given seasons is returned (the
    crawl re-scans weeks 0..18; inserts dedupe). Otherwise only leagues
    that have not yet been fetched through week 18.
    """
    from dashboard_services.db import get_conn

    completeness = "" if full else """
              AND (last_crawled_week IS NULL OR last_crawled_week < 18)
    """
    with get_conn() as conn:
        return conn.execute(
            f"""
            SELECT league_id, season, last_crawled_week, league_type
            FROM trade_intel_leagues
            WHERE crawl_enabled = TRUE
              AND league_type IN (0, 1, 2)
              AND season = ANY(%s)
              {completeness}
            ORDER BY season DESC, discovered_at ASC, league_id ASC
            LIMIT %s
            """,
            (list(seasons), batch_size),
        ).fetchall()


def _start_week_for(row: dict, full: bool) -> int:
    if full or row.get("last_crawled_week") is None:
        return 0  # week 0 = Sleeper's offseason trade bucket
    return int(row["last_crawled_week"]) + 1


def run_backfill(seasons: list[int] | None = None, batch_size: int = 1000,
                 workers: int = 8, full: bool = False,
                 dry_run: bool = False) -> dict:
    """Crawl the selected prior-season leagues over their full seasons."""
    from data_building.trade_intel.trade_crawler import (
        _WEEKS_PER_SEASON,
        _current_nfl_season,
        _mark_crawled_batch,
        crawl_league,
    )

    current_season = _current_nfl_season()
    if seasons is None:
        # The four seasons before the current one: chain depth in
        # discovery reaches ~12, but trade value in older seasons decays
        # fast; four back keeps the one-time cost bounded.
        seasons = [current_season - i for i in range(1, 5)]
    seasons = sorted({int(s) for s in seasons if int(s) < current_season},
                     reverse=True)
    if not seasons:
        logger.info("[backfill] No prior seasons requested.")
        return {"leagues": 0, "new_trades": 0, "by_season": {}}

    leagues = _historical_leagues(seasons, batch_size, full=full)
    logger.info(
        "[backfill] %d incomplete prior-season leagues (seasons %s, full=%s)",
        len(leagues), seasons, full,
    )
    if dry_run or not leagues:
        return {"leagues": len(leagues), "new_trades": 0, "by_season": {}}

    def _one(row):
        start = _start_week_for(row, full)
        try:
            n = crawl_league(row["league_id"], int(row["season"]),
                             start_week=start,
                             end_week=_WEEKS_PER_SEASON)
        except Exception as exc:
            logger.warning("[backfill] League %s failed: %s",
                           row["league_id"], exc)
            return row, start, 0, False
        return row, start, n, True

    total = 0
    by_season: dict[int, int] = {}
    mark_batch: list[tuple[int, str, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, row) for row in leagues]
        for fut in as_completed(futures):
            row, start, n, ok = fut.result()
            if not ok:
                # Leave last_crawled_week untouched so a later run retries.
                continue
            total += n
            by_season[int(row["season"])] = by_season.get(int(row["season"]), 0) + n
            # The crawl covered this league through week 18, starting at
            # its previous frontier (or week 0 when full/never crawled),
            # so marking week 18 is honest.
            mark_batch.append((_WEEKS_PER_SEASON, row["league_id"], n))
            if len(mark_batch) >= 50:
                _mark_crawled_batch(mark_batch)
                mark_batch = []
    if mark_batch:
        _mark_crawled_batch(mark_batch)

    logger.info("[backfill] Done. %d new trades across %d leagues. By season: %s",
                total, len(leagues), by_season)
    return {"leagues": len(leagues), "new_trades": total,
            "by_season": by_season}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill prior-season trades (full-season crawls).")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated seasons, e.g. 2025,2024. "
                             "Default: the four seasons before the current one.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Max leagues this run. Default 1000.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent leagues. Default 8.")
    parser.add_argument("--full", action="store_true",
                        help="Re-scan weeks 0..18 even for leagues already "
                             "marked complete (inserts dedupe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report how many leagues qualify; crawl nothing.")
    args = parser.parse_args()

    seasons = None
    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    result = run_backfill(seasons=seasons, batch_size=args.batch_size,
                          workers=args.workers, full=args.full,
                          dry_run=args.dry_run)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
