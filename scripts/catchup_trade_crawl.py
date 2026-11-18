"""
Catch-up crawl for current-season dynasty + redraft trades.

Why this exists
---------------
From 2026-11-05 the nightly trade-recrawl (crawl_mode="both") inserted
zero dynasty (league_type 2) and zero redraft (0) trades for 13 days
while keeper kept flowing: last_trade_at was stamped with the crawl
time, so the velocity-ordered batch was captured by whichever format
yielded last, and already-marked weeks were never refetched. The
crawler fix (per-format batch guarantees, real trade timestamps,
last-week refetch) stops it happening again, but the missed trades
still have to be collected - this job collects them.

It selects current-season dynasty + redraft leagues that have traded
before (total_trades > 0) and re-scans each one over its full season:
weeks 0..season-end (week 0 is Sleeper's offseason bucket). Inserts
are idempotent (ON CONFLICT transaction_id DO NOTHING), so only the
trades missed since Nov 5 actually land. Leagues are picked
least-recently-crawled first and marked as each batch completes, so
re-running the job resumes where the last run stopped; a full pass
ends when a selection comes back empty.

Runs by hand (Render WEB shell, after the crawler fix is deployed):

    python scripts/catchup_trade_crawl.py                  # drain the pool
    python scripts/catchup_trade_crawl.py --max-batches 2  # two batches only
    python scripts/catchup_trade_crawl.py --dry-run        # report only
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

# Dynasty (2) and redraft (0): the two formats the starved recrawl
# missed. Keeper (1) kept flowing and needs no catch-up.
_CATCHUP_LEAGUE_TYPES = (0, 2)
_FORMAT_NAMES = {0: "Redraft", 2: "Dynasty"}


def _catchup_leagues(season: int, batch_size: int) -> list[dict]:
    """Current-season dynasty/redraft leagues due a full rescan.

    Least-recently-crawled first: completed leagues are marked with a
    fresh last_crawled_at, so the next selection naturally continues
    where the previous batch stopped.
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        return conn.execute(
            """
            SELECT league_id, season, last_crawled_week, league_type
            FROM trade_intel_leagues
            WHERE crawl_enabled = TRUE
              AND season = %s
              AND league_type IN (0, 2)
              AND total_trades > 0
            ORDER BY last_crawled_at ASC NULLS FIRST, league_id ASC
            LIMIT %s
            """,
            (int(season), batch_size),
        ).fetchall()


def run_catchup(season: int | None = None, batch_size: int = 500,
                workers: int = 8, max_batches: int = 0,
                dry_run: bool = False) -> dict:
    """Full-rescan current-season dynasty + redraft leagues, in batches."""
    from data_building.trade_intel.trade_crawler import (
        _current_nfl_season,
        _current_nfl_week,
        _mark_crawled_batch,
        _season_end_week,
        crawl_league_detailed,
    )

    if season is None:
        season = _current_nfl_season()
    season = int(season)
    current_week = _current_nfl_week()
    current_season = _current_nfl_season()
    end_week = _season_end_week(season, current_season, current_week)

    if dry_run:
        leagues = _catchup_leagues(season, batch_size)
        counts = {0: 0, 2: 0}
        for row in leagues:
            counts[int(row["league_type"])] = \
                counts.get(int(row["league_type"]), 0) + 1
        print(f"[catchup] Dry run: season {season}, first batch of "
              f"{batch_size} holds {len(leagues)} leagues "
              f"(dynasty {counts.get(2, 0)}, redraft {counts.get(0, 0)}). "
              f"Nothing crawled.", flush=True)
        return {"leagues": len(leagues), "new_trades": 0,
                "by_format": counts, "batches": 0}

    def _one(row):
        try:
            n, max_created_at = crawl_league_detailed(
                row["league_id"], int(row["season"]),
                start_week=0, end_week=end_week)
        except Exception as exc:
            logger.warning("[catchup] League %s failed: %s",
                           row["league_id"], exc)
            return row, 0, None, False
        return row, n, max_created_at, True

    total_trades = 0
    total_leagues = 0
    by_format: dict[int, int] = {0: 0, 2: 0}
    batches = 0
    while True:
        if max_batches and batches >= max_batches:
            break
        leagues = _catchup_leagues(season, batch_size)
        if not leagues:
            break
        batches += 1
        batch_trades = 0
        batch_by_format: dict[int, int] = {0: 0, 2: 0}
        mark_batch: list[tuple] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_one, row) for row in leagues]
            for fut in as_completed(futures):
                row, n, max_created_at, ok = fut.result()
                if not ok:
                    # Leave the league unmarked so a later run retries.
                    continue
                fmt = int(row["league_type"])
                batch_trades += n
                batch_by_format[fmt] = batch_by_format.get(fmt, 0) + n
                mark_batch.append(
                    (end_week, row["league_id"], n, max_created_at))
                if len(mark_batch) >= 50:
                    _mark_crawled_batch(mark_batch)
                    mark_batch = []
        if mark_batch:
            _mark_crawled_batch(mark_batch)
        total_trades += batch_trades
        total_leagues += len(leagues)
        for fmt, n in batch_by_format.items():
            by_format[fmt] = by_format.get(fmt, 0) + n
        print(f"[catchup] Batch {batches}: {batch_trades} new trades "
              f"across {len(leagues)} leagues "
              f"(dynasty {batch_by_format.get(2, 0)}, "
              f"redraft {batch_by_format.get(0, 0)}).", flush=True)

    print(f"[catchup] Done. {total_trades} new trades across "
          f"{total_leagues} leagues in {batches} batch(es) "
          f"(dynasty {by_format.get(2, 0)}, "
          f"redraft {by_format.get(0, 0)}).", flush=True)
    return {"leagues": total_leagues, "new_trades": total_trades,
            "by_format": by_format, "batches": batches}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Catch up current-season dynasty + redraft trades "
                    "missed since 2026-11-05 (full-season rescans).")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to catch up. Default: the current "
                             "NFL season.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Leagues per batch. Default 500.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent leagues. Default 8.")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Stop after this many batches. "
                             "Default 0 = run until the pool is drained.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the first batch's selection counts "
                             "by format; crawl nothing.")
    args = parser.parse_args()

    result = run_catchup(season=args.season, batch_size=args.batch_size,
                         workers=args.workers, max_batches=args.max_batches,
                         dry_run=args.dry_run)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
