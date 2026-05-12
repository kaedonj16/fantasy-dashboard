"""
Draft ADP crawler for the Trade Intelligence Engine.

For every dynasty league in trade_intel_leagues, fetches completed startup and
rookie drafts from the Sleeper API and stores the raw pick data.  After each
batch the aggregated ADP table (draft_adp) is recomputed via a single SQL
upsert so callers always get up-to-date numbers.

Classification:
  - startup  : drafts with rounds >= 10  (full keeper/dynasty startup)
  - rookie   : drafts with rounds 1-5    (annual rookie-only draft)
  Drafts with 6-9 rounds are skipped — they're ambiguous.

Idempotent: draft_adp_drafts.draft_id is a PRIMARY KEY, so re-running never
double-counts picks.  Leagues are stamped with last_draft_adp_crawled_at so
the daily run only re-checks leagues whose stamp is older than RECRAWL_DAYS.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLEEPER_BASE = "https://api.sleeper.app/v1"

# Re-use a session with retry/backoff identical to trade_crawler
_SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=_retry)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)
_SESSION.headers.update({"User-Agent": "fantasy-draft-adp/1.0"})

_RATE_LIMIT_BACKOFF = 60  # seconds to wait on 429

# How many days before we re-check a league's draft list for new drafts
RECRAWL_DAYS = 30


def _get(path: str) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = _SESSION.get(url, timeout=10)
        if resp.status_code == 429:
            logger.warning("[draft_adp] Rate limited — sleeping %ds", _RATE_LIMIT_BACKOFF)
            time.sleep(_RATE_LIMIT_BACKOFF)
            resp = _SESSION.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[draft_adp] %s failed: %s", path, exc)
        return None


def _classify_draft(draft_meta: dict) -> Optional[str]:
    """Return 'startup', 'rookie', or None for ambiguous/unknown drafts."""
    rounds = (draft_meta.get("settings") or {}).get("rounds") or 0
    try:
        rounds = int(rounds)
    except (TypeError, ValueError):
        return None
    if rounds >= 10:
        return "startup"
    if 1 <= rounds <= 5:
        return "rookie"
    return None


def crawl_league_drafts(league_id: str, is_superflex: bool, num_teams: int) -> int:
    """
    Fetch completed drafts for one league and persist new picks.

    Returns the number of newly inserted pick rows.
    """
    drafts = _get(f"/league/{league_id}/drafts")
    if not drafts:
        return 0

    new_picks = 0

    for draft in drafts:
        draft_id = str(draft.get("draft_id") or "")
        status = draft.get("status", "")
        if not draft_id or status != "complete":
            continue

        draft_type = _classify_draft(draft)
        if not draft_type:
            continue

        # Skip drafts already indexed
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM draft_adp_drafts WHERE draft_id = %s",
                (draft_id,),
            ).fetchone()
        if existing:
            continue

        season_raw = draft.get("season")
        try:
            season = int(season_raw)
        except (TypeError, ValueError):
            continue

        settings = draft.get("settings") or {}
        rounds = settings.get("rounds")
        try:
            rounds = int(rounds) if rounds is not None else None
        except (TypeError, ValueError):
            rounds = None

        picks = _get(f"/draft/{draft_id}/picks")
        if not picks:
            continue

        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO draft_adp_drafts
                    (draft_id, league_id, season, draft_type, num_teams,
                     is_superflex, rounds, status, total_picks, crawled_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (draft_id) DO NOTHING
                """,
                (
                    draft_id, league_id, season, draft_type,
                    num_teams, is_superflex, rounds, status, len(picks),
                ),
            )

            rows = []
            for pick in picks:
                player_id = pick.get("player_id")
                if not player_id:
                    continue
                pick_no = pick.get("pick_no")
                if pick_no is None:
                    continue
                rows.append((
                    draft_id,
                    str(player_id),
                    int(pick_no),
                    pick.get("round"),
                    pick.get("draft_slot"),
                    str(pick.get("roster_id")) if pick.get("roster_id") is not None else None,
                ))

            if rows:
                conn.execute(
                    "INSERT INTO draft_adp_picks "
                    "    (draft_id, player_id, pick_no, round, pick_in_round, roster_id) "
                    "VALUES " + ",".join(["(%s,%s,%s,%s,%s,%s)"] * len(rows)) +
                    " ON CONFLICT (draft_id, pick_no) DO NOTHING",
                    [v for row in rows for v in row],
                )
                new_picks += len(rows)

    return new_picks


def _mark_leagues_crawled(league_ids: list[str]) -> None:
    if not league_ids:
        return
    with get_conn() as conn:
        conn.execute(
            "UPDATE trade_intel_leagues "
            "SET last_draft_adp_crawled_at = NOW() "
            "WHERE league_id = ANY(%s)",
            (league_ids,),
        )


def compute_adp() -> int:
    """
    Recompute the draft_adp aggregate table from raw pick data.

    Uses a single SQL upsert so this is safe to call repeatedly.
    Returns the total number of rows in draft_adp after the update.
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO draft_adp
                (player_id, draft_type, season, is_superflex, num_teams,
                 avg_pick, std_pick, avg_round, sample_size, updated_at)
            SELECT
                p.player_id,
                d.draft_type,
                d.season,
                COALESCE(d.is_superflex, FALSE),
                COALESCE(d.num_teams, 12),
                ROUND(AVG(p.pick_no)::numeric, 2),
                ROUND(STDDEV(p.pick_no)::numeric, 2),
                ROUND(AVG(p.round)::numeric, 2),
                COUNT(*),
                NOW()
            FROM draft_adp_picks p
            JOIN draft_adp_drafts d ON d.draft_id = p.draft_id
            GROUP BY p.player_id, d.draft_type, d.season,
                     COALESCE(d.is_superflex, FALSE), COALESCE(d.num_teams, 12)
            ON CONFLICT (player_id, draft_type, season, is_superflex, num_teams)
            DO UPDATE SET
                avg_pick    = EXCLUDED.avg_pick,
                std_pick    = EXCLUDED.std_pick,
                avg_round   = EXCLUDED.avg_round,
                sample_size = EXCLUDED.sample_size,
                updated_at  = EXCLUDED.updated_at
            """
        )
        row = conn.execute("SELECT COUNT(*) AS n FROM draft_adp").fetchone()
        return row["n"] if row else 0


def _leagues_to_crawl(batch_size: int, crawl_mode: str = "both", recrawl_days: int = 2) -> list[dict]:
    """
    Return dynasty leagues based on crawl mode.

    crawl_mode:
      - "new": Only uncrawled leagues
      - "existing": Previously crawled leagues not crawled in recrawl_days
      - "both": Mix of new and existing leagues
    """
    with get_conn() as conn:
        if crawl_mode == "new":
            # Only uncrawled dynasty leagues
            query = """
                SELECT league_id,
                       COALESCE(num_teams, 12) AS num_teams,
                       COALESCE(is_superflex, FALSE) AS is_superflex
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type = 2          -- dynasty only
                  AND last_draft_adp_crawled_at IS NULL
                ORDER BY discovered_at DESC
                LIMIT %s
            """
            params = (batch_size,)
        elif crawl_mode == "existing":
            # Only previously crawled leagues, but not recently
            query = """
                SELECT league_id,
                       COALESCE(num_teams, 12) AS num_teams,
                       COALESCE(is_superflex, FALSE) AS is_superflex
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type = 2          -- dynasty only
                  AND last_draft_adp_crawled_at IS NOT NULL
                  AND last_draft_adp_crawled_at < NOW() - INTERVAL '%s days'
                ORDER BY last_draft_adp_crawled_at DESC
                LIMIT %s
            """
            params = (recrawl_days, batch_size)
        else:  # both
            # Mix of new and existing, prioritize new
            query = """
                (SELECT league_id, num_teams, is_superflex, 1 as priority
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type = 2
                  AND last_draft_adp_crawled_at IS NULL
                ORDER BY discovered_at DESC
                LIMIT %s)
                UNION ALL
                (SELECT league_id, num_teams, is_superflex, 2 as priority
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type = 2
                  AND last_draft_adp_crawled_at IS NOT NULL
                  AND last_draft_adp_crawled_at < NOW() - INTERVAL '%s days'
                ORDER BY last_draft_adp_crawled_at DESC
                LIMIT %s)
                ORDER BY priority ASC, league_id
                LIMIT %s
            """
            # Split batch between new and existing (70% new, 30% existing)
            new_batch = int(batch_size * 0.7)
            existing_batch = batch_size - new_batch
            params = (new_batch, recrawl_days, existing_batch, batch_size)
        
        return conn.execute(query, params).fetchall()


def run_draft_adp_crawl(batch_size: int = 2000, workers: int = 10, crawl_mode: str = "new", recrawl_days: int = 30) -> dict:
    """
    Crawl draft pick data from all eligible dynasty leagues and recompute ADP.

    batch_size : max leagues to process per run
    workers    : concurrent league crawlers

    Returns a summary dict with new_picks and adp_entries.
    """
    leagues = _leagues_to_crawl(batch_size, crawl_mode, recrawl_days)
    if not leagues:
        print("[draft_adp] No leagues need crawling right now.")
        return {"new_picks": 0, "adp_entries": 0}

    print(f"[draft_adp] Crawling drafts for {len(leagues)} leagues with {workers} workers")

    total_new_picks = 0
    crawled_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                crawl_league_drafts,
                row["league_id"],
                bool(row["is_superflex"]),
                int(row["num_teams"]),
            ): row["league_id"]
            for row in leagues
        }
        for fut in as_completed(futures):
            league_id = futures[fut]
            try:
                n = fut.result()
                total_new_picks += n
            except Exception as exc:
                logger.warning("[draft_adp] League %s failed: %s", league_id, exc)
            crawled_ids.append(league_id)

            # Flush stamps in batches of 50 to avoid large single-update delays
            if len(crawled_ids) >= 50:
                _mark_leagues_crawled(crawled_ids)
                crawled_ids = []

    if crawled_ids:
        _mark_leagues_crawled(crawled_ids)

    print(f"[draft_adp] {total_new_picks} new picks stored — recomputing ADP...")
    adp_count = compute_adp()
    print(f"[draft_adp] Done. {adp_count} ADP entries across all segments.")

    return {"new_picks": total_new_picks, "adp_entries": adp_count}


def run_draft_adp_crawl_continuous(
    batch_size: int = 2000,
    workers: int = 10,
    interval_minutes: int = 30,
    hours: float = 4.0,
) -> dict:
    """
    Run draft ADP crawl continuously for a given time period.
    
    Parameters:
    - batch_size: max leagues to process per batch
    - workers: concurrent league crawlers
    - interval_minutes: minutes between crawl batches
    - hours: total hours to run
    
    Returns summary dict with cumulative results.
    """
    from datetime import datetime, timedelta
    
    deadline = datetime.now() + timedelta(hours=hours)
    logger.info("Starting continuous draft ADP crawl. Deadline: %s", deadline.strftime("%H:%M:%S"))
    
    batch_num = 0
    total_new_picks = 0
    total_adp_entries = 0
    
    while datetime.now() < deadline:
        batch_num += 1
        remaining_minutes = (deadline - datetime.now()).total_seconds() / 60
        
        logger.info(
            "Batch %d | batch_size=%d | time remaining=%.1f min",
            batch_num,
            batch_size,
            remaining_minutes,
        )
        
        result = run_draft_adp_crawl(batch_size=batch_size, workers=workers)
        new_picks = result.get("new_picks", 0)
        adp_entries = result.get("adp_entries", 0)
        total_new_picks += new_picks
        total_adp_entries = adp_entries  # ADP entries are total count, not cumulative
        
        logger.info(
            "Batch %d done: %d new picks, %d ADP entries (cumulative picks: %d)",
            batch_num, new_picks, adp_entries, total_new_picks,
        )
        
        if datetime.now() >= deadline:
            break
        
        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        if next_run >= deadline:
            break
        
        sleep_secs = (next_run - datetime.now()).total_seconds()
        logger.info("Sleeping %.0f seconds until next batch...", sleep_secs)
        time.sleep(max(sleep_secs, 0))
    
    logger.info(
        "Continuous crawl complete. %d batches | %d total new picks | %d final ADP entries",
        batch_num, total_new_picks, total_adp_entries,
    )
    
    return {
        "batches": batch_num,
        "total_new_picks": total_new_picks,
        "final_adp_entries": total_adp_entries,
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Draft ADP crawler")
    parser.add_argument("--continuous", action="store_true", help="Run continuously for given time")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between batches (continuous mode)")
    parser.add_argument("--hours", type=float, default=4.0, help="Hours to run (continuous mode)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Leagues per batch")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if args.continuous:
        result = run_draft_adp_crawl_continuous(
            batch_size=args.batch_size,
            workers=args.workers,
            interval_minutes=args.interval,
            hours=args.hours,
        )
    else:
        result = run_draft_adp_crawl(batch_size=args.batch_size, workers=args.workers)
    
    print(result)
