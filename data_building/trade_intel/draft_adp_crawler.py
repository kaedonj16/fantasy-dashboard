"""
Draft ADP crawler for the Trade Intelligence Engine.

For every dynasty (league_type 2) and true-redraft (league_type 0) league in
trade_intel_leagues, fetches completed drafts from the Sleeper API and stores the
raw pick data.  After each batch the aggregated ADP table (draft_adp) is
recomputed via a single SQL upsert so callers always get up-to-date numbers.

Classification (depends on Sleeper league type - a redraft full draft looks like
a dynasty startup by round count):
  - dynasty (type 2): rounds >= 10 -> 'startup', rounds 1-5 -> 'rookie'
  - redraft (type 0): rounds >= 10 -> 'redraft'
  - keeper (type 1) is NOT crawled: kept rosters skew its draft toward rookies,
    so it does not represent redraft ADP.
  Ambiguous drafts (6-9 rounds, or short drafts in redraft leagues) are skipped.

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

def get_conn():
    """Lazy DB handle so importing this module (e.g. for the pure unit tests,
    which have no psycopg) doesn't pull in the driver until a query runs."""
    from dashboard_services.db import get_conn as _get_conn
    return _get_conn()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLEEPER_BASE = "https://api.sleeper.app/v1"

# HTTP session is built lazily (and `requests`/urllib3 imported only then) so
# importing this module for the pure unit tests doesn't require those packages.
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "fantasy-draft-adp/1.0"})
        _SESSION = s
    return _SESSION


_RATE_LIMIT_BACKOFF = 60  # seconds to wait on 429

# How many days before we re-check a league's draft list for new drafts
RECRAWL_DAYS = 30


def _get(path: str) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    session = _session()
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 429:
            logger.warning("[draft_adp] Rate limited - sleeping %ds", _RATE_LIMIT_BACKOFF)
            time.sleep(_RATE_LIMIT_BACKOFF)
            resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[draft_adp] %s failed: %s", path, exc)
        return None


def _classify_draft(draft_meta: dict, league_type: int = 2) -> Optional[str]:
    """Return the draft_type, or None for ambiguous/unknown drafts.

    Classification depends on the Sleeper league type (settings.type: 0=redraft,
    1=keeper, 2=dynasty) because a full draft looks identical by round count
    across formats:
      - dynasty (2): rounds >= 10 -> 'startup', 1-5 -> 'rookie'.
      - redraft (0): a full-roster draft (rounds >= 10) is a 'redraft' board.
      - keeper (1): NOT usable as redraft ADP — most veterans are kept, so the
        draft pool is rookies + replacements and a top rookie goes ~1.01, which
        badly skews "redraft" ADP. Skipped entirely.
    Short drafts on the redraft axis are ambiguous and skipped.
    """
    rounds = (draft_meta.get("settings") or {}).get("rounds") or 0
    try:
        rounds = int(rounds)
    except (TypeError, ValueError):
        return None
    if league_type == 2:          # dynasty
        if rounds >= 10:
            return "startup"
        if 1 <= rounds <= 5:
            return "rookie"
        return None
    if league_type == 0:          # true redraft (no keepers)
        if rounds >= 10:
            return "redraft"
        return None
    # Keeper (1) or unknown: their drafts don't represent redraft ADP.
    return None


def crawl_league_drafts(league_id: str, is_superflex: bool, num_teams: int,
                        league_type: int = 2) -> tuple[int, int]:
    """
    Fetch completed drafts for one league and persist new picks.

    league_type drives draft classification (dynasty startup/rookie vs
    true-redraft). Returns (new_picks, new_drafts).
    """
    drafts = _get(f"/league/{league_id}/drafts")
    if not drafts:
        return 0, 0

    new_picks  = 0
    new_drafts = 0

    for draft in drafts:
        draft_id = str(draft.get("draft_id") or "")
        status = draft.get("status", "")
        if not draft_id or status != "complete":
            continue

        draft_type = _classify_draft(draft, league_type)
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
                new_drafts += 1

    return new_picks, new_drafts


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
            # Only uncrawled leagues (dynasty startup/rookie + true redraft)
            query = """
                SELECT league_id,
                       COALESCE(num_teams, 12) AS num_teams,
                       COALESCE(is_superflex, FALSE) AS is_superflex,
                       COALESCE(league_type, 2) AS league_type
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type IN (0, 2)    -- 2=dynasty, 0=true redraft (keeper=1 excluded)
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
                       COALESCE(is_superflex, FALSE) AS is_superflex,
                       COALESCE(league_type, 2) AS league_type
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type IN (0, 2)    -- 2=dynasty, 0=true redraft (keeper=1 excluded)
                  AND last_draft_adp_crawled_at IS NOT NULL
                  AND last_draft_adp_crawled_at < NOW() - INTERVAL '%s days'
                ORDER BY last_draft_adp_crawled_at DESC
                LIMIT %s
            """
            params = (recrawl_days, batch_size)
        else:  # both
            # Mix of new and existing, prioritize new
            query = """
                (SELECT league_id, num_teams, is_superflex,
                        COALESCE(league_type, 2) AS league_type, 1 as priority
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type IN (0, 2)
                  AND last_draft_adp_crawled_at IS NULL
                ORDER BY discovered_at DESC
                LIMIT %s)
                UNION ALL
                (SELECT league_id, num_teams, is_superflex,
                        COALESCE(league_type, 2) AS league_type, 2 as priority
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND league_type IN (0, 2)
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

    total_new_picks  = 0
    total_new_drafts = 0
    crawled_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                crawl_league_drafts,
                row["league_id"],
                bool(row["is_superflex"]),
                int(row["num_teams"]),
                int(row["league_type"]) if row["league_type"] is not None else 2,
            ): row["league_id"]
            for row in leagues
        }
        for fut in as_completed(futures):
            league_id = futures[fut]
            try:
                n_picks, n_drafts = fut.result()
                total_new_picks  += n_picks
                total_new_drafts += n_drafts
            except Exception as exc:
                logger.warning("[draft_adp] League %s failed: %s", league_id, exc)
            crawled_ids.append(league_id)

            if len(crawled_ids) >= 50:
                _mark_leagues_crawled(crawled_ids)
                crawled_ids = []

    if crawled_ids:
        _mark_leagues_crawled(crawled_ids)

    print(f"[draft_adp] {total_new_drafts} new drafts, {total_new_picks} new picks - recomputing ADP...")
    adp_count = compute_adp()
    print(f"[draft_adp] Done. {adp_count} ADP entries across all segments.")

    return {"new_picks": total_new_picks, "new_drafts": total_new_drafts, "adp_entries": adp_count}


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
    
    batch_num         = 0
    total_new_picks   = 0
    total_new_drafts  = 0
    total_adp_entries = 0

    while datetime.now() < deadline:
        batch_num += 1
        remaining_minutes = (deadline - datetime.now()).total_seconds() / 60

        logger.info(
            "Batch %d | batch_size=%d | time remaining=%.1f min",
            batch_num, batch_size, remaining_minutes,
        )

        result = run_draft_adp_crawl(batch_size=batch_size, workers=workers)
        new_picks   = result.get("new_picks", 0)
        new_drafts  = result.get("new_drafts", 0)
        adp_entries = result.get("adp_entries", 0)
        total_new_picks   += new_picks
        total_new_drafts  += new_drafts
        total_adp_entries  = adp_entries  # total count, not cumulative

        logger.info(
            "Batch %d done: %d new drafts, %d new picks, %d ADP entries "
            "(cumulative: %d drafts, %d picks)",
            batch_num, new_drafts, new_picks, adp_entries,
            total_new_drafts, total_new_picks,
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
        "Continuous crawl complete. %d batches | %d total new drafts | %d total new picks | %d final ADP entries",
        batch_num, total_new_drafts, total_new_picks, total_adp_entries,
    )

    return {
        "batches":           batch_num,
        "total_new_drafts":  total_new_drafts,
        "total_new_picks":   total_new_picks,
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
    parser.add_argument("--crawl-mode", choices=["new", "existing", "both"], default="new",
                        help="'new' = only uncrawled leagues (default); 'existing' = re-crawl "
                             "already-crawled ones; 'both' = a mix. Use existing/both to revisit "
                             "leagues that were already stamped (e.g. to pick up new drafts).")
    parser.add_argument("--recrawl-days", type=int, default=30,
                        help="For existing/both: only re-crawl leagues not crawled in this many days")
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
        result = run_draft_adp_crawl(batch_size=args.batch_size, workers=args.workers,
                                     crawl_mode=args.crawl_mode, recrawl_days=args.recrawl_days)
    
    print(result)
