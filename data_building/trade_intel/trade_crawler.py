"""
Trade crawler for Trade Intelligence Engine.

For each known dynasty league, fetches all transactions of type 'trade'
across every week of the season and stores raw asset data in Postgres.

Idempotent: UNIQUE constraint on transaction_id prevents duplicates.
Tracks last_crawled_week so incremental runs only fetch new weeks.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from typing import Any

import requests

from dashboard_services.api import get_transactions
from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLEEPER_BASE = "https://api.sleeper.app/v1"
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure session with larger connection pool
SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(
    pool_connections=20,  # Increase from default 10
    pool_maxsize=20,      # Increase from default 10  
    max_retries=retry_strategy
)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
SESSION.headers.update({"User-Agent": "fantasy-trade-intel/1.0"})

_WEEKS_PER_SEASON = 18
# No artificial sleep — rely on 429 retry. Aggressive but polite per Sleeper docs.
_RATE_LIMIT_BACKOFF = 60   # seconds to wait on 429


def _get(path: str) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, timeout=10)
        if resp.status_code == 429:
            logger.warning("[crawler] Rate limited — sleeping %ds", _RATE_LIMIT_BACKOFF)
            time.sleep(_RATE_LIMIT_BACKOFF)
            resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[crawler] %s failed: %s", path, exc)
        return None


def _current_nfl_week() -> int:
    state = _get("/state/nfl")
    if state and "week" in state:
        week = int(state["week"])
        return week if week > 0 else _WEEKS_PER_SEASON
    return _WEEKS_PER_SEASON


def _pick_order(pick: dict) -> str | None:
    order = pick.get("order")
    if order is None:
        return None
    if order <= 4:
        return "early"
    if order <= 8:
        return "mid"
    return "late"


def _extract_assets(txn: dict) -> list[dict]:
    assets: list[dict] = []
    adds: dict[str, Any] = txn.get("adds") or {}
    drops: dict[str, Any] = txn.get("drops") or {}
    draft_picks: list[dict] = txn.get("draft_picks") or []

    all_roster_ids = sorted(set([str(r) for r in list(adds.values()) + list(drops.keys() if drops else [])]))
    if len(all_roster_ids) < 2:
        pick_rosters = sorted({str(p.get("owner_id", "")) for p in draft_picks if p.get("owner_id")})
        all_roster_ids = pick_rosters or all_roster_ids

    side_map: dict[str, str] = {}
    for i, rid in enumerate(all_roster_ids[:2]):
        side_map[str(rid)] = "a" if i == 0 else "b"

    for player_id, receiver_roster_id in adds.items():
        side = side_map.get(str(receiver_roster_id), "a")
        assets.append({
            "side": side,
            "asset_type": "player",
            "player_id": str(player_id),
            "pick_season": None,
            "pick_round": None,
            "pick_order": None,
        })

    for pick in draft_picks:
        receiver = str(pick.get("owner_id", ""))
        side = side_map.get(receiver, "a")
        assets.append({
            "side": side,
            "asset_type": "pick",
            "player_id": None,
            "pick_season": pick.get("season"),
            "pick_round": pick.get("round"),
            "pick_order": _pick_order(pick),
        })

    return assets


def _fetch_week(league_id: str, week: int) -> tuple[int, list[dict]]:
    """Fetch one week of transactions. Returns (week, trades_list)."""
    transactions = get_transactions(league_id, week)
    if not transactions:
        return week, []
    trades = [
        t for t in transactions
        if t.get("type") == "trade" and t.get("status") == "complete"
    ]
    return week, trades


def crawl_league(
    league_id: str,
    season: int,
    start_week: int = 1,
    end_week: int | None = None,
    week_workers: int = 2,
) -> int:
    """
    Crawl all trades for one league. Returns count of newly inserted trades.

    Fetches up to `week_workers` weeks in parallel, then bulk-inserts into
    a single DB connection.
    """
    if end_week is None:
        end_week = _current_nfl_week()
    elif end_week == 0:
        end_week = 1

    weeks = list(range(start_week, end_week + 1))
    if not weeks:
        return 0

    # Fetch all weeks in parallel
    week_trades: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(week_workers, len(weeks))) as pool:
        futures = {pool.submit(_fetch_week, league_id, w): w for w in weeks}
        for fut in as_completed(futures):
            try:
                w, trades = fut.result()
                if trades:
                    week_trades[w] = trades
            except Exception as exc:
                logger.debug("[crawler] week fetch failed for %s: %s", league_id, exc)

    if not week_trades:
        return 0

    # Write everything in one DB connection
    new_trades = 0
    with get_conn() as conn:
        for week, trades in sorted(week_trades.items()):
            for txn in trades:
                txn_id = str(txn.get("transaction_id", ""))
                if not txn_id:
                    continue

                created_ms = txn.get("created")
                created_at = (
                    datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
                    if created_ms else None
                )

                result = conn.execute(
                    """
                    INSERT INTO trade_intel_trades
                        (league_id, transaction_id, season, week, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                    RETURNING id
                    """,
                    (league_id, txn_id, season, week, txn.get("status", "complete"), created_at)
                ).fetchone()

                if not result:
                    continue

                trade_db_id = result["id"]
                assets = _extract_assets(txn)
                if assets:
                    conn.execute(
                        """
                        INSERT INTO trade_intel_assets
                            (trade_id, side, asset_type, player_id,
                             pick_season, pick_round, pick_order)
                        VALUES """ + ",".join(["(%s,%s,%s,%s,%s,%s,%s)"] * len(assets)),
                        [v for a in assets for v in (
                            trade_db_id, a["side"], a["asset_type"], a["player_id"],
                            a["pick_season"], a["pick_round"], a["pick_order"],
                        )]
                    )
                new_trades += 1

    return new_trades


def _leagues_to_crawl(batch_size: int = 500) -> list[dict]:
    """Return leagues ordered by least-recently crawled."""
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT league_id, season, last_crawled_week
            FROM trade_intel_leagues
            WHERE crawl_enabled = TRUE
            ORDER BY last_crawled_at ASC NULLS FIRST
            LIMIT %s
            """,
            (batch_size,)
        ).fetchall()


def _mark_crawled_batch(updates: list[tuple[int, str]]) -> None:
    """Batch-update last_crawled_at/week for multiple leagues in one query."""
    if not updates:
        return
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE trade_intel_leagues AS t
            SET last_crawled_at = NOW(),
                last_crawled_week = v.week
            FROM (VALUES """ + ",".join(["(%s::int, %s::text)"] * len(updates)) + """) AS v(week, league_id)
            WHERE t.league_id = v.league_id
            """,
            [val for week, lid in updates for val in (week, lid)]
        )


def _crawl_one(row: dict, end_week: int) -> tuple[str, int]:
    """Crawl a single league. Runs inside a thread pool worker."""
    league_id = row["league_id"]
    season = row["season"]
    start_week = (row["last_crawled_week"] or 0) + 1
    if start_week > end_week:
        return league_id, 0
    try:
        n = crawl_league(league_id, season, start_week=start_week, end_week=end_week)
        return league_id, n
    except Exception as exc:
        logger.warning("[crawler] League %s failed: %s", league_id, exc)
        return league_id, 0


def run_crawl(batch_size: int = 500, workers: int = 10) -> dict:
    """
    Crawl one batch of leagues in parallel.

    workers: concurrent leagues (default 20 — safe for I/O-bound HTTP workload)
    Each league internally fetches its weeks in parallel (up to 8 concurrent week requests).
    """
    current_week = _current_nfl_week()
    leagues = _leagues_to_crawl(batch_size)
    total_trades = 0
    total_leagues = 0
    mark_batch: list[tuple[int, str]] = []

    logger.info("[crawler] Starting crawl. Leagues=%d, Week=%d, Workers=%d", len(leagues), current_week, workers)
    logger.info("[crawler] Checkpoint: Beginning parallel crawl of %d leagues", len(leagues))
    print(f"[crawler] Crawling {len(leagues)} leagues with {workers} workers, week={current_week}")

    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        logger.info("[crawler] Checkpoint: Submitting %d leagues to thread pool", len(leagues))
        futures = {
            executor.submit(_crawl_one, row, current_week): row["league_id"]
            for row in leagues
        }
        logger.info("[crawler] Checkpoint: All leagues submitted, waiting for completion")
        
        for future in as_completed(futures):
            completed_count += 1
            league_id, n = future.result()
            mark_batch.append((current_week, league_id))
            
            if n > 0:
                total_trades += n
                total_leagues += 1
                logger.info("[crawler] Checkpoint: League %s (%d/%d): +%d trades (Total: %d trades, %d leagues)", 
                           league_id, completed_count, len(leagues), n, total_trades, total_leagues)
            else:
                logger.info("[crawler] Checkpoint: League %s (%d/%d): No new trades", league_id, completed_count, len(leagues))

            # Flush mark batch every 50 to avoid holding too many updates
            if len(mark_batch) >= 50:
                logger.info("[crawler] Checkpoint: Flushing mark batch of %d completed leagues", len(mark_batch))
                _mark_crawled_batch(mark_batch)
                mark_batch = []
                logger.info("[crawler] Checkpoint: Mark batch flushed")

    if mark_batch:
        logger.info("[crawler] Checkpoint: Final flush of %d remaining completed leagues", len(mark_batch))
        _mark_crawled_batch(mark_batch)
        logger.info("[crawler] Checkpoint: Final mark batch flushed")

    logger.info("[crawler] Checkpoint: Crawl complete. Processed %d leagues total", completed_count)
    logger.info("[crawler] Done. %d new trades across %d leagues.", total_trades, total_leagues)
    print(f"[crawler] Done. {total_trades} new trades across {total_leagues} leagues.")
    return {"leagues_crawled": total_leagues, "new_trades": total_trades}


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    result = run_crawl()
    print(result)
