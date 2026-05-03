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
from typing import Any, Optional, Tuple

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
    pool_connections=4,
    pool_maxsize=8,
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


def _fetch_draft_slot_map(league_id: str) -> dict[tuple, int]:
    """
    Returns {(season_str, roster_id_str): slot} by reading the draft_order
    from every draft in this league.

    Uses slot_to_roster_id (slot → roster_id) from the draft detail, inverted
    to roster_id → slot. draft_order maps user_id → slot (not roster_id).
    The slot is round-agnostic: a team at slot 6 has pick X.06 in every round.

    Only populated once the commissioner has set the draft order for a season.
    """
    drafts = _get(f"/league/{league_id}/drafts")
    if not drafts:
        return {}

    slot_map: dict[tuple, int] = {}
    for d in drafts:
        season   = str(d.get("season", ""))
        draft_id = d.get("draft_id")
        if not draft_id or not season:
            continue
        detail = _get(f"/draft/{draft_id}")
        if not detail:
            continue
        slot_to_roster: dict = detail.get("slot_to_roster_id") or {}
        for slot, roster_id in slot_to_roster.items():
            if slot and roster_id:
                slot_map[(season, str(roster_id))] = int(slot)

    return slot_map


def _extract_assets(txn: dict, slot_map: dict[tuple, int] | None = None) -> list[dict]:
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
        receiver  = str(pick.get("owner_id", ""))
        side      = side_map.get(receiver, "a")
        roster_id = pick.get("roster_id")
        p_season  = pick.get("season")
        p_round   = pick.get("round")
        slot: Optional[int] = None
        if slot_map and roster_id is not None and p_season is not None:
            slot = slot_map.get((str(p_season), str(roster_id)))
        assets.append({
            "side":            side,
            "asset_type":      "pick",
            "player_id":       None,
            "pick_season":     p_season,
            "pick_round":      p_round,
            "pick_order":      _pick_order(pick),
            "pick_roster_id":  str(roster_id) if roster_id is not None else None,
            "pick_slot":       slot,
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
    end_week: Optional[int] = None,
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

    # Build slot map once per league so every pick gets its exact draft position
    slot_map = _fetch_draft_slot_map(league_id)

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
                assets = _extract_assets(txn, slot_map=slot_map)
                if assets:
                    conn.execute(
                        """
                        INSERT INTO trade_intel_assets
                            (trade_id, side, asset_type, player_id,
                             pick_season, pick_round, pick_order,
                             pick_roster_id, pick_slot)
                        VALUES """ + ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s,%s)"] * len(assets)),
                        [v for a in assets for v in (
                            trade_db_id, a["side"], a["asset_type"], a["player_id"],
                            a["pick_season"], a["pick_round"],
                            a["pick_order"], a.get("pick_roster_id"),
                            a.get("pick_slot"),
                        )]
                    )
                new_trades += 1

    return new_trades


def _leagues_to_crawl(batch_size: int = 500, crawl_mode: str = "new", recrawl_days: int = 7) -> list[dict]:
    """Return leagues based on crawl mode."""
    with get_conn() as conn:
        if crawl_mode == "new":
            # Only uncrawled dynasty leagues
            query = """
                SELECT league_id, season, last_crawled_week, league_type
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND last_crawled_week IS NULL
                  AND league_type IN (1, 2)  -- dynasty and redraft
                ORDER BY discovered_at ASC
                LIMIT %s
            """
            params = (batch_size,)
        elif crawl_mode == "existing":
            # Only previously crawled dynasty leagues, but not recently
            query = """
                SELECT league_id, season, last_crawled_week, league_type
                FROM trade_intel_leagues
                WHERE crawl_enabled = TRUE
                  AND last_crawled_week IS NOT NULL
                  AND (last_crawled_at IS NULL OR last_crawled_at < NOW() - INTERVAL '%s days')
                  AND league_type IN (1, 2)  -- dynasty and redraft
                ORDER BY last_crawled_at ASC NULLS FIRST
                LIMIT %s
            """
            params = (recrawl_days, batch_size)
        else:  # both
            # Mix of new and existing, prioritize new
            query = """
                WITH new_leagues AS (
                    SELECT league_id, season, last_crawled_week, league_type, 1 as priority
                    FROM trade_intel_leagues
                    WHERE crawl_enabled = TRUE
                      AND last_crawled_week IS NULL
                      AND league_type = 2
                    ORDER BY discovered_at ASC
                    LIMIT %s
                ),
                existing_leagues AS (
                    SELECT league_id, season, last_crawled_week, league_type, 2 as priority
                    FROM trade_intel_leagues
                    WHERE crawl_enabled = TRUE
                      AND last_crawled_week IS NOT NULL
                      AND (last_crawled_at IS NULL OR last_crawled_at < NOW() - INTERVAL '%s days')
                      AND league_type = 2
                    ORDER BY last_crawled_at ASC NULLS FIRST
                    LIMIT %s
                ),
                combined AS (
                    SELECT * FROM new_leagues
                    UNION ALL
                    SELECT * FROM existing_leagues
                )
                SELECT league_id, season, last_crawled_week, league_type
                FROM combined
                ORDER BY priority ASC, league_id
                LIMIT %s
            """
            # Split batch between new and existing (70% new, 30% existing)
            new_batch = int(batch_size * 0.7)
            existing_batch = batch_size - new_batch
            params = (new_batch, recrawl_days, existing_batch, batch_size)
        
        return conn.execute(query, params).fetchall()


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


def _crawl_one(row: dict, end_week: int, override_start_week: Optional[int] = None) -> Tuple[str, int, str]:
    """Crawl a single league. Runs inside a thread pool worker."""
    league_id = row["league_id"]
    season = row["season"]
    league_type = row.get("league_type", 2)
    if override_start_week is not None:
        start_week = override_start_week
    else:
        start_week = (row["last_crawled_week"] or 0) + 1
    if start_week > end_week:
        return league_id, 0, league_type
    try:
        n = crawl_league(league_id, season, start_week=start_week, end_week=end_week)
        return league_id, n, league_type
    except Exception as exc:
        logger.warning("[crawler] League %s failed: %s", league_id, exc)
        return league_id, 0, league_type


def run_crawl(batch_size: int = 500, workers: int = 10, crawl_mode: str = "new", recrawl_days: int = 7) -> dict:
    """
    Crawl one batch of leagues in parallel.

    workers: concurrent leagues (default 20 — safe for I/O-bound HTTP workload)
    Each league internally fetches its weeks in parallel (up to 8 concurrent week requests).
    
    crawl_mode: 'new' (uncrawled leagues), 'existing' (re-crawl), 'both' (mixed)
    recrawl_days: for 'existing' mode, only re-crawl leagues not crawled in X days
    """
    current_week = _current_nfl_week()
    leagues = _leagues_to_crawl(batch_size, crawl_mode, recrawl_days)
    dynasty_trades = 0
    redraft_trades = 0
    dynasty_leagues = 0
    redraft_leagues = 0
    mark_batch: list[tuple[int, str]] = []

    # For existing-mode re-crawls, always start from week 1 so we pick up the
    # full season — including any weeks whose start_week would otherwise exceed
    # current_week (common in the offseason when last_crawled_week == 18).
    start_week_override = 1 if crawl_mode == "existing" else None

    print(f"[crawler] Crawling {len(leagues)} leagues with {workers} workers, week={current_week}")

    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_crawl_one, row, current_week, start_week_override): row["league_id"]
            for row in leagues
        }
        
        for future in as_completed(futures):
            completed_count += 1
            league_id, n, league_type = future.result()
            mark_batch.append((current_week, league_id))
            
            if n > 0:
                if league_type == 2:  # dynasty
                    dynasty_trades += n
                    dynasty_leagues += 1
                else:  # redraft
                    redraft_trades += n
                    redraft_leagues += 1

            # Flush mark batch every 50 to avoid holding too many updates
            if len(mark_batch) >= 50:
                _mark_crawled_batch(mark_batch)
                mark_batch = []

    if mark_batch:
        _mark_crawled_batch(mark_batch)

    # Print summary by league type
    print(f"[crawler] Dynasty: {dynasty_trades} trades from {dynasty_leagues} leagues")
    print(f"[crawler] Redraft: {redraft_trades} trades from {redraft_leagues} leagues")
    print(f"[crawler] Done. {dynasty_trades + redraft_trades} new trades across {dynasty_leagues + redraft_leagues} leagues.")
    return {"dynasty_trades": dynasty_trades, "redraft_trades": redraft_trades, "dynasty_leagues": dynasty_leagues, "redraft_leagues": redraft_leagues}


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    result = run_crawl()
    print(result)
