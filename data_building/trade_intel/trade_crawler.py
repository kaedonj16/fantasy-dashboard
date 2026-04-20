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
from datetime import datetime, timezone
from typing import Any

import requests

from dashboard_services.api import get_transactions
from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

SLEEPER_BASE = "https://api.sleeper.app/v1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fantasy-trade-intel/1.0"})

_REQUEST_DELAY = 0.05   # seconds between calls
_WEEKS_PER_SEASON = 18  # fetch weeks 1-18


def _get(path: str) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, timeout=10)
        if resp.status_code == 429:
            logger.warning("[crawler] Rate limited — sleeping 60s")
            time.sleep(60)
            resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[crawler] %s failed: %s", path, exc)
        return None


def _current_nfl_week() -> int:
    state = _get("/state/nfl")
    if state and "week" in state:
        return int(state["week"])
    return _WEEKS_PER_SEASON


def _parse_side(adds: dict, drops: dict, roster_id: str) -> str:
    """
    In Sleeper trades, `adds` maps player_id -> roster_id (who received them).
    Return 'a' for the lower roster_id side, 'b' for the other, keyed by
    which side received this roster_id's players.
    """
    return "a" if roster_id == sorted(adds.values())[0] else "b"


def _extract_assets(txn: dict) -> list[dict]:
    """
    Returns a flat list of asset dicts:
      {side, asset_type, player_id, pick_season, pick_round, pick_order}
    """
    assets: list[dict] = []
    adds: dict[str, Any] = txn.get("adds") or {}
    drops: dict[str, Any] = txn.get("drops") or {}
    draft_picks: list[dict] = txn.get("draft_picks") or []

    # Determine the two roster IDs involved
    all_roster_ids = sorted(set([str(r) for r in list(adds.values()) + list(drops.keys() if drops else [])]))
    # adds: player_id -> roster_id_that_receives
    # We label the side that appears first (lower) as 'a'
    if len(all_roster_ids) < 2:
        # Try to infer from draft_picks
        pick_rosters = sorted({str(p.get("owner_id", "")) for p in draft_picks if p.get("owner_id")})
        all_roster_ids = pick_rosters or all_roster_ids

    side_map: dict[str, str] = {}
    for i, rid in enumerate(all_roster_ids[:2]):
        side_map[str(rid)] = "a" if i == 0 else "b"

    # Player assets — goes to the side that *received* them
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

    # Pick assets
    for pick in draft_picks:
        # owner_id is who receives the pick
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


def _pick_order(pick: dict) -> str | None:
    order = pick.get("order")
    if order is None:
        return None
    if order <= 4:
        return "early"
    if order <= 8:
        return "mid"
    return "late"


def _ingest_transaction(conn, league_id: str, season: int, week: int, txn: dict) -> bool:
    """Insert one trade transaction + its assets. Returns True if newly inserted."""
    txn_id = str(txn.get("transaction_id", ""))
    if not txn_id:
        return False

    status = txn.get("status", "complete")
    created_ms = txn.get("created")
    created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc) if created_ms else None

    result = conn.execute(
        """
        INSERT INTO trade_intel_trades
            (league_id, transaction_id, season, week, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
        RETURNING id
        """,
        (league_id, txn_id, season, week, status, created_at)
    ).fetchone()

    if not result:
        return False  # already existed

    trade_db_id = result["id"]
    assets = _extract_assets(txn)
    for asset in assets:
        conn.execute(
            """
            INSERT INTO trade_intel_assets
                (trade_id, side, asset_type, player_id, pick_season, pick_round, pick_order)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                trade_db_id,
                asset["side"],
                asset["asset_type"],
                asset["player_id"],
                asset["pick_season"],
                asset["pick_round"],
                asset["pick_order"],
            )
        )
    return True


def crawl_league(league_id: str, season: int, start_week: int = 1, end_week: int | None = None) -> int:
    """Crawl all trades for one league. Returns count of newly inserted trades."""
    if end_week is None:
        end_week = _current_nfl_week()
    elif end_week == 0:
        end_week = 1

    new_trades = 0
    for week in range(start_week, end_week + 1):
        time.sleep(_REQUEST_DELAY)
        transactions = get_transactions(league_id, week)
        if not transactions:
            continue

        with get_conn() as conn:
            for txn in transactions:
                if txn.get("type") != "trade":
                    continue
                if txn.get("status") != "complete":
                    continue
                if _ingest_transaction(conn, league_id, season, week, txn):
                    new_trades += 1

    return new_trades


def _leagues_to_crawl(batch_size: int = 200) -> list[dict]:
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


def _mark_crawled(league_id: str, week: int) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE trade_intel_leagues
            SET last_crawled_at = NOW(), last_crawled_week = %s
            WHERE league_id = %s
            """,
            (week, league_id)
        )


def run_crawl(batch_size: int = 200) -> dict:
    """
    Crawl one batch of leagues. Designed to be called on a schedule
    (e.g. every hour). Incremental: only fetches weeks since last crawl.
    """
    current_week = _current_nfl_week()
    leagues = _leagues_to_crawl(batch_size)
    total_trades = 0
    total_leagues = 0

    logger.info("[crawler] Crawling %d leagues (current week: %d)", len(leagues), current_week)
    print(f"[crawler] Crawling {len(leagues)} leagues")

    for row in leagues:
        league_id = row["league_id"]
        season = row["season"]
        start_week = (row["last_crawled_week"] or 0) + 1

        if start_week > current_week:
            _mark_crawled(league_id, current_week)
            continue

        try:
            n = crawl_league(league_id, season, start_week=start_week, end_week=current_week)
            _mark_crawled(league_id, current_week)
            total_trades += n
            total_leagues += 1
            if n > 0:
                logger.info("[crawler] %s: +%d trades", league_id, n)
        except Exception as exc:
            logger.warning("[crawler] League %s failed: %s", league_id, exc)

    logger.info("[crawler] Done. %d new trades across %d leagues.", total_trades, total_leagues)
    print(f"[crawler] Done. {total_trades} new trades across {total_leagues} leagues.")
    return {"leagues_crawled": total_leagues, "new_trades": total_trades}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_crawl()
    print(result)
