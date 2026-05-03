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


def _leagues_to_crawl(batch_size: int) -> list[dict]:
    """
    Return dynasty leagues that haven't had their drafts indexed recently.

    Priority:
      1. Never crawled (last_draft_adp_crawled_at IS NULL)
      2. Not crawled in RECRAWL_DAYS days
    """
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT league_id,
                   COALESCE(num_teams, 12) AS num_teams,
                   COALESCE(is_superflex, FALSE) AS is_superflex
            FROM trade_intel_leagues
            WHERE crawl_enabled = TRUE
              AND league_type = 2          -- dynasty only
              AND (
                  last_draft_adp_crawled_at IS NULL
                  OR last_draft_adp_crawled_at < NOW() - INTERVAL '%s days'
              )
            ORDER BY last_draft_adp_crawled_at ASC NULLS FIRST
            LIMIT %s
            """,
            (RECRAWL_DAYS, batch_size),
        ).fetchall()


def run_draft_adp_crawl(batch_size: int = 500, workers: int = 10) -> dict:
    """
    Crawl draft pick data from all eligible dynasty leagues and recompute ADP.

    batch_size : max leagues to process per run
    workers    : concurrent league crawlers

    Returns a summary dict with new_picks and adp_entries.
    """
    leagues = _leagues_to_crawl(batch_size)
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


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_draft_adp_crawl()
    print(result)
