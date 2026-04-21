"""
League discovery for Trade Intelligence Engine.

Strategy (no Sleeper search API exists):
1. Seed from Sleeper trending players endpoint — each trending entry includes
   league_ids that recently touched the player.
2. From each discovered league, pull rosters -> owner user_ids -> fetch their
   leagues -> expand the frontier.
3. Filter to dynasty leagues only (league_type == 2).
4. Persist discovered leagues to trade_intel_leagues for the crawler.
"""
from __future__ import annotations

import logging
import time
from typing import Set

import requests

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

SLEEPER_BASE = "https://api.sleeper.app/v1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fantasy-trade-intel/1.0"})

_REQUEST_DELAY = 0.1   # seconds between Sleeper calls — stay well under rate limits
_MAX_LEAGUES = 5_000   # target ceiling per crawl run


def _get(path: str, params: dict | None = None) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            logger.warning("[discovery] Rate limited — sleeping 60s")
            time.sleep(60)
            resp = SESSION.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("[discovery] %s failed: %s", path, exc)
        return None


def _current_season() -> int:
    state = _get("/state/nfl")
    if state and "season" in state:
        return int(state["season"])
    return 2024


def _seed_league_ids(season: int) -> Set[str]:
    """
    Seed the discovery frontier from leagues already in the DB.

    Sleeper's trending endpoint only returns {player_id, count} — it does NOT
    embed league IDs, so we can't use it for seeding.  Instead we BFS-expand
    from whatever leagues are already stored (populated by manual inserts or
    previous discovery runs).  On a completely fresh DB the frontier will be
    empty; the user must insert at least one league_id manually to bootstrap.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT league_id FROM trade_intel_leagues
            WHERE season IN (%s, %s)
            ORDER BY last_crawled_at ASC NULLS FIRST
            LIMIT 200
            """,
            (season, season - 1)
        ).fetchall()
    seeds = {r["league_id"] for r in rows}
    logger.info("[discovery] DB seeds: %d leagues to BFS-expand from (season %d or %d)",
                len(seeds), season, season - 1)
    return seeds


def _user_leagues(user_id: str, season: int) -> list[str]:
    ids: list[str] = []
    for yr in {season, season + 1}:  # also check next year — offseason leagues created early
        data = _get(f"/user/{user_id}/leagues/nfl/{yr}")
        if data:
            ids.extend(str(lg["league_id"]) for lg in data if lg.get("league_id"))
    return ids


def _league_meta(league_id: str) -> dict | None:
    return _get(f"/league/{league_id}")


def _roster_owner_ids(league_id: str) -> list[str]:
    rosters = _get(f"/league/{league_id}/rosters")
    if not rosters:
        return []
    return [str(r["owner_id"]) for r in rosters if r.get("owner_id")]


def _already_known(season: int) -> Set[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT league_id FROM trade_intel_leagues WHERE season = %s",
            (season,)
        ).fetchall()
    return {r["league_id"] for r in rows}


def _save_leagues(leagues: list[dict]) -> int:
    if not leagues:
        return 0
    with get_conn() as conn:
        for lg in leagues:
            conn.execute(
                """
                INSERT INTO trade_intel_leagues
                    (league_id, season, num_teams, scoring_type, league_type,
                     is_superflex, crawl_enabled)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (league_id) DO UPDATE SET
                    crawl_enabled = TRUE,
                    is_superflex  = EXCLUDED.is_superflex
                """,
                (
                    lg["league_id"],
                    lg["season"],
                    lg.get("num_teams"),
                    lg.get("scoring_type"),
                    lg.get("league_type"),
                    lg.get("is_superflex", False),
                )
            )
    return len(leagues)


def _classify_scoring(settings: dict) -> str:
    ppr = float((settings.get("scoring_settings") or {}).get("rec", 0))
    if ppr >= 1.0:
        return "ppr"
    if ppr >= 0.5:
        return "half"
    return "std"


def _is_superflex(meta: dict) -> bool:
    """True if the league has a SUPER_FLEX roster slot."""
    rp = meta.get("roster_positions") or []
    return any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in rp)


def run_discovery(target: int = _MAX_LEAGUES, season: int | None = None) -> int:
    """
    Discover up to `target` dynasty Sleeper leagues and store them.
    Returns total count of newly inserted leagues.
    """
    if season is None:
        season = _current_season()

    known = _already_known(season)
    # Seed BFS from leagues already in the DB. These won't be re-saved,
    # but their roster owners are expanded to discover new connected leagues.
    seeds: Set[str] = _seed_league_ids(season)
    # Tracks which leagues need owner-expansion (seeds + newly found dynasty leagues)
    to_expand: Set[str] = set(seeds)
    # Frontier holds league IDs we haven't processed yet (new, not in DB)
    frontier: Set[str] = set()
    visited_users: Set[str] = set()
    to_save: list[dict] = []
    total_new = 0

    logger.info("[discovery] Starting. Known=%d, Seeds=%d", len(known), len(seeds))

    # First pass: expand all seed leagues to populate the frontier
    for league_id in to_expand:
        time.sleep(_REQUEST_DELAY)
        for owner_id in _roster_owner_ids(league_id):
            if owner_id in visited_users:
                continue
            visited_users.add(owner_id)
            time.sleep(_REQUEST_DELAY)
            for new_lid in _user_leagues(owner_id, season):
                if new_lid not in known:
                    frontier.add(new_lid)

    logger.info("[discovery] After seed expansion: %d leagues in frontier", len(frontier))

    while frontier and total_new < target:
        league_id = frontier.pop()
        if league_id in known:
            continue

        time.sleep(_REQUEST_DELAY)
        meta = _league_meta(league_id)
        if not meta:
            continue

        # Only dynasty leagues
        if meta.get("settings", {}).get("type") != 2:
            known.add(league_id)  # mark so we don't revisit
            continue

        lg_season = int(meta.get("season") or season)
        to_save.append({
            "league_id":   league_id,
            "season":      lg_season,
            "num_teams":   meta.get("total_rosters"),
            "scoring_type": _classify_scoring(meta),
            "league_type": 2,
            "is_superflex": _is_superflex(meta),
        })
        known.add(league_id)

        # Expand frontier via roster owners
        if len(frontier) < 2000:
            time.sleep(_REQUEST_DELAY)
            for owner_id in _roster_owner_ids(league_id):
                if owner_id in visited_users:
                    continue
                visited_users.add(owner_id)
                time.sleep(_REQUEST_DELAY)
                for new_lid in _user_leagues(owner_id, season):
                    if new_lid not in known:
                        frontier.add(new_lid)

        # Flush every 100
        if len(to_save) >= 100:
            n = _save_leagues(to_save)
            total_new += n
            logger.info("[discovery] Saved batch. New total: %d", total_new)
            to_save = []

    if to_save:
        total_new += _save_leagues(to_save)

    logger.info("[discovery] Done. %d new leagues discovered this run.", total_new)
    return total_new


def backfill_superflex(batch_size: int = 500) -> int:
    """
    Fetch roster_positions for existing leagues that don't have is_superflex set yet
    and update them.  Run once after adding the column.

    Returns the number of leagues updated.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT league_id FROM trade_intel_leagues
            WHERE is_superflex IS NULL
            ORDER BY discovered_at ASC
            LIMIT %s
            """,
            (batch_size,),
        ).fetchall()

    if not rows:
        logger.info("[backfill_superflex] Nothing to update.")
        return 0

    updated = 0
    for row in rows:
        league_id = row["league_id"]
        time.sleep(_REQUEST_DELAY)
        meta = _league_meta(league_id)
        if meta is None:
            # Can't reach league — mark False so we don't keep retrying
            is_sf = False
        else:
            is_sf = _is_superflex(meta)

        with get_conn() as conn:
            conn.execute(
                "UPDATE trade_intel_leagues SET is_superflex = %s WHERE league_id = %s",
                (is_sf, league_id),
            )
        updated += 1

    logger.info("[backfill_superflex] Updated %d leagues.", updated)
    return updated


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        print(f"Backfilled {backfill_superflex()} leagues.")
    else:
        count = run_discovery()
        print(f"Discovered {count} new leagues.")
