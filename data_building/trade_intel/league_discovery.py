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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set

import requests

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
              AND league_type = 2
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


def bootstrap_from_usernames(usernames: list[str], season: int | None = None) -> int:
    """
    Seed the DB from one or more Sleeper usernames.

    For each username: look up the user, fetch their leagues for the current
    (and next) season, filter to dynasty (type==2), and insert them into
    trade_intel_leagues so that subsequent BFS discovery has a non-empty frontier.

    Returns the number of new leagues inserted.
    """
    if season is None:
        season = _current_season()

    known = _already_known(season)
    to_save: list[dict] = []

    for username in usernames:
        user = _get(f"/user/{username}")
        if not user or not user.get("user_id"):
            logger.warning("[bootstrap] Username '%s' not found or no user_id returned", username)
            continue
        user_id = str(user["user_id"])
        logger.info("[bootstrap] User '%s' → user_id=%s", username, user_id)

        league_ids = _user_leagues(user_id, season)
        logger.info("[bootstrap] Found %d leagues for user '%s'", len(league_ids), username)

        for lid in league_ids:
            if lid in known:
                continue
            time.sleep(_REQUEST_DELAY)
            meta = _league_meta(lid)
            if not meta:
                continue
            league_type = meta.get("settings", {}).get("type")
            if league_type != 2:
                continue
            lg_season = int(meta.get("season") or season)
            to_save.append({
                "league_id":    lid,
                "season":       lg_season,
                "num_teams":    meta.get("total_rosters", 0),
                "scoring_type": _classify_scoring(meta),
                "league_type":  2,
                "is_superflex": _is_superflex(meta),
            })
            known.add(lid)
            logger.info("[bootstrap] Seeded dynasty league %s (%d teams) from user '%s'",
                        lid, meta.get("total_rosters", 0), username)

    n = _save_leagues(to_save)
    logger.info("[bootstrap] Inserted %d new dynasty league(s) as BFS seeds.", n)
    return n


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

    logger.info("[discovery] Starting. Known=%d, Seeds=%d, Target=%d", len(known), len(seeds), target)
    logger.info("[discovery] Checkpoint: Beginning seed expansion phase")

    # First pass: expand all seed leagues to populate the frontier (parallelized)
    def expand_seed_league(league_id: str) -> tuple[str, list[str], dict]:
        """Expand a single seed league and return (league_id, new_leagues, league_type_counts)"""
        time.sleep(_REQUEST_DELAY)
        owner_ids = _roster_owner_ids(league_id)
        new_leagues = []
        league_type_counts = {"dynasty": 0, "redraft": 0, "other": 0, "no_meta": 0}
        
        for owner_id in owner_ids:
            time.sleep(_REQUEST_DELAY)
            user_leagues = _user_leagues(owner_id, season)
            for lid in user_leagues:
                if lid not in known:
                    new_leagues.append(lid)
                    # Quick type check for logging
                    meta = _league_meta(lid)
                    if meta:
                        league_type = meta.get("settings", {}).get("type")
                        if league_type == 2:
                            league_type_counts["dynasty"] += 1
                        elif league_type == 1:
                            league_type_counts["redraft"] += 1
                        else:
                            league_type_counts["other"] += 1
                    else:
                        league_type_counts["no_meta"] += 1
        
        return league_id, new_leagues, league_type_counts

    logger.info("[discovery] Checkpoint: Processing %d seed leagues with 10 workers", len(to_expand))
    seed_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(expand_seed_league, lid): lid for lid in to_expand}
        for future in as_completed(futures):
            league_id, new_leagues, league_type_counts = future.result()
            seed_results.append((league_id, len(new_leagues)))
            for new_lid in new_leagues:
                frontier.add(new_lid)
            
            # Format league type breakdown
            total_found = len(new_leagues)
            if total_found > 0:
                breakdown = f"{total_found} total: {league_type_counts['dynasty']} dynasty, {league_type_counts['redraft']} redraft, {league_type_counts['other']} other, {league_type_counts['no_meta']} no_meta"
            else:
                breakdown = "0 new leagues"
            
            logger.info("[discovery] Checkpoint: Seed %s: found %s", league_id, breakdown)

    total_new_from_seeds = sum(count for _, count in seed_results)
    logger.info("[discovery] Checkpoint: Seed expansion complete. %d leagues in frontier, %d new from seeds", len(frontier), total_new_from_seeds)
    logger.info("[discovery] Checkpoint: Beginning main discovery loop")

    def process_frontier_batch(batch_leagues: list[str]) -> tuple[list[dict], list[str], int]:
        """Process a batch of frontier leagues and return (to_save, new_frontier_leagues, processed_count)"""
        batch_to_save = []
        batch_new_frontier = []
        dynasty_count = 0
        redraft_count = 0
        other_count = 0
        
        def process_single_frontier_league(league_id: str) -> tuple[dict | None, list[str], str]:
            """Process a single frontier league and return (league_data, new_frontier_leagues, league_type_label)"""
            time.sleep(_REQUEST_DELAY)
            meta = _league_meta(league_id)
            if not meta:
                return None, [], "no_meta"
            
            # Check league type
            league_type = meta.get("settings", {}).get("type")
            if league_type == 2:
                league_type_label = "dynasty"
            elif league_type == 1:
                league_type_label = "redraft"
            else:
                league_type_label = f"other_{league_type}"
            
            # Only dynasty leagues proceed to full processing
            if league_type != 2:
                return None, [], league_type_label
            
            lg_season = int(meta.get("season") or season)
            num_teams = meta.get("total_rosters", 0)
            scoring_type = _classify_scoring(meta)
            is_sf = _is_superflex(meta)
            
            league_data = {
                "league_id":   league_id,
                "season":      lg_season,
                "num_teams":   num_teams,
                "scoring_type": scoring_type,
                "league_type": 2,
                "is_superflex": is_sf,
            }
            
            # Expand frontier via roster owners (only if frontier is small)
            new_frontier_leagues = []
            if len(frontier) < 2000:
                owner_ids = _roster_owner_ids(league_id)
                for owner_id in owner_ids:
                    if owner_id in visited_users:
                        continue
                    visited_users.add(owner_id)
                    time.sleep(_REQUEST_DELAY)
                    user_leagues = _user_leagues(owner_id, season)
                    new_leagues = [lid for lid in user_leagues if lid not in known]
                    new_frontier_leagues.extend(new_leagues)
            
            return league_data, new_frontier_leagues, league_type_label
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_single_frontier_league, lid): lid for lid in batch_leagues}
            for future in as_completed(futures):
                league_data, new_frontier, league_type_label = future.result()
                
                # Count league types
                if league_type_label == "dynasty":
                    dynasty_count += 1
                elif league_type_label == "redraft":
                    redraft_count += 1
                elif league_type_label == "no_meta":
                    other_count += 1
                else:
                    other_count += 1
                
                if league_data:
                    batch_to_save.append(league_data)
                    known.add(league_data["league_id"])
                batch_new_frontier.extend(new_frontier)
        
        return batch_to_save, batch_new_frontier, len(batch_leagues), dynasty_count, redraft_count, other_count

    processed_count = 0
    batch_size = 50  # Process frontier in batches
    
    while frontier and total_new < target:
        # Get next batch from frontier
        batch_leagues = []
        for _ in range(min(batch_size, len(frontier))):
            if not frontier:
                break
            league_id = frontier.pop()
            if league_id not in known:
                batch_leagues.append(league_id)
        
        if not batch_leagues:
            continue
            
        logger.info("[discovery] Checkpoint: Processing batch of %d frontier leagues (Frontier size: %d, New so far: %d/%d)", 
                   len(batch_leagues), len(frontier), total_new, target)
        
        batch_to_save, batch_new_frontier, batch_processed, dynasty_count, redraft_count, other_count = process_frontier_batch(batch_leagues)
        processed_count += batch_processed
        
        logger.info("[discovery] Checkpoint: Batch complete - %d total leagues: %d dynasty, %d redraft, %d other/no_meta | %d new frontier leagues", 
                   len(batch_leagues), dynasty_count, redraft_count, other_count, len(batch_new_frontier))
        
        # Add new leagues to save and frontier
        to_save.extend(batch_to_save)
        for new_lid in batch_new_frontier:
            if new_lid not in known:
                frontier.add(new_lid)
        
        # Log details for each discovered league
        for league_data in batch_to_save:
            logger.info("[discovery] Checkpoint: League %s: Dynasty found - %d teams, %s, %s", 
                       league_data["league_id"], league_data["num_teams"], 
                       league_data["scoring_type"], 
                       "Superflex" if league_data["is_superflex"] else "1QB")

        # Flush every 100
        if len(to_save) >= 100:
            logger.info("[discovery] Checkpoint: Flushing batch of %d leagues to database", len(to_save))
            n = _save_leagues(to_save)
            total_new += n
            logger.info("[discovery] Checkpoint: Batch saved. New total: %d/%d target", total_new, target)
            to_save = []

    if to_save:
        logger.info("[discovery] Checkpoint: Final flush of %d leagues to database", len(to_save))
        total_new += _save_leagues(to_save)
        logger.info("[discovery] Checkpoint: Final batch saved. Total new leagues: %d", total_new)

    logger.info("[discovery] Checkpoint: Discovery complete. Processed %d frontier leagues, visited %d users", processed_count, len(visited_users))
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
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        print(f"Backfilled {backfill_superflex()} leagues.")
    else:
        count = run_discovery()
        print(f"Discovered {count} new leagues.")
