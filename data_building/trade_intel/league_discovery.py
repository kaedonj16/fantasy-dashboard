"""
League discovery for Trade Intelligence Engine.

Strategy (no Sleeper search API exists):
1. Seed from Sleeper trending players endpoint - each trending entry includes
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
from typing import Set, Optional, List, Dict, Tuple

import requests

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLEEPER_BASE = "https://api.sleeper.app/v1"

# Configure session with larger connection pool to match trade crawler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

_REQUEST_DELAY = 0.1   # seconds between Sleeper calls - stay well under rate limits
_MAX_LEAGUES = 5_000   # target ceiling per crawl run


def _get(path: str, params: dict | None = None) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            logger.warning("[discovery] Rate limited - sleeping 60s")
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

    Sleeper's trending endpoint only returns {player_id, count} - it does NOT
    embed league IDs, so we can't use it for seeding.  Instead we BFS-expand
    from whatever leagues are already stored (populated by manual inserts or
    previous discovery runs).  On a completely fresh DB the frontier will be
    empty; the user must insert at least one league_id manually to bootstrap.
    Includes both dynasty (2) and true-redraft (0) leagues as BFS seeds.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT league_id FROM trade_intel_leagues
            WHERE season IN (%s, %s)
              AND league_type IN (0, 2)
            ORDER BY last_crawled_at ASC NULLS FIRST
            LIMIT 2000
            """,
            (season, season - 1)
        ).fetchall()
    seeds = {r["league_id"] for r in rows}
    logger.info("[discovery] DB seeds: %d leagues to BFS-expand from",
                len(seeds))
    return seeds


def _user_leagues(user_id: str, season: int) -> List[str]:
    ids: List[str] = []
    for yr in {season, season + 1}:  # also check next year - offseason leagues created early
        data = _get(f"/user/{user_id}/leagues/nfl/{yr}")
        if data:
            ids.extend(str(lg["league_id"]) for lg in data if lg.get("league_id"))
    return ids


def _league_meta(league_id: str) -> Optional[Dict]:
    return _get(f"/league/{league_id}")


def _roster_owner_ids(league_id: str) -> List[str]:
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


def _save_users(user_ids: List[str], source: str = "bfs", usernames: Optional[Dict[str, str]] = None) -> None:
    """Upsert user IDs into trade_intel_users. Skips on conflict (first write wins)."""
    if not user_ids:
        return
    usernames = usernames or {}
    
    import time
    import random
    from psycopg import errors
    
    # Prepare data for batch processing
    values = [(uid, usernames.get(uid), source) for uid in user_ids]
    
    # Write in batches with connection recovery to prevent timeouts
    BATCH = 500
    for batch_start in range(0, len(values), BATCH):
        batch = values[batch_start : batch_start + BATCH]
        
        # Retry each batch up to 3 times with fresh connections
        for attempt in range(3):
            try:
                with get_conn(autocommit=True) as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        """
                        INSERT INTO trade_intel_users (user_id, username, source)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        batch
                    )
                    print(f"[_save_users] Written batch {batch_start}-{batch_start + len(batch) - 1} ({len(batch)} users)")
                    break  # Success, exit retry loop
                    
            except errors.DeadlockDetected:
                if attempt == 2:  # Last attempt failed
                    print(f"[_save_users] Deadlock in batch {batch_start}-{batch_start + len(batch) - 1} after 3 attempts, skipping.")
                    break
                else:
                    # Add jittered exponential backoff for deadlocks
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[_save_users] Deadlock in batch {batch_start}-{batch_start + len(batch) - 1} (attempt {attempt + 1}/3). Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    print(f"[_save_users] Failed to write batch {batch_start}-{batch_start + len(batch) - 1} after 3 attempts, skipping. Error: {e}")
                    # Continue with next batch instead of failing completely
                    break
                else:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) + 1
                    print(f"[_save_users] Batch {batch_start}-{batch_start + len(batch) - 1} failed (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)


def _save_leagues(leagues: list[dict]) -> int:
    if not leagues:
        return 0
    
    import time
    import random
    from psycopg import errors
    
    # Prepare data for batch processing
    values = [
        (
            lg["league_id"],
            lg["season"],
            lg.get("num_teams"),
            lg.get("scoring_type"),
            lg.get("league_type"),
            lg.get("is_superflex", False),
            True
        )
        for lg in leagues
    ]
    
    written = 0
    
    # Write in batches with connection recovery to prevent timeouts
    BATCH = 500
    for batch_start in range(0, len(values), BATCH):
        batch = values[batch_start : batch_start + BATCH]
        
        # Retry each batch up to 3 times with fresh connections
        for attempt in range(3):
            try:
                with get_conn(autocommit=True) as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        """
                        INSERT INTO trade_intel_leagues
                            (league_id, season, num_teams, scoring_type, league_type,
                             is_superflex, crawl_enabled)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (league_id) DO UPDATE SET
                            crawl_enabled = TRUE,
                            is_superflex  = EXCLUDED.is_superflex,
                            league_type   = EXCLUDED.league_type
                        """,
                        batch
                    )
                    written += len(batch)
                    print(f"[_save_leagues] Written batch {batch_start}-{batch_start + len(batch) - 1} ({len(batch)} leagues) - Total: {written} / {len(leagues)}")
                    break  # Success, exit retry loop
                    
            except errors.DeadlockDetected:
                if attempt == 2:  # Last attempt failed
                    print(f"[_save_leagues] Deadlock in batch {batch_start}-{batch_start + len(batch) - 1} after 3 attempts, skipping.")
                    break
                else:
                    # Add jittered exponential backoff for deadlocks
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[_save_leagues] Deadlock in batch {batch_start}-{batch_start + len(batch) - 1} (attempt {attempt + 1}/3). Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    print(f"[_save_leagues] Failed to write batch {batch_start}-{batch_start + len(batch) - 1} after 3 attempts, skipping. Error: {e}")
                    # Continue with next batch instead of failing completely
                    break
                else:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) + 1
                    print(f"[_save_leagues] Batch {batch_start}-{batch_start + len(batch) - 1} failed (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
    
    return written


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


def bootstrap_from_usernames(usernames: List[str], season: Optional[int] = None) -> int:
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
            if league_type not in (0, 2):
                continue
            lg_season = int(meta.get("season") or season)
            to_save.append({
                "league_id":    lid,
                "season":       lg_season,
                "num_teams":    meta.get("total_rosters", 0),
                "scoring_type": _classify_scoring(meta),
                "league_type":  league_type,
                "is_superflex": _is_superflex(meta),
            })
            known.add(lid)
            mode = "dynasty" if league_type == 2 else "redraft"
            logger.info("[bootstrap] Seeded %s league %s (%d teams) from user '%s'",
                        mode, lid, meta.get("total_rosters", 0), username)

    n = _save_leagues(to_save)
    logger.info("[bootstrap] Inserted %d new league(s) as BFS seeds.", n)
    return n


def seed_user(user_id: str, username: Optional[str] = None, season: Optional[int] = None) -> int:
    """
    Seed dynasty leagues for a single Sleeper user_id into trade_intel_leagues,
    and record the user in trade_intel_users.  Safe to call on every login -
    ON CONFLICT DO NOTHING means repeat visits are a no-op.

    Returns the number of new dynasty leagues inserted.
    """
    if season is None:
        season = _current_season()

    _save_users([user_id], source="login", usernames={user_id: username} if username else None)

    known = _already_known(season)
    league_ids = _user_leagues(user_id, season)
    to_save: list[dict] = []

    for lid in league_ids:
        if lid in known:
            continue
        time.sleep(_REQUEST_DELAY)
        meta = _league_meta(lid)
        if not meta:
            continue
        league_type = meta.get("settings", {}).get("type")
        if league_type not in (0, 2):
            continue
        lg_season = int(meta.get("season") or season)
        to_save.append({
            "league_id":    lid,
            "season":       lg_season,
            "num_teams":    meta.get("total_rosters", 0),
            "scoring_type": _classify_scoring(meta),
            "league_type":  league_type,
            "is_superflex": _is_superflex(meta),
        })

    n = _save_leagues(to_save)
    if n:
        logger.info("[seed_user] user=%s inserted %d new dynasty league(s)", user_id, n)

    with get_conn() as conn:
        conn.execute(
            "UPDATE trade_intel_users SET last_seeded_at = NOW() WHERE user_id = %s",
            (user_id,)
        )
    return n


def seed_from_stored_users(batch_size: int = 200, season: Optional[int] = None) -> int:
    """
    Pull users from trade_intel_users that haven't been seeded recently,
    fetch their Sleeper leagues, and insert any new dynasty leagues.

    Prioritises users that have never been seeded (last_seeded_at IS NULL).
    Returns total new leagues inserted.
    """
    if season is None:
        season = _current_season()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT user_id, username FROM trade_intel_users
            ORDER BY last_seeded_at ASC NULLS FIRST
            LIMIT %s
            """,
            (batch_size,)
        ).fetchall()

    if not rows:
        logger.info("[seed_from_stored_users] No stored users to seed from.")
        return 0

    known = _already_known(season)
    to_save: list[dict] = []

    for row in rows:
        user_id = row["user_id"]
        league_ids = _user_leagues(user_id, season)
        for lid in league_ids:
            if lid in known:
                continue
            time.sleep(_REQUEST_DELAY)
            meta = _league_meta(lid)
            if not meta:
                continue
            league_type = meta.get("settings", {}).get("type")
            if league_type not in (0, 2):
                continue
            lg_season = int(meta.get("season") or season)
            to_save.append({
                "league_id":    lid,
                "season":       lg_season,
                "num_teams":    meta.get("total_rosters", 0),
                "scoring_type": _classify_scoring(meta),
                "league_type":  league_type,
                "is_superflex": _is_superflex(meta),
            })
            known.add(lid)

        with get_conn() as conn:
            conn.execute(
                "UPDATE trade_intel_users SET last_seeded_at = NOW() WHERE user_id = %s",
                (user_id,)
            )

    n = _save_leagues(to_save)
    logger.info("[seed_from_stored_users] %d users → %d new dynasty leagues", len(rows), n)
    return n


def run_discovery(target: int = _MAX_LEAGUES, season: Optional[int] = None) -> int:
    """
    Discover up to `target` dynasty Sleeper leagues and store them.
    Returns total count of newly inserted leagues.
    """
    if season is None:
        season = _current_season()

    known = _already_known(season)
    seeds: Set[str] = _seed_league_ids(season)
    to_expand: Set[str] = set(seeds)
    frontier: Set[str] = set()
    visited_users: Set[str] = set()
    to_save: List[Dict] = []
    total_new = 0
    dynasty_count = 0
    redraft_count = 0

    print(f"[discovery] Starting. Known={len(known)}, Seeds={len(seeds)}, Target={target}")

    # First pass: expand all seed leagues to populate the frontier (parallelized)
    seed_user_ids: List[str] = []

    def expand_seed_league(league_id: str) -> Tuple[str, List[str], List[str]]:
        """Expand a single seed league and return (league_id, new_leagues, owner_ids)"""
        time.sleep(_REQUEST_DELAY)
        owner_ids = _roster_owner_ids(league_id)
        new_leagues = []

        for owner_id in owner_ids:
            time.sleep(_REQUEST_DELAY)
            user_leagues = _user_leagues(owner_id, season)
            for lid in user_leagues:
                if lid not in known:
                    new_leagues.append(lid)

        return league_id, new_leagues, owner_ids

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(expand_seed_league, lid): lid for lid in to_expand}
        for future in as_completed(futures):
            league_id, new_leagues, owner_ids = future.result()
            seed_user_ids.extend(owner_ids)
            for new_lid in new_leagues:
                frontier.add(new_lid)

    # Flush all seed-phase users in one batch
    if seed_user_ids:
        _save_users(seed_user_ids, source="bfs")

    print(f"[discovery] Seed expansion complete. {len(frontier)} leagues in frontier")

    def process_frontier_batch(batch_leagues: List[str]) -> Tuple[List[Dict], List[str]]:
        """Process a batch of frontier leagues and return (to_save, new_frontier_leagues)"""
        batch_to_save = []
        batch_new_frontier = []
        
        def process_single_frontier_league(league_id: str) -> Tuple[Optional[Dict], List[str], List[str]]:
            """Process a single frontier league and return (league_data, new_frontier_leagues, dynasty_owner_ids)"""
            time.sleep(_REQUEST_DELAY)
            meta = _league_meta(league_id)
            if not meta:
                return None, [], []
            
            league_type = meta.get("settings", {}).get("type")
            if league_type not in (0, 2):
                return None, [], []

            lg_season = int(meta.get("season") or season)
            num_teams = meta.get("total_rosters", 0)
            scoring_type = _classify_scoring(meta)
            is_sf = _is_superflex(meta)

            league_data = {
                "league_id":   league_id,
                "season":      lg_season,
                "num_teams":   num_teams,
                "scoring_type": scoring_type,
                "league_type": league_type,
                "is_superflex": is_sf,
            }
            
            new_frontier_leagues = []
            dynasty_owner_ids: List[str] = []
            if len(frontier) < 2000:
                owner_ids = _roster_owner_ids(league_id)
                if league_type == 2:
                    dynasty_owner_ids = owner_ids
                for owner_id in owner_ids:
                    if owner_id in visited_users:
                        continue
                    visited_users.add(owner_id)
                    time.sleep(_REQUEST_DELAY)
                    user_leagues = _user_leagues(owner_id, season)
                    new_leagues = [lid for lid in user_leagues if lid not in known]
                    new_frontier_leagues.extend(new_leagues)

            return league_data, new_frontier_leagues, dynasty_owner_ids

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_single_frontier_league, lid): lid for lid in batch_leagues}
            batch_dynasty_users: List[str] = []
            for future in as_completed(futures):
                league_data, new_frontier, dynasty_users = future.result()
                if league_data:
                    batch_to_save.append(league_data)
                    known.add(league_data["league_id"])
                batch_new_frontier.extend(new_frontier)
                batch_dynasty_users.extend(dynasty_users)
        if batch_dynasty_users:
            _save_users(batch_dynasty_users, source="bfs")
        
        return batch_to_save, batch_new_frontier

    processed_count = 0
    batch_size = 50
    
    while frontier and total_new < target:
        batch_leagues = []
        for _ in range(min(batch_size, len(frontier))):
            if not frontier:
                break
            league_id = frontier.pop()
            if league_id not in known:
                batch_leagues.append(league_id)
        
        if not batch_leagues:
            continue
            
        batch_to_save, batch_new_frontier = process_frontier_batch(batch_leagues)
        processed_count += len(batch_leagues)
        
        # Count league types in this batch
        batch_dynasty = sum(1 for lg in batch_to_save if lg["league_type"] == 2)
        batch_redraft = sum(1 for lg in batch_to_save if lg["league_type"] == 1)
        dynasty_count += batch_dynasty
        redraft_count += batch_redraft
        
        to_save.extend(batch_to_save)
        for new_lid in batch_new_frontier:
            if new_lid not in known:
                frontier.add(new_lid)

        # Flush every 100
        if len(to_save) >= 100:
            n = _save_leagues(to_save)
            total_new += n
            to_save = []

    if to_save:
        total_new += _save_leagues(to_save)

    print(f"[discovery] Done. {total_new} new leagues: {dynasty_count} dynasty, {redraft_count} redraft")
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
            # Can't reach league - mark False so we don't keep retrying
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
