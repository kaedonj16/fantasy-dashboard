"""
League discovery for Trade Intelligence Engine.

Strategy (no Sleeper search API exists):
1. Seed from Sleeper trending players endpoint - each trending entry includes
   league_ids that recently touched the player.
2. From each discovered league, pull rosters -> owner user_ids -> fetch their
   leagues -> expand the frontier.
3. Retain true-redraft (0) and dynasty (2) leagues; exclude keeper (1).
4. Persist discovered leagues to trade_intel_leagues for the crawler.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Callable, Iterable, Set, Optional, List, Dict, Tuple

import requests

from dashboard_services.db import get_conn
from data_building.trade_intel.league_types import LeagueType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLEEPER_BASE = "https://api.sleeper.app/v1"

# Keep the pool sized to the worker cap. A 20-conn pool plus 10 workers each
# holding a parsed Sleeper user-leagues payload is what OOM'd the 512Mi cron.
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
# Do not retry read timeouts: urllib3 keeps the partial body, and the
# league-discovery cron OOM'd while Retry(total=2) was re-downloading
# /user/.../leagues payloads that had already timed out at 10s.
retry_strategy = Retry(
    total=1,
    connect=1,
    read=False,
    redirect=0,
    status=1,
    backoff_factor=0.3,
    status_forcelist=[502, 503, 504],
    raise_on_status=False,
)
adapter = HTTPAdapter(
    pool_connections=4,
    pool_maxsize=4,
    max_retries=retry_strategy,
)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
SESSION.headers.update({"User-Agent": "fantasy-trade-intel/1.0"})

_REQUEST_DELAY = 0.1   # seconds between Sleeper calls - stay well under rate limits
_MAX_LEAGUES = 5_000   # target ceiling per crawl run

# Discovery used to BFS-expand 2000 seed leagues with 10 workers, submitting
# every future up front. Each seed fetches full roster JSON plus every owner's
# full /user/.../leagues payload (not just ids) — that peak blew the 512Mi cap
# before crawl even started. Bound workers, in-flight futures, and seed count
# so we only expand enough to fill a frontier for `target`.
_DISCOVERY_WORKERS = 2
_DISCOVERY_IN_FLIGHT = 4
_MAX_SEEDS_PER_RUN = 80
_FRONTIER_CAP = 1500
_MAX_LEAGUES_PER_USER = 60
_MAX_OWNERS_PER_LEAGUE = 32
# Abort scanning a user-leagues / rosters body after this many raw bytes so a
# single oversized Sleeper payload cannot materialize as a multi-MB Python tree.
_MAX_SCAN_BYTES = 1_048_576
_STREAM_CHUNK = 32 * 1024
_ID_OVERLAP = 64
_LEAGUE_ID_RE = re.compile(rb'"league_id"\s*:\s*"?(\d{6,20})"?')
_OWNER_ID_RE = re.compile(rb'"owner_id"\s*:\s*"?(\d{6,20})"?')


def _log_rss(label: str) -> None:
    """Linux ru_maxrss is KB. Best-effort; never fail the run for a log line."""
    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        logger.info("[discovery] %s rss=%.0fMi", label, rss_mb)
    except Exception:
        pass


def _seed_limit_for_target(target: int) -> int:
    """How many DB seeds to BFS-expand this run.

    Finding `target` new leagues does not require walking the whole pool.
    Expanding 2000 seeds was the OOM; a few dozen well-chosen seeds fill a
    frontier of thousands of candidate league ids.
    """
    return min(_MAX_SEEDS_PER_RUN, max(32, int(target) // 3))


def _frontier_cap_for_target(target: int) -> int:
    """Max unseen league ids to hold while filtering keepers / fetching meta.

    3x target covers attrition (keeper leagues, fetch failures) without
    retaining an unbounded BFS frontier in the 512Mi process.
    """
    return min(_FRONTIER_CAP, max(int(target) * 3, 64))


def _league_ids_from_payload(data) -> List[str]:
    """Pull league_id strings out of a Sleeper user-leagues list and drop the rest.

    The payload is full league objects (scoring_settings, roster_positions, …).
    Callers must not keep `data` after this returns. Prefer `_ids_from_chunks`
    on the raw HTTP body so the object tree is never built.
    """
    if not data or not isinstance(data, list):
        return []
    ids: List[str] = []
    seen: Set[str] = set()
    for lg in data:
        if not isinstance(lg, dict):
            continue
        lid = lg.get("league_id")
        if not lid:
            continue
        lid = str(lid)
        if lid in seen:
            continue
        seen.add(lid)
        ids.append(lid)
        if len(ids) >= _MAX_LEAGUES_PER_USER:
            break
    return ids


def _ids_from_chunks(
    chunks: Iterable[bytes],
    pattern: re.Pattern[bytes],
    limit: int,
    max_bytes: int = _MAX_SCAN_BYTES,
) -> List[str]:
    """Scan raw JSON chunks for id fields. Peak memory is one chunk + overlap.

    Used for /user/.../leagues and /rosters so we never `resp.json()` a
    multi-league payload (that parse tree is what OOM'd league-discovery).
    """
    if limit <= 0:
        return []
    ids: List[str] = []
    seen: Set[str] = set()
    leftover = b""
    scanned = 0
    for chunk in chunks:
        if not chunk:
            continue
        scanned += len(chunk)
        buf = leftover + chunk
        for m in pattern.finditer(buf):
            lid = m.group(1).decode("ascii")
            if lid in seen:
                continue
            seen.add(lid)
            ids.append(lid)
            if len(ids) >= limit:
                return ids
        leftover = buf[-_ID_OVERLAP:] if len(buf) > _ID_OVERLAP else buf
        if scanned >= max_bytes:
            break
    return ids


def _ids_from_stream(path: str, pattern: re.Pattern[bytes], limit: int) -> List[str]:
    """GET `path` with stream=True and extract ids without parsing JSON."""
    if limit <= 0:
        return []
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, timeout=10, stream=True)
        try:
            if resp.status_code == 429:
                logger.warning("[discovery] Rate limited - sleeping 60s")
                time.sleep(60)
                resp.close()
                resp = SESSION.get(url, timeout=10, stream=True)
            resp.raise_for_status()
            return _ids_from_chunks(
                resp.iter_content(chunk_size=_STREAM_CHUNK),
                pattern,
                limit,
            )
        finally:
            resp.close()
    except Exception as exc:
        logger.debug("[discovery] %s failed: %s", path, exc)
        return []


def _get(path: str, params: dict | None = None) -> list | dict | None:
    url = f"{SLEEPER_BASE}{path}"
    try:
        resp = SESSION.get(url, params=params, timeout=10)
        try:
            if resp.status_code == 429:
                logger.warning("[discovery] Rate limited - sleeping 60s")
                time.sleep(60)
                resp.close()
                resp = SESSION.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        finally:
            resp.close()
    except Exception as exc:
        logger.debug("[discovery] %s failed: %s", path, exc)
        return None


def _current_season() -> int:
    state = _get("/state/nfl")
    if state and "season" in state:
        return int(state["season"])
    return 2024


def _seed_league_ids(season: int, limit: int = _MAX_SEEDS_PER_RUN) -> List[str]:
    """
    Seed the discovery frontier from leagues already in the DB.

    Sleeper's trending endpoint only returns {player_id, count} - it does NOT
    embed league IDs, so we can't use it for seeding.  Instead we BFS-expand
    from whatever leagues are already stored (populated by manual inserts or
    previous discovery runs).  On a completely fresh DB the frontier will be
    empty; the user must insert at least one league_id manually to bootstrap.
    Includes both dynasty (2) and true-redraft (0) leagues as BFS seeds.

    Returns a list (SQL order: least-recently crawled first) capped at `limit`.
    Expanding the whole pool is neither necessary to hit `target` nor safe on
    the 512Mi cron.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT league_id FROM trade_intel_leagues
            WHERE season IN (%s, %s)
              AND league_type IN (0, 2)
            ORDER BY last_crawled_at ASC NULLS FIRST
            LIMIT %s
            """,
            (season, season - 1, int(limit))
        ).fetchall()
    seeds = [r["league_id"] for r in rows]
    logger.info("[discovery] DB seeds: %d leagues to BFS-expand from (limit=%d)",
                len(seeds), limit)
    return seeds


def _user_leagues(user_id: str, season: int) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    # Tuple not set: stable order. Stream-scan ids so the full league objects
    # (scoring_settings, roster_positions, …) are never parsed into a tree.
    for yr in (season, season + 1):
        remaining = _MAX_LEAGUES_PER_USER - len(ids)
        for lid in _ids_from_stream(
            f"/user/{user_id}/leagues/nfl/{yr}",
            _LEAGUE_ID_RE,
            remaining,
        ):
            if lid not in seen:
                seen.add(lid)
                ids.append(lid)
                if len(ids) >= _MAX_LEAGUES_PER_USER:
                    return ids
    return ids


def _league_meta(league_id: str) -> Optional[Dict]:
    return _get(f"/league/{league_id}")


def _roster_owner_ids(league_id: str) -> List[str]:
    return _ids_from_stream(
        f"/league/{league_id}/rosters",
        _OWNER_ID_RE,
        _MAX_OWNERS_PER_LEAGUE,
    )


def _already_known(season: int) -> Set[str]:
    # Stream through a server-side cursor and fold straight into the set. This set
    # scales with the whole season's league pool and is rebuilt every run, so
    # avoid also materializing a full fetchall() list beside it (that doubled the
    # peak of the largest allocation in discovery as the pool grew).
    known: Set[str] = set()
    with get_conn() as conn:
        with conn.cursor(name="ti_already_known") as cur:
            cur.itersize = 10000
            cur.execute(
                "SELECT league_id FROM trade_intel_leagues WHERE season = %s",
                (season,),
            )
            for r in cur:
                known.add(r["league_id"])
    return known


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
    from utils.lineup_slots import is_superflex_lineup
    return is_superflex_lineup(meta.get("roster_positions") or [])


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


def _expand_one_seed(
    league_id: str, season: int, known: Set[str]
) -> Tuple[List[str], List[str]]:
    """Fetch owners of one seed league and return (owner_ids, unseen league ids)."""
    time.sleep(_REQUEST_DELAY)
    owner_ids = _roster_owner_ids(league_id)
    new_leagues: List[str] = []
    seen_local: Set[str] = set()
    for owner_id in owner_ids:
        time.sleep(_REQUEST_DELAY)
        for lid in _user_leagues(owner_id, season):
            if lid in known or lid in seen_local:
                continue
            seen_local.add(lid)
            new_leagues.append(lid)
    return owner_ids, new_leagues


def _expand_seeds_into_frontier(
    seeds: List[str],
    known: Set[str],
    frontier_cap: int,
    expand_league: Callable[[str], Tuple[List[str], List[str]]],
    workers: int = _DISCOVERY_WORKERS,
    in_flight_cap: int = _DISCOVERY_IN_FLIGHT,
) -> Tuple[Set[str], List[str], int]:
    """BFS-expand seeds until the frontier hits `frontier_cap`.

    Only `in_flight_cap` futures are alive at once (the old all-at-once submit
    of every seed is what grew peak memory with the seed LIMIT). Stops submitting
    once the frontier is full so leftover seeds are never expanded.

    Returns (frontier, owner_ids, seeds_expanded).
    """
    frontier: Set[str] = set()
    owner_ids_out: List[str] = []
    seeds_expanded = 0
    seed_iter = iter(seeds)
    workers = max(1, int(workers))
    in_flight_cap = max(workers, int(in_flight_cap))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict = {}

        def fill() -> None:
            while len(futures) < in_flight_cap and len(frontier) < frontier_cap:
                try:
                    lid = next(seed_iter)
                except StopIteration:
                    return
                futures[executor.submit(expand_league, lid)] = lid

        fill()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut, None)
                seeds_expanded += 1
                try:
                    owner_ids, new_ids = fut.result()
                except Exception as exc:
                    logger.debug("[discovery] seed expand failed: %s", exc)
                    continue
                owner_ids_out.extend(owner_ids)
                if len(frontier) >= frontier_cap:
                    continue
                for nid in new_ids:
                    if nid not in known:
                        frontier.add(nid)
                        if len(frontier) >= frontier_cap:
                            break
            if len(frontier) < frontier_cap:
                fill()

    return frontier, owner_ids_out, seeds_expanded


def run_discovery(target: int = _MAX_LEAGUES, season: Optional[int] = None) -> int:
    """
    Discover up to `target` dynasty Sleeper leagues and store them.
    Returns total count of newly inserted leagues.
    """
    if season is None:
        season = _current_season()

    seed_limit = _seed_limit_for_target(target)
    frontier_cap = _frontier_cap_for_target(target)

    known = _already_known(season)
    seeds: List[str] = _seed_league_ids(season, limit=seed_limit)
    visited_users: Set[str] = set()
    to_save: List[Dict] = []
    total_new = 0
    dynasty_count = 0
    redraft_count = 0

    print(
        f"[discovery] Starting. Known={len(known)}, Seeds={len(seeds)}, "
        f"Target={target}, frontier_cap={frontier_cap}"
    )
    _log_rss("start")

    def expand_league(league_id: str) -> Tuple[List[str], List[str]]:
        return _expand_one_seed(league_id, season, known)

    frontier, seed_user_ids, seeds_expanded = _expand_seeds_into_frontier(
        seeds, known, frontier_cap, expand_league,
    )
    _log_rss("after seed expand")

    if seed_user_ids:
        _save_users(list(dict.fromkeys(seed_user_ids)), source="bfs")

    print(
        f"[discovery] Seed expansion complete. {seeds_expanded}/{len(seeds)} seeds "
        f"expanded, {len(frontier)} leagues in frontier"
    )

    def process_frontier_batch(
        batch_leagues: List[str], expand_owners: bool
    ) -> Tuple[List[Dict], List[str]]:
        """Process a batch of frontier leagues and return (to_save, new_frontier_leagues)."""
        batch_to_save = []
        batch_new_frontier = []
        visited_snapshot = set(visited_users)

        def process_single_frontier_league(
            league_id: str,
        ) -> Tuple[Optional[Dict], List[str], List[str]]:
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

            new_frontier_leagues: List[str] = []
            walked_owners: List[str] = []
            # Only walk owners when the frontier still needs filling. Fetching
            # every owner's full league list after we already have enough
            # candidates was unbounded extra JSON in the 512Mi process.
            if expand_owners:
                owner_ids = _roster_owner_ids(league_id)
                for owner_id in owner_ids:
                    if owner_id in visited_snapshot:
                        continue
                    walked_owners.append(owner_id)
                    time.sleep(_REQUEST_DELAY)
                    user_leagues = _user_leagues(owner_id, season)
                    new_frontier_leagues.extend(
                        lid for lid in user_leagues if lid not in known
                    )

            return league_data, new_frontier_leagues, walked_owners

        with ThreadPoolExecutor(max_workers=_DISCOVERY_WORKERS) as executor:
            futures = {
                executor.submit(process_single_frontier_league, lid): lid
                for lid in batch_leagues
            }
            batch_users: List[str] = []
            for future in as_completed(futures):
                league_data, new_frontier, walked_owners = future.result()
                if league_data:
                    batch_to_save.append(league_data)
                    known.add(league_data["league_id"])
                batch_new_frontier.extend(new_frontier)
                batch_users.extend(walked_owners)
                visited_users.update(walked_owners)
        if batch_users:
            _save_users(list(dict.fromkeys(batch_users)), source="bfs")

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

        remaining = target - total_new
        expand_owners = len(frontier) < min(frontier_cap, max(remaining * 2, 32))
        batch_to_save, batch_new_frontier = process_frontier_batch(
            batch_leagues, expand_owners,
        )
        processed_count += len(batch_leagues)

        batch_dynasty = sum(1 for lg in batch_to_save if lg["league_type"] == LeagueType.DYNASTY)
        batch_redraft = sum(1 for lg in batch_to_save if lg["league_type"] == LeagueType.REDRAFT)
        dynasty_count += batch_dynasty
        redraft_count += batch_redraft

        to_save.extend(batch_to_save)
        if expand_owners:
            for new_lid in batch_new_frontier:
                if new_lid not in known and len(frontier) < frontier_cap:
                    frontier.add(new_lid)

        if len(to_save) >= 100:
            n = _save_leagues(to_save)
            total_new += n
            to_save = []

    if to_save:
        total_new += _save_leagues(to_save)

    _log_rss("done")
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
