"""
Backfill pick_slot and pick_order for trade_intel_assets rows where they are null.

The Sleeper /traded_picks endpoint gives us the original pick owner (roster_id)
but NOT the slot. The slot comes from the draft's slot_to_roster_id mapping,
fetched via /draft/{draft_id}.

Three stages:
  Stage 1 — populate pick_roster_id for existing rows by re-fetching the
             original Sleeper transactions (matched by league_id + week).
             Parallelized: fetches up to --workers (league, week) pairs at once.

  Stage 2 — for each league with known pick_roster_id, fetch the Sleeper
             draft details to get slot_to_roster_id, then write pick_slot
             and pick_order (early/mid/late). Parallelized per league.

  Stage 3 — for leagues whose draft order isn't set yet (future picks),
             fall back to estimating early/mid/late from current roster
             standings (wins/points). Parallelized per league.

Run:
    python scripts/backfill_pick_order.py [--stage 1|2|3|all] [--dry-run] [--workers N]
"""
import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dashboard_services.db import get_conn

SLEEPER_BASE     = "https://api.sleeper.app/v1"
_RATE_LIMIT_WAIT = 60  # seconds to back off on 429

# Shared session with large pool for concurrent requests
_SESSION = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]),
)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)


def _get(url: str) -> dict | list | None:
    try:
        r = _SESSION.get(url, timeout=10)
        if r.status_code == 429:
            print(f"  [rate limit] sleeping {_RATE_LIMIT_WAIT}s")
            time.sleep(_RATE_LIMIT_WAIT)
            r = _SESSION.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] {url} → {e}")
        return None


def _slot_to_order(slot: int, num_teams: int) -> str:
    third = num_teams / 3
    if slot <= third:
        return "early"
    if slot <= third * 2:
        return "mid"
    return "late"


def _bulk_update(conn, table_update_sql: str, rows: list[tuple]) -> None:
    """Write updates in chunks of 5000 using parameterized queries."""
    chunk_size = 5000
    for i in range(0, len(rows), chunk_size):
        for params in rows[i : i + chunk_size]:
            conn.execute(table_update_sql, params)


# ---------------------------------------------------------------------------
# Shared: fetch slot_to_roster_id for a league across all its seasons
# ---------------------------------------------------------------------------

def _build_league_draft_orders(league_id: str) -> dict[str, dict[str, int]]:
    """
    Returns {season_str: {roster_id_str: slot_int}} for every draft in
    this league that has slot_to_roster_id set.
    """
    drafts = _get(f"{SLEEPER_BASE}/league/{league_id}/drafts")
    if not drafts:
        return {}

    result: dict[str, dict[str, int]] = {}
    for d in drafts:
        season   = str(d.get("season", ""))
        draft_id = d.get("draft_id")
        if not draft_id or not season:
            continue
        detail = _get(f"{SLEEPER_BASE}/draft/{draft_id}")
        if not detail:
            continue
        # slot_to_roster_id: {slot_str: roster_id} — invert to roster_id→slot
        slot_to_roster = detail.get("slot_to_roster_id") or {}
        if slot_to_roster:
            result[season] = {
                str(roster_id): int(slot)
                for slot, roster_id in slot_to_roster.items()
                if slot and roster_id
            }

    return result


# ---------------------------------------------------------------------------
# Stage 1: populate pick_roster_id — parallelized by (league_id, week)
# ---------------------------------------------------------------------------

def _fetch_group(
    league_id: str,
    week: int,
    trade_list: list,
    asset_map: dict[int, list[int]],
) -> list[tuple[str, int]]:
    """
    Fetch one (league_id, week) batch from Sleeper.
    Returns [(roster_id_str, asset_id), ...] pairs ready to write.
    """
    url  = f"{SLEEPER_BASE}/league/{league_id}/transactions/{week}"
    txns = _get(url)
    if not txns:
        return []

    txn_index = {str(t.get("transaction_id") or ""): t for t in txns}
    updates: list[tuple[str, int]] = []

    for r in trade_list:
        txn = txn_index.get(str(r["transaction_id"]))
        if not txn:
            continue
        picks     = txn.get("draft_picks") or []
        asset_ids = asset_map.get(r["trade_db_id"], [])
        for asset_id, pick in zip(asset_ids, picks):
            roster_id = pick.get("roster_id")
            if roster_id is not None:
                updates.append((str(roster_id), asset_id))

    return updates


def stage1(dry_run: bool = False, workers: int = 20) -> None:
    """Re-fetch original Sleeper transactions to extract roster_id for each pick."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT tit.league_id, tit.season, tit.week,
                            tit.transaction_id, tit.id AS trade_db_id
            FROM trade_intel_trades tit
            JOIN trade_intel_assets tia ON tia.trade_id = tit.id
            WHERE tia.asset_type = 'pick'
              AND tia.pick_roster_id IS NULL
              AND tia.pick_season IS NOT NULL
            ORDER BY tit.league_id, tit.season, tit.week
            """
        ).fetchall()

        if not rows:
            print("Stage 1: nothing to do — all picks already have pick_roster_id.")
            return

        print(f"Stage 1: {len(rows)} trades with picks missing roster_id")

        # Pre-fetch all pick asset IDs per trade in one DB pass to avoid per-trade queries
        trade_ids = list({r["trade_db_id"] for r in rows})
        asset_map: dict[int, list[int]] = defaultdict(list)
        batch = 1000
        for i in range(0, len(trade_ids), batch):
            chunk        = trade_ids[i : i + batch]
            placeholders = ",".join(["%s"] * len(chunk))
            for ar in conn.execute(
                f"""
                SELECT trade_id, id FROM trade_intel_assets
                WHERE trade_id IN ({placeholders})
                  AND asset_type = 'pick'
                  AND pick_roster_id IS NULL
                ORDER BY trade_id, id
                """,
                chunk,
            ).fetchall():
                asset_map[ar["trade_id"]].append(ar["id"])

    # Group by (league_id, week) — one API call per group covers all trades that week
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["league_id"], r["week"])].append(r)

    total_groups = len(groups)
    print(f"Stage 1: {total_groups} (league, week) groups → parallel fetch with {workers} workers")

    all_updates: list[tuple[str, int]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_group, league_id, week, trade_list, asset_map): (league_id, week)
            for (league_id, week), trade_list in groups.items()
        }
        for fut in as_completed(futures):
            completed += 1
            if completed % 2000 == 0:
                print(f"  {completed}/{total_groups} groups done, {len(all_updates)} updates queued")
            try:
                all_updates.extend(fut.result())
            except Exception as e:
                print(f"  [WARN] group {futures[fut]} failed: {e}")

    print(f"Stage 1: {len(all_updates)} roster_id values to write")

    if not dry_run and all_updates:
        with get_conn() as conn:
            _bulk_update(
                conn,
                "UPDATE trade_intel_assets SET pick_roster_id = %s WHERE id = %s",
                all_updates,
            )

    print(f"Stage 1: {'would update' if dry_run else 'updated'} {len(all_updates)} rows with roster_id")


# ---------------------------------------------------------------------------
# Stage 2: populate pick_slot + pick_order — parallelized per league
# ---------------------------------------------------------------------------

def stage2(dry_run: bool = False, workers: int = 20) -> None:
    """
    For picks with known pick_roster_id, fetch each league's draft slot map
    and write pick_slot (1-based) and pick_order (early/mid/late).
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT tia.id AS asset_id, tit.league_id,
                   tia.pick_season, tia.pick_roster_id, til.num_teams
            FROM trade_intel_assets tia
            JOIN trade_intel_trades tit ON tit.id = tia.trade_id
            LEFT JOIN trade_intel_leagues til ON til.league_id = tit.league_id
            WHERE tia.asset_type = 'pick'
              AND tia.pick_slot IS NULL
              AND tia.pick_roster_id IS NOT NULL
              AND tia.pick_season IS NOT NULL
            """
        ).fetchall()

    if not rows:
        print("Stage 2: nothing to do.")
        return

    league_ids = {r["league_id"] for r in rows}
    print(f"Stage 2: {len(rows)} picks across {len(league_ids)} leagues — fetching draft orders in parallel")

    league_orders: dict[str, dict[str, dict[str, int]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_league_draft_orders, lid): lid for lid in league_ids}
        for fut in as_completed(futures):
            lid = futures[fut]
            try:
                league_orders[lid] = fut.result()
            except Exception as e:
                print(f"  [WARN] league {lid} draft fetch failed: {e}")
                league_orders[lid] = {}

    all_updates: list[tuple[int, str, int]] = []  # (slot, pick_order, asset_id)
    skipped = 0

    for r in rows:
        season_orders = league_orders.get(r["league_id"], {})
        order_map     = season_orders.get(str(r["pick_season"]), {})
        slot          = order_map.get(str(r["pick_roster_id"]))
        if slot is None:
            skipped += 1
            continue
        num_teams  = r["num_teams"] or 12
        all_updates.append((slot, _slot_to_order(slot, num_teams), r["asset_id"]))

    print(f"Stage 2: {len(all_updates)} picks resolved, {skipped} skipped (draft order not set)")

    if not dry_run and all_updates:
        with get_conn() as conn:
            _bulk_update(
                conn,
                "UPDATE trade_intel_assets SET pick_slot = %s, pick_order = %s WHERE id = %s",
                all_updates,
            )

    print(f"Stage 2: {'would update' if dry_run else 'updated'} {len(all_updates)} picks")


# ---------------------------------------------------------------------------
# Stage 3: fallback — estimate early/mid/late from standings, parallelized
# ---------------------------------------------------------------------------

def _fetch_standings(league_id: str) -> tuple[str, list[str]]:
    """Returns (league_id, [roster_id, ...] sorted worst→best record)."""
    rosters = _get(f"{SLEEPER_BASE}/league/{league_id}/rosters")
    if not rosters:
        return league_id, []

    def sort_key(r: dict) -> tuple:
        s = r.get("settings") or {}
        return (s.get("wins", 0), s.get("fpts", 0) + s.get("fpts_decimal", 0) * 0.01)

    return league_id, [str(r["roster_id"]) for r in sorted(rosters, key=sort_key)]


def stage3(dry_run: bool = False, workers: int = 20) -> None:
    """
    For picks that still have no pick_order after stage 2, estimate from
    current standings: worst record → early pick. Only meaningful for future
    picks where the draft order hasn't been decided yet.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT tia.id AS asset_id, tit.league_id,
                   tia.pick_season, tia.pick_roster_id, til.num_teams
            FROM trade_intel_assets tia
            JOIN trade_intel_trades tit ON tit.id = tia.trade_id
            LEFT JOIN trade_intel_leagues til ON til.league_id = tit.league_id
            WHERE tia.asset_type = 'pick'
              AND tia.pick_order IS NULL
              AND tia.pick_roster_id IS NOT NULL
              AND tia.pick_season IS NOT NULL
            """
        ).fetchall()

    if not rows:
        print("Stage 3: nothing to do.")
        return

    league_ids = {r["league_id"] for r in rows}
    print(f"Stage 3: {len(rows)} picks across {len(league_ids)} leagues — fetching standings in parallel")

    standings: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_standings, lid): lid for lid in league_ids}
        for fut in as_completed(futures):
            lid = futures[fut]
            try:
                _, order = fut.result()
                standings[lid] = order
            except Exception as e:
                print(f"  [WARN] league {lid} standings fetch failed: {e}")
                standings[lid] = []

    all_updates: list[tuple[int, int]] = []  # (slot, asset_id)
    skipped = 0

    for r in rows:
        order = standings.get(r["league_id"], [])
        if not order:
            skipped += 1
            continue
        try:
            slot = order.index(str(r["pick_roster_id"])) + 1
        except ValueError:
            skipped += 1
            continue
        all_updates.append((slot, r["asset_id"]))

    print(f"Stage 3: {len(all_updates)} picks estimated, {skipped} skipped")

    if not dry_run and all_updates:
        with get_conn() as conn:
            _bulk_update(
                conn,
                "UPDATE trade_intel_assets SET pick_slot = %s WHERE id = %s",
                all_updates,
            )

    print(f"Stage 3: {'would update' if dry_run else 'updated'} {len(all_updates)} picks (standings estimate)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",   choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel workers for HTTP fetches (default 20)")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] no DB writes will be made\n")

    if args.stage in ("1", "all"):
        stage1(dry_run=args.dry_run, workers=args.workers)

    if args.stage in ("2", "all"):
        stage2(dry_run=args.dry_run, workers=args.workers)

    if args.stage in ("3", "all"):
        stage3(dry_run=args.dry_run, workers=args.workers)

    print("\nDone.")
