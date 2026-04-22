"""
Backfill pick_slot and pick_order for trade_intel_assets rows where they are null.

The Sleeper /traded_picks endpoint gives us the original pick owner (roster_id)
but NOT the slot. The slot comes from the draft's draft_order mapping
(roster_id → slot), fetched via /draft/{draft_id}.

Three stages:
  Stage 1 — populate pick_roster_id for existing rows by re-fetching the
             original Sleeper transactions (matched by league_id + week +
             transaction_id, which we store).

  Stage 2 — for each league with known pick_roster_id, fetch the Sleeper
             draft details to get the draft_order map, then write pick_slot
             and pick_order (early/mid/late).

  Stage 3 — for leagues whose draft_order isn't set yet (future picks),
             fall back to estimating early/mid/late from current roster
             standings (wins/points).

Run:
    python scripts/backfill_pick_order.py [--stage 1|2|3|all] [--dry-run]
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from dashboard_services.db import get_conn

SLEEPER_BASE = "https://api.sleeper.app/v1"
RATE_SLEEP   = 0.25
BATCH_SIZE   = 100  # Process updates in batches


def _get(url: str) -> Union[dict, list, None]:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            print("  [rate limit] sleeping 60s")
            time.sleep(60)
            r = requests.get(url, timeout=10)
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


# ---------------------------------------------------------------------------
# Shared: fetch draft_order for a league across all its seasons
# ---------------------------------------------------------------------------

def _build_league_draft_orders(league_id: str) -> dict[str, dict[str, int]]:
    """
    Returns {season_str: {roster_id_str: slot_int}} for every draft in
    this league that has a draft_order set.
    """
    drafts = _get(f"{SLEEPER_BASE}/league/{league_id}/drafts")
    time.sleep(RATE_SLEEP)
    if not drafts:
        return {}

    result: dict[str, dict[str, int]] = {}
    for d in drafts:
        season   = str(d.get("season", ""))
        draft_id = d.get("draft_id")
        if not draft_id or not season:
            continue
        detail = _get(f"{SLEEPER_BASE}/draft/{draft_id}")
        time.sleep(RATE_SLEEP)
        if not detail:
            continue
        slot_to_roster = detail.get("slot_to_roster_id") or {}
        if slot_to_roster:
            result[season] = {str(roster_id): int(slot) for slot, roster_id in slot_to_roster.items()}

    return result


# ---------------------------------------------------------------------------
# Stage 1: populate pick_roster_id for rows that have null
# ---------------------------------------------------------------------------

def stage1(dry_run: bool = False):
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

    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["league_id"], r["season"], r["week"])].append(r)

    # Collect all updates for bulk operations
    bulk_updates = []
    updated = 0
    for (league_id, season, week), trade_list in groups.items():
        url  = f"{SLEEPER_BASE}/league/{league_id}/transactions/{week}"
        txns = _get(url)
        time.sleep(RATE_SLEEP)
        if not txns:
            continue

        txn_index = {str(t.get("transaction_id") or ""): t for t in txns}

        for r in trade_list:
            txn = txn_index.get(str(r["transaction_id"]))
            if not txn:
                continue
            picks = txn.get("draft_picks") or []
            if not picks:
                continue

            with get_conn() as conn:
                asset_rows = conn.execute(
                    """
                    SELECT id FROM trade_intel_assets
                    WHERE trade_id = %s AND asset_type = 'pick'
                      AND pick_roster_id IS NULL
                    ORDER BY id
                    """,
                    (r["trade_db_id"],),
                ).fetchall()

                for asset, pick in zip(asset_rows, picks):
                    roster_id = pick.get("roster_id")
                    if roster_id is None:
                        continue
                    bulk_updates.append((str(roster_id), asset["id"]))
                    updated += 1

    # Perform bulk updates
    if bulk_updates and not dry_run:
        with get_conn() as conn:
            # Update in batches to avoid large transactions
            for i in range(0, len(bulk_updates), BATCH_SIZE):
                batch = bulk_updates[i:i + BATCH_SIZE]
                values_str = ",".join(["(%s, %s)"] * len(batch))
                flat_values = [v for pair in batch for v in pair]
                
                conn.execute(f"""
                    UPDATE trade_intel_assets 
                    SET pick_roster_id = v.roster_id
                    FROM (VALUES {values_str}) AS v(roster_id, asset_id)
                    WHERE trade_intel_assets.id = v.asset_id
                """, flat_values)

    print(f"Stage 1: {'would update' if dry_run else 'updated'} {updated} rows with roster_id")


# ---------------------------------------------------------------------------
# Stage 2: populate pick_slot + pick_order via draft_order
# ---------------------------------------------------------------------------

def stage2(dry_run: bool = False):
    """
    For picks with known pick_roster_id, fetch each league's draft_order and
    write pick_slot (exact position, e.g. 6) and pick_order (early/mid/late).
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

    print(f"Stage 2: {len(rows)} picks to update")

    # Build draft orders once per league
    league_orders: dict[str, dict[str, dict[str, int]]] = {}
    league_ids = {r["league_id"] for r in rows}
    for lid in league_ids:
        league_orders[lid] = _build_league_draft_orders(lid)
        print(f"  {lid}: {len(league_orders[lid])} season(s) with draft order")

    # Collect bulk updates
    bulk_updates = []
    updated = skipped = 0
    
    for r in rows:
        season_orders = league_orders.get(r["league_id"], {})
        order_map     = season_orders.get(str(r["pick_season"]), {})
        slot          = order_map.get(str(r["pick_roster_id"]))
        if slot is None:
            skipped += 1
            continue

        num_teams  = r["num_teams"] or 12
        pick_order = _slot_to_order(slot, num_teams)
        bulk_updates.append((slot, pick_order, r["asset_id"]))
        updated += 1

    # Perform bulk updates
    if bulk_updates and not dry_run:
        with get_conn() as conn:
            for i in range(0, len(bulk_updates), BATCH_SIZE):
                batch = bulk_updates[i:i + BATCH_SIZE]
                values_str = ",".join(["(%s, %s, %s)"] * len(batch))
                flat_values = [v for triple in batch for v in triple]
                
                conn.execute(f"""
                    UPDATE trade_intel_assets 
                    SET pick_slot = v.slot, pick_order = v.order
                    FROM (VALUES {values_str}) AS v(slot, order, asset_id)
                    WHERE trade_intel_assets.id = v.asset_id
                """, flat_values)

    print(f"Stage 2: {'would update' if dry_run else 'updated'} {updated}, "
          f"skipped {skipped} (draft order not yet set)")


# ---------------------------------------------------------------------------
# Stage 3: fallback — estimate early/mid/late from current standings
# ---------------------------------------------------------------------------

def stage3(dry_run: bool = False):
    """
    For picks that still have no pick_order after stage 2 (draft order not set),
    estimate from current league standings: worst record → early pick.
    Only used for future picks where the draft order hasn't been decided.
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

    print(f"Stage 3: {len(rows)} picks to estimate from standings")

    # Cache standings per league
    standings_cache: dict[str, list[str]] = {}  # league_id → [roster_id, ...] worst→best

    def get_standings(league_id: str, num_teams: int) -> list[str]:
        if league_id in standings_cache:
            return standings_cache[league_id]
        rosters = _get(f"{SLEEPER_BASE}/league/{league_id}/rosters")
        time.sleep(RATE_SLEEP)
        if not rosters:
            standings_cache[league_id] = []
            return []
        # Sort by wins ASC then points ASC → worst team first (earliest pick)
        def sort_key(r):
            s = r.get("settings") or {}
            return (s.get("wins", 0), s.get("fpts", 0) + s.get("fpts_decimal", 0) * 0.01)
        sorted_rosters = sorted(rosters, key=sort_key)
        order = [str(r["roster_id"]) for r in sorted_rosters]
        standings_cache[league_id] = order
        return order

    # Collect bulk updates
    bulk_updates = []
    updated = skipped = 0
    
    for r in rows:
        num_teams = r["num_teams"] or 12
        order     = get_standings(r["league_id"], num_teams)
        if not order:
            skipped += 1
            continue
        try:
            slot = order.index(str(r["pick_roster_id"])) + 1  # 1-based
        except ValueError:
            skipped += 1
            continue

        pick_order = _slot_to_order(slot, num_teams)
        bulk_updates.append((pick_order, r["asset_id"]))
        updated += 1

    # Perform bulk updates
    if bulk_updates and not dry_run:
        with get_conn() as conn:
            for i in range(0, len(bulk_updates), BATCH_SIZE):
                batch = bulk_updates[i:i + BATCH_SIZE]
                values_str = ",".join(["(%s, %s)"] * len(batch))
                flat_values = [v for pair in batch for v in pair]
                
                conn.execute(f"""
                    UPDATE trade_intel_assets 
                    SET pick_order = v.order
                    FROM (VALUES {values_str}) AS v(order, asset_id)
                    WHERE trade_intel_assets.id = v.asset_id
                """, flat_values)

    print(f"Stage 3 (standings estimate): "
          f"{'would update' if dry_run else 'updated'} {updated}, skipped {skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] no DB writes will be made\n")

    if args.stage in ("1", "all"):
        stage1(dry_run=args.dry_run)

    if args.stage in ("2", "all"):
        stage2(dry_run=args.dry_run)

    if args.stage in ("3", "all"):
        stage3(dry_run=args.dry_run)

    print("\nDone.")
