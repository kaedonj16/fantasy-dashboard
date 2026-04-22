"""
Backfill pick_order (early/mid/late) for trade_intel_assets rows where it is null.

Two-stage process:
  Stage 1 — populate pick_roster_id for existing rows by re-fetching the original
             Sleeper transactions (we have league_id + week + transaction_id).
  Stage 2 — for each league/season with known pick_roster_id, fetch the Sleeper
             draft's draft_order mapping (roster_id → slot) and update pick_order.

Run:
    python scripts/backfill_pick_order.py [--stage 1|2|both] [--dry-run]
"""
import argparse
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from dashboard_services.db import get_conn

SLEEPER_BASE = "https://api.sleeper.app/v1"
RATE_SLEEP   = 0.3   # seconds between Sleeper API calls


def _get(url: str) -> dict | list | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] GET {url} failed: {e}")
        return None


def _pick_order_from_slot(slot: int, num_teams: int) -> str:
    """Convert a 1-based draft slot to early/mid/late."""
    third = num_teams / 3
    if slot <= third:
        return "early"
    if slot <= third * 2:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Stage 1: populate pick_roster_id for rows that have null
# ---------------------------------------------------------------------------

def stage1_populate_roster_id(dry_run: bool = False):
    """
    Re-fetch original Sleeper transactions to extract roster_id for each pick.
    Groups work by (league_id, season, week) to minimise API calls.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT tit.league_id, tit.season, tit.week, tit.transaction_id, tit.id AS trade_db_id
            FROM trade_intel_trades tit
            JOIN trade_intel_assets tia ON tia.trade_id = tit.id
            WHERE tia.asset_type = 'pick'
              AND tia.pick_roster_id IS NULL
              AND tia.pick_season IS NOT NULL
            ORDER BY tit.league_id, tit.season, tit.week
            """
        ).fetchall()

    if not rows:
        print("Stage 1: no rows need pick_roster_id — already done or no picks.")
        return

    print(f"Stage 1: {len(rows)} trades with picks missing roster_id")

    # Group by (league_id, season, week) so we fetch each week once
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["league_id"], r["season"], r["week"])].append(r)

    updated = 0
    for (league_id, season, week), trade_list in groups.items():
        url = f"{SLEEPER_BASE}/league/{league_id}/transactions/{week}"
        txns = _get(url)
        time.sleep(RATE_SLEEP)
        if not txns:
            continue

        # Build index: transaction_id → pick list
        txn_index = {str(t.get("transaction_id") or ""): t for t in txns}

        for r in trade_list:
            txn = txn_index.get(str(r["transaction_id"]))
            if not txn:
                continue
            picks = txn.get("draft_picks") or []
            if not picks:
                continue

            with get_conn() as conn:
                # Fetch asset rows for this trade
                asset_rows = conn.execute(
                    """
                    SELECT id, pick_season, pick_round
                    FROM trade_intel_assets
                    WHERE trade_id = %s AND asset_type = 'pick' AND pick_roster_id IS NULL
                    ORDER BY id
                    """,
                    (r["trade_db_id"],),
                ).fetchall()

                # Match picks by order (both lists should align since we insert in order)
                for asset, pick in zip(asset_rows, picks):
                    roster_id = pick.get("roster_id")
                    if roster_id is None:
                        continue
                    if not dry_run:
                        conn.execute(
                            "UPDATE trade_intel_assets SET pick_roster_id = %s WHERE id = %s",
                            (str(roster_id), asset["id"]),
                        )
                    updated += 1

    print(f"Stage 1: {'would update' if dry_run else 'updated'} {updated} pick rows with roster_id")


# ---------------------------------------------------------------------------
# Stage 2: populate pick_order using Sleeper draft_order
# ---------------------------------------------------------------------------

def stage2_populate_pick_order(dry_run: bool = False):
    """
    For each league/season with picks that have pick_roster_id but no pick_order,
    fetch the Sleeper draft for that season and map roster_id → draft_slot.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT tit.league_id, tia.pick_season, tia.pick_round,
                            tia.pick_roster_id, tia.id AS asset_id,
                            til.num_teams
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
        print("Stage 2: no rows need pick_order — already done or missing roster_id.")
        return

    print(f"Stage 2: {len(rows)} pick assets to update")

    # Cache draft orders: (league_id, season) → {roster_id: slot}
    draft_order_cache: dict[tuple, dict] = {}

    def get_draft_order(league_id: str, season: int) -> dict:
        key = (league_id, season)
        if key in draft_order_cache:
            return draft_order_cache[key]

        drafts = _get(f"{SLEEPER_BASE}/league/{league_id}/drafts")
        time.sleep(RATE_SLEEP)
        if not drafts:
            draft_order_cache[key] = {}
            return {}

        # Find the rookie/regular draft for this season
        order = {}
        for d in drafts:
            if str(d.get("season")) != str(season):
                continue
            draft_id = d.get("draft_id")
            if not draft_id:
                continue
            detail = _get(f"{SLEEPER_BASE}/draft/{draft_id}")
            time.sleep(RATE_SLEEP)
            if not detail:
                continue
            do = detail.get("draft_order") or {}
            if do:
                # draft_order maps roster_id (as string) → slot (1-based)
                order = {str(k): int(v) for k, v in do.items()}
                break  # use first draft that has an order

        draft_order_cache[key] = order
        return order

    updated = 0
    skipped = 0
    for r in rows:
        num_teams = r["num_teams"] or 12
        order_map = get_draft_order(r["league_id"], r["pick_season"])
        slot = order_map.get(str(r["pick_roster_id"]))
        if slot is None:
            skipped += 1
            continue

        pick_order = _pick_order_from_slot(slot, num_teams)
        if not dry_run:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE trade_intel_assets SET pick_order = %s WHERE id = %s",
                    (pick_order, r["asset_id"]),
                )
        updated += 1

    print(f"Stage 2: {'would update' if dry_run else 'updated'} {updated} rows, skipped {skipped} (draft order not found)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill pick_order for trade_intel_assets")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] no DB writes will be made")

    if args.stage in ("1", "both"):
        stage1_populate_roster_id(dry_run=args.dry_run)

    if args.stage in ("2", "both"):
        stage2_populate_pick_order(dry_run=args.dry_run)

    print("Done.")
