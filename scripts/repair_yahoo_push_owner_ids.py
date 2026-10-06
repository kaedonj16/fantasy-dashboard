#!/usr/bin/env python3
"""Repair Yahoo push_subscriptions.owner_id values that point at the wrong manager.

Background: an earlier backfill (scripts/archive/backfills/backfill_push_owner_ids.py)
and the subscribe endpoint's fallback both resolved a Yahoo subscription's
owner_id as the "most recently seen guid" from yahoo_league_owners. That table
records EVERY Yahoo user who ever authorized while viewing a league (it exists
so background jobs can fetch a league on any authorized viewer's token), so
"most recent" is not the subscriber's identity. Rows ended up carrying another
manager's guid, and those devices got TD alerts for players on someone else's
roster (and missed their own players' alerts).

This script re-points each Yahoo row at the subscriber's OWN Yahoo guid,
resolved via the row's account_key -> account_identities (the account's linked
Yahoo identity, which has an anti-steal guarantee). It never writes another
manager's guid:

- row.owner_id already equals one of the account's linked Yahoo guids -> keep
- account has a linked Yahoo guid but the row points elsewhere -> fix to it
- no linked Yahoo identity for the account -> set owner_id NULL. NULL never
  matches a roster owner, so no wrong alerts go out; the next subscribe with a
  live Yahoo session re-resolves the correct guid.

Usage:
    python scripts/repair_yahoo_push_owner_ids.py [--dry-run]

Logs every change as:
    [repair] league_id=<id> endpoint=<prefix>... owner_id <old> -> <new>
and a summary line at the end.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("repair_yahoo_push_owner_ids")


def _account_yahoo_guids(conn, account_key) -> list[str]:
    """The account's OWN linked Yahoo guids (never another viewer's)."""
    try:
        account_id = int(str(account_key or "").strip())
    except (TypeError, ValueError):
        return []
    if not account_id:
        return []
    try:
        rows = conn.execute(
            "SELECT platform_user_id FROM account_identities "
            "WHERE account_id = %s AND platform = 'yahoo'",
            (account_id,),
        ).fetchall()
    except Exception as exc:
        logger.warning("[repair] account_identities lookup failed: %s", exc)
        return []
    guids = []
    for r in rows or []:
        guid = r["platform_user_id"] if isinstance(r, dict) else r[0]
        guid = str(guid or "").strip()
        if guid:
            guids.append(guid)
    return guids


def _decide_owner_id(owner_id, account_guids):
    """Pure per-row decision: (new_owner_id, action).

    - owner_id already one of the account's own linked Yahoo guids -> keep it
    - account has a linked Yahoo guid but the row points elsewhere -> fix to it
    - no linked Yahoo identity -> NULL (never another manager's guid; the
      next subscribe with a live Yahoo session re-resolves correctly)
    """
    owner_id = str(owner_id or "").strip() or None
    guids = [str(g or "").strip() for g in account_guids or [] if str(g or "").strip()]
    if owner_id and owner_id in guids:
        return owner_id, "kept"
    if guids:
        return guids[0], "fixed"
    return None, "nulled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="log what would change without writing")
    args = ap.parse_args()

    from dashboard_services.db import get_conn

    checked = 0
    kept = 0
    fixed = 0
    nulled = 0
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT endpoint, league_id, owner_id, account_key "
                "FROM push_subscriptions "
                "WHERE COALESCE(platform, '') = 'yahoo' "
                "AND COALESCE(league_id, '') <> ''"
            ).fetchall()
        except Exception as exc:
            logger.error("[repair] could not read push_subscriptions: %s", exc)
            return 1

        for r in rows:
            if isinstance(r, dict):
                endpoint, league_id = r["endpoint"], r["league_id"]
                owner_id, account_key = r["owner_id"], r.get("account_key")
            else:
                endpoint, league_id, owner_id, account_key = r[0], r[1], r[2], r[3]
            checked += 1
            guids = _account_yahoo_guids(conn, account_key)
            new_owner, action = _decide_owner_id(owner_id, guids)
            if action == "kept":
                kept += 1
                continue
            logger.info(
                "[repair] league_id=%s endpoint=%s... owner_id %s -> %s",
                league_id, str(endpoint)[:24], owner_id, new_owner,
            )
            if action == "fixed":
                fixed += 1
            else:
                nulled += 1
            if not args.dry_run:
                conn.execute(
                    "UPDATE push_subscriptions SET owner_id = %s "
                    "WHERE endpoint = %s AND league_id = %s",
                    (new_owner, endpoint, str(league_id)),
                )
        if not args.dry_run:
            conn.commit()

    logger.info(
        "[repair] done: yahoo_rows_checked=%d kept=%d fixed=%d nulled=%d%s",
        checked, kept, fixed, nulled, " (dry run)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
