#!/usr/bin/env python3
"""Backfill push_subscriptions.owner_id with the correct per-league platform owner ID.

Background: the subscribe endpoint used to store one session-scoped owner_id
(typically the Sleeper viewer_user_id) for every league in a bulk subscribe.
Each platform uses a different roster owner_id namespace, so cross-platform
leagues never matched in _broadcast_owner and owner-targeted pushes
(ScoreZone TD alerts, value drops, etc.) were silently dropped.

- sleeper: roster owner_id is the Sleeper user_id (stored value was correct)
- yahoo: roster owner_id is the manager guid from _yahoo_owner_id, NOT team_id
- espn/fleaflicker/mfl: roster owner_id is the team_id

This script updates Yahoo rows whose stored owner_id is not a known guid for
that league, setting it to the most recently seen guid from
yahoo_league_owners. It is idempotent: running it twice changes nothing the
second time.

Usage:
    python scripts/backfill_push_owner_ids.py [--dry-run]

Logs every change as:
    [backfill] league_id=<id> platform=yahoo owner_id <old> -> <new>
and a summary line at the end.
"""
from __future__ import annotations

import argparse
import logging
import sys

# DEPRECATED -- DO NOT RUN. This script writes the "most recently seen guid"
# from yahoo_league_owners onto every subscriber's Yahoo row, but that table
# records EVERY Yahoo user who ever authorized while viewing a league, so the
# guid is frequently another manager's. Running it reintroduces the bug where
# devices get TD alerts for players on someone else's roster. Use
# scripts/repair_yahoo_push_owner_ids.py instead, which resolves each row via
# the subscriber's own linked account identity and never writes a foreign guid.
if "--i-understand-this-is-deprecated" not in sys.argv:
    print(
        "REFUSING TO RUN: this backfill is deprecated because it assigns other "
        "managers' Yahoo guids to subscribers' push rows. Use "
        "scripts/repair_yahoo_push_owner_ids.py instead.",
        file=sys.stderr,
    )
    sys.exit(2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_push_owner_ids")


def _yahoo_guids_by_league(conn) -> dict:
    """Most recently seen manager guid per Yahoo league_id."""
    out = {}
    try:
        for r in conn.execute(
            "SELECT league_id, guid FROM yahoo_league_owners "
            "ORDER BY updated_at DESC"
        ).fetchall():
            lid = str(r["league_id"]) if isinstance(r, dict) else str(r[0])
            guid = r["guid"] if isinstance(r, dict) else r[1]
            if lid and guid and lid not in out:
                out[lid] = str(guid)
    except Exception as exc:
        logger.warning("[backfill] yahoo_league_owners lookup failed: %s", exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="log what would change without writing")
    args = ap.parse_args()

    from dashboard_services.db import get_conn

    guids = {}
    changed = 0
    checked = 0
    skipped = 0
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT DISTINCT league_id, platform FROM push_subscriptions "
                "WHERE COALESCE(league_id, '') <> ''"
            ).fetchall()
        except Exception as exc:
            logger.error("[backfill] could not read push_subscriptions: %s", exc)
            return 1

        guids = _yahoo_guids_by_league(conn)

        for r in rows:
            league_id = r["league_id"] if isinstance(r, dict) else r[0]
            platform = (r["platform"] if isinstance(r, dict) else r[1]) or "sleeper"
            platform = str(platform).lower()
            checked += 1

            # Only Yahoo needs the fix: Sleeper rows already store the
            # user_id, and espn/fleaflicker/mfl use team_id which matches.
            if platform != "yahoo":
                skipped += 1
                continue

            guid = guids.get(str(league_id))
            if not guid:
                logger.info(
                    "[backfill] league_id=%s platform=yahoo: no guid in "
                    "yahoo_league_owners, skipping", league_id)
                skipped += 1
                continue

            subs = conn.execute(
                "SELECT endpoint, owner_id FROM push_subscriptions "
                "WHERE league_id = %s",
                (str(league_id),),
            ).fetchall()
            for s in subs:
                endpoint = s["endpoint"] if isinstance(s, dict) else s[0]
                owner_id = s["owner_id"] if isinstance(s, dict) else s[1]
                if str(owner_id or "") == guid:
                    continue  # already correct; idempotent
                logger.info(
                    "[backfill] league_id=%s platform=yahoo owner_id %s -> %s "
                    "(endpoint %s...)",
                    league_id, owner_id, guid, str(endpoint)[:24])
                changed += 1
                if not args.dry_run:
                    conn.execute(
                        "UPDATE push_subscriptions SET owner_id = %s "
                        "WHERE league_id = %s AND endpoint = %s",
                        (guid, str(league_id), endpoint),
                    )
        if not args.dry_run:
            conn.commit()

    logger.info(
        "[backfill] done: leagues_checked=%d skipped=%d rows_updated=%d%s",
        checked, skipped, changed, " (dry run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
