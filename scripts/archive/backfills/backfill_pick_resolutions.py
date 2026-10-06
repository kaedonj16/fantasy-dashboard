#!/usr/bin/env python3
"""Bounded, idempotent refresh of verified traded-pick selections.

Only Sleeper's completed fantasy draft results are used.  Existing verified
rows are retained when the provider is unavailable; reruns update authoritative
corrections and newly completed drafts.
"""
from __future__ import annotations

import argparse
import logging

from dashboard_services.api import build_league_history_map
from dashboard_services.db import get_conn
from dashboard_services.player_league_trades import build_draft_resolution_map

log = logging.getLogger(__name__)


def backfill(*, league_id: str | None = None, season: int | None = None,
             limit: int = 100, dry_run: bool = True) -> dict[str, int]:
    limit = max(1, min(int(limit), 1000))
    params: list[object] = []
    where = ["a.asset_type='pick'", "a.pick_roster_id IS NOT NULL"]
    if league_id:
        where.append("t.league_id=%s")
        params.append(str(league_id))
    if season:
        where.append("a.pick_season=%s")
        params.append(int(season))
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.id, a.pick_season, a.pick_round, a.pick_roster_id, "
            "t.league_id, t.season AS trade_season FROM trade_intel_assets a "
            "JOIN trade_intel_trades t ON t.id=a.trade_id WHERE "
            + " AND ".join(where) + " ORDER BY a.id LIMIT %s", params,
        ).fetchall()

    # One provider lookup per league lineage, never one per trade row.
    maps: dict[str, dict] = {}
    for row in rows:
        lid = str(row["league_id"])
        if lid in maps:
            continue
        lineage = build_league_history_map("sleeper", lid, int(row["trade_season"])) or {int(row["trade_season"]): lid}
        merged: dict = {}
        conflicts: set = set()
        for hist_lid in set(lineage.values()):
            for key, value in build_draft_resolution_map("sleeper", str(hist_lid)).items():
                if key in merged and merged[key].get("draft_id") != value.get("draft_id"):
                    conflicts.add(key)
                else:
                    merged[key] = value
        maps[lid] = {k: v for k, v in merged.items() if k not in conflicts}

    updates = []
    for row in rows:
        key = (int(row["pick_season"]), int(row["pick_round"]), str(row["pick_roster_id"]))
        hit = maps.get(str(row["league_id"]), {}).get(key)
        if hit:
            updates.append((hit["draft_id"], hit["player_id"], hit["pick_no"], hit["within_round"], row["id"]))
    if updates and not dry_run:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE trade_intel_assets SET resolved_draft_id=%s, resolved_player_id=%s, "
                    "resolved_pick_no=%s, resolved_round_slot=%s, resolved_at=NOW() WHERE id=%s",
                    updates,
                )
    return {"scanned": len(rows), "resolved": len(updates), "updated": 0 if dry_run else len(updates)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-id")
    parser.add_argument("--season", type=int)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    log.info("pick resolution backfill: %s", backfill(
        league_id=args.league_id, season=args.season, limit=args.limit, dry_run=not args.execute,
    ))
