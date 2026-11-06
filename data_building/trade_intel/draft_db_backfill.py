"""Backfill Draft Database metadata + grade summaries for existing rows.

Drafts crawled before migration 056 have NULL league_name / scoring_type /
league_format / slot_to_roster_id / draft_order_type / roster_names and no
grade_summary. This script re-fetches the Sleeper draft + league + users +
rosters payloads for the stored draft_ids / league_ids and fills those
columns, then (re)computes every draft's persisted grade summary via
draft_adp_crawler.compute_and_store_grade_summary.

Rows whose payloads Sleeper no longer serves (deleted leagues, expired
drafts) keep their NULLs: the Draft Database API/UI degrade honestly for
them (league name falls back to a "League <id>" label, format filters do
not match NULL rows, cards simply omit the grade superlatives).

How to run (on the trade-intel cron service / daily-job shell, from the
repo root):

    python -m data_building.trade_intel.draft_db_backfill
    python -m data_building.trade_intel.draft_db_backfill --limit 500
    python -m data_building.trade_intel.draft_db_backfill --grades-only
    python -m data_building.trade_intel.draft_db_backfill --league-id 1234567890

--grades-only skips the Sleeper re-fetch and only recomputes grade
summaries from data already in the DB (cheap; safe to re-run any time
the segment ADP corpus grows).
"""
from __future__ import annotations

import argparse
import json
import logging

from data_building.trade_intel.draft_adp_crawler import (
    _get, compute_and_store_grade_summary, extract_draft_metadata, get_conn,
)

logger = logging.getLogger(__name__)


def backfill_metadata(limit: int | None = None,
                      league_id: str | None = None) -> dict:
    """Fill migration-056 metadata columns for drafts that lack them."""
    q = (
        "SELECT draft_id, league_id FROM draft_adp_drafts "
        "WHERE (league_name IS NULL OR slot_to_roster_id IS NULL "
        "       OR roster_names IS NULL)"
    )
    params: list = []
    if league_id:
        q += " AND league_id = %s"
        params.append(league_id)
    q += " ORDER BY crawled_at"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()

    league_cache: dict[str, dict] = {}
    filled = failed = 0
    for row in rows:
        draft_id, lid = str(row["draft_id"]), str(row["league_id"])
        try:
            if lid not in league_cache:
                league_cache[lid] = {
                    "league": _get(f"/league/{lid}") or {},
                    "users": _get(f"/league/{lid}/users") or [],
                    "rosters": _get(f"/league/{lid}/rosters") or [],
                    # The draft list carries slot_to_roster_id + type;
                    # index it by draft_id for this league.
                    "drafts": {
                        str(d.get("draft_id")): d
                        for d in (_get(f"/league/{lid}/drafts") or [])
                        if d.get("draft_id")
                    },
                }
            ctx = league_cache[lid]
            draft = ctx["drafts"].get(draft_id)
            if draft is None:
                # The single-draft endpoint still serves most drafts even
                # when the league draft list no longer includes them.
                draft = _get(f"/draft/{draft_id}") or {}
            md = extract_draft_metadata(
                draft, ctx["league"], ctx["users"], ctx["rosters"])
            if not any(md.values()):
                failed += 1
                continue
            with get_conn() as conn:
                conn.execute(
                    "UPDATE draft_adp_drafts SET "
                    "league_name = COALESCE(league_name, %s), "
                    "scoring_type = COALESCE(scoring_type, %s), "
                    "league_format = COALESCE(league_format, %s), "
                    "slot_to_roster_id = COALESCE(slot_to_roster_id, %s::jsonb), "
                    "draft_order_type = COALESCE(draft_order_type, %s), "
                    "roster_names = COALESCE(roster_names, %s::jsonb), "
                    "roster_owners = COALESCE(roster_owners, %s::jsonb) "
                    "WHERE draft_id = %s",
                    (md["league_name"], md["scoring_type"], md["league_format"],
                     json.dumps(md["slot_to_roster_id"]) if md["slot_to_roster_id"] else None,
                     md["draft_order_type"],
                     json.dumps(md["roster_names"]) if md["roster_names"] else None,
                     json.dumps(md["roster_owners"]) if md["roster_owners"] else None,
                     draft_id),
                )
            filled += 1
        except Exception:
            logger.warning("[draft-db-backfill] metadata failed for %s",
                           draft_id, exc_info=True)
            failed += 1
    return {"considered": len(rows), "filled": filled, "failed": failed}


def backfill_grades(limit: int | None = None,
                    league_id: str | None = None) -> dict:
    """(Re)compute grade summaries for stored drafts."""
    q = "SELECT draft_id FROM draft_adp_drafts WHERE status = 'complete'"
    params: list = []
    if league_id:
        q += " AND league_id = %s"
        params.append(league_id)
    q += " ORDER BY crawled_at"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    done = 0
    for row in rows:
        with get_conn() as conn:
            if compute_and_store_grade_summary(conn, str(row["draft_id"])):
                done += 1
    return {"considered": len(rows), "graded": done}


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--league-id", default=None)
    parser.add_argument("--grades-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")
    result = {}
    if not args.grades_only:
        result["metadata"] = backfill_metadata(args.limit, args.league_id)
    if not args.metadata_only:
        result["grades"] = backfill_grades(args.limit, args.league_id)
    print(result)
    return result


if __name__ == "__main__":
    main()
