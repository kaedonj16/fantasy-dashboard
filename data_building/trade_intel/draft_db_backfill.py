"""Backfill Draft Database metadata + grade summaries for existing rows.

Drafts crawled before migration 056 have NULL league_name / scoring_type /
league_format / slot_to_roster_id / draft_order_type / roster_names and no
grade_summary. This script re-fetches the Sleeper draft + league + users +
rosters payloads for the stored draft_ids / league_ids and fills those
columns, then computes every draft's persisted grade summary via
draft_adp_crawler.compute_and_store_grade_summary.

Rows whose payloads Sleeper no longer serves (deleted leagues, expired
drafts) keep their NULLs: the Draft Database API/UI degrade honestly for
them (league name falls back to a "League <id>" label, format filters do
not match NULL rows, cards simply omit the grade superlatives).

Scale + parallelism (2026-11 rework): the corpus is ~134k drafts across
~124k leagues, and the Sleeper cost unit is the LEAGUE (4 API calls:
league, users, rosters, drafts). The original single-threaded version
processed leagues in crawled_at order (oldest, mostly-dead leagues first)
and printed nothing until the end; at ~500k Sleeper calls it could never
finish inside a shell session. This version:

  * fans leagues out over a ThreadPoolExecutor (--workers, default 8);
    each league's payloads are fetched ONCE and fill all of its pending
    drafts. Per-draft UPDATEs commit individually, so progress persists
    per draft and a killed run loses at most the in-flight league -
    re-running resumes from the rows still NULL.
  * works RECENT FIRST (season / draft_started_at descending), so the
    current seasons' boards get metadata and grades early instead of
    after the dead-league tail.
  * logs progress (units done/total, drafts filled/graded, failed,
    drafts/min pace, elapsed) instead of running silent.
  * grades only drafts still missing a grade_summary by default, so a
    re-run resumes instead of recomputing all ~134k summaries;
    --recompute-all restores the original recompute-everything behavior.

How to run (on the trade-intel cron service / daily-job shell, from the
repo root):

    python -m data_building.trade_intel.draft_db_backfill
    python -m data_building.trade_intel.draft_db_backfill --limit 500
    python -m data_building.trade_intel.draft_db_backfill --grades-only
    python -m data_building.trade_intel.draft_db_backfill --league-id 1234567890

--grades-only skips the Sleeper re-fetch and only computes grade
summaries from data already in the DB (cheap; safe to re-run any time
the segment ADP corpus grows).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from data_building.trade_intel.draft_adp_crawler import (
    _get, compute_and_store_grade_summary, extract_draft_metadata, get_conn,
)

logger = logging.getLogger(__name__)

# Metadata fan-out default. 8 workers multiplies the old single-thread
# throughput while keeping worst-case request rate far under Sleeper's
# ~1000 calls/min/IP guidance (8 threads x 4 calls, each gated by real
# latency), and bounds how many workers can sit parked in _get's 60s
# 429 backoff at once. The cap exists so a typo can't turn the flag into
# a self-inflicted rate-limit storm.
DEFAULT_WORKERS = 8
MAX_WORKERS = 32

# Grades are DB-bound, not API-bound: each summary runs several aggregate
# queries over draft_adp_picks against a small Postgres, so the grade
# pool is deliberately smaller than the metadata pool.
DEFAULT_GRADE_WORKERS = 4

# Progress cadence: log when this many units complete OR this many
# seconds pass, whichever comes first.
_PROGRESS_EVERY_LEAGUES = 25
_PROGRESS_EVERY_DRAFTS = 100
_PROGRESS_EVERY_SECONDS = 30.0


def _clamp_workers(workers) -> int:
    try:
        workers = int(workers)
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(workers, MAX_WORKERS))


class _Progress:
    """Main-thread progress logger for one phase.

    Ticked only from the driver loop as futures complete, so it needs
    no locking. Pace is drafts processed (filled/graded + failed) per
    minute - the number that tells the operator whether the run is
    keeping up with Sleeper or drowning in 429 backoffs.
    """

    def __init__(self, phase: str, unit: str, total_units: int, every: int):
        self.phase = phase
        self.unit = unit
        self.total_units = total_units
        self.every = every
        self.units_done = 0
        self.ok = 0
        self.failed = 0
        self.started = time.monotonic()
        self._last_log_at = self.started
        self._last_log_units = 0

    def tick(self, ok: int = 0, failed: int = 0) -> None:
        self.units_done += 1
        self.ok += ok
        self.failed += failed
        now = time.monotonic()
        if (self.units_done - self._last_log_units >= self.every
                or now - self._last_log_at >= _PROGRESS_EVERY_SECONDS):
            self._log(now)
            self._last_log_at = now
            self._last_log_units = self.units_done

    def finish(self) -> None:
        self._log(time.monotonic(), final=True)

    def _log(self, now: float, final: bool = False) -> None:
        elapsed = now - self.started
        processed = self.ok + self.failed
        pace = processed / (elapsed / 60.0) if elapsed > 0 else 0.0
        logger.info(
            "[draft-db-backfill] %s %s: %s %d/%d, drafts ok %d, failed %d, "
            "%.1f drafts/min, elapsed %s",
            self.phase, "done" if final else "progress", self.unit,
            self.units_done, self.total_units, self.ok, self.failed,
            pace, timedelta(seconds=int(elapsed)),
        )


def _run_pool(work_fn, units: dict, workers: int, progress: _Progress):
    """Run work_fn(key, payload) over a dict's items, recent-first.

    work_fn must never raise (per-unit failure isolation lives inside
    it); it returns (ok, failed) draft counts. With workers <= 1 the
    pool is skipped entirely so ordering is strictly deterministic.
    """
    if workers <= 1 or len(units) <= 1:
        for key, payload in units.items():
            ok, failed = work_fn(key, payload)
            progress.tick(ok=ok, failed=failed)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(work_fn, key, payload): key
            for key, payload in units.items()
        }
        for fut in as_completed(futures):
            ok, failed = fut.result()
            progress.tick(ok=ok, failed=failed)


# ---------------------------------------------------------------------------
# Metadata phase
# ---------------------------------------------------------------------------

_METADATA_PENDING_WHERE = (
    "(league_name IS NULL OR slot_to_roster_id IS NULL "
    "OR roster_names IS NULL)"
)

# Recent first: a league's newest drafts carry the current ADP signal and
# are the boards users actually open. The old ORDER BY crawled_at burned
# the first hours of every run on the oldest, mostly-dead leagues.
_RECENT_FIRST = "season DESC, draft_started_at DESC NULLS LAST, crawled_at ASC"

_UPDATE_METADATA_SQL = (
    "UPDATE draft_adp_drafts SET "
    "league_name = COALESCE(league_name, %s), "
    "scoring_type = COALESCE(scoring_type, %s), "
    "league_format = COALESCE(league_format, %s), "
    "slot_to_roster_id = COALESCE(slot_to_roster_id, %s::jsonb), "
    "draft_order_type = COALESCE(draft_order_type, %s), "
    "roster_names = COALESCE(roster_names, %s::jsonb), "
    "roster_owners = COALESCE(roster_owners, %s::jsonb) "
    "WHERE draft_id = %s"
)


def _pending_metadata_rows(limit: int | None = None,
                           league_id: str | None = None) -> list:
    q = ("SELECT draft_id, league_id FROM draft_adp_drafts "
         f"WHERE {_METADATA_PENDING_WHERE}")
    params: list = []
    if league_id:
        q += " AND league_id = %s"
        params.append(league_id)
    q += f" ORDER BY {_RECENT_FIRST}"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()


def _group_by_league(rows) -> dict:
    """draft rows -> {league_id: [draft_id, ...]}, recent-first order.

    The rows arrive sorted recent-first, so a league's insertion slot is
    set by its newest pending draft - dict order IS the work order.
    """
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["league_id"]), []).append(
            str(row["draft_id"]))
    return grouped


def _fetch_league_ctx(lid: str) -> dict:
    """The 4 per-league Sleeper payloads, fetched once per league."""
    return {
        "league": _get(f"/league/{lid}") or {},
        "users": _get(f"/league/{lid}/users") or [],
        "rosters": _get(f"/league/{lid}/rosters") or [],
        # The draft list carries slot_to_roster_id + type; index it by
        # draft_id for this league.
        "drafts": {
            str(d.get("draft_id")): d
            for d in (_get(f"/league/{lid}/drafts") or [])
            if d.get("draft_id")
        },
    }


def _fill_one_draft(draft_id: str, ctx: dict) -> bool:
    draft = ctx["drafts"].get(draft_id)
    if draft is None:
        # The single-draft endpoint still serves most drafts even when
        # the league draft list no longer includes them.
        draft = _get(f"/draft/{draft_id}") or {}
    md = extract_draft_metadata(
        draft, ctx["league"], ctx["users"], ctx["rosters"])
    if not any(md.values()):
        return False
    with get_conn() as conn:
        conn.execute(
            _UPDATE_METADATA_SQL,
            (md["league_name"], md["scoring_type"], md["league_format"],
             json.dumps(md["slot_to_roster_id"]) if md["slot_to_roster_id"] else None,
             md["draft_order_type"],
             json.dumps(md["roster_names"]) if md["roster_names"] else None,
             json.dumps(md["roster_owners"]) if md["roster_owners"] else None,
             draft_id),
        )
    return True


def _fill_league(lid: str, draft_ids: list) -> tuple:
    """Fill one league's pending drafts. Returns (filled, failed).

    Never raises: a dead league (or a 429 storm that makes every fetch
    come back None) counts its drafts failed - they keep their NULLs
    and are retried on the next run - and the other ~124k leagues
    proceed. Rate-limit policy stays in the crawler's _get; the driver
    adds no retries of its own, so a storm degrades to failed rows,
    not to a failure spiral.
    """
    try:
        ctx = _fetch_league_ctx(lid)
    except Exception:
        logger.warning("[draft-db-backfill] payload fetch failed for "
                       "league %s", lid, exc_info=True)
        return 0, len(draft_ids)
    filled = failed = 0
    for draft_id in draft_ids:
        try:
            if _fill_one_draft(draft_id, ctx):
                filled += 1
            else:
                failed += 1
        except Exception:
            logger.warning("[draft-db-backfill] metadata failed for %s",
                           draft_id, exc_info=True)
            failed += 1
    return filled, failed


def backfill_metadata(limit: int | None = None,
                      league_id: str | None = None,
                      workers: int = DEFAULT_WORKERS) -> dict:
    """Fill migration-056 metadata columns for drafts that lack them."""
    rows = _pending_metadata_rows(limit, league_id)
    grouped = _group_by_league(rows)
    progress = _Progress("metadata", "leagues", len(grouped),
                         _PROGRESS_EVERY_LEAGUES)
    logger.info("[draft-db-backfill] metadata: %d drafts pending across "
                "%d leagues, %d workers, recent first",
                len(rows), len(grouped), _clamp_workers(workers))
    _run_pool(_fill_league, grouped, _clamp_workers(workers), progress)
    progress.finish()
    return {"considered": len(rows), "filled": progress.ok,
            "failed": progress.failed}


# ---------------------------------------------------------------------------
# Grades phase
# ---------------------------------------------------------------------------

def _pending_grade_rows(limit: int | None = None,
                        league_id: str | None = None,
                        recompute_all: bool = False) -> list:
    q = "SELECT draft_id FROM draft_adp_drafts WHERE status = 'complete'"
    params: list = []
    if not recompute_all:
        # Default: only drafts still missing a summary, so re-running
        # the backfill resumes instead of recomputing every stored
        # draft. --recompute-all preserves the original behavior.
        q += " AND grade_summary IS NULL"
    if league_id:
        q += " AND league_id = %s"
        params.append(league_id)
    q += f" ORDER BY {_RECENT_FIRST}"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()


def _grade_one(_index, draft_id) -> tuple:
    """Grade one draft. Returns (graded, failed); never raises."""
    draft_id = str(draft_id)
    try:
        with get_conn() as conn:
            return (1, 0) if compute_and_store_grade_summary(
                conn, draft_id) else (0, 1)
    except Exception:
        logger.warning("[draft-db-backfill] grade failed for %s",
                       draft_id, exc_info=True)
        return 0, 1


def backfill_grades(limit: int | None = None,
                    league_id: str | None = None,
                    workers: int = DEFAULT_GRADE_WORKERS,
                    recompute_all: bool = False) -> dict:
    """Compute grade summaries for stored drafts (missing ones by
    default; every complete draft with recompute_all=True)."""
    rows = _pending_grade_rows(limit, league_id, recompute_all)
    # _run_pool iterates a dict; key the drafts by index to keep the
    # recent-first row order as the work order.
    units = {i: str(row["draft_id"]) for i, row in enumerate(rows)}
    progress = _Progress("grades", "drafts", len(units),
                         _PROGRESS_EVERY_DRAFTS)
    logger.info("[draft-db-backfill] grades: %d drafts to grade, "
                "%d workers, recent first%s",
                len(units), _clamp_workers(workers),
                "" if recompute_all else " (missing summaries only)")
    _run_pool(_grade_one, units, _clamp_workers(workers), progress)
    progress.finish()
    return {"considered": len(rows), "graded": progress.ok}


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--league-id", default=None)
    parser.add_argument("--grades-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="parallel leagues for the metadata phase "
                             f"(default {DEFAULT_WORKERS}, max {MAX_WORKERS})")
    parser.add_argument("--grade-workers", type=int,
                        default=DEFAULT_GRADE_WORKERS,
                        help="parallel drafts for the grades phase "
                             f"(default {DEFAULT_GRADE_WORKERS})")
    parser.add_argument("--recompute-all", action="store_true",
                        help="grades: recompute summaries for ALL "
                             "complete drafts, not just ones missing a "
                             "grade_summary")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")
    result = {}
    if not args.grades_only:
        result["metadata"] = backfill_metadata(
            args.limit, args.league_id, workers=args.workers)
    if not args.metadata_only:
        result["grades"] = backfill_grades(
            args.limit, args.league_id, workers=args.grade_workers,
            recompute_all=args.recompute_all)
    print(result)
    return result


if __name__ == "__main__":
    main()
