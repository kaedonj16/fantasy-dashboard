#!/usr/bin/env python3
"""Repair stale breakout grade rows shadowed by reconstructed/backtest runs.

Background
----------
``weekly_breakout_grades`` is keyed by (player_id, season, as_of_week,
scoring_version) and the grader treats any non-"ungraded" row as terminal:
it is never recomputed (see ``load_existing_grade_keys`` and the
``ON CONFLICT ... DO NOTHING`` insert in
``data_building/breakout_engine/weekly_grading.py``).

A reconstructed (backtest) run used to be able to write grade rows first;
when the live weekly run later published its snapshot, the grader skipped
the live calls as "already graded" and the live grades were silently
dropped.  The track record then served reconstructed scores (e.g. a 21.0
that was never surfaced) instead of the live surfaced scores.

PR #2347 prevents this going forward (a live ``publish_weekly_snapshot``
deletes that week's grade rows so the grader re-grades from live calls),
but weeks that are already poisoned need this one-time cleanup.

What this script does
--------------------
For every (season, as_of_week, scoring_version) present in
``weekly_breakout_grades`` it compares the grade rows against the currently
published snapshot in ``weekly_breakout_scores`` and the run record in
``weekly_breakout_runs``.  A week's grades are STALE (deleted) when:

  A. a completed *live* (non-reconstructed) run owns the snapshot, the
     grades were written *before* that run published, and the grade scores
     differ from the snapshot scores for the same players; or
  B. the run is still flagged reconstructed but the snapshot no longer
     matches the grades: either the scores differ for overlapping players,
     or the graded player *population* barely overlaps the snapshot
     population (under 50% overlap and not merely in-progress grading).
     A live snapshot published under the same key while reconstructed
     grades shadow it is the typical cause.

Weeks that are reconstructed with no live snapshot are LEFT ALONE
(backtest-only history is legitimate).  Weeks whose grades already match
the published snapshot are LEFT ALONE.

After deletion, the next grader run re-grades those weeks from the live
calls automatically.

Usage (run in the Render shell from the repo root):
    python scripts/repair_breakout_grades.py                  # dry run
    python scripts/repair_breakout_grades.py --apply           # actually delete
    python scripts/repair_breakout_grades.py --season 2026 --week 1
    python scripts/repair_breakout_grades.py --apply --json    # machine output

Safe by default: without --apply nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard_services.db import get_conn

GRADES_TABLE = "weekly_breakout_grades"
SCORES_TABLE = "weekly_breakout_scores"
RUNS_TABLE = "weekly_breakout_runs"


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _iso(dt) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _run_reconstructed(detail) -> bool:
    """Mirror of the app's reconstructed-week check (runs.detail JSONB)."""
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except (ValueError, TypeError):
            return False
    if isinstance(detail, dict):
        return str(detail.get("reconstructed", "false")).lower() == "true"
    return False


OVERLAP_STALE_THRESHOLD = 0.5


def decide_week(grade_scores, snap_scores, run_status, run_reconstructed,
                newest_grade_at, published_at):
    """Pure stale-grade decision for one (season, week, version) group.

    ``grade_scores`` / ``snap_scores`` map player_id -> score (or None).
    ``run_status`` is None when there is no run record. Returns
    ``(decision, reason, details)``; ``decision`` is "delete" or "keep" and
    ``details`` carries overlap stats for the human report.
    """
    grade_set = set(grade_scores)
    snap_set = set(snap_scores)
    overlap = [p for p in grade_scores if p in snap_scores]
    mismatched = [
        p for p in overlap
        if grade_scores[p] is not None and snap_scores[p] is not None
        and abs(grade_scores[p] - snap_scores[p]) > 0.001
    ]
    denom = max(len(grade_set), len(snap_set))
    overlap_ratio = (len(overlap) / denom) if denom else 1.0
    # Grades drawn from a different player population than the published
    # snapshot. The subset guard excludes in-progress grading (every graded
    # player still inside the snapshot set) from this signal.
    populations_differ = (
        bool(snap_set)
        and overlap_ratio < OVERLAP_STALE_THRESHOLD
        and not grade_set.issubset(snap_set)
    )
    details = {
        "overlapping_players": len(overlap),
        "overlap_ratio": round(overlap_ratio, 4),
        "mismatched_players": len(mismatched),
        "populations_differ": populations_differ,
    }

    run_live = run_status in ("completed", "success") and not run_reconstructed

    decision = "keep"
    reason = "grades match the published snapshot"
    if run_status is None:
        reason = "no run record; keeping (nothing to re-grade from)"
    elif run_reconstructed:
        if snap_set and (mismatched or populations_differ):
            decision = "delete"
            if populations_differ and not mismatched:
                reason = (
                    "run is flagged reconstructed and the graded player set "
                    "barely overlaps the published snapshot "
                    f"({overlap_ratio:.0%} overlap): reconstructed grades are "
                    "shadowing a different (live) snapshot population"
                )
            else:
                reason = (
                    "run is flagged reconstructed but the published snapshot "
                    "scores differ from the grade scores: reconstructed grades "
                    "are shadowing a live snapshot"
                )
        elif snap_set:
            reason = "reconstructed week with no live snapshot; keeping backtest history"
        else:
            reason = "reconstructed week with no snapshot rows; keeping"
    elif run_live:
        if not snap_set:
            reason = "live run but no snapshot rows; keeping (nothing to re-grade from)"
        elif (newest_grade_at is not None and published_at is not None
              and newest_grade_at < published_at):
            if mismatched or populations_differ:
                decision = "delete"
                reason = (
                    "grades were written before the live snapshot published "
                    "and do not match it: stale reconstructed grades shadowing live data"
                )
            else:
                reason = "grades predate the live publish but scores still match; keeping"
        else:
            reason = "grades were written after the live snapshot published; keeping"
    else:
        reason = "run is not a completed live run; keeping"
    return decision, reason, details


def analyze(season: int | None = None, week: int | None = None) -> list[dict]:
    """Return one report dict per (season, as_of_week, scoring_version) with grades."""
    with get_conn() as conn:
        where = []
        params: list = []
        if season is not None:
            where.append("season = %s")
            params.append(int(season))
        if week is not None:
            where.append("as_of_week = %s")
            params.append(int(week))
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        grade_keys = conn.execute(
            f"SELECT DISTINCT season, as_of_week, scoring_version "
            f"FROM {GRADES_TABLE} "
            f"{where_sql} "
            f"ORDER BY season, as_of_week, scoring_version",
            tuple(params),
        ).fetchall()

        reports: list[dict] = []
        for gk in grade_keys:
            s, w, ver = int(gk["season"]), int(gk["as_of_week"]), gk["scoring_version"]

            grades = conn.execute(
                f"SELECT player_id, player_name, breakout_score, grade, graded_at "
                f"FROM {GRADES_TABLE} "
                f"WHERE season = %s AND as_of_week = %s AND scoring_version = %s",
                (s, w, ver),
            ).fetchall()

            run = conn.execute(
                f"SELECT id, status, calculated_at, completed_at, detail "
                f"FROM {RUNS_TABLE} "
                f"WHERE season = %s AND as_of_week = %s AND scoring_version = %s "
                f"ORDER BY calculated_at DESC LIMIT 1",
                (s, w, ver),
            ).fetchone()

            snaps = conn.execute(
                f"SELECT player_id, player_name, breakout_score "
                f"FROM {SCORES_TABLE} "
                f"WHERE season = %s AND as_of_week = %s AND scoring_version = %s",
                (s, w, ver),
            ).fetchall()

            grade_scores = {str(r["player_id"]): _f(r["breakout_score"]) for r in grades}
            snap_scores = {str(r["player_id"]): _f(r["breakout_score"]) for r in snaps}

            graded_ats = [r["graded_at"] for r in grades if r["graded_at"] is not None]
            newest_grade = max(graded_ats) if graded_ats else None
            run_status = run.get("status") if run else None
            run_recon = bool(run) and _run_reconstructed(run.get("detail"))
            published_at = run.get("calculated_at") if run else None

            decision, reason, details = decide_week(
                grade_scores, snap_scores, run_status, run_recon,
                newest_grade, published_at,
            )

            # Samples for the report (names help Kaedon eyeball it).
            names = {str(r["player_id"]): r.get("player_name") for r in grades}
            snap_names = {str(r["player_id"]): r.get("player_name") for r in snaps}
            grade_set = set(grade_scores)
            snap_set = set(snap_scores)
            sample_mm = [
                {
                    "player": names.get(p, p),
                    "grade_score": grade_scores[p],
                    "snapshot_score": snap_scores[p],
                }
                for p in sorted(grade_set & snap_set)
                if (grade_scores[p] is not None and snap_scores[p] is not None
                    and abs(grade_scores[p] - snap_scores[p]) > 0.001)
            ][:5]
            grade_only_sample = [names.get(p, p) for p in sorted(grade_set - snap_set)[:5]]
            snap_only_sample = [snap_names.get(p, p) for p in sorted(snap_set - grade_set)[:5]]

            reports.append(
                {
                    "season": s,
                    "as_of_week": w,
                    "scoring_version": ver,
                    "grade_rows": len(grades),
                    "snapshot_rows": len(snaps),
                    "run_status": run_status,
                    "run_reconstructed": run_recon,
                    "run_published_at": _iso(published_at),
                    "newest_grade_at": _iso(newest_grade),
                    "overlapping_players": details["overlapping_players"],
                    "overlap_ratio": details["overlap_ratio"],
                    "mismatched_players": details["mismatched_players"],
                    "populations_differ": details["populations_differ"],
                    "mismatch_sample": sample_mm,
                    "grade_only_sample": grade_only_sample,
                    "snapshot_only_sample": snap_only_sample,
                    "decision": decision,
                    "reason": reason,
                }
            )
        return reports


def apply_delete(report: dict) -> int:
    """Delete one stale (season, week, version) group of grade rows. Returns rows deleted."""
    with get_conn() as conn:
        # get_conn defaults to autocommit=False; commit explicitly on success.
        cur = conn.execute(
            f"DELETE FROM {GRADES_TABLE} "
            f"WHERE season = %s AND as_of_week = %s AND scoring_version = %s",
            (report["season"], report["as_of_week"], report["scoring_version"]),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        return deleted


def _print_human(reports: list[dict], deleted: dict[tuple, int] | None) -> None:
    to_delete = [r for r in reports if r["decision"] == "delete"]
    kept = [r for r in reports if r["decision"] != "delete"]
    print(f"Breakout grade repair: {len(reports)} week/version group(s) with grade rows")
    print(f"  {len(to_delete)} stale, {len(kept)} healthy")
    print()
    for r in reports:
        key = (r["season"], r["as_of_week"], r["scoring_version"])
        tag = "DELETE" if r["decision"] == "delete" else "keep"
        print(
            f"[{tag}] season={r['season']} week={r['as_of_week']} "
            f"version={r['scoring_version']} grades={r['grade_rows']} "
            f"snapshots={r['snapshot_rows']} run={r['run_status'] or 'none'}"
            + (" reconstructed" if r["run_reconstructed"] else "")
        )
        print(f"       {r['reason']}")
        if r.get("overlap_ratio") is not None:
            print(
                f"       player overlap: {r['overlapping_players']} "
                f"({r['overlap_ratio']:.0%} of larger set)"
            )
        if r["mismatch_sample"]:
            for m in r["mismatch_sample"]:
                print(
                    f"       - {m['player']}: grade score {m['grade_score']} "
                    f"vs snapshot {m['snapshot_score']}"
                )
            extra = r["mismatched_players"] - len(r["mismatch_sample"])
            if extra > 0:
                print(f"       ... and {extra} more mismatched player(s)")
        if r.get("populations_differ"):
            if r.get("grade_only_sample"):
                print(f"       only in grades: {', '.join(r['grade_only_sample'])}")
            if r.get("snapshot_only_sample"):
                print(f"       only in snapshot: {', '.join(r['snapshot_only_sample'])}")
        if deleted is not None and key in deleted:
            print(f"       deleted {deleted[key]} row(s)")
    print()
    if deleted is None:
        if to_delete:
            print("DRY RUN: nothing deleted. Re-run with --apply to delete.")
        else:
            print("DRY RUN: nothing to delete.")
    else:
        total = sum(deleted.values())
        print(f"Deleted {total} stale grade row(s) across {len(deleted)} week(s).")
        print("The next grader run will re-grade these weeks from the live calls.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Delete stale breakout grade rows shadowed by reconstructed runs "
        "(dry run unless --apply)."
    )
    ap.add_argument("--apply", action="store_true",
                    help="actually delete stale grade rows (default is dry run)")
    ap.add_argument("--season", type=int, default=None,
                    help="only inspect this season")
    ap.add_argument("--week", type=int, default=None,
                    help="only inspect this as_of_week")
    ap.add_argument("--json", action="store_true",
                    help="print the full report as JSON")
    args = ap.parse_args(argv)

    try:
        reports = analyze(season=args.season, week=args.week)
    except Exception as exc:  # noqa: BLE001 - surfaced cleanly for shell use
        print(f"ERROR: could not analyze grade rows: {exc}", file=sys.stderr)
        return 1

    deleted: dict[tuple, int] = {}
    if args.apply:
        for r in reports:
            if r["decision"] != "delete":
                continue
            try:
                n = apply_delete(r)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"ERROR deleting season={r['season']} week={r['as_of_week']} "
                    f"version={r['scoring_version']}: {exc}",
                    file=sys.stderr,
                )
                return 1
            deleted[(r["season"], r["as_of_week"], r["scoring_version"])] = n

    if args.json:
        out = {"dry_run": not args.apply, "weeks": reports}
        if args.apply:
            out["deleted_rows"] = {
                f"{s}/w{w}/{v}": n for (s, w, v), n in deleted.items()
            }
            out["note"] = "The next grader run will re-grade these weeks from the live calls."
        print(json.dumps(out, indent=2, default=str))
        return 0

    _print_human(reports, deleted if args.apply else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
