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
  B. the run is still flagged reconstructed but the snapshot scores no
     longer match the grade scores (a live snapshot is present under the
     same key while reconstructed grades shadow it).

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


def analyze(season: int | None = None, week: int | None = None) -> list[dict]:
    """Return one report dict per (season, as_of_week, scoring_version) with grades."""
    with get_conn() as conn:
        grade_keys = conn.execute(
            f"SELECT DISTINCT season, as_of_week, scoring_version "
            f"FROM {GRADES_TABLE} "
            f"WHERE (%s IS NULL OR season = %s) "
            f"  AND (%s IS NULL OR as_of_week = %s) "
            f"ORDER BY season, as_of_week, scoring_version",
            (season, season, week, week),
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
                f"SELECT player_id, breakout_score "
                f"FROM {SCORES_TABLE} "
                f"WHERE season = %s AND as_of_week = %s AND scoring_version = %s",
                (s, w, ver),
            ).fetchall()

            grade_scores = {str(r["player_id"]): _f(r["breakout_score"]) for r in grades}
            snap_scores = {str(r["player_id"]): _f(r["breakout_score"]) for r in snaps}
            overlap = [p for p in grade_scores if p in snap_scores]
            mismatched = [
                p for p in overlap
                if grade_scores[p] is not None and snap_scores[p] is not None
                and abs(grade_scores[p] - snap_scores[p]) > 0.001
            ]

            graded_ats = [r["graded_at"] for r in grades if r["graded_at"] is not None]
            newest_grade = max(graded_ats) if graded_ats else None

            run_recon = bool(run) and _run_reconstructed(run.get("detail"))
            run_live = bool(run) and (run.get("status") in ("completed", "success")) and not run_recon
            published_at = run.get("calculated_at") if run else None

            decision = "keep"
            reason = "grades match the published snapshot"
            if not run:
                reason = "no run record; keeping (nothing to re-grade from)"
            elif run_recon:
                if snaps and mismatched:
                    decision = "delete"
                    reason = (
                        "run is flagged reconstructed but the published snapshot "
                        "scores differ from the grade scores: reconstructed grades "
                        "are shadowing a live snapshot"
                    )
                elif snaps:
                    reason = "reconstructed week with no live snapshot; keeping backtest history"
                else:
                    reason = "reconstructed week with no snapshot rows; keeping"
            elif run_live:
                if not snaps:
                    reason = "live run but no snapshot rows; keeping (nothing to re-grade from)"
                elif newest_grade is not None and published_at is not None and newest_grade < published_at:
                    if mismatched or not overlap:
                        decision = "delete"
                        reason = (
                            "grades were written before the live snapshot published "
                            "and do not match it: stale reconstructed grades shadowing live data"
                        )
                    else:
                        reason = "grades predate the live publish but scores still match; keeping"
                else:
                    reason = "grades were written after the live snapshot published; keeping"

            # Sample a few mismatches for the report (names help Kaedon eyeball it).
            names = {str(r["player_id"]): r.get("player_name") for r in grades}
            sample_mm = [
                {
                    "player": names.get(p, p),
                    "grade_score": grade_scores[p],
                    "snapshot_score": snap_scores[p],
                }
                for p in mismatched[:5]
            ]

            reports.append(
                {
                    "season": s,
                    "as_of_week": w,
                    "scoring_version": ver,
                    "grade_rows": len(grades),
                    "snapshot_rows": len(snaps),
                    "run_status": run.get("status") if run else None,
                    "run_reconstructed": run_recon,
                    "run_published_at": _iso(published_at),
                    "newest_grade_at": _iso(newest_grade),
                    "overlapping_players": len(overlap),
                    "mismatched_players": len(mismatched),
                    "mismatch_sample": sample_mm,
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
        if r["mismatch_sample"]:
            for m in r["mismatch_sample"]:
                print(
                    f"       - {m['player']}: grade score {m['grade_score']} "
                    f"vs snapshot {m['snapshot_score']}"
                )
            extra = r["mismatched_players"] - len(r["mismatch_sample"])
            if extra > 0:
                print(f"       ... and {extra} more mismatched player(s)")
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
