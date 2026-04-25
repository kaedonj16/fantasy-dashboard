#!/usr/bin/env python3
"""
Populate historical_prospect_grades with model-computed prospect scores.

Runs the real prospect model (via the backtest pipeline) against each
seeded player using nflverse roster + combine data and CFBD college stats,
then writes the scores back to the DB.

Usage:
    python scripts/populate_historical_grades.py
    python scripts/populate_historical_grades.py --years 2021 2022 2023
    python scripts/populate_historical_grades.py --years 2024 2025 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_prospect_model import (
    _load_draft_class,
    _load_combine_athleticism,
    _load_cfbd_college_stats,
    _build_prospect_dicts,
    SKILL_POS,
)
from data_building.rookie_pipeline.prospect_model import score_all_prospects
from data_building.rookie_pipeline.value_translation import assign_tier


def _slug(name: str) -> str:
    """Convert a player name to the HIST_{YEAR}_{SLUG} format."""
    import re
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _run_year(draft_year: int, dry_run: bool = False) -> int:
    """
    Score one draft class and upsert into historical_prospect_grades.
    Returns number of rows written.
    """
    print(f"\n[populate] ── {draft_year} ──────────────────────────────────────")

    draft_class = _load_draft_class(draft_year)
    if not draft_class:
        print(f"[populate] No nflverse data for {draft_year}, skipping.")
        return 0

    print(f"[populate] {len(draft_class)} skill-position draftees from nflverse")

    athleticism   = _load_combine_athleticism(draft_year)
    college_stats = _load_cfbd_college_stats(draft_year, draft_class)
    prospects, consensus_map = _build_prospect_dicts(
        draft_class, athleticism, draft_year, college_stats
    )

    print(f"[populate] Scoring {len(prospects)} prospects…")
    scores = score_all_prospects(prospects, consensus_map, skip_sagarin=True)

    # Build lookup: gsis_id / name-slug → draft_class row (for actual pick)
    dc_by_id = {
        (p["gsis_id"] or p["name"].lower().replace(" ", "-")): p
        for p in draft_class
    }

    # Load the seeded rows from DB so we can match by name and update only those
    from dashboard_services.db import get_conn
    try:
        with get_conn() as conn:
            seeded = conn.execute(
                "SELECT player_id, name FROM historical_prospect_grades WHERE draft_class_year = %s",
                (draft_year,),
            ).fetchall()
    except Exception as e:
        print(f"[populate] DB read failed: {e}")
        return 0

    seeded_by_norm: Dict[str, str] = {}
    for row in seeded:
        norm = row["name"].lower().strip().replace("'", "").replace(".", "")
        seeded_by_norm[norm] = row["player_id"]

    # Build position counters for position_rank
    pos_by_score: Dict[str, List[Dict]] = {}
    for sc in scores:
        pos = (sc.get("position") or "").upper()
        if pos in SKILL_POS:
            pos_by_score.setdefault(pos, []).append(sc)
    for pos_list in pos_by_score.values():
        pos_list.sort(key=lambda x: x.get("prospect_score", 0), reverse=True)

    pos_rank_map: Dict[str, int] = {}
    for pos_list in pos_by_score.values():
        for rank, sc in enumerate(pos_list, 1):
            pos_rank_map[sc["player_id"]] = rank

    scores_sorted = sorted(scores, key=lambda x: x.get("prospect_score", 0), reverse=True)
    overall_rank_map: Dict[str, int] = {
        sc["player_id"]: i + 1 for i, sc in enumerate(scores_sorted)
    }

    rows_written = 0
    for sc in scores:
        pid_backtest = sc["player_id"]

        # Match to a seeded row by name
        p = next((p for p in prospects if p["player_id"] == pid_backtest), {})
        name = p.get("name", "")
        norm = name.lower().strip().replace("'", "").replace(".", "")

        db_player_id = seeded_by_norm.get(norm)
        if not db_player_id:
            # Try partial match (last name)
            last = norm.split()[-1] if norm else ""
            candidates = [k for k in seeded_by_norm if last in k]
            if len(candidates) == 1:
                db_player_id = seeded_by_norm[candidates[0]]

        if not db_player_id:
            # Not in our seeded set — skip (we only score what we seeded)
            continue

        score = float(sc.get("prospect_score") or 0)
        tier_num, tier_label = assign_tier(score)

        row_data = {
            "player_id":             db_player_id,
            "prospect_score":        round(score, 2),
            "tier":                  tier_num,
            "tier_label":            tier_label,
            "overall_rank":          overall_rank_map.get(pid_backtest),
            "position_rank":         pos_rank_map.get(pid_backtest),
            "production_score":      round(float(sc.get("production_score") or 0), 2),
            "efficiency_score":      round(float(sc.get("efficiency_score") or 0), 2),
            "age_score":             round(float(sc.get("age_score") or 0), 2),
            "breakout_profile_score":round(float(sc.get("breakout_profile_score") or 0), 2),
            "athleticism_score":     round(float(sc.get("athleticism_score") or 0), 2),
            "competition_score":     round(float(sc.get("competition_score") or 0), 2),
            "draft_capital_score":   round(float(sc.get("projected_draft_capital_score") or 0), 2),
            "confidence_score":      round(float(sc.get("confidence_score") or 0), 2),
        }

        if dry_run:
            print(
                f"  [DRY] {name:<28} {sc.get('position',''):>2}  "
                f"score={score:>5.1f}  tier={tier_num}  "
                f"rank={overall_rank_map.get(pid_backtest, '?')}  "
                f"→ db_id={db_player_id}"
            )
            rows_written += 1
            continue

        try:
            with get_conn() as conn:
                conn.execute(
                    """
                    UPDATE historical_prospect_grades SET
                        prospect_score        = %(prospect_score)s,
                        tier                  = %(tier)s,
                        tier_label            = %(tier_label)s,
                        overall_rank          = %(overall_rank)s,
                        position_rank         = %(position_rank)s,
                        production_score      = %(production_score)s,
                        efficiency_score      = %(efficiency_score)s,
                        age_score             = %(age_score)s,
                        breakout_profile_score= %(breakout_profile_score)s,
                        athleticism_score     = %(athleticism_score)s,
                        competition_score     = %(competition_score)s,
                        draft_capital_score   = %(draft_capital_score)s,
                        confidence_score      = %(confidence_score)s
                    WHERE player_id = %(player_id)s
                    """,
                    row_data,
                )
                conn.commit()
            print(
                f"  ✓ {name:<28} {sc.get('position',''):>2}  "
                f"score={score:>5.1f}  tier={tier_num}  "
                f"rank={overall_rank_map.get(pid_backtest, '?')}"
            )
            rows_written += 1
        except Exception as e:
            print(f"  ✗ {name}: DB write failed: {e}")

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate historical_prospect_grades with model-computed scores."
    )
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(range(2016, 2026)),
        help="Draft years to process (default: 2016-2025)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without touching the DB",
    )
    args = parser.parse_args()

    total = 0
    for year in sorted(args.years):
        n = _run_year(year, dry_run=args.dry_run)
        total += n

    action = "would update" if args.dry_run else "updated"
    print(f"\n[populate] Done — {action} {total} rows across {len(args.years)} classes.")


if __name__ == "__main__":
    main()
