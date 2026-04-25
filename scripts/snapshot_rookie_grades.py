#!/usr/bin/env python3
"""
Snapshot current rookie rankings into historical_prospect_grades.

Reads live scores from rookie_rankings / rookie_prospects and upserts
them into historical_prospect_grades so you can track grade history
over time and use them as comparables for future classes.

Usage:
    python scripts/snapshot_rookie_grades.py
    python scripts/snapshot_rookie_grades.py --year 2026
    python scripts/snapshot_rookie_grades.py --year 2026 --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _slug(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _norm(name: str) -> str:
    n = name.lower()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot rookie rankings into historical_prospect_grades."
    )
    parser.add_argument("--year", type=int, default=None,
                        help="Draft year to snapshot (default: active rookie class)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching the DB")
    args = parser.parse_args()

    from data_building.rookie_pipeline.pipeline import (
        get_active_rookie_class,
        get_rookie_rankings_from_db,
    )
    from dashboard_services.db import get_conn

    year = args.year or get_active_rookie_class()
    rows = get_rookie_rankings_from_db(year)

    if not rows:
        print(f"[snapshot] No rankings found for {year}. Run populate_rookie_data.py first.")
        return

    print(f"[snapshot] {len(rows)} ranked prospects for {year} draft class")

    # Load existing historical rows for this year to detect existing player_ids
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT player_id, name FROM historical_prospect_grades WHERE draft_class_year = %s",
            (year,),
        ).fetchall()

    existing_by_norm = {_norm(r["name"]): r["player_id"] for r in existing}
    existing_ids = {r["player_id"] for r in existing}

    written = skipped = 0
    for r in rows:
        name = r.get("name") or ""
        position = (r.get("position") or "").upper()
        if not name or not position:
            continue

        # Determine player_id: reuse existing HIST id if name matches, else create new
        norm_name = _norm(name)
        hist_id = existing_by_norm.get(norm_name) or f"HIST_{year}_{_slug(name)}"

        prospect_score = float(r.get("prospect_score") or 0)
        tier = r.get("tier")
        tier_label = r.get("tier_label")
        overall_rank = r.get("overall_rank")
        position_rank = r.get("position_rank")
        production_score = float(r.get("production_score") or 0)
        efficiency_score = float(r.get("efficiency_score") or 0)
        age_score = float(r.get("age_score") or 0)
        breakout_profile_score = float(r.get("breakout_profile_score") or 0)
        athleticism_score = float(r.get("athleticism_score") or 0)
        competition_score = float(r.get("competition_score") or 0)
        draft_capital_score = float(r.get("projected_draft_capital_score") or 0)
        confidence_score = float(r.get("confidence_score") or 0)
        school = r.get("school")
        headshot_url = r.get("headshot_url")
        actual_pick = r.get("actual_pick")
        actual_round = r.get("actual_round")
        actual_nfl_team = r.get("actual_nfl_team")

        if args.dry_run:
            action = "UPDATE" if hist_id in existing_ids else "INSERT"
            print(
                f"  [{action}] {name:<28} {position:>2}  "
                f"score={prospect_score:>5.1f}  tier={tier}  "
                f"rank={overall_rank}  → {hist_id}"
            )
            written += 1
            continue

        try:
            with get_conn() as conn:
                if hist_id in existing_ids:
                    conn.execute(
                        """
                        UPDATE historical_prospect_grades SET
                            prospect_score          = %(prospect_score)s,
                            tier                    = %(tier)s,
                            tier_label              = %(tier_label)s,
                            overall_rank            = %(overall_rank)s,
                            position_rank           = %(position_rank)s,
                            production_score        = %(production_score)s,
                            efficiency_score        = %(efficiency_score)s,
                            age_score               = %(age_score)s,
                            breakout_profile_score  = %(breakout_profile_score)s,
                            athleticism_score       = %(athleticism_score)s,
                            competition_score       = %(competition_score)s,
                            draft_capital_score     = %(draft_capital_score)s,
                            confidence_score        = %(confidence_score)s,
                            actual_pick             = %(actual_pick)s,
                            actual_round            = %(actual_round)s,
                            actual_nfl_team         = %(actual_nfl_team)s,
                            headshot_url            = %(headshot_url)s
                        WHERE player_id = %(player_id)s
                        """,
                        {
                            "player_id": hist_id,
                            "prospect_score": round(prospect_score, 2),
                            "tier": tier,
                            "tier_label": tier_label,
                            "overall_rank": overall_rank,
                            "position_rank": position_rank,
                            "production_score": round(production_score, 2),
                            "efficiency_score": round(efficiency_score, 2),
                            "age_score": round(age_score, 2),
                            "breakout_profile_score": round(breakout_profile_score, 2),
                            "athleticism_score": round(athleticism_score, 2),
                            "competition_score": round(competition_score, 2),
                            "draft_capital_score": round(draft_capital_score, 2),
                            "confidence_score": round(confidence_score, 2),
                            "actual_pick": actual_pick,
                            "actual_round": actual_round,
                            "actual_nfl_team": actual_nfl_team,
                            "headshot_url": headshot_url,
                        },
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO historical_prospect_grades (
                            player_id, name, position, draft_class_year, school,
                            prospect_score, tier, tier_label, overall_rank, position_rank,
                            production_score, efficiency_score, age_score,
                            breakout_profile_score, athleticism_score,
                            competition_score, draft_capital_score, confidence_score,
                            actual_pick, actual_round, actual_nfl_team, headshot_url
                        ) VALUES (
                            %(player_id)s, %(name)s, %(position)s, %(draft_class_year)s, %(school)s,
                            %(prospect_score)s, %(tier)s, %(tier_label)s, %(overall_rank)s, %(position_rank)s,
                            %(production_score)s, %(efficiency_score)s, %(age_score)s,
                            %(breakout_profile_score)s, %(athleticism_score)s,
                            %(competition_score)s, %(draft_capital_score)s, %(confidence_score)s,
                            %(actual_pick)s, %(actual_round)s, %(actual_nfl_team)s, %(headshot_url)s
                        )
                        ON CONFLICT (player_id) DO UPDATE SET
                            prospect_score          = EXCLUDED.prospect_score,
                            tier                    = EXCLUDED.tier,
                            tier_label              = EXCLUDED.tier_label,
                            overall_rank            = EXCLUDED.overall_rank,
                            position_rank           = EXCLUDED.position_rank,
                            production_score        = EXCLUDED.production_score,
                            efficiency_score        = EXCLUDED.efficiency_score,
                            age_score               = EXCLUDED.age_score,
                            breakout_profile_score  = EXCLUDED.breakout_profile_score,
                            athleticism_score       = EXCLUDED.athleticism_score,
                            competition_score       = EXCLUDED.competition_score,
                            draft_capital_score     = EXCLUDED.draft_capital_score,
                            confidence_score        = EXCLUDED.confidence_score,
                            actual_pick             = EXCLUDED.actual_pick,
                            actual_round            = EXCLUDED.actual_round,
                            actual_nfl_team         = EXCLUDED.actual_nfl_team,
                            headshot_url            = EXCLUDED.headshot_url
                        """,
                        {
                            "player_id": hist_id,
                            "name": name,
                            "position": position,
                            "draft_class_year": year,
                            "school": school,
                            "prospect_score": round(prospect_score, 2),
                            "tier": tier,
                            "tier_label": tier_label,
                            "overall_rank": overall_rank,
                            "position_rank": position_rank,
                            "production_score": round(production_score, 2),
                            "efficiency_score": round(efficiency_score, 2),
                            "age_score": round(age_score, 2),
                            "breakout_profile_score": round(breakout_profile_score, 2),
                            "athleticism_score": round(athleticism_score, 2),
                            "competition_score": round(competition_score, 2),
                            "draft_capital_score": round(draft_capital_score, 2),
                            "confidence_score": round(confidence_score, 2),
                            "actual_pick": actual_pick,
                            "actual_round": actual_round,
                            "actual_nfl_team": actual_nfl_team,
                            "headshot_url": headshot_url,
                        },
                    )
                conn.commit()
            written += 1
        except Exception as e:
            print(f"  ✗ {name}: DB write failed: {e}")
            skipped += 1

    action = "would write" if args.dry_run else "wrote"
    print(f"\n[snapshot] Done — {action} {written} rows for {year} draft class" +
          (f" ({skipped} failed)" if skipped else "") + ".")


if __name__ == "__main__":
    main()
