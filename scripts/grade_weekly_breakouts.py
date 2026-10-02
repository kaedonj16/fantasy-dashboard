#!/usr/bin/env python3
"""
Grade matured weekly breakout calls against realized outcomes.

For each season, grades every persisted weekly_breakout_scores call whose
3 outcome weeks exist in player_weekly_metrics, writing one immutable row
per call to weekly_breakout_grades (idempotent: re-running grades only
newly matured calls), then prints hit / partial / miss rates grouped by
classification and scoring version for threshold review.

Each run also snapshots the live Sleeper injury designations for the
current NFL week (player_injury_snapshots): a call whose outcome window
was wiped out by injury (< 2 games played with IR/PUP/NFI or OUT/
DOUBTFUL designations) grades "injured" - terminal, and excluded from
the hit-rate denominators.

Usage:
    python scripts/grade_weekly_breakouts.py
    python scripts/grade_weekly_breakouts.py --season 2026
    python scripts/grade_weekly_breakouts.py --season 2026 --as-of-week 6
    python scripts/grade_weekly_breakouts.py --season 2026 --min-sample 5

Also runs automatically at the end of each in-season weekly scoring run
(see calculate_breakouts_with_real_data.main).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def _default_season() -> int:
    try:
        from data_building.sleeper_data import get_nfl_state
        state = get_nfl_state() or {}
        season = state.get("season")
        if season:
            return int(season)
    except Exception:
        pass
    from datetime import date
    return date.today().year


def _print_bucket(label: str, bucket: dict) -> None:
    def _fmt(rate) -> str:
        return "n/a" if rate is None else f"{rate:.1%}"

    print(
        f"  {label:<28} calls={bucket['calls']:<5} graded={bucket['graded']:<5} "
        f"hit={bucket['hit']:<4} partial={bucket['partial']:<4} "
        f"miss={bucket['miss']:<4} ungraded={bucket['ungraded']:<4} "
        f"injured={bucket.get('injured', 0):<4} "
        f"hit_rate={_fmt(bucket['hit_rate'])} "
        f"partial_rate={_fmt(bucket['partial_rate'])} "
        f"miss_rate={_fmt(bucket['miss_rate'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="Season to grade (default: current NFL season)")
    parser.add_argument("--as-of-week", type=int, default=None,
                        help="Grade through this week (default: latest week "
                             "present in player_weekly_metrics)")
    parser.add_argument("--min-sample", type=int, default=None,
                        help="Minimum graded calls before a group's rates "
                             "are reported (default: engine minimum)")
    args = parser.parse_args()

    from data_building.breakout_engine.weekly_grading import (
        MIN_SUMMARY_SAMPLE,
        grade_weekly_breakouts,
        summarize_grades,
    )

    season = args.season if args.season is not None else _default_season()
    min_sample = (args.min_sample if args.min_sample is not None
                  else MIN_SUMMARY_SAMPLE)

    print(f"Grading weekly breakout calls for season {season}"
          + (f" through week {args.as_of_week}" if args.as_of_week else "")
          + "...")
    result = grade_weekly_breakouts(season, through_week=args.as_of_week)
    print(json.dumps(result, indent=2, default=str))

    summary = summarize_grades(season, min_sample=min_sample)
    print(f"\nSeason {season} grade summary "
          f"(rates need >= {summary['min_sample']} graded calls):")
    _print_bucket("overall", summary["overall"])
    for label, groups in (("by classification", summary["by_classification"]),
                          ("by scoring version", summary["by_scoring_version"])):
        print(f"{label}:")
        for name, bucket in groups.items():
            _print_bucket(name, bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
