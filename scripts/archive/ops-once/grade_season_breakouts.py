#!/usr/bin/env python3
"""
Grade season-breakout preseason calls against realized outcomes.

Usage:
    python scripts/grade_season_breakouts.py [--season 2026] [--stage both]

For each season, the FINAL pre-season snapshot per player in
breakout_opportunity_scores is taken as the call, and graded against
player_weekly_metrics in two immutable stages: early (weeks 1-8, once week
8 exists, provisional) and final (full season, once the stored week
reaches 17). The hit definition is the season engine's own backtest label
(backtest_multitask.get_breakout_pids); see
data_building/breakout_engine/season_grading.py for the full rule set.
Grades are written idempotently - re-running is a no-op for calls that
already have a grade at a stage.

Unlike the weekly grader's CLI, running with no --season grades EVERY
season that has stored calls (2022 onward): the point of this script is
the one-pass retroactive backfill. The breakout runner also invokes the
grader for the current season on every weekly run.
"""
import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from data_building.breakout_engine.season_grading import (
    MIN_SUMMARY_SAMPLE,
    STAGE_EARLY,
    STAGE_FINAL,
    grade_season_breakouts,
    load_call_seasons,
    summarize_grades,
)


def _print_bucket(label, bucket):
    def rate(value):
        return "n/a" if value is None else f"{value:.1%}"
    print(
        f"  {label:<28} calls={bucket['calls']:>4} graded={bucket['graded']:>4} "
        f"hit={bucket['hit']:>3} partial={bucket['partial']:>3} "
        f"miss={bucket['miss']:>3} ungraded={bucket['ungraded']:>3} "
        f"hit_rate={rate(bucket['hit_rate'])} "
        f"partial_rate={rate(bucket['partial_rate'])} "
        f"miss_rate={rate(bucket['miss_rate'])}"
    )


def _print_summary(title, summary):
    print(title)
    _print_bucket("overall", summary["overall"])
    for label, group_title in (
        ("by_phase", "by phase:"),
        ("by_season", "by season:"),
        ("by_score_band", "by score band:"),
    ):
        if summary[label]:
            print(f"  {group_title}")
            for key, bucket in summary[label].items():
                _print_bucket(f"  {key}", bucket)


def main():
    parser = argparse.ArgumentParser(
        description="Grade season breakout preseason calls"
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Season to grade (default: every season with stored calls)",
    )
    parser.add_argument(
        "--stage", choices=["early", "final", "both"], default="both",
        help="Which grading stage(s) to write (default: both)",
    )
    parser.add_argument(
        "--min-sample", type=int, default=MIN_SUMMARY_SAMPLE,
        help="Minimum graded calls before a group's rates are reported",
    )
    args = parser.parse_args()

    stages = (
        (STAGE_EARLY, STAGE_FINAL) if args.stage == "both" else (args.stage,)
    )
    seasons = (
        [args.season] if args.season is not None else load_call_seasons()
    )
    if not seasons:
        print("No seasons with stored breakout calls found.")
        return

    for season in seasons:
        result = grade_season_breakouts(season, stages=stages)
        print(f"grade_season_breakouts result: {json.dumps(result, default=str)}")
        for stage in stages:
            _print_summary(
                f"Season-breakout grades for {season} (stage: {stage}, "
                f"rates need >= {args.min_sample} graded calls):",
                summarize_grades(
                    season=season, stage=stage, min_sample=args.min_sample
                ),
            )
    if args.season is None:
        for stage in stages:
            _print_summary(
                f"Season-breakout grades, all seasons (stage: {stage}, "
                f"rates need >= {args.min_sample} graded calls):",
                summarize_grades(stage=stage, min_sample=args.min_sample),
            )


if __name__ == "__main__":
    main()
