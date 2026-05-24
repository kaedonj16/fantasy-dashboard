#!/usr/bin/env python3
"""
Backfill historical breakout scores for past seasons.

Runs build_historical_scores for each specified stats season so the
breakout_opportunity_scores table contains predictions for 2023, 2024,
and 2025 — required for the peer comparison comp database.

Usage:
    python scripts/backfill_breakout_history.py
    python scripts/backfill_breakout_history.py --seasons 2022 2023 2024
    python scripts/backfill_breakout_history.py --seasons 2023 --min-score 50
    python scripts/backfill_breakout_history.py --dry-run
"""

import argparse
import sys
import os
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# Stats season → as_of_date used for roster change detection.
# Set to early May of the *prediction* year (stats_season + 1) so the
# engine sees that offseason's departures/arrivals rather than today's.
_AS_OF_DATES: dict[int, date] = {
    2022: date(2023, 5, 1),
    2023: date(2024, 5, 1),
    2024: date(2025, 5, 1),
    2025: date(2026, 5, 1),
}

# Default stats seasons to backfill (covers prediction seasons 2023-2026)
_DEFAULT_SEASONS = [2022, 2023, 2024, 2025]


def run_season(stats_season: int, min_score: float, dry_run: bool) -> bool:
    prediction_season = stats_season + 1
    as_of = _AS_OF_DATES.get(stats_season, date(prediction_season, 5, 1))

    print(f"\n{'='*60}")
    print(f"Stats season : {stats_season}")
    print(f"Predicts     : {prediction_season}")
    print(f"As-of date   : {as_of}")
    print(f"Min score    : {min_score}")
    if dry_run:
        print("[DRY RUN] Skipping actual scoring.")
        return True
    print(f"{'='*60}")

    try:
        from data_building.breakout_engine.build_historical_scores import run
        result = run(
            seasons=[stats_season],
            min_score=min_score,
            as_of_date_override=as_of,
        )
        saved = result.get("saved", 0) if isinstance(result, dict) else "?"
        print(f"✓ Season {stats_season} complete — {saved} records saved")
        return True
    except Exception as exc:
        print(f"✗ Season {stats_season} failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical breakout scores")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=_DEFAULT_SEASONS,
        help=f"Stats seasons to process (default: {_DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=45.0,
        help="Minimum breakout score to save (default: 45.0 — wider than live 55 for richer comp pool)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing",
    )
    args = parser.parse_args()

    print(f"Breakout history backfill — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Seasons to process: {args.seasons}")

    passed, failed = [], []
    for s in sorted(args.seasons):
        ok = run_season(s, args.min_score, args.dry_run)
        (passed if ok else failed).append(s)

    print(f"\n{'='*60}")
    print(f"Done.  Passed: {passed}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
