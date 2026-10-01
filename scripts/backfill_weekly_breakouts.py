#!/usr/bin/env python3
"""
Re-score past weekly breakout boards under the current SCORING_VERSION
using as-of inputs reconstructed from nflverse (see
data_building.breakout_engine.reconstruction).

For every completed week of the season that lacks an original
current-version run, the week's calls are re-scored with the rosters,
depth charts and injury reports as they stood on the original run's
date, and published flagged as reconstructions. The original snapshots
keep serving the week selector; live track-record rates exclude the
reconstructed calls, which the breakout sidebar reports as a separate
labeled backtest line.

This is step one of two. This script only scores and publishes;
grading stays with the existing path:

    python -m scripts.grade_weekly_breakouts --season 2026

(or the next regular pipeline run, which grades automatically).
Reconstructed week 1 calls are mature immediately; later weeks grade
as their 3-week outcome windows complete.

Usage:
    python -m scripts.backfill_weekly_breakouts
    python -m scripts.backfill_weekly_breakouts --season 2026
    python -m scripts.backfill_weekly_breakouts --season 2026 --weeks 1,2,3
"""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="Season to backfill (default: current NFL season)")
    parser.add_argument("--weeks", type=str, default=None,
                        help="Comma-separated weeks to reconstruct "
                             "(default: every completed week lacking an "
                             "original current-version run)")
    args = parser.parse_args()

    season = args.season or _default_season()
    weeks = None
    if args.weeks:
        weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    from data_building.breakout_engine import reconstruction

    print(f"[backfill] reconstructing weekly breakouts for {season}"
          + (f" weeks {weeks}" if weeks else " (all eligible weeks)"))
    results = reconstruction.reconstruct_season(season, weeks)
    if not results:
        print("[backfill] no completed weeks found; nothing to do")
        return 0
    for result in results:
        line = (f"  week {result.get('as_of_week')}: "
                f"status={result.get('status')}")
        if result.get("records_saved") is not None:
            line += f" records_saved={result.get('records_saved')}"
        if result.get("reason"):
            line += f" ({result['reason']})"
        print(line)
    print("[backfill] done. Grade the reconstructed calls with: "
          f"python -m scripts.grade_weekly_breakouts --season {season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
