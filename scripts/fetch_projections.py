#!/usr/bin/env python3
"""
Manually fetch + cache weekly projections from Sleeper (the same work the daily
cron does in cron_daily.py step 2).

Examples:
    python scripts/fetch_projections.py                 # auto: current + next week
    python scripts/fetch_projections.py --all            # all 18 weeks (offseason refresh)
    python scripts/fetch_projections.py --season 2025 --weeks 10 11
    python scripts/fetch_projections.py --weeks 12       # current season, just week 12

Writes cache/projections/projections_s{season}_w{week}.json.
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from dashboard_services.api import get_nfl_state
from utils.utils import fetch_week_projections, save_week_projections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="season year (default: current NFL state)")
    ap.add_argument("--weeks", type=int, nargs="+", help="specific week(s) to fetch")
    ap.add_argument("--all", action="store_true", help="fetch all 18 weeks")
    args = ap.parse_args()

    state = get_nfl_state() or {}
    season = args.season or int(state.get("season"))

    if args.all:
        weeks = list(range(1, 19))
    elif args.weeks:
        weeks = sorted(set(args.weeks))
    else:
        # Match the in-season cron: current + next week.
        week = int(state.get("week") or 1)
        weeks = sorted({week, min(week + 1, 18)})

    print(f"Fetching projections for season {season}, weeks {weeks}")
    total = 0
    for w in weeks:
        data = fetch_week_projections(season, w)
        save_week_projections(season, w, data)
        n = len(data or {})
        total += n
        print(f"  week {w}: {n} players")
    print(f"Done — {total} player-week projections cached.")


if __name__ == "__main__":
    main()
