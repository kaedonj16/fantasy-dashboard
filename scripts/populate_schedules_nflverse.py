#!/usr/bin/env python3
"""
Populate NFL schedules from the free nflverse feed (no Tank01 API key needed).

A drop-in alternative to scripts/populate_schedules.py that works for any season
back to 1999 with complete coverage. Writes the same cache files the game-logs
reader consumes (cache/schedule/schedule_s{Y}_w{W}.json).

Usage:
    python -m scripts.populate_schedules_nflverse 2016 2017 2018 2019 2020 2021
    python -m scripts.populate_schedules_nflverse --from 2016 --to 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_building.external_data.nflverse_schedules import write_schedules_for_season


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Populate NFL schedules from nflverse.")
    p.add_argument("seasons", nargs="*", type=int, help="Season years to fetch.")
    p.add_argument("--from", dest="from_year", type=int, help="Start of inclusive range.")
    p.add_argument("--to", dest="to_year", type=int, help="End of inclusive range.")
    args = p.parse_args(argv)

    if args.seasons:
        seasons = sorted(set(args.seasons))
    elif args.from_year and args.to_year:
        seasons = list(range(args.from_year, args.to_year + 1))
    else:
        p.error("provide season years or --from/--to")
        return 1

    total = 0
    for season in seasons:
        total += write_schedules_for_season(season)
    print(f"Done. Total weeks written: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
