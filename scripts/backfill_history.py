#!/usr/bin/env python3
"""
Backfill historical NFL data across all three layers of the dashboard:

  1. Game logs   -> cache/sleeper_stats/sleeper_stats_s{Y}_w{W}.json  (Sleeper API)
                    + cache/sleeper_stats/redzone_stats_{Y}.json
  2. Schedules   -> cache/schedule/schedule_s{Y}_w{W}.json            (nflverse, free)
                    (provides opponent + date on each game log; no Tank01 key)
  3. Adv metrics -> player_advanced_metrics DB table                  (usage + snaps +
                    target share + air yards)
  4. NGS + FTN   -> player_advanced_metrics DB table                  (NGS separation/
                    cushion/YAC-over-exp + FTN drop_rate/contested_catch_rate;
                    redistributable, public-safe)
  5. PFF metrics -> player_advanced_metrics DB table                  (PFF grades, YPRR,
                    elusive rating, BTT%, etc. — OFF by default; --with-pff to
                    enable. License-restricted, not displayed publicly.)

It simply orchestrates the existing per-season builders so all the layers
line up for the same set of seasons.

Data-source floors (why you can't go arbitrarily far back):
  - Sleeper game logs:        ~2009+
  - Real snap counts:         2012+  (estimated from touches before then)
  - Next Gen Stats air yards: 2016+  (advanced metrics lose fidelity before this)
  - PFF summary exports:      2014+  (grades go back further, but premium stats ~2014)

So 2016 is the practical floor for full-fidelity advanced metrics.

PFF note: the PFF layer needs a PFF_COOKIE env var to download fresh CSVs from
premium.pff.com. Without it, it reads local CSVs from data/pff_nfl_{season}/ (or
data/{facet}_summary_{season}.csv) and skips seasons whose CSVs are missing.

Safe to re-run: game logs/schedules are cached (skip if present), and the
advanced-metrics + PFF saves use upsert (ON CONFLICT DO UPDATE).

Usage:
    # Default: backfill the seasons you don't have yet (2018-2021)
    python scripts/backfill_history.py

    # Explicit seasons
    python scripts/backfill_history.py 2018 2019 2020 2021

    # Inclusive range
    python scripts/backfill_history.py --from 2018 --to 2021

    # Only certain layers
    python scripts/backfill_history.py 2018 2019 --logs-only
    python scripts/backfill_history.py 2018 2019 --no-metrics

    # Include the license-restricted PFF layer (private use only)
    python scripts/backfill_history.py 2018 2019 --with-pff
"""

import logging
import argparse
import os
import sys
import traceback
from pathlib import Path

# Make sure the project root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_building.external_data.sleeper_bulk_stats import (
    fetch_week_stats,
    fetch_season_redzone_stats,
)

# Data-source fidelity floors (informational warnings only).
SNAP_COUNT_FLOOR = 2012
NGS_AIR_YARDS_FLOOR = 2016
SLEEPER_FLOOR = 2009

REGULAR_SEASON_WEEKS = range(1, 19)  # weeks 1-18


def backfill_game_logs(season: int) -> int:
    """Fetch and cache Sleeper game-log stats for all regular-season weeks."""
    print(f"  [logs] Fetching weeks 1-18 from Sleeper...")
    weeks_ok = 0
    for week in REGULAR_SEASON_WEEKS:
        try:
            data = fetch_week_stats(season, week)
            if isinstance(data, dict) and data:
                weeks_ok += 1
        except Exception as e:
            print(f"  [logs] week {week} failed: {e}")
    # Red-zone splits are season-level enrichment used by the metrics layer.
    try:
        fetch_season_redzone_stats(season)
    except Exception as e:
        print(f"  [logs] red-zone fetch failed: {e}")
    print(f"  [logs] {weeks_ok}/18 weeks cached")
    return weeks_ok


def backfill_schedules(season: int) -> None:
    """Fetch and cache schedules (opponent + date) for the season.

    Uses the free nflverse source (no Tank01 API key, complete back to 1999).
    """
    from data_building.external_data.nflverse_schedules import write_schedules_for_season
    print(f"  [schedules] Fetching weeks 1-18 from nflverse...")
    try:
        write_schedules_for_season(season)
    except Exception as e:
        print(f"  [schedules] failed: {e}")


def backfill_metrics(season: int, players_index: dict) -> int:
    """Calculate and upsert advanced metrics for the season."""
    from scripts.backfill_advanced_metrics import backfill_season
    if season < NGS_AIR_YARDS_FLOOR:
        print(f"  [metrics] NOTE: {season} < {NGS_AIR_YARDS_FLOOR}; "
              f"air-yards/NGS metrics will be blank or estimated.")
    print(f"  [metrics] Building advanced metrics...")
    try:
        return backfill_season(season, players_index)
    except Exception as e:
        print(f"  [metrics] failed: {e}")
        traceback.print_exc()
        return 0


def backfill_nflverse(season: int) -> None:
    """Upsert redistributable NGS + FTN receiving metrics for the season."""
    from scripts.sync_nflverse_metrics import main as nflverse_main
    print(f"  [nflverse] Syncing NGS + FTN metrics...")
    try:
        nflverse_main(["--season", str(season)])
    except SystemExit:
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    except Exception as e:
        print(f"  [nflverse] failed: {e}")


def backfill_pff(season: int) -> None:
    """Download (if PFF_COOKIE set) and upsert PFF advanced metrics for the season."""
    from scripts.sync_pff_advanced_metrics import main as pff_main
    print(f"  [pff] Syncing PFF advanced metrics...")
    try:
        pff_main(["--season", str(season)])
    except SystemExit:
        # argparse-style exits inside the PFF script shouldn't abort the run.
        pass
    except Exception as e:
        print(f"  [pff] failed: {e}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill historical NFL data.")
    p.add_argument("seasons", nargs="*", type=int,
                   help="Explicit season years (e.g. 2018 2019 2020 2021).")
    p.add_argument("--from", dest="from_year", type=int,
                   help="Start of inclusive season range.")
    p.add_argument("--to", dest="to_year", type=int,
                   help="End of inclusive season range.")
    p.add_argument("--logs-only", action="store_true",
                   help="Only fetch game logs (skip schedules + metrics).")
    p.add_argument("--no-logs", action="store_true", help="Skip game logs.")
    p.add_argument("--no-schedules", action="store_true", help="Skip schedules.")
    p.add_argument("--no-metrics", action="store_true", help="Skip advanced metrics.")
    p.add_argument("--no-nflverse", action="store_true",
                   help="Skip NGS + FTN (nflverse) metrics.")
    p.add_argument("--with-pff", action="store_true",
                   help="Also import PFF (premium) metrics. Off by default: PFF "
                        "data is license-restricted and not displayed publicly.")
    return p.parse_args(argv)


def resolve_seasons(args: argparse.Namespace) -> list[int]:
    if args.seasons:
        return sorted(set(args.seasons))
    if args.from_year and args.to_year:
        return list(range(args.from_year, args.to_year + 1))
    # Default: the seasons typically missing (you already have 2022-2025).
    return [2018, 2019, 2020, 2021]


def main() -> None:
    args = parse_args(sys.argv[1:])
    seasons = resolve_seasons(args)

    do_logs = not args.no_logs
    do_schedules = not args.no_schedules and not args.logs_only
    do_metrics = not args.no_metrics and not args.logs_only
    do_nflverse = not args.no_nflverse and not args.logs_only
    do_pff = args.with_pff and not args.logs_only

    print(f"Backfilling seasons: {seasons}")
    print(f"Layers: logs={do_logs} schedules={do_schedules} "
          f"metrics={do_metrics} nflverse={do_nflverse} pff={do_pff}")
    print("=" * 60)

    if do_pff and not os.getenv("PFF_COOKIE"):
        print("[pff] NOTE: PFF_COOKIE not set; will read local CSVs from data/ "
              "and skip seasons whose CSVs are missing.")

    players_index = {}
    if do_metrics:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
        if not players_index:
            print("[error] Could not load players index; metrics layer disabled.")
            do_metrics = False

    totals = {"logs": 0, "metrics": 0}
    for season in seasons:
        if season < SLEEPER_FLOOR:
            print(f"[skip] {season}: before Sleeper data floor ({SLEEPER_FLOOR}).")
            continue
        print(f"\n=== Season {season} ===")
        if do_logs:
            totals["logs"] += backfill_game_logs(season)
        if do_schedules:
            backfill_schedules(season)
        if do_metrics:
            totals["metrics"] += backfill_metrics(season, players_index)
        if do_nflverse:
            backfill_nflverse(season)
        if do_pff:
            backfill_pff(season)

    print("\n" + "=" * 60)
    print(f"Done. Game-log weeks cached: {totals['logs']}, "
          f"advanced-metric rows saved: {totals['metrics']}")
    if do_metrics:
        print("\nVerify metrics with:")
        print("  python -c \"from dashboard_services.db import get_conn; "
              "[print(dict(r)) for r in get_conn().__enter__().execute("
              "'SELECT season, COUNT(*) cnt FROM player_advanced_metrics "
              "GROUP BY season ORDER BY season DESC').fetchall()]\"")


if __name__ == "__main__":
    main()
