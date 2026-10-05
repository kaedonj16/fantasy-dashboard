#!/usr/bin/env python3
"""
Verify a backfill produced what's expected, per season.

Checks four things and prints a table:
  1. Schedules  — weeks + total games cached (a full reg season is ~256 games)
  2. Game logs  — weeks of sleeper_stats cached (full season = 18)
  3. Metrics    — players with NGS / FTN drop_rate / EPA populated in the DB
  4. PFF purge  — PFF snapshot rows that still hold shared values (should be 0)

The file checks (schedules, game logs) always run. The DB checks run only if a
database is reachable (DATABASE_URL set); otherwise they're skipped.

Usage:
    python -m scripts.verify_backfill
    python -m scripts.verify_backfill 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
"""

import logging
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def schedule_stats(season: int):
    files = glob.glob(os.path.join("cache", "schedule", f"schedule_s{season}_w*.json"))
    weeks, games = 0, 0
    for f in files:
        try:
            data = json.load(open(f))
            if isinstance(data, list) and data:
                weeks += 1
                games += len(data)
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    return weeks, games


def gamelog_weeks(season: int) -> int:
    files = glob.glob(os.path.join("cache", "sleeper_stats", f"sleeper_stats_s{season}_w*.json"))
    return sum(1 for f in files if "redzone" not in os.path.basename(f))


def db_metric_counts(seasons):
    """Return {season: {...}} of DB coverage, or None if DB unreachable."""
    try:
        from dashboard_services.db import get_conn
    except Exception as e:
        print(f"[db] import failed ({e}); skipping DB checks")
        return None

    out = {}
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT season,
                    COUNT(*) FILTER (WHERE ngs_avg_separation IS NOT NULL) AS ngs,
                    COUNT(*) FILTER (WHERE drop_rate IS NOT NULL)          AS drop_rate,
                    COUNT(*) FILTER (WHERE contested_catch_rate IS NOT NULL) AS contested,
                    COUNT(*) FILTER (WHERE epa_per_play IS NOT NULL)       AS qb_epa,
                    COUNT(*) FILTER (WHERE receiving_epa IS NOT NULL)      AS rec_epa
                FROM player_advanced_metrics
                WHERE season = ANY(%s)
                GROUP BY season
                """,
                (list(seasons),),
            ).fetchall()
            for r in rows:
                out[int(r["season"])] = dict(r)

            # PFF purge check: shared values still present on the PFF (Feb-15) row.
            purge = conn.execute(
                """
                SELECT season,
                    COUNT(*) FILTER (WHERE drop_rate IS NOT NULL
                                     OR contested_catch_rate IS NOT NULL
                                     OR breakaway_percentage IS NOT NULL
                                     OR explosive_runs_10_plus IS NOT NULL
                                     OR avg_depth_of_target IS NOT NULL) AS pff_shared_left
                FROM player_advanced_metrics
                WHERE season = ANY(%s) AND as_of_date = make_date(season + 1, 2, 15)
                GROUP BY season
                """,
                (list(seasons),),
            ).fetchall()
            for r in purge:
                out.setdefault(int(r["season"]), {})["pff_shared_left"] = int(r["pff_shared_left"])
    except Exception as e:
        print(f"[db] query failed ({e}); skipping DB checks")
        return None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify backfill coverage per season.")
    p.add_argument("seasons", nargs="*", type=int,
                   help="Seasons to check (default 2016-2025).")
    args = p.parse_args(argv)
    seasons = sorted(set(args.seasons)) or list(range(2016, 2026))

    db = db_metric_counts(seasons)

    print(f"{'Season':>6} | {'Sched wk/games':>15} | {'Log wks':>7} | "
          f"{'NGS':>5} {'Drop':>5} {'Cont':>5} {'QBepa':>6} {'Recepa':>6} | {'PFF left':>8}")
    print("-" * 92)
    for s in seasons:
        wk, games = schedule_stats(s)
        logs = gamelog_weeks(s)
        d = (db or {}).get(s, {}) if db is not None else {}
        ngs = d.get("ngs", "-"); drop = d.get("drop_rate", "-")
        cont = d.get("contested", "-"); qep = d.get("qb_epa", "-")
        rep = d.get("rec_epa", "-")
        pff = d.get("pff_shared_left", "-") if db is not None else "skip"
        print(f"{s:>6} | {f'{wk}/{games}':>15} | {logs:>7} | "
              f"{str(ngs):>5} {str(drop):>5} {str(cont):>5} {str(qep):>6} {str(rep):>6} | {str(pff):>8}")

    print("\nExpected: schedules 17-18 wks / 256 games (16-game era) or 272 (2021+);")
    print("log wks=18 (completed seasons);")
    print("NGS only 2016+, Drop/Cont (FTN) only 2022+, EPA all seasons; 'PFF left' should be 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
