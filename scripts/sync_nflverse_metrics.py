#!/usr/bin/env python3
"""
Sync redistributable NGS + FTN receiving metrics into player_advanced_metrics.

Pulls Next Gen Stats receiving tracking metrics and FTN-charting-derived
drop_rate / contested_catch_rate (see data_building/external_data/nflverse_metrics.py)
and upserts them as a per-season snapshot row. The season reader coalesces all
rows for a season, so these merge with the computed and PFF snapshots.

Unlike the PFF sync, this needs no cookie/CSV — the data comes from the open
nflverse releases via nfl_data_py.

Usage:
    python -m scripts.sync_nflverse_metrics                 # most recent 3 seasons
    python -m scripts.sync_nflverse_metrics --season 2024
    python -m scripts.sync_nflverse_metrics --seasons 2022,2023,2024,2025
    python -m scripts.sync_nflverse_metrics --last-n 8
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Iterable, List, Optional

from dashboard_services.api import get_nfl_state
from dashboard_services.db import get_conn
from data_building.advanced_metrics import init_advanced_metrics_db, _normalize_position
from data_building.external_data.nflverse_metrics import build_nflverse_metrics_for_season
from utils.utils import load_players_index


def resolve_seasons(explicit: Optional[str], last_n: int) -> List[int]:
    if explicit:
        seasons = []
        for tok in explicit.split(","):
            tok = tok.strip()
            if tok:
                seasons.append(int(tok))
        return sorted(set(seasons))
    nfl_state = get_nfl_state() or {}
    anchor = int(nfl_state.get("season") or date.today().year)
    return list(range(anchor - last_n + 1, anchor + 1))


def upsert_season(season: int, players_index: dict) -> int:
    """Build and upsert NGS + FTN metrics for one season. Returns rows written."""
    by_pid = build_nflverse_metrics_for_season(season)
    if not by_pid:
        print(f"  No nflverse metrics resolved for {season}")
        return 0

    # Distinct snapshot date per season; later than the computed (01-10) and PFF
    # (02-15) snapshots so the coalescing reader prefers these public-safe values.
    as_of_date = date(season + 1, 3, 1).isoformat()
    count = 0

    with get_conn() as conn:
        for pid, cols in by_pid.items():
            if not cols:
                continue
            meta = players_index.get(pid) or players_index.get(str(pid)) or {}
            pos = _normalize_position((meta.get("pos") or meta.get("position") or "").upper()) or None

            keys = list(cols.keys())
            db_cols = ["player_id", "as_of_date", "season", "position"] + keys
            vals = [pid, as_of_date, season, pos] + [cols[k] for k in keys]
            placeholders = ", ".join(["%s"] * len(db_cols))
            set_clause = ", ".join(
                f"{c}=EXCLUDED.{c}" for c in ["season", "position", *keys]
            )
            conn.execute(
                f"""
                INSERT INTO player_advanced_metrics ({', '.join(db_cols)})
                VALUES ({placeholders})
                ON CONFLICT (player_id, as_of_date)
                DO UPDATE SET {set_clause}
                """,
                vals,
            )
            count += 1

    print(f"  Upserted {count} players for {season} (date={as_of_date})")
    return count


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync NGS + FTN receiving metrics into player_advanced_metrics")
    parser.add_argument("--season", type=int, help="Single season to sync")
    parser.add_argument("--seasons", type=str, help="Comma-separated seasons")
    parser.add_argument("--last-n", type=int, default=3,
                        help="When season(s) omitted, sync the most recent N (default 3)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    init_advanced_metrics_db()
    players_index = load_players_index() or {}

    seasons_arg = str(args.season) if args.season is not None else args.seasons
    seasons = resolve_seasons(seasons_arg, args.last_n)
    print(f"Syncing nflverse (NGS + FTN) metrics for: {seasons}")

    total = 0
    for season in seasons:
        print(f"=== Season {season} ===")
        try:
            total += upsert_season(season, players_index)
        except Exception as e:
            import traceback
            print(f"  [error] {season} failed: {e}")
            traceback.print_exc()

    print(f"Done. Total player rows upserted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
