#!/usr/bin/env python3
"""
Download PFF CSV exports and upsert selected columns into player_advanced_metrics.

Required env var when downloading automatically:
  PFF_COOKIE  — your premium.pff.com session cookie (copy from browser DevTools)

Optional:
  PFF_AUTH_HEADER  (e.g. 'Bearer ...')

When PFF_COOKIE is set, the script downloads fresh CSVs directly from PFF's API
before syncing. Without it, the script reads from local files in data/.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
from datetime import date
from typing import Dict, Iterable, Optional, List

import requests

from dashboard_services.db import get_conn
from dashboard_services.api import get_nfl_state
from data_building.advanced_metrics import init_advanced_metrics_db, _normalize_position
from utils.utils import load_players_index, normalize_name
from scripts.fix_advanced_metrics_ids import _build_index_maps, _resolve

OUTPUT_DIR = "data"
PFF_BASE = "https://premium.pff.com"

# All regular-season weeks + playoff rounds (WC=28, Div=29, Conf=30, SB=32).
# PFF just returns whatever weeks have data, so passing the full list is safe
# mid-season.
_ALL_WEEKS = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,28,29,30,32"

RECEIVING_COLS = {
    "yards_after_catch": "yards_after_catch",
    "yards_after_catch_per_reception": "yards_after_catch_per_reception",
    "avg_depth_of_target": "avg_depth_of_target",
    "contested_catch_rate": "contested_catch_rate",
    "avoided_tackles": "avoided_tackles",
    "drop_rate": "drop_rate",
    "slot_rate": "slot_rate",
    "wide_rate": "wide_rate",
    "inline_rate": "inline_rate",
    "pass_block_rate": "pass_block_rate",
    "grades_offense": "grades_offense",
    "yprr": "yprr",
}

RUSHING_COLS = {
    "explosive": "explosive_runs_10_plus",
    "breakaway_percent": "breakaway_percentage",
    "elusive_rating": "elusive_rating",
    "grades_run": "pff_rushing_grade",
    "grades_offense": "grades_offense",
    "avoided_tackles": "avoided_tackles",
}

PASSING_COLS = {
    "grades_pass": "pff_passing_grade",
    "grades_offense": "grades_offense",
    "btt_rate": "big_time_throw_rate",
    "completion_percent": "adjusted_completion_rate",
    "pressure_to_sack_rate": "pressure_to_sack_rate",
    "qb_rating": "nfl_passer_rating",
}


def _headers() -> Dict[str, str]:
    h = {"User-Agent": "fantasy-dashboard-pff-sync/1.0"}
    if os.getenv("PFF_COOKIE"):
        h["Cookie"] = os.getenv("PFF_COOKIE", "")
    if os.getenv("PFF_AUTH_HEADER"):
        h["Authorization"] = os.getenv("PFF_AUTH_HEADER", "")
    return h


def download_pff_csv(facet: str, season: int, dest_path: str) -> bool:
    """Download a PFF summary CSV for the given facet and season.

    Returns True on success, False if the download should be skipped
    (no cookie configured, HTTP error, or empty response).
    """
    if not os.getenv("PFF_COOKIE"):
        return False

    url = (
        f"{PFF_BASE}/api/v1/facet/{facet}/summary"
        f"?league=nfl&season={season}&week={_ALL_WEEKS}&export=true"
    )
    try:
        resp = requests.get(url, headers=_headers(), timeout=60)
        if resp.status_code == 401 or resp.status_code == 403:
            print(f"    [warn] PFF returned {resp.status_code} for {facet} — "
                  f"cookie may be expired")
            return False
        resp.raise_for_status()
        content = resp.content
        if len(content) < 50:
            print(f"    [warn] PFF returned empty/tiny response for {facet}")
            return False
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as fh:
            fh.write(content)
        print(f"    Downloaded {facet} → {dest_path} ({len(content):,} bytes)")
        return True
    except requests.RequestException as exc:
        print(f"    [warn] Failed to download {facet}: {exc}")
        return False


def resolve_seasons(explicit_seasons: Optional[str], last_n: int) -> List[int]:
    if explicit_seasons:
        vals: List[int] = []
        for tok in explicit_seasons.split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(int(tok))
        return sorted(set(vals), reverse=True)

    nfl_state = get_nfl_state() or {}
    anchor = int(nfl_state.get("season") or date.today().year)
    return [anchor - i for i in range(max(1, last_n))]


def build_player_lookup() -> Dict[str, str]:
    idx = load_players_index() or {}
    lookup: Dict[str, str] = {}
    for pid, meta in idx.items():
        name = normalize_name(meta.get("name") or "")
        pos = (meta.get("pos") or "").upper()
        team = (meta.get("team") or "").upper()
        if not name:
            continue
        lookup[f"{name}|{pos}|{team}"] = str(pid)
        lookup[f"{name}|{pos}|"] = str(pid)
        lookup[f"{name}||{team}"] = str(pid)
        lookup[f"{name}||"] = str(pid)
    return lookup


def _f(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace("%", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def upsert_csv(
    csv_path: str,
    season: int,
    mapping: Dict[str, str],
    position_hint: str,
    index_maps: tuple,
) -> int:
    # Keep one stable snapshot date per season so we can store multi-year history
    # in a table keyed by (player_id, as_of_date).
    season_as_of_date = date(season + 1, 2, 15).isoformat()
    by_name, by_lastname = index_maps
    count = 0
    unmatched = 0

    with open(csv_path, "r", encoding="utf-8") as f, get_conn() as conn:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row.get("player") or "")
            pos = _normalize_position((row.get("position") or "").upper())
            team = (row.get("team_name") or "").upper()

            if not name:
                continue
            pos = pos or position_hint

            # The CSV player_id is PFF's id; resolve it to the Sleeper id used
            # everywhere else (players_index, leaderboard name lookup, the
            # computed snapshot rows). Without this the rows are orphaned: names
            # render as "Unknown" and they never combine with the computed row
            # for the same player/season. Skip rows we can't resolve rather than
            # writing an unresolvable id.
            player_id = _resolve(
                row.get("player"), row.get("position"),
                row.get("team_name"), by_name, by_lastname,
            )
            if not player_id:
                unmatched += 1
                continue

            update_data = {}
            for source_col, target_col in mapping.items():
                if source_col in row:
                    val = _f(row.get(source_col))
                    if val is not None:
                        update_data[target_col] = val

            if not update_data:
                continue

            cols = ["player_id", "as_of_date", "season", "position"] + list(update_data.keys())
            vals = [player_id, season_as_of_date, season, pos] + list(update_data.values())
            placeholders = ", ".join(["%s"] * len(cols))
            set_clause = ", ".join([f"{c}=EXCLUDED.{c}" for c in ["season", "position", *update_data.keys()]])

            conn.execute(
                f"""
                INSERT INTO player_advanced_metrics ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (player_id, as_of_date)
                DO UPDATE SET {set_clause}
                """,
                vals,
            )
            count += 1

    if unmatched:
        print(f"    [warn] {unmatched} rows in {os.path.basename(csv_path)} "
              f"could not be matched to a Sleeper id (skipped)")
    return count


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync PFF CSV metrics into player_advanced_metrics")
    parser.add_argument("--season", type=int, help="Single season to sync")
    parser.add_argument("--seasons", type=str, help="Comma-separated seasons to sync (e.g. 2025,2024,2023)")
    parser.add_argument("--last-n", type=int, default=3, help="When season(s) omitted, sync the most recent N seasons (default: 3)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_advanced_metrics_db()

    has_cookie = bool(os.getenv("PFF_COOKIE"))
    if has_cookie:
        print("PFF_COOKIE set — will download fresh CSVs from PFF")
    else:
        print("No PFF_COOKIE — reading from local files only")

    # Build the PFF-name -> Sleeper-id resolver from the same index the
    # leaderboard uses to render names, so synced rows share the Sleeper id of
    # the computed snapshot and combine under one player/season.
    idx = load_players_index() or {}
    index_maps = _build_index_maps(idx)

    seasons_arg = str(args.season) if args.season is not None else args.seasons
    seasons = resolve_seasons(seasons_arg, args.last_n)
    print(f"Syncing seasons: {', '.join(str(s) for s in seasons)}")

    total_r = total_w = total_p = 0
    for season in seasons:
        season_dir = os.path.join(OUTPUT_DIR, f"pff_nfl_{season}")

        def _resolve_file(facet: str) -> Optional[str]:
            # If we have a cookie, download fresh and use that path.
            download_path = os.path.join(season_dir, f"{facet}_summary.csv")
            if has_cookie:
                if download_pff_csv(facet, season, download_path):
                    return download_path
            # Fall back to any existing local file.
            candidates = [
                download_path,
                os.path.join(OUTPUT_DIR, f"{facet}_summary_{season}.csv"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return None

        rushing_csv  = _resolve_file("rushing")
        receiving_csv = _resolve_file("receiving")
        passing_csv  = _resolve_file("passing")

        if rushing_csv:
            total_r += upsert_csv(rushing_csv, season, RUSHING_COLS, "RB", index_maps)
        if receiving_csv:
            total_w += upsert_csv(receiving_csv, season, RECEIVING_COLS, "WR", index_maps)
        if passing_csv:
            total_p += upsert_csv(passing_csv, season, PASSING_COLS, "QB", index_maps)

    print(f"Synced PFF metrics rows: rushing={total_r}, receiving={total_w}, passing={total_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
