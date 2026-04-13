#!/usr/bin/env python3
"""
Download PFF CSV exports and upsert selected columns into player_advanced_metrics.

Expected env vars:
  PFF_RUSHING_CSV_URL
  PFF_RECEIVING_CSV_URL
  PFF_PASSING_CSV_URL
Optional auth:
  PFF_COOKIE
  PFF_AUTH_HEADER  (e.g. 'Bearer ...')

URL env vars may include a `{season}` token if your export links are season-scoped,
for example:
  PFF_RUSHING_CSV_URL="https://.../rushing?season={season}&export=csv"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import date
from typing import Dict, Iterable, Optional, List

import requests

from dashboard_services.db import get_conn
from dashboard_services.api import get_nfl_state
from data_building.advanced_metrics import init_advanced_metrics_db
from utils.utils import load_players_index, normalize_name

OUTPUT_DIR = "data"
PFF_BASE = "https://premium.pff.com"

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
    "grades_pass_block": "grades_pass_block",
}

RUSHING_COLS = {
    "explosive": "explosive_runs_10_plus",
    "explosive_runs_10_plus": "explosive_runs_10_plus",
    "breakaway_percent": "breakaway_percentage",
    "breakaway_percentage": "breakaway_percentage",
    "elusive_rating": "elusive_rating",
    "grades_run": "pff_rushing_grade",
    "pff_rushing_grade": "pff_rushing_grade",
    "grades_offense": "grades_offense",
}

PASSING_COLS = {
    "grades_pass": "pff_passing_grade",
    "pff_passing_grade": "pff_passing_grade",
    "grades_offense": "grades_offense",
    "big_time_throws": "big_time_throw_rate",
    "big_time_throw_rate": "big_time_throw_rate",
    "completion_percent": "adjusted_completion_rate",
    "adjusted_completion_rate": "adjusted_completion_rate",
    "pressure_to_sack_rate": "pressure_to_sack_rate",
    "qb_rating": "nfl_passer_rating",
    "nfl_passer_rating": "nfl_passer_rating",
}


def _headers() -> Dict[str, str]:
    h = {"User-Agent": "fantasy-dashboard-pff-sync/1.0"}
    if os.getenv("PFF_COOKIE"):
        h["Cookie"] = os.getenv("PFF_COOKIE", "")
    if os.getenv("PFF_AUTH_HEADER"):
        h["Authorization"] = os.getenv("PFF_AUTH_HEADER", "")
    return h


def download_csv(url: str, out_path: str) -> str:
    if not url:
        raise ValueError("Missing CSV URL env var")

    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(resp.text)

    return out_path


def _default_page_url(kind: str, season: int) -> str:
    if kind == "receiving":
        return f"{PFF_BASE}/nfl/positions/{season}/REGPO/receiving?position=WR,TE,RB&minimum=20p"
    if kind == "rushing":
        return f"{PFF_BASE}/nfl/positions/{season}/REGPO/rushing?position=RB&minimum=20p"
    if kind == "passing":
        return f"{PFF_BASE}/nfl/positions/{season}/REGPO/passing?position=QB&minimum=20p"
    raise ValueError(f"Unknown PFF kind: {kind}")


def _seasonize_page_url(url: str, season: int) -> str:
    if "{season}" in url:
        return url.format(season=season)
    # Handle links like /positions/2016/REGPO/...
    return re.sub(r"/positions/\d{4}/", f"/positions/{season}/", url)


def _ensure_csv_export(url: str) -> str:
    if "export=csv" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}export=csv"


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


def _player_key(row: Dict[str, str]) -> tuple[str, str, str]:
    name = normalize_name(row.get("player") or row.get("name") or "")
    pos = (row.get("position") or row.get("pos") or "").upper()
    team = (row.get("team") or row.get("team_name") or "").upper()
    return name, pos, team


def upsert_csv(
    csv_path: str,
    season: int,
    mapping: Dict[str, str],
    position_hint: str,
    lookup: Dict[str, str],
) -> int:
    # Keep one stable snapshot date per season so we can store multi-year history
    # in a table keyed by (player_id, as_of_date).
    season_as_of_date = date(season + 1, 2, 15).isoformat()
    count = 0

    with open(csv_path, "r", encoding="utf-8") as f, get_conn() as conn:
        reader = csv.DictReader(f)
        for row in reader:
            row_season_raw = row.get("season") or row.get("Season") or row.get("year") or row.get("Year")
            if row_season_raw:
                try:
                    if int(float(str(row_season_raw).strip())) != season:
                        continue
                except ValueError:
                    pass

            name, pos, team = _player_key(row)
            if not name:
                continue
            pos = pos or position_hint
            player_id = lookup.get(f"{name}|{pos}|{team}") or lookup.get(f"{name}|{pos}|") or lookup.get(f"{name}||{team}") or lookup.get(f"{name}||")
            if not player_id:
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

    return count


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync PFF CSV metrics into player_advanced_metrics")
    parser.add_argument("--season", type=int, help="Single season to sync")
    parser.add_argument("--seasons", type=str, help="Comma-separated seasons to sync (e.g. 2025,2024,2023)")
    parser.add_argument("--last-n", type=int, default=3, help="When season(s) omitted, sync the most recent N seasons (default: 3)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_advanced_metrics_db()
    lookup = build_player_lookup()

    seasons_arg = str(args.season) if args.season is not None else args.seasons
    seasons = resolve_seasons(seasons_arg, args.last_n)
    print(f"Syncing seasons: {', '.join(str(s) for s in seasons)}")

    total_r = total_w = total_p = 0
    for season in seasons:
        rushing_page_url = os.getenv("PFF_RUSHING_CSV_URL", "") or _default_page_url("rushing", season)
        receiving_page_url = os.getenv("PFF_RECEIVING_CSV_URL", "") or _default_page_url("receiving", season)
        passing_page_url = os.getenv("PFF_PASSING_CSV_URL", "") or _default_page_url("passing", season)

        rushing_url = _ensure_csv_export(_seasonize_page_url(rushing_page_url, season))
        receiving_url = _ensure_csv_export(_seasonize_page_url(receiving_page_url, season))
        passing_url = _ensure_csv_export(_seasonize_page_url(passing_page_url, season))

        rushing_csv = download_csv(rushing_url, os.path.join(OUTPUT_DIR, f"pff_nfl_rushing_{season}.csv"))
        receiving_csv = download_csv(receiving_url, os.path.join(OUTPUT_DIR, f"pff_nfl_receiving_{season}.csv"))
        passing_csv = download_csv(passing_url, os.path.join(OUTPUT_DIR, f"pff_nfl_passing_{season}.csv"))

        total_r += upsert_csv(rushing_csv, season, RUSHING_COLS, "RB", lookup)
        total_w += upsert_csv(receiving_csv, season, RECEIVING_COLS, "WR", lookup)
        total_p += upsert_csv(passing_csv, season, PASSING_COLS, "QB", lookup)

    print(f"Synced PFF metrics rows: rushing={total_r}, receiving={total_w}, passing={total_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
