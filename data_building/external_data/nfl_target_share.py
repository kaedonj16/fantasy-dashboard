# dashboard_services/nfl_target_share.py

import concurrent.futures
import json
import time
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

from utils.paths import DATA_DIR

FOOTBALLGUYS_TEAM_TARGETS_URL = (
    "https://www.footballguys.com/stats/targets/teams?team={team}&year={year}"
)

# Standard team codes as used by the site (matches what you showed: ARI, ATL, etc.)
NFL_TEAM_CODES = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

MAX_WORKERS_TARGETS = 6  # tune this based on how aggressive you want to be


def fetch_team_target_share(
        team: str,
        season: int,
        session: requests.Session,
) -> Dict[str, Tuple[float, float]]:
    """
    Scrape Footballguys Team Targets page for one team and season.

    Returns:
        { player_name: (total_targets, target_share) }

    where target_share is player_total / team_total.
    """
    sess = session or requests
    url = FOOTBALLGUYS_TEAM_TARGETS_URL.format(team=team, year=season)

    resp = sess.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    # Parse all tables from page; first one is the targets table
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        print(f"[target_share] No tables found for team {team}")
        return {}

    df = tables[0]

    # Clean up column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "name" not in df.columns or "total" not in df.columns:
        print(f"[target_share] Unexpected column names for team {team}: {df.columns}")
        return {}

    # Filter out totals rows and any non-player rows
    df = df[~df["name"].str.contains("totals", case=False, na=False)].copy()

    # Make sure "total" is numeric
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)

    # Compute team total targets
    team_total = float(df["total"].sum())
    if team_total <= 0:
        print(f"[target_share] Team {team} has zero total targets, skipping.")
        return {}

    ts_map: Dict[str, Tuple[float, float]] = {}
    # iterrows is fine here; table is small
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        total_targets = float(row["total"])
        if total_targets <= 0:
            continue
        target_share = total_targets / team_total
        ts_map[name] = (total_targets, target_share)

    return ts_map


def fetch_league_target_share(season: int) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Fetch target share for all teams, in parallel.

    Uses daily caching to avoid scraping 32 teams on every run.

    Returns:
        { (team, player_name): { "total_targets": x, "target_share": y } }
    """
    # Check cache first (daily cache)
    cache_dir = Path(DATA_DIR).parent / "cache" / "target_share"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"target_share_{season}.json"

    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 86400:
        try:
            print(f"[target_share] Loading from cache: {cache_path.name}")
            with cache_path.open("r") as f:
                cached_data = json.load(f)
                league_map = {}
                for key_str, value in cached_data.items():
                    team, name = json.loads(key_str)
                    league_map[(team, name)] = value
                print(f"[target_share] Loaded {len(league_map)} player-team combos from cache")
                return league_map
        except Exception as e:
            print(f"[target_share] Cache read failed: {e}, fetching fresh data")

    league_map: Dict[Tuple[str, str], Dict[str, float]] = {}

    print(f"[target_share] Fetching targets for all teams (season {season})")

    session = requests.Session()

    def worker(team: str):
        try:
            team_map = fetch_team_target_share(team, season, session=session)
            return team, team_map, None
        except Exception as e:
            return team, {}, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_TARGETS) as executor:
        futures = {executor.submit(worker, team): team for team in NFL_TEAM_CODES}

        for fut in concurrent.futures.as_completed(futures):
            team, team_map, err = fut.result()
            if err:
                print(f"[target_share] ERROR fetching {team}: {err}")
                continue

            for name, (total, share) in team_map.items():
                league_map[(team, name)] = {
                    "total_targets": float(total),
                    "target_share": float(share),
                }

    print(f"[target_share] Built target share map for {len(league_map)} (team, player) combos")

    # Save to cache (convert tuple keys to strings for JSON)
    cache_data = {json.dumps([team, name]): value for (team, name), value in league_map.items()}
    with cache_path.open("w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"[target_share] Cached results to {cache_path.name}")

    return league_map


def enrich_value_table_with_target_share(season: int) -> None:
    """
    Loads value_table_{season}.json, adds Footballguys target share data
    as usage["total_targets"] and usage["target_share"], and writes back.

    Matching is done on (team, name) to minimize collisions.
    """
    value_table_path = DATA_DIR / "usage_table.json"
    if not value_table_path.exists():
        raise FileNotFoundError(f"No usage table found at {value_table_path}")

    print(f"[target_share] Loading value table from {value_table_path}")
    with value_table_path.open("r", encoding="utf-8") as f:
        players = json.load(f)

    ts_map = fetch_league_target_share(season)

    updated_count = 0
    for p in players:
        name = p.get("name")
        team = p.get("team")

        if not name or not team:
            continue

        key = (team, name)
        ts_info = ts_map.get(key)

        if not ts_info:
            # Fall back to name-only match if needed
            # (This is optional but can help when team codes differ.)
            for (t2, n2), info in ts_map.items():
                if n2 == name:
                    ts_info = info
                    break

        if not ts_info:
            continue

        usage = p.setdefault("usage", {})
        usage["total_targets"] = ts_info["total_targets"]
        usage["target_share"] = ts_info["target_share"]
        updated_count += 1

    print(f"[target_share] Updated target share for {updated_count} players")

    # Write back to disk (overwrite)
    with value_table_path.open("w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"[target_share] Saved enriched value table to {value_table_path}")


if __name__ == "__main__":
    # quick CLI usage:
    enrich_value_table_with_target_share(2025)
