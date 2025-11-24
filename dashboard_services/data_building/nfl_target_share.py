# dashboard_services/nfl_target_share.py

import json
import pandas as pd
import requests
import time
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

from dashboard_services.data_building.value_model_training import DATA_DIR

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


def fetch_team_target_share(team: str, season: int) -> Dict[str, Tuple[float, float]]:
    """
    Scrape Footballguys Team Targets page for one team and season.

    Returns:
        { player_name: (total_targets, target_share) }

    where target_share is player_total / team_total.
    """
    url = FOOTBALLGUYS_TEAM_TARGETS_URL.format(team=team, year=season)
    print(f"[target_share] Fetching {team} targets from {url}")

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Parse all tables from page; first one is the targets table
    tables = pd.read_html(resp.text)
    if not tables:
        print(f"[target_share] No tables found for team {team}")
        return {}

    df = tables[0]

    # The table should look like:
    # name | Wk 1 | Wk 2 | ... | total
    # There will also be rows like "RB Totals", "WR Totals", etc.
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
    team_total = df["total"].sum()
    if team_total <= 0:
        print(f"[target_share] Team {team} has zero total targets, skipping.")
        return {}

    ts_map: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        total_targets = float(row["total"])
        if total_targets <= 0:
            continue
        target_share = total_targets / team_total
        ts_map[name] = (total_targets, target_share)

    print(f"[target_share] Team {team}: {len(ts_map)} players, team_total={team_total}")
    return ts_map


def fetch_league_target_share(season: int) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Fetch target share for all teams.

    Returns:
        { (team, player_name): { "total_targets": x, "target_share": y } }
    """
    league_map: Dict[Tuple[str, str], Dict[str, float]] = {}

    for team in NFL_TEAM_CODES:
        try:
            team_map = fetch_team_target_share(team, season)
        except Exception as e:
            print(f"[target_share] ERROR fetching {team}: {e}")
            continue

        for name, (total, share) in team_map.items():
            league_map[(team, name)] = {
                "total_targets": total,
                "target_share": share,
            }

        # Be nice to their servers
        time.sleep(1.0)

    print(f"[target_share] Built target share map for {len(league_map)} (team, player) combos")
    return league_map


def enrich_value_table_with_target_share(season: int) -> None:
    """
    Loads value_table_{season}.json, adds Footballguys target share data
    as usage["total_targets"] and usage["target_share"], and writes back.

    Matching is done on (team, name) to minimize collisions.
    """
    value_table_path = DATA_DIR / f"value_table_{date.today().isoformat()}.json"
    if not value_table_path.exists():
        raise FileNotFoundError(f"No value table found at {value_table_path}")

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
