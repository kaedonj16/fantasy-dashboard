import json
import os

import requests

from data_building.external_data.sleeper_bulk_stats import CACHE_DIR
from utils.utils import (
    load_players_index,
    load_teams_index,
    path_players_index,
    write_json,
)

TANK01_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLPlayerList"
TANK01_HEADERS = {
    "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com",
    "x-rapidapi-key": os.environ.get("TANK01_API_KEY", ""),
}

# Map Tank01 raw positions -> your canonical IDP buckets
TANK_IDP_POS_MAP = {
    "S": "DB",
    "CB": "DB",
    "DE": "DL",
    "DT": "DL",
    "DL": "DL",  # just in case they send this
    "LB": "LB",
}


def build_idp_players_index() -> dict:
    """
    Build an IDP player index by merging:
      - Tank01 player list (S/CB/DE/DT/LB -> DB/DL/LB)
      - Sleeper player IDs (for key)
      - teams_index.json for bye weeks

    Output format (idp_players_index.json):
      {
        "12507": {
          "name": "Omarion Hampton",
          "team": "LAC",
          "tankId": "4685382",
          "byeWeek": 12,
          "pos": "DL",         # canonical: DL / LB / DB
          "bDay": "3/16/2003",
          "espnID": "12345"
        },
        ...
      }
    """
    outfile = CACHE_DIR / "idp_players_index.json"

    print("Fetching Tank01 player list...")
    try:
        response = requests.get(TANK01_URL, headers=TANK01_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching Tank01 player list: {e}")
        return {}

    # Load bye-week mapping from teams_index.json
    teams_index = load_teams_index()

    tank_players = data.get("body", [])
    idp_index: dict[str, dict] = {}

    for p in tank_players:
        raw_pos = p.get("pos")
        canonical_pos = TANK_IDP_POS_MAP.get(raw_pos)

        # Skip non-IDP positions
        if canonical_pos is None:
            continue

        name = p.get("espnName")
        team = p.get("team")  # e.g. "LAC"
        tank_id = p.get("playerID") or p.get("playerId")
        birthday = p.get("bDay") or ""
        espn_id = p.get("espnID") or p.get("espnId") or None
        sleeper_id = p.get("sleeperBotID")

        if not name or not team:
            continue

        if not sleeper_id:
            # Optionally log unmatched Tank01 IDPs for debugging
            continue

        bye_week = teams_index.get(team, {}).get("byeWeek")

        idp_index[sleeper_id] = {
            "name": name,
            "team": team,
            "tankId": tank_id,
            "byeWeek": bye_week,
            "pos": canonical_pos,  # DB / DL / LB
            "bDay": birthday,
            "espnID": espn_id,
        }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(idp_index, f, indent=2)

    print(f"IDP index built. Total IDPs: {len(idp_index)} → {outfile}")
    return idp_index


def add_espn_id_to_players_index() -> dict:
    """
    Loads players_index.json and adds "espnID" to each player (all positions),
    matched by Sleeper player id (Tank01 sleeperBotID).
    """
    players_index = load_players_index()
    output_path = path_players_index()

    # 2) Fetch Tank01 player list once
    print("Fetching Tank01 player list...")
    try:
        response = requests.get(TANK01_URL, headers=TANK01_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Error fetching Tank01 player list: {e}") from e

    tank_players = data.get("body", []) or []

    # 3) Build Sleeper -> ESPN map (all positions)
    sleeper_to_espn: dict[str, str] = {}
    for p in tank_players:
        sleeper_id = p.get("sleeperBotID")
        if not sleeper_id:
            continue

        espn_id = p.get("espnID") or p.get("espnId")
        if not espn_id:
            continue

        sleeper_to_espn[str(sleeper_id)] = str(espn_id)

    # 4) Update players_index.json entries
    updated = 0
    missing = 0

    for sleeper_id, info in players_index.items():
        espn_id = sleeper_to_espn.get(str(sleeper_id))
        if espn_id:
            # add / overwrite
            if info.get("espnID") != espn_id:
                info["espnID"] = espn_id
                updated += 1
        else:
            # optionally ensure the key exists; comment out if you don't want nulls
            if "espnID" not in info:
                info["espnID"] = None
            missing += 1

    # 5) Save result
    write_json(path_players_index(), players_index)

    print(
        f"players_index updated → {output_path}\n"
        f"ESPN IDs applied/changed: {updated}\n"
        f"No ESPN match found: {missing}"
    )
    return players_index


# If you want to run it directly:
if __name__ == "__main__":
    add_espn_id_to_players_index()
