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
