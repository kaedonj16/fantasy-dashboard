import requests
import json
from pathlib import Path

from dashboard_services.data_building.sleeper_bulk_stats import CACHE_DIR
from dashboard_services.utils import normalize_name, load_teams_index, DATA_DIR

TANK01_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLPlayerList"
TANK01_HEADERS = {
    "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com",
    "x-rapidapi-key": "a31667ff00msh6d542faa96aa36bp1513aajsn612c819feca4",
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


if __name__ == '__main__':
    sleeper_players = get_players_map()  # or however you load them
    build_idp_players_index(sleeper_players)