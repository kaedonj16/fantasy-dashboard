# dashboard_services/players_cleanup.py

from __future__ import annotations

import json
import os
import os
import requests
from typing import Dict, Any
from typing import Dict, Any

from utils import load_players_index

TANK01_API_KEY = os.environ.get("TANK01_API_KEY")
TANK01_HOST = os.environ.get("TANK01_HOST", "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com")
BASE = f"https://{TANK01_HOST}"


class Tank01Error(Exception):
    pass


def fetch_tank01_players(season: int | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Call the Tank01 'NFL players' endpoint and return a map:

        { "<tankId>": { "pos": "WR", "team": "SEA", ... }, ... }

    Adjust `url` and query params based on your actual RapidAPI docs.
    """

    if not TANK01_API_KEY:
        raise Tank01Error("TANK01_API_KEY environment variable is not set.")
    if not TANK01_HOST or "<your-tank01-host-here>" in TANK01_HOST:
        raise Tank01Error("TANK01_HOST environment variable is not set or placeholder.")

    # TODO: change this path to your actual Tank01 players endpoint
    url = f"{BASE}/getNFLPlayerList"  # or getNFLPlayerInfo, etc.

    querystring: Dict[str, Any] = {}
    if season is not None:
        querystring["season"] = str(season)

    headers = {
        "x-rapidapi-key": TANK01_API_KEY,
        "x-rapidapi-host": TANK01_HOST,
    }

    resp = requests.get(url, headers=headers, params=querystring, timeout=20)
    if resp.status_code != 200:
        raise Tank01Error(f"Tank01 players error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    status_code = data.get("statusCode")
    if status_code not in (200, None):
        # some Tank01 endpoints just omit statusCode
        raise Tank01Error(f"Tank01 players statusCode={status_code}: {data}")

    body = data.get("body")
    players_map: Dict[str, Dict[str, Any]] = {}

    if isinstance(body, list):
        iterable = body
    elif isinstance(body, dict):
        # sometimes keyed by playerID or something similar
        iterable = body.values()
    else:
        iterable = []

    for p in iterable:
        if not isinstance(p, dict):
            continue

        pid = p.get("playerID") or p.get("playerId") or p.get("id")
        if not pid:
            continue
        pid = str(pid)

        pos = p.get("position") or p.get("pos")
        team = p.get("team") or p.get("teamAbv")
        bDay = p.get("bDay")

        players_map[pid] = {
            "pos": pos,
            "team": team,
            "bDay": bDay,
            # keep the raw object if you want:
            # "_raw": p,
        }

    print(f"[tank01_players] Built players map with {len(players_map)} entries")
    return players_map


ALLOWED_POS = {"QB", "RB", "WR", "TE", "FB"}

# adjust to wherever your players_index.json actually lives
DEFAULT_PLAYERS_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "cache", "players_index.json"
)


def save_players_index(players_index: Dict[str, Dict[str, Any]],
                       path: str = DEFAULT_PLAYERS_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(players_index, f, indent=2)
    print(f"[players_cleanup] Wrote {len(players_index)} players to {os.path.abspath(path)}")


def enrich_and_filter_players_index(
        season: int | None = None,
        players_index_path: str = DEFAULT_PLAYERS_INDEX_PATH,
) -> None:
    """
    - Load your existing players_index
    - Add Tank01 'pos' to each entry that has a tankId
    - Drop any players whose pos is not QB/RB/WR/TE/FB
    - Save back to disk
    """

    players_index = load_players_index()
    print(f"[players_cleanup] Loaded {len(players_index)} players from {players_index_path}")

    tank_players = fetch_tank01_players(season=season)

    new_index: Dict[str, Dict[str, Any]] = {}
    missing_pos = 0
    missing_tank = 0

    for sleeper_id, meta in players_index.items():
        tank_id = meta.get("tankId")
        if not tank_id:
            # if you want to keep players without a tankId, move this into new_index instead
            missing_tank += 1
            continue

        tank_id_str = str(tank_id)
        tinfo = tank_players.get(tank_id_str)
        if not tinfo:
            missing_pos += 1
            continue

        bday = (tinfo.get("bDay") or "").upper()

        updated = dict(meta)
        updated["bDay"] = bday  # add the position into your index
        new_index[sleeper_id] = updated

    print(f"[players_cleanup] Kept {len(new_index)} players")
    print(f"[players_cleanup] Skipped {missing_pos} with no Tank01 bday, {missing_tank} with no tankId")

    save_players_index(new_index, path=players_index_path)


if __name__ == "__main__":
    # use the season Tank01 expects; or None if the endpoint doesn't need it
    enrich_and_filter_players_index(season=2025)
