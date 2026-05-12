#!/usr/bin/env python3
"""
Fetch Tank01 player list and patch exp + draft_year into every existing
players_index.json entry.  Also adds players that are missing entirely.

Usage:
    cd /home/user/fantasy-dashboard
    python3 update_players_exp.py
"""

import json
import datetime
from pathlib import Path
import requests

from utils.utils import TANK01_API_HOST, TANK01_API_KEY, path_players_index


def update_players_exp():
    players_index_path = Path(path_players_index())
    with players_index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"Loaded {len(index)} existing players")

    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }

    print("Fetching Tank01 player list...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("body", [])
    print(f"Received {len(data)} players from Tank01")

    current_nfl_season = datetime.datetime.now().year

    updated = 0
    added = 0

    for p in data:
        sleeper_id = str(
            p.get("sleeperBotID")
            or p.get("sleeperId")
            or p.get("sleeper_id")
            or p.get("sleeperid")
            or ""
        )
        if not sleeper_id:
            continue

        raw_exp = p.get("exp") or p.get("espnYrsPro")
        if raw_exp is None:
            continue

        try:
            exp_int = int(raw_exp)
        except (TypeError, ValueError):
            continue

        if exp_int <= 0:
            continue

        draft_year = current_nfl_season - exp_int + 1

        if sleeper_id in index:
            entry = index[sleeper_id]
            if entry.get("exp") != exp_int or entry.get("draft_year") != draft_year:
                entry["exp"] = exp_int
                entry["draft_year"] = draft_year
                updated += 1
        else:
            # Also add genuinely new players we haven't seen before
            tank_id = str(p.get("playerID") or p.get("playerId") or p.get("id") or "")
            name = p.get("espnName") or p.get("fullName") or p.get("name") or ""
            index[sleeper_id] = {
                "name": name,
                "team": p.get("team") or p.get("proTeam") or "",
                "tankId": tank_id,
                "bDay": p.get("bDay"),
                "espnID": tank_id,
                "pos": p.get("position") or p.get("pos") or "",
                "exp": exp_int,
                "draft_year": draft_year,
            }
            added += 1

    with players_index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Updated exp/draft_year for {updated} existing players")
    print(f"  Added {added} new players")
    print(f"  Total players: {len(index)}")


if __name__ == "__main__":
    update_players_exp()
