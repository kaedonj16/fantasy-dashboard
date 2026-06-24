#!/usr/bin/env python3
"""
One-shot refresh of player team assignments and ESPN headshots.

Pulls a single Tank01 player list and updates the `team` and `espnHeadshot`
fields in both cache indexes (players_index.json + players_index_relevant.json).
Uses the cwd-independent path helpers, so it can be run from anywhere:

    python -m data_building.updates.refresh_player_data

Requires TANK01_API_KEY in the environment (see utils.utils).
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from utils.utils import (
    TANK01_API_HOST,
    TANK01_API_KEY,
    path_players_index,
    path_relevant_index,
)


def _fetch_tank01_player_list() -> list[dict]:
    """Fetch the full Tank01 NFL player list (one API call)."""
    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }
    print("📡 Fetching Tank01 player list...")
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("body", [])


def _build_maps(body: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build two lookups from the Tank01 body:
      team_by_sleeper:    sleeper_id -> team abbreviation (WSH normalized to WAS)
      headshot_by_tankid: tankId     -> espnHeadshot URL
    """
    team_by_sleeper: dict[str, str] = {}
    headshot_by_tankid: dict[str, str] = {}
    for p in body:
        sleeper_id = (
            p.get("sleeperBotID")
            or p.get("sleeperId")
            or p.get("sleeper_id")
            or p.get("sleeperid")
        )
        if sleeper_id:
            team = p.get("team", p.get("proTeam", ""))
            if team == "WSH":
                team = "WAS"
            team_by_sleeper[str(sleeper_id)] = team

        tank_id = str(p.get("playerID") or p.get("playerId") or p.get("id") or "")
        headshot = p.get("espnHeadshot")
        if tank_id and headshot:
            headshot_by_tankid[tank_id] = headshot

    print(
        f"   • {len(team_by_sleeper)} team entries, "
        f"{len(headshot_by_tankid)} headshot entries"
    )
    return team_by_sleeper, headshot_by_tankid


def _refresh_index(
    path: Path,
    team_by_sleeper: dict[str, str],
    headshot_by_tankid: dict[str, str],
) -> tuple[int, int]:
    """Update teams + headshots in one index file. Returns (teams_changed, headshots_set)."""
    if not path.exists():
        print(f"⚠️  {path} not found, skipping")
        return 0, 0

    with path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    teams_changed = 0
    headshots_set = 0
    for sleeper_id, meta in index.items():
        # Team (keyed by sleeper id)
        new_team = team_by_sleeper.get(str(sleeper_id))
        if new_team and meta.get("team", "") != new_team:
            meta["team"] = new_team
            teams_changed += 1

        # Headshot (keyed by the player's tankId). Refresh whenever the source
        # has a URL so team-change art (new uniforms) also gets picked up.
        tank_id = str(meta.get("tankId", ""))
        if tank_id and tank_id in headshot_by_tankid:
            if meta.get("espnHeadshot") != headshot_by_tankid[tank_id]:
                meta["espnHeadshot"] = headshot_by_tankid[tank_id]
                headshots_set += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(
        f"✅ {path.name}: {teams_changed} team updates, "
        f"{headshots_set} headshot updates"
    )
    return teams_changed, headshots_set


def refresh_player_data() -> None:
    body = _fetch_tank01_player_list()
    team_by_sleeper, headshot_by_tankid = _build_maps(body)
    for path in (Path(path_players_index()), Path(path_relevant_index())):
        _refresh_index(path, team_by_sleeper, headshot_by_tankid)
    print("\nDone.")


if __name__ == "__main__":
    refresh_player_data()
