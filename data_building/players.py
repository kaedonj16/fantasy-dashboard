import json
import requests
from pathlib import Path
from typing import Dict, Any, Union

from dashboard_services.utils import path_players_index

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def fetch_sleeper_players() -> dict:
    res = requests.get(SLEEPER_PLAYERS_URL, timeout=30)
    res.raise_for_status()
    return res.json()


def update_player_teams_from_sleeper(
        player_index_path: Union[str, Path],
        *,
        write: bool = True,
        keep_missing_team: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Update the `team` field in your existing player index using fresh Sleeper data.

    Expected existing structure:
    {
      "5248": {
        "name": "Gus Edwards",
        "team": "LAC",
        "tankId": "3051926",
        "byeWeek": 12,
        "pos": "RB",
        "bDay": "4/13/1995",
        "espnID": "3051926"
      },
      ...
    }

    Returns:
      {
        "updated_index": <dict>,
        "changed_players": <list[dict]>,
        "unchanged_count": <int>,
        "changed_count": <int>,
      }
    """
    player_index_path = Path(player_index_path)

    if not player_index_path.exists():
        raise FileNotFoundError(f"Player index file not found: {player_index_path}")

    with player_index_path.open("r", encoding="utf-8") as f:
        player_index = json.load(f)

    sleeper_players = fetch_sleeper_players()

    changed_players = []
    unchanged_count = 0

    for sleeper_id, existing in player_index.items():
        if not isinstance(existing, dict):
            continue

        fresh = sleeper_players.get(str(sleeper_id))
        if not isinstance(fresh, dict):
            continue

        old_team = str(existing.get("team") or "").strip().upper()
        new_team_raw = str(fresh.get("team") or "").strip().upper()

        if not new_team_raw and keep_missing_team:
            unchanged_count += 1
            continue

        new_team = new_team_raw or ""

        if old_team != new_team:
            existing["team"] = new_team
            changed_players.append({
                "sleeper_id": str(sleeper_id),
                "name": existing.get("name") or fresh.get("full_name"),
                "old_team": old_team,
                "new_team": new_team,
            })
        else:
            unchanged_count += 1

    if write:
        with player_index_path.open("w", encoding="utf-8") as f:
            json.dump(player_index, f, ensure_ascii=False, indent=2)

    return {
        "updated_index": player_index,
        "changed_players": changed_players,
        "unchanged_count": unchanged_count,
        "changed_count": len(changed_players),
    }


if __name__ == "__main__":
    update_player_teams_from_sleeper(path_players_index())
