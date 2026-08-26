"""Refresh current NFL team affiliations from Sleeper.

Used by the daily cron so trades / signings / cuts land in:
  1. cache/players_index.json (+ relevant index) on the cron disk
  2. player_current_team (shared Postgres — what the web service reads)
  3. player_values.team for players that already have a values row

Sleeper is the primary source: free, keyed by the same player_id the rest of
the app uses. Tank01 headshot refresh stays in refresh_player_data.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from data_building.external_data.player_current_team import (
    normalize_nfl_team,
    update_player_values_teams,
    upsert_current_teams,
)
from utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def path_players_index() -> str:
    return str(CACHE_DIR / "players_index.json")


def path_relevant_index() -> str:
    return str(CACHE_DIR / "players_index_relevant.json")


def _write_json(path: Union[str, Path], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def fetch_sleeper_players() -> dict:
    # Lazy import: unit CI installs a slim dep set without requests; callers that
    # only pass sleeper_players=... (tests) never need the network client.
    import requests

    res = requests.get(SLEEPER_PLAYERS_URL, timeout=60)
    res.raise_for_status()
    return res.json()

def _bye_by_team() -> Dict[str, int]:
    """team abbrev -> bye week from teams_index (best-effort)."""
    try:
        from utils.utils import load_teams_index
        teams = load_teams_index() or {}
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for abv, meta in teams.items():
        if not isinstance(meta, dict):
            continue
        bye = meta.get("byeWeek")
        if bye is None:
            continue
        try:
            out[normalize_nfl_team(abv)] = int(bye)
        except (TypeError, ValueError):
            continue
    return out


def _apply_teams_to_index(
    index: Dict[str, Any],
    sleeper_players: dict,
    *,
    keep_missing_team: bool,
    bye_by_team: Dict[str, int],
) -> List[dict]:
    """Mutate index in place; return list of change dicts."""
    changed: List[dict] = []
    for sleeper_id, existing in index.items():
        if not isinstance(existing, dict):
            continue

        fresh = sleeper_players.get(str(sleeper_id))
        if not isinstance(fresh, dict):
            continue

        old_team = normalize_nfl_team(existing.get("team"))
        new_team_raw = fresh.get("team")
        if new_team_raw is None or str(new_team_raw).strip() == "":
            if keep_missing_team:
                continue
            new_team = ""
        else:
            new_team = normalize_nfl_team(new_team_raw)

        if old_team == new_team:
            # Keep byeWeek in sync even when the team didn't change.
            if new_team and bye_by_team.get(new_team) is not None:
                if existing.get("byeWeek") != bye_by_team[new_team]:
                    existing["byeWeek"] = bye_by_team[new_team]
            continue

        existing["team"] = new_team
        if new_team and bye_by_team.get(new_team) is not None:
            existing["byeWeek"] = bye_by_team[new_team]
        changed.append({
            "sleeper_id": str(sleeper_id),
            "name": existing.get("name") or fresh.get("full_name"),
            "old_team": old_team,
            "new_team": new_team,
        })
    return changed


def update_player_teams_from_sleeper(
    player_index_path: Union[str, Path],
    *,
    write: bool = True,
    keep_missing_team: bool = True,
    sleeper_players: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Update the `team` field in an existing player index using Sleeper data.

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

    if sleeper_players is None:
        sleeper_players = fetch_sleeper_players()

    bye = _bye_by_team()
    changed_players = _apply_teams_to_index(
        player_index,
        sleeper_players,
        keep_missing_team=keep_missing_team,
        bye_by_team=bye,
    )
    unchanged_count = max(0, len(player_index) - len(changed_players))

    if write:
        _write_json(player_index_path, player_index)

    return {
        "updated_index": player_index,
        "changed_players": changed_players,
        "unchanged_count": unchanged_count,
        "changed_count": len(changed_players),
    }


def refresh_current_nfl_teams(
    *,
    keep_missing_team: bool = True,
    write_files: bool = True,
    write_db: bool = True,
) -> dict[str, Any]:
    """Daily entry point: refresh both indexes + shared DB + player_values.team.

    Fetches Sleeper once and applies to players_index + players_index_relevant.
    Always upserts the *full* current team map for indexed players into
    player_current_team so the web overlay stays complete (not just deltas).
    """
    print("[nfl-teams] Fetching Sleeper player list...")
    sleeper_players = fetch_sleeper_players()
    print(f"[nfl-teams] Sleeper returned {len(sleeper_players)} players")

    paths = [Path(path_players_index()), Path(path_relevant_index())]
    all_changed: List[dict] = []
    seen_change_ids = set()
    full_team_map: Dict[str, str] = {}

    for path in paths:
        if not path.exists():
            print(f"[nfl-teams] {path.name} not found, skipping")
            continue
        result = update_player_teams_from_sleeper(
            path,
            write=write_files,
            keep_missing_team=keep_missing_team,
            sleeper_players=sleeper_players,
        )
        print(
            f"[nfl-teams] {path.name}: {result['changed_count']} team changes "
            f"({result['unchanged_count']} unchanged)"
        )
        for c in result["changed_players"]:
            sid = c["sleeper_id"]
            if sid not in seen_change_ids:
                seen_change_ids.add(sid)
                all_changed.append(c)
                print(f"   • {c.get('name')}: {c['old_team'] or '—'} → {c['new_team'] or 'FA'}")

        # Collect full team map from the primary index (or whichever we have).
        for pid, meta in (result["updated_index"] or {}).items():
            if isinstance(meta, dict):
                full_team_map[str(pid)] = normalize_nfl_team(meta.get("team"))

    db_upserted = 0
    values_updated = 0
    if write_db and full_team_map:
        db_upserted = upsert_current_teams(full_team_map)
        print(f"[nfl-teams] Upserted {db_upserted} rows into player_current_team")
        # Only patch player_values for players whose team actually changed —
        # avoids rewriting the whole values table every day.
        changed_map = {
            c["sleeper_id"]: c["new_team"] for c in all_changed
        }
        if changed_map:
            values_updated = update_player_values_teams(changed_map)
            print(f"[nfl-teams] Updated player_values.team for {values_updated} players")

    summary = {
        "changed_count": len(all_changed),
        "changed_players": all_changed,
        "db_upserted": db_upserted,
        "player_values_updated": values_updated,
        "indexed_players": len(full_team_map),
    }
    print(
        f"[nfl-teams] Done — {summary['changed_count']} players changed teams "
        f"(indexed={summary['indexed_players']})"
    )
    return summary


# Back-compat alias used by older one-off scripts / docs.
update_player_teams = update_player_teams_from_sleeper


if __name__ == "__main__":
    refresh_current_nfl_teams()
