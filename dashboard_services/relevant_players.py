from __future__ import annotations

import json
from typing import Dict

from dashboard_services.players import get_league_rostered_player_ids
from dashboard_services.utils import load_players_index, load_usage_table


POS_WHITELIST = {"QB", "RB", "WR", "TE"}


def build_relevant_players_index(
    league_id: str,
) -> Dict[str, dict]:
    """
    Return only fantasy-relevant players, with usage stats attached:

      {
        pid: {
          ...players_index[pid] fields...,
          "usage": { ... usage stats ... }
        }
      }
    """
    players_index = load_players_index()

    # ----------------------------
    # Normalize usage table
    # ----------------------------
    raw_usage = load_usage_table()
    usage_table: dict[str, dict] = {}

    if isinstance(raw_usage, dict):
        # already {pid: usage}
        usage_table = {str(pid): (u or {}) for pid, u in raw_usage.items()}
    elif isinstance(raw_usage, list):
        # list of objects: {id, usage, ...}
        for obj in raw_usage:
            pid2 = obj.get("id")
            if pid2 is not None:
                usage_table[str(pid2)] = obj.get("usage") or {}
    else:
        raise TypeError("usage_table must be dict or list")

    # ----------------------------
    # Roster map: who is actually on teams
    # ----------------------------
    rostered_by_team = get_league_rostered_player_ids(league_id)
    # flatten set of rostered pids
    rostered_pids: set[str] = {
        str(pid) for pids in rostered_by_team.values() for pid in pids
    }

    def is_fantasy_relevant(pid: str, meta: dict, u: dict) -> bool:
        pos = meta.get("pos") or meta.get("position")
        if pos not in POS_WHITELIST:
            return False

        # always keep rostered players
        if pid in rostered_pids:
            return True

        if not u:
            return False

        # All of these are small numeric conversions; keep as-is but localize .get calls
        get_u = u.get
        games = float(get_u("games") or 0.0)
        ppr_ppg = float(get_u("ppr_ppg") or 0.0)
        avg_snaps = float(get_u("avg_off_snaps") or 0.0)
        avg_tgt = float(get_u("avg_targets") or 0.0)
        avg_car = float(get_u("avg_carries") or 0.0)

        # usage + production thresholds
        if games >= 3:
            return True
        if ppr_ppg >= 6.0:
            return True
        if avg_snaps >= 20:
            return True
        if (avg_tgt + avg_car) >= 3:
            return True

        return False

    # ----------------------------
    # Filter relevant players AND attach usage
    # ----------------------------
    relevant: Dict[str, dict] = {}
    get_usage = usage_table.get

    for pid, meta in players_index.items():
        pid_s = str(pid)
        u = get_usage(pid_s, {})

        if is_fantasy_relevant(pid_s, meta, u):
            merged = dict(meta)
            merged["usage"] = u
            relevant[pid_s] = merged

    return relevant


def write_relevant_players_index(
    league_id: str,
    out_path: str = "cache/players_index_relevant.json",
):
    relevant_index = build_relevant_players_index(league_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relevant_index, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(relevant_index)} fantasy-relevant players to {out_path}")
