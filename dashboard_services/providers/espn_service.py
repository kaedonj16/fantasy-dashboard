from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

# ESPN lineupSlotId conventions (most leagues):
# 20 = Bench, 21 = IR
ESPN_BENCH_SLOT = 20
ESPN_IR_SLOT = 21

def _record_and_streak(team_raw: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns:
      record_str like 'LLWLLLLLWWWWWW' (chronological order not guaranteed by ESPN)
      streak_str like '6W'
    """
    rec = (team_raw.get("record") or {}).get("overall") or {}
    streak_len = int(rec.get("streakLength") or 0)
    streak_type = (rec.get("streakType") or "").upper()  # WIN / LOSS / TIE

    streak = "0"
    if streak_len > 0 and streak_type:
        streak = f"{streak_len}{'W' if streak_type == 'WIN' else ('L' if streak_type == 'LOSS' else 'T')}"

    # ESPN does not always provide a clean game-by-game string.
    # Some leagues expose "recordByPeriod" / "outcomes" in other views,
    # but in your raw snippet I only see aggregates.
    #
    # If you later pull a per-week outcomes list, plug it in here.
    record_str = ""  # default empty when we don't have per-week results

    return record_str, streak


def transform_espn_roster_to_unified(
    espn_roster: Dict[str, Any],
    *,
    league_id: str,
    owner_map: Optional[Dict[str, str]] = None,
    pro_team_id_to_abbrev: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    espn_roster: one item from your espn get_rosters() output:
      {
        "roster_id": "1",
        "owner_id": "{GUID}",
        "players": ["4362238", "-16026", ...],
        "raw": {... team object incl roster.entries ...}
      }

    Returns your Sleeper-shaped roster dict.
    """

    team_raw = espn_roster.get("raw") or {}

    # owner_id: map ESPN owner GUID -> your unified user id if provided
    espn_owner = espn_roster.get("owner_id")
    unified_owner = owner_map.get(espn_owner, espn_owner) if owner_map else espn_owner

    # roster_id: your target example uses an int
    rid = espn_roster.get("roster_id")
    roster_id_int = int(rid) if rid is not None and str(rid).isdigit() else rid

    # Points + record aggregates
    overall = (team_raw.get("record") or {}).get("overall") or {}
    wins = int(overall.get("wins") or 0)
    losses = int(overall.get("losses") or 0)
    ties = int(overall.get("ties") or 0)

    fpts = team_raw.get("points")
    fpts_against = overall.get("pointsAgainst")

    # If ESPN returns floats, split decimals like Sleeper does
    def _split_points(x: Any) -> Tuple[int, int]:
        if x is None:
            return 0, 0
        try:
            val = float(x)
        except Exception:
            return 0, 0
        whole = int(val)
        dec = int(round((val - whole) * 100))
        return whole, dec

    fpts_whole, fpts_dec = _split_points(fpts)
    fpa_whole, fpa_dec = _split_points(fpts_against)

    record_str, streak_str = _record_and_streak(team_raw)

    # starters / reserve using roster.entries
    entries = ((team_raw.get("roster") or {}).get("entries")) or []
    starters: List[str] = []
    reserve: List[str] = []

    # player list: use entries so it includes everyone (and so we can map DST abbrev if desired)
    players: List[str] = []

    for e in entries:
        if not isinstance(e, dict):
            continue
        pid = e.get("playerId")
        if pid is None:
            continue

        pid_str = str(pid)

        # Optional DST mapping:
        # ESPN DST is often a negative playerId, but the entry usually includes player.proTeamId
        if pro_team_id_to_abbrev and isinstance(pid, int) and pid < 0:
            pte = (e.get("playerPoolEntry") or {}).get("player") or {}
            pro_team_id = pte.get("proTeamId")
            if isinstance(pro_team_id, int) and pro_team_id in pro_team_id_to_abbrev:
                pid_str = pro_team_id_to_abbrev[pro_team_id]

        players.append(pid_str)

        slot = e.get("lineupSlotId")
        try:
            slot = int(slot)
        except Exception:
            slot = None

        if slot == ESPN_IR_SLOT:
            reserve.append(pid_str)
        elif slot is not None and slot != ESPN_BENCH_SLOT:
            starters.append(pid_str)

    # If entries weren’t present for some reason, fall back to espn_roster["players"]
    if not players:
        players = [str(p) for p in (espn_roster.get("players") or [])]

    return {
        "co_owners": None,
        "keepers": None,
        "league_id": str(league_id),
        "metadata": {
            "record": record_str,     # will be "" unless you fetch per-week outcomes elsewhere
            "streak": streak_str,     # e.g. "2W"
        },
        "owner_id": unified_owner,
        "player_map": None,
        "players": players,
        "reserve": reserve or [],
        "roster_id": roster_id_int,
        "settings": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "fpts": fpts_whole,
            "fpts_decimal": fpts_dec,
            "fpts_against": fpa_whole,
            "fpts_against_decimal": fpa_dec,
            # not available from your shown ESPN payload; keep defaults:
            "ppts": 0,
            "ppts_decimal": 0,
            "total_moves": int(team_raw.get("transactionCounter") or 0) if team_raw.get("transactionCounter") is not None else 0,
            "waiver_budget_used": 0,
            "waiver_position": 0,
        },
        "starters": starters or [],
        "taxi": None,
    }
