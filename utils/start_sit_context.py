"""Pure adapters from existing weekly/team caches into Start/Sit inputs."""
from __future__ import annotations

from statistics import mean, pstdev

from utils.nfl_stadiums import normalize_nfl_team

# Display-only absence notes (teammate / opponent injuries). These never feed
# the start/sit score: Sleeper's projections already redistribute opportunity
# when a teammate is out, so scoring them again would double-count.
_ABSENCE_SKILL_POS = ("QB", "RB", "WR", "TE")
_ABSENCE_DEF_POS = ("DL", "DE", "DT", "NT", "EDGE", "LB", "OLB", "ILB", "MLB",
                    "DB", "CB", "S", "FS", "SS")
_ABSENCE_STATUSES = frozenset({"IR", "PUP", "NFI", "SUSP", "SUS", "OUT",
                               "DOUBTFUL", "NA"})
_ABSENCE_STATUS_LABEL = {
    "OUT": "Out", "DOUBTFUL": "Doubtful", "IR": "IR", "PUP": "PUP",
    "NFI": "NFI", "SUSP": "Suspended", "SUS": "Suspended", "NA": "Out",
}


def _absence_entry(pid: str, p: dict) -> dict | None:
    """One display note for a seriously-hurt player, or None when not notable."""
    if not isinstance(p, dict):
        return None
    status = str(p.get("injury_status") or p.get("status") or "").strip().upper()
    if status not in _ABSENCE_STATUSES:
        return None
    pos = str(p.get("position") or p.get("pos") or "").strip().upper()
    name = str(p.get("full_name") or p.get("name") or "").strip()
    if not name:
        return None
    body = str(p.get("injury_body_part") or "").strip()
    label = _ABSENCE_STATUS_LABEL.get(status, status.title())
    text = f"{name} ({label}{', ' + body if body else ''})"
    try:
        order = p.get("depth_chart_order")
        order = float(order) if order is not None else None
    except (TypeError, ValueError):
        order = None
    return {"pid": str(pid), "name": name, "pos": pos, "status": status,
            "text": text, "depth_order": order}


def build_absence_index(full_players: dict) -> dict:
    """Precompute per-team serious-injury lists from a Sleeper players map.

    Returns ``{TEAM: {"skill": [...], "defense": [...]}}`` where each entry is
    the ``_absence_entry`` dict above. One pass over the map so per-row
    lookups stay cheap.
    """
    index: dict = {}
    for pid, p in (full_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").strip().upper()
        if not team:
            continue
        entry = _absence_entry(pid, p)
        if entry is None:
            continue
        bucket = index.setdefault(team, {"skill": [], "defense": []})
        if entry["pos"] in _ABSENCE_SKILL_POS:
            bucket["skill"].append(entry)
        elif entry["pos"] in _ABSENCE_DEF_POS:
            bucket["defense"].append(entry)
    return index


def absence_notes(index: dict, team: str, opponent: str, *,
                  exclude_pid: str | None = None, max_defense: int = 3) -> dict:
    """Display-only absence notes for one player's game.

    ``teammates``: seriously-hurt skill-position teammates (QB/RB/WR/TE),
    excluding the player themselves. ``opponents``: seriously-hurt defenders
    on the opposing team, capped at ``max_defense`` so the note never gets
    noisy. Empty lists when nothing is notable.
    """
    team = str(team or "").strip().upper()
    opponent = str(opponent or "").strip().upper()
    exclude = str(exclude_pid) if exclude_pid is not None else None

    def _sort_key(e: dict):
        order = e.get("depth_order")
        return (order is None, order if order is not None else 0.0,
                e.get("name") or "")

    teammates = []
    if team:
        for e in sorted((index or {}).get(team, {}).get("skill", []),
                        key=_sort_key):
            if exclude is not None and e.get("pid") == exclude:
                continue
            teammates.append(e)
    opponents = []
    if opponent:
        for e in sorted((index or {}).get(opponent, {}).get("defense", []),
                        key=_sort_key)[:max(0, int(max_defense))]:
            opponents.append(e)
    return {"teammates": teammates, "opponents": opponents}


def expected_plays_context(team_rows: dict, team: str, opponent: str, nfl_avg) -> dict:
    """Blend actual offense plays with opponent plays faced; no invented pace."""
    team = normalize_nfl_team(team)
    opponent = normalize_nfl_team(opponent)
    try:
        avg = float(nfl_avg)
    except (TypeError, ValueError):
        return {}
    own = (team_rows or {}).get(team) or {}
    opp = (team_rows or {}).get(opponent) or {}
    try:
        offense = float(own["off_plays_pg"])
        allowed = float(opp.get("plays_faced_l4_pg") or opp["plays_faced_pg"])
    except (KeyError, TypeError, ValueError):
        return {}
    # Shrink both observations toward league average, then blend evenly. This
    # guards against one anomalous recent game while retaining real possession.
    expected = mean((0.7 * offense + 0.3 * avg, 0.7 * allowed + 0.3 * avg))
    return {"expected_team_plays": round(expected, 1), "league_average_plays": round(avg, 1),
            "source": "team_play_volume"}


def role_confidence_from_trend(trend: dict) -> float | None:
    """0..1 recent role stability, preserving confirmed promotions."""
    series = trend.get("series") if isinstance(trend, dict) else None
    if not isinstance(series, list) or len(series) < 2:
        return None
    try:
        values = [float(v) for v in series[-3:]]
    except (TypeError, ValueError):
        return None
    level = max(1.0, mean(values))
    stability = max(0.0, 1.0 - pstdev(values) / level)
    # A promoted player with two consecutive elevated readings is not punished
    # for the old low baseline that made them interesting in the first place.
    if len(values) >= 2 and values[-1] >= values[-2] >= mean(series[:-2] or values):
        stability = max(stability, 0.75)
    return round(min(1.0, stability), 3)
