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
# Sleeper gives offensive linemen no depth data at all (depth_chart_order is
# None for starters and practice-squadders alike), so "starting" linemen are
# identified by snap share instead (see starting_lineman_pids).
_OL_POSITIONS = frozenset({"OL", "T", "G", "C", "OT", "OG"})
_ABSENCE_STATUSES = frozenset({"IR", "PUP", "NFI", "SUSP", "SUS", "OUT",
                               "DOUBTFUL", "NA"})
_ABSENCE_STATUS_LABEL = {
    "OUT": "Out", "DOUBTFUL": "Doubtful", "IR": "IR", "PUP": "PUP",
    "NFI": "NFI", "SUSP": "Suspended", "SUS": "Suspended", "NA": "Out",
}


def _absence_entry(pid: str, p: dict, *, productive: bool = False,
                   starting_lineman: bool = False) -> dict | None:
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
    if pos in _OL_POSITIONS:
        # The "OL · " tag rides inside the parenthetical so the existing
        # renderers (which split the text at " (") show it in the muted
        # status part: "Tyler Smith (OL · IR, Thumb)".
        text = f"{name} (OL · {label}{', ' + body if body else ''})"
    else:
        text = f"{name} ({label}{', ' + body if body else ''})"
    try:
        order = p.get("depth_chart_order")
        order = float(order) if order is not None else None
    except (TypeError, ValueError):
        order = None
    # Importance gate: only absences a viewer would actually care about.
    # Skill positions: a proven producer (pooled weekly points) always counts;
    # otherwise depth order must say starter/immediate backup. Depth order for
    # skill spots is polluted by the injury itself (a hurt starter slides
    # down), which is why the production signal exists. QBs are stricter:
    # only the starter matters. Defense depth order is per-slot and clean
    # (deep/IR players are None), so starter-or-backup is the whole test and
    # production never applies.
    if pos in _ABSENCE_SKILL_POS:
        if not productive:
            if pos == "QB":
                if order != 1:
                    return None
            elif order is None or order > 2:
                return None
    elif pos in _ABSENCE_DEF_POS:
        if order is None or order > 2:
            return None
    elif pos in _OL_POSITIONS:
        # No depth signal exists for linemen; only a snap-share-identified
        # starter counts (callers pass the set via build_absence_index).
        if not starting_lineman:
            return None
    return {"pid": str(pid), "name": name, "pos": pos, "status": status,
            "text": text, "depth_order": order}


def productive_pids_from_weekly_points(*weekly_maps, min_games: int = 4,
                                       min_ppg: float = 6.0) -> set:
    """Player ids with a real production track record, pooled across seasons.

    Each map is ``{pid: [weekly fantasy points]}`` (one entry per game with a
    stat line). A player qualifies with at least ``min_games`` pooled games
    averaging at least ``min_ppg``. Pure and defensive: junk input is skipped,
    never raised on.
    """
    pooled: dict = {}
    for weekly in weekly_maps:
        if not isinstance(weekly, dict):
            continue
        for pid, pts in weekly.items():
            if not isinstance(pts, (list, tuple)):
                continue
            try:
                vals = [float(v) for v in pts]
            except (TypeError, ValueError):
                continue
            if vals:
                pooled.setdefault(str(pid), []).extend(vals)
    out = set()
    for pid, vals in pooled.items():
        if len(vals) >= min_games and sum(vals) / len(vals) >= min_ppg:
            out.add(pid)
    return out


def starting_lineman_pids(snap_totals, positions_by_pid, *,
                          min_games: int = 2, min_share: float = 0.5) -> set:
    """Ids of starting offensive linemen, identified by snap share.

    ``snap_totals`` is a list of per-season maps
    ``{pid: (off_snaps, team_snaps, games)}`` pooled across seasons;
    ``positions_by_pid`` maps pid -> Sleeper position. A lineman qualifies
    with at least ``min_games`` pooled games and a pooled offensive snap
    share of at least ``min_share`` (healthy starters sit near 100%, and
    hurt/backup linemen have no snap line at all). Sleeper carries no depth
    data for linemen, so this is the only "starting" signal. Pure and
    defensive: junk input is skipped, never raised on.
    """
    pooled: dict = {}
    if isinstance(snap_totals, (list, tuple)):
        for season_map in snap_totals:
            if not isinstance(season_map, dict):
                continue
            for pid, totals in season_map.items():
                if not isinstance(totals, (list, tuple)) or len(totals) < 3:
                    continue
                try:
                    off, team_snaps, games = (float(totals[0]),
                                              float(totals[1]),
                                              float(totals[2]))
                except (TypeError, ValueError):
                    continue
                acc = pooled.setdefault(str(pid), [0.0, 0.0, 0.0])
                acc[0] += off
                acc[1] += team_snaps
                acc[2] += games
    positions = positions_by_pid if isinstance(positions_by_pid, dict) else {}
    out = set()
    for pid, (off, team_snaps, games) in pooled.items():
        if games < min_games or team_snaps <= 0:
            continue
        if off / team_snaps < min_share:
            continue
        pos = str(positions.get(pid) or "").strip().upper()
        if pos in _OL_POSITIONS:
            out.add(pid)
    return out


def build_absence_index(full_players: dict, *, productive_pids=None,
                        starting_linemen=None) -> dict:
    """Precompute per-team serious-injury lists from a Sleeper players map.

    Returns ``{TEAM: {"skill": [...], "defense": [...], "line": [...]}}``
    where each entry is the ``_absence_entry`` dict above. One pass over the
    map so per-row lookups stay cheap. ``productive_pids`` (ids from
    ``productive_pids_from_weekly_points``) lets a proven producer count as
    notable even when his depth-chart order slid because of the injury.
    ``starting_linemen`` (ids from ``starting_lineman_pids``) is the only
    way an offensive lineman counts: Sleeper gives linemen no depth data.
    """
    productive_ids = {str(pid) for pid in productive_pids} if productive_pids else set()
    lineman_ids = {str(pid) for pid in starting_linemen} if starting_linemen else set()
    index: dict = {}
    for pid, p in (full_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").strip().upper()
        if not team:
            continue
        entry = _absence_entry(pid, p, productive=str(pid) in productive_ids,
                               starting_lineman=str(pid) in lineman_ids)
        if entry is None:
            continue
        bucket = index.setdefault(team, {"skill": [], "defense": [], "line": []})
        if entry["pos"] in _ABSENCE_SKILL_POS:
            bucket["skill"].append(entry)
        elif entry["pos"] in _ABSENCE_DEF_POS:
            bucket["defense"].append(entry)
        elif entry["pos"] in _OL_POSITIONS:
            bucket["line"].append(entry)
    return index


def absence_notes(index: dict, team: str, opponent: str, *,
                  exclude_pid: str | None = None, max_defense: int = 3,
                  max_teammates: int = 4, max_linemen: int = 2) -> dict:
    """Display-only absence notes for one player's game.

    ``teammates``: seriously-hurt skill-position teammates (QB/RB/WR/TE),
    excluding the player themselves, capped at ``max_teammates``; then
    seriously-hurt starting offensive linemen (snap-share identified),
    sorted by name under their own ``max_linemen`` cap so they never
    consume skill slots. ``opponents``: seriously-hurt defenders on the
    opposing team, capped at ``max_defense`` so the note never gets noisy.
    Empty lists when nothing is notable.
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
        teammates = teammates[:max(0, int(max_teammates))]
        linemen = []
        for e in sorted((index or {}).get(team, {}).get("line", []) or [],
                        key=lambda e: e.get("name") or ""):
            if exclude is not None and e.get("pid") == exclude:
                continue
            linemen.append(e)
        teammates = teammates + linemen[:max(0, int(max_linemen))]
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
