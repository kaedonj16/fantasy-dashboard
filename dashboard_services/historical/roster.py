"""Preseason roster-spot analog from ADP rank among teammates (pure).

Official depth charts are not a warehouse feature, and same-season snaps /
carries would leak. The preseason market already ranks a team's RBs (WRs,
TEs, QBs) by ADP: lowest ADP is the expected RB1, third-or-later is RB3+.

Missing ADP stays unknown and is omitted from the ranking. This module must
stay dependency-free (no pandas, Flask, nfl_data_py, or I/O).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    SKILL_POSITIONS,
    normalize_adp,
    normalize_team_abbr,
    _optional_int,
)

ROSTER_SPOT_STARTER = 1
ROSTER_SPOT_SECOND = 2
ROSTER_SPOT_DEPTH = 3
ROSTER_SPOTS: tuple[int, ...] = (
    ROSTER_SPOT_STARTER,
    ROSTER_SPOT_SECOND,
    ROSTER_SPOT_DEPTH,
)


def normalize_roster_spot(value: Any) -> Optional[int]:
    """Clamp a 1-based teammate rank to 1, 2, or 3 (3+). Unknown stays None."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip().lower().rstrip("+")
        n = _optional_int(text)
    else:
        n = _optional_int(value)
    if n is None or n <= 0:
        return None
    if n <= ROSTER_SPOT_STARTER:
        return ROSTER_SPOT_STARTER
    if n == ROSTER_SPOT_SECOND:
        return ROSTER_SPOT_SECOND
    return ROSTER_SPOT_DEPTH


def roster_spot_label(position: Any, spot: Any) -> str:
    """Display label like RB1 / RB3+. Never contains '_'."""
    pos = str(position or "").upper().strip() or "SK"
    n = normalize_roster_spot(spot)
    if n is None:
        return ""
    if n >= ROSTER_SPOT_DEPTH:
        return f"{pos}3+"
    return f"{pos}{n}"


def _adp_of(row: Mapping[str, Any]) -> Optional[float]:
    adp = normalize_adp(row.get("adp") if row.get("adp") is not None else row.get("adp_overall"))
    if adp is not None:
        return adp
    feats = row.get("feats") if isinstance(row.get("feats"), Mapping) else {}
    return normalize_adp(feats.get("adp") if feats.get("adp") is not None else feats.get("adp_overall"))


def _sort_key(row: Mapping[str, Any]) -> tuple:
    adp = _adp_of(row)
    return (
        float(adp) if adp is not None else 1e9,
        str(row.get("name") or ""),
        str(row.get("pid") or row.get("sleeper_id") or row.get("id") or ""),
    )


def rank_roster_spots(members: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Map member id → roster spot for one (season, pos, team) group.

    Members without ADP are omitted. Ties break on name, then id.
    """
    ranked = [row for row in members if isinstance(row, Mapping) and _adp_of(row) is not None]
    ranked.sort(key=_sort_key)
    out: dict[str, int] = {}
    for i, row in enumerate(ranked):
        pid = str(row.get("pid") or row.get("sleeper_id") or row.get("id") or "")
        if not pid:
            continue
        out[pid] = normalize_roster_spot(i + 1) or ROSTER_SPOT_DEPTH
    return out


def _stamp_group(members: Sequence[dict], *, on_feats: bool, overwrite: bool) -> int:
    ranked = [row for row in members if _adp_of(row) is not None]
    ranked.sort(key=_sort_key)
    stamped = 0
    for i, row in enumerate(ranked):
        spot = normalize_roster_spot(i + 1)
        if spot is None:
            continue
        if on_feats:
            feats = row.get("feats")
            if not isinstance(feats, dict):
                feats = {}
                row["feats"] = feats
            if not overwrite and feats.get("roster_spot") is not None:
                continue
            feats["roster_spot"] = spot
        else:
            if not overwrite and row.get("roster_spot") is not None:
                continue
            row["roster_spot"] = spot
        stamped += 1
    return stamped


def assign_observation_roster_spots(observations: Sequence[Any]) -> int:
    """Stamp ``feats.roster_spot`` from ADP rank among teammates that season.

    Existing values win. Groups are (season, position, team). Returns how many
    observations were newly stamped.
    """
    groups: dict[tuple, list[dict]] = {}
    for obs in observations or []:
        if not isinstance(obs, dict):
            continue
        feats = obs.get("feats")
        if not isinstance(feats, dict):
            continue
        if normalize_roster_spot(feats.get("roster_spot")) is not None:
            continue
        season = _optional_int(obs.get("season"))
        pos = str(obs.get("pos") or feats.get("position") or "").upper()
        team = normalize_team_abbr(feats.get("team") or obs.get("team"))
        if season is None or pos not in SKILL_POSITIONS or not team:
            continue
        if _adp_of(obs) is None:
            continue
        groups.setdefault((season, pos, team), []).append(obs)
    stamped = 0
    for members in groups.values():
        stamped += _stamp_group(members, on_feats=True, overwrite=False)
    return stamped


def stamp_roster_spots_on_queries(queries: Sequence[Any]) -> int:
    """Overwrite live ``roster_spot`` from current-board ADP among teammates.

    Live ADP is the preseason signal for the upcoming season. Queries without
    team, position, or ADP are left unknown. Returns how many were stamped.
    """
    groups: dict[tuple, list[dict]] = {}
    for query in queries or []:
        if not isinstance(query, dict):
            continue
        pos = str(query.get("position") or "").upper()
        team = normalize_team_abbr(query.get("team") or query.get("nfl_team"))
        if pos not in SKILL_POSITIONS or not team:
            query.pop("roster_spot", None)
            continue
        if _adp_of(query) is None:
            query.pop("roster_spot", None)
            continue
        groups.setdefault((pos, team), []).append(query)
    stamped = 0
    for members in groups.values():
        stamped += _stamp_group(members, on_feats=False, overwrite=True)
    return stamped
