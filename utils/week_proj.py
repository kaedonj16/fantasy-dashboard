"""Week-projection map helpers (Flask-free).

Matchups and Scout both need to unwrap ``proj_by_week[week]`` into a pid →
value map. Keep this module free of ``dashboard_services.api`` / Flask so unit
jobs that only install ruff + pytest can still exercise Scout.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


def week_proj_map_from_bundles(projections: Any, week: Any) -> Dict[str, Any]:
    """Unwrap ``proj_by_week[week]`` into a pid → value map.

    ``build_projections_by_week`` stores ``{week: {"projections": {pid: float}}}``.
    Some callers historically passed a flat map or used string week keys; Scout
    also falls back to the raw multi-variant file. Accept all of those shapes so
    Matchup Preview never silently shows wall-to-wall ``0.0``.
    """
    if not isinstance(projections, dict):
        return {}
    container = projections.get(week)
    if container is None:
        try:
            container = projections.get(int(week))
        except (TypeError, ValueError):
            container = None
    if container is None:
        container = projections.get(str(week))
    if not isinstance(container, dict):
        return {}
    nested = container.get("projections")
    if isinstance(nested, dict):
        return nested
    # Flat pid → float (or raw multi-variant entries). Drop meta keys.
    return {k: v for k, v in container.items() if k not in ("projections", "_available")}


def _read_week_projection_disk(season: int, week: int) -> dict:
    """Read a cached week file only. Never fetch — team-modal must stay snappy."""
    try:
        from utils.utils import path_week_proj
        path = Path(path_week_proj(int(season), int(week)))
        if not path.exists() or path.stat().st_size < 16:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _pos_of(pos_by_pid: Optional[Mapping], pid: str) -> str:
    meta = (pos_by_pid or {}).get(pid)
    if isinstance(meta, dict):
        return str(meta.get("pos") or meta.get("position") or "")
    return str(meta or "")


def league_player_week_projections(
    season: int,
    weeks: int,
    player_ids: Iterable,
    scoring_settings: Optional[Mapping] = None,
    pos_by_pid: Optional[Mapping] = None,
    *,
    load_week: Optional[Callable[[int, int], dict]] = None,
) -> Dict[str, Dict[str, float]]:
    """``{week: {pid: pts}}`` for league roster players.

    Disk-cached Sleeper weekly projections, scored with the league's settings
    so the schedule-tab lineup rows match Matchup Preview / Sleeper. Missing
    weeks (file omitted a player) fall back to that player's median of other
    weeks; an explicit 0 (bye / inactive) is kept. Never invents RNG points.
    """
    from utils.fantasy_scoring import weekly_projection_points

    load = load_week or _read_week_projection_disk
    wanted = []
    seen = set()
    for raw in player_ids or []:
        pid = str(raw or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        wanted.append(pid)
    if not wanted or int(weeks) < 1:
        return {}

    settings = dict(scoring_settings or {})
    raw_weeks: Dict[int, Dict[str, Optional[float]]] = {}
    known: Dict[str, list] = {pid: [] for pid in wanted}
    for w in range(1, int(weeks) + 1):
        try:
            file_map = load(int(season), w) or {}
        except Exception:
            file_map = {}
        week_vals: Dict[str, Optional[float]] = {}
        for pid in wanted:
            pts = weekly_projection_points(
                file_map, pid, settings, _pos_of(pos_by_pid, pid),
            )
            week_vals[pid] = None if pts is None else round(float(pts), 1)
            if pts is not None and float(pts) > 0.5:
                known[pid].append(float(pts))
        raw_weeks[w] = week_vals

    fallback = {
        pid: round(float(median(vals)), 1)
        for pid, vals in known.items() if vals
    }
    out: Dict[str, Dict[str, float]] = {}
    for w, week_vals in raw_weeks.items():
        filled = {}
        for pid, pts in week_vals.items():
            if pts is not None:
                filled[pid] = pts
            elif pid in fallback:
                filled[pid] = fallback[pid]
        if filled:
            out[str(w)] = filled
    return out
