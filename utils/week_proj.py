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


def flatten_proj_by_week(proj_by_week: Any) -> Dict[str, Dict[str, float]]:
    """``{week: {pid: float}}`` from the matchups-page ``proj_by_week`` bundles."""
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(proj_by_week, dict):
        return out
    for week in proj_by_week:
        if str(week).startswith("_"):
            continue
        try:
            w = int(week)
        except (TypeError, ValueError):
            continue
        filled: Dict[str, float] = {}
        for pid, val in (week_proj_map_from_bundles(proj_by_week, w) or {}).items():
            try:
                filled[str(pid)] = float(val)
            except (TypeError, ValueError):
                continue
        if filled:
            out[str(w)] = filled
    return out


def matchup_proj_pts(week_proj_map: Optional[Mapping], pid: Any) -> float:
    """Primary pid lookup used by the matchups page (``_proj_value_for_pid``)."""
    if pid is None or not isinstance(week_proj_map, Mapping):
        return 0.0
    key = str(pid)
    if key in week_proj_map:
        try:
            return float(week_proj_map.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if pid in week_proj_map and pid != key:
        try:
            return float(week_proj_map.get(pid) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


_SLOT_LABELS = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K",
    "DEF": "DEF", "DST": "DEF", "FLEX": "FLEX",
    "WRRB_FLEX": "W/R", "WRRB": "W/R", "RB_WR": "W/R",
    "REC_FLEX": "W/T", "WRTE_FLEX": "W/T",
    "SUPER_FLEX": "SFLEX", "SUPERFLEX": "SFLEX", "QB_WR_RB_TE": "SFLEX",
    "IDP_FLEX": "IDP", "DL": "DL", "LB": "LB", "DB": "DB",
    "DE": "DE", "DT": "DT", "CB": "CB", "S": "S",
}


def _slot_label(code: Any) -> str:
    key = str(code or "").strip().upper()
    return _SLOT_LABELS.get(key, key or "—")


def _blank_starter(label: str = "—") -> Dict[str, Any]:
    return {"id": None, "name": "—", "pos": label or "FLEX", "label": label or "—", "points": 0.0}


def _starter_from_matchup(starter: Any, week_proj_map: Mapping, label: str) -> Dict[str, Any]:
    if not isinstance(starter, dict):
        return _blank_starter(label)
    pid = starter.get("pid")
    if pid is None:
        pid = starter.get("id")
    if pid is None:
        pid = starter.get("player_id")
    if pid is None or str(pid).strip() in ("", "0"):
        return _blank_starter(label)
    pos = str(starter.get("pos") or label or "FLEX")
    pts = round(matchup_proj_pts(week_proj_map, pid), 1)
    return {
        "id": str(pid),
        "name": str(starter.get("name") or "—"),
        "pos": pos,
        "label": label or pos or "—",
        "points": pts,
    }


def team_schedule_from_matchups(
    roster_id: Any,
    matchups_by_week: Optional[Mapping],
    proj_by_week: Optional[Mapping] = None,
    starter_slots: Optional[Iterable] = None,
) -> list:
    """One schedule row per week from the same matchup previews the matchups page uses.

    Starters are the actual weekly lineup (not a greedy optimal), and points come
    from ``proj_by_week`` via the same pid lookup as Matchup Preview.
    """
    rid = str(roster_id)
    slots = [str(s) for s in (starter_slots or [])]
    by_week = matchups_by_week if isinstance(matchups_by_week, dict) else {}
    weeks: list = []
    seen = set()
    ordered = []
    for key in by_week:
        try:
            w = int(key)
        except (TypeError, ValueError):
            continue
        if w in seen:
            continue
        seen.add(w)
        ordered.append(w)
    for week in sorted(ordered):
        rows = by_week.get(week)
        if rows is None:
            rows = by_week.get(str(week))
        me_team = None
        opp_team = None
        for m in rows or []:
            if not isinstance(m, dict):
                continue
            left = m.get("left") or {}
            right = m.get("right") or {}
            if str(left.get("roster_id")) == rid:
                me_team, opp_team = left, right
                break
            if str(right.get("roster_id")) == rid:
                me_team, opp_team = right, left
                break
        if me_team is None:
            continue
        opp_team = opp_team if isinstance(opp_team, dict) else {}
        week_map = week_proj_map_from_bundles(proj_by_week or {}, week)
        me_raw = list(me_team.get("starters") or [])
        opp_raw = list(opp_team.get("starters") or [])
        n = max(len(me_raw), len(opp_raw), len(slots))
        me_starters = []
        opp_starters = []
        for i in range(n):
            label = _slot_label(slots[i]) if i < len(slots) else ""
            me_p = me_raw[i] if i < len(me_raw) else None
            opp_p = opp_raw[i] if i < len(opp_raw) else None
            if not label:
                pos = ""
                if isinstance(me_p, dict):
                    pos = str(me_p.get("pos") or "")
                if not pos and isinstance(opp_p, dict):
                    pos = str(opp_p.get("pos") or "")
                label = pos or "—"
            me_starters.append(_starter_from_matchup(me_p, week_map, label))
            opp_starters.append(_starter_from_matchup(opp_p, week_map, label))
        me_total = round(sum(float(s.get("points") or 0) for s in me_starters), 1)
        opp_total = round(sum(float(s.get("points") or 0) for s in opp_starters), 1)
        opp_rid = opp_team.get("roster_id")
        weeks.append({
            "week": week,
            "me": {
                "roster_id": str(me_team.get("roster_id") or rid),
                "name": me_team.get("name") or "",
                "starters": me_starters,
                "total": me_total,
            },
            "opp": {
                "roster_id": str(opp_rid) if opp_rid is not None else None,
                "name": opp_team.get("name") or "BYE",
                "starters": opp_starters,
                "total": opp_total,
            },
        })
    return weeks

