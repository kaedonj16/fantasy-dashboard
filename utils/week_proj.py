"""Week-projection map helpers (Flask-free).

Matchups and Scout both need to unwrap ``proj_by_week[week]`` into a pid →
value map. Keep this module free of ``dashboard_services.api`` / Flask so unit
jobs that only install ruff + pytest can still exercise Scout.
"""
from __future__ import annotations

from typing import Any, Dict


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
