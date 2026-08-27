"""Shared league-chrome labels: name, format, and week.

The top nav chip (and ``window.__brctx``) use this so every page reads the same
league + week instead of restating them in page titles.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def format_label(size: int, is_sf: bool) -> str:
    """e.g. ``12tm SF`` or ``10tm 1QB``. Size omitted when unknown."""
    kind = "SF" if is_sf else "1QB"
    try:
        n = int(size or 0)
    except (TypeError, ValueError):
        n = 0
    if n >= 2:
        return f"{n}tm {kind}"
    return kind


def week_label(week: int, *, season_type: str = "", offseason: bool = False) -> str:
    """Keep week and season-state text out of the persistent league chrome."""
    return ""


def build_league_chrome(
    *,
    name: str = "",
    size: int = 0,
    roster_positions: Optional[list] = None,
    week: int = 0,
    season_type: str = "",
    offseason: bool = False,
    is_sf: Optional[bool] = None,
) -> Dict[str, Any]:
    """Dict the nav chip and ``__brctx`` both consume."""
    if is_sf is None:
        try:
            from utils.lineup_slots import is_superflex_lineup
            is_sf = is_superflex_lineup(roster_positions or [])
        except Exception:
            is_sf = False
    try:
        size_n = int(size or 0)
    except (TypeError, ValueError):
        size_n = 0
    try:
        week_n = int(week or 0)
    except (TypeError, ValueError):
        week_n = 0
    fmt = format_label(size_n, bool(is_sf))
    wk = week_label(week_n, season_type=season_type, offseason=offseason)
    display = (name or "").strip() or "This league"
    return {
        "name": display,
        "raw_name": (name or "").strip(),
        "week": week_n,
        "week_label": wk,
        "size": size_n if size_n >= 2 else 0,
        "sf": bool(is_sf),
        "format": fmt,
    }
