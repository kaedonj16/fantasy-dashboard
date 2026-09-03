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


def _int(value, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def has_format_signal(
    roster_positions: Optional[list] = None,
    settings: Optional[dict] = None,
) -> bool:
    """True when we can tell 1QB vs Superflex from slots or settings."""
    if roster_positions:
        return True
    settings = settings or {}
    return any(
        key in settings
        for key in ("slots_super_flex", "slots_sf", "slots_qb")
    )


def is_sf_from_league(
    roster_positions: Optional[list] = None,
    settings: Optional[dict] = None,
) -> bool:
    """Superflex from roster slots and/or Sleeper ``slots_super_flex``."""
    try:
        from utils.lineup_slots import is_superflex_lineup
        if is_superflex_lineup(roster_positions or []):
            return True
    except Exception:
        pass
    settings = settings or {}
    return _int(settings.get("slots_super_flex") or settings.get("slots_sf")) > 0


def fields_from_provider_league(league: Optional[dict]) -> Dict[str, Any]:
    """Name / size / slots / SF from a Sleeper-shaped league dict."""
    league = league if isinstance(league, dict) else {}
    settings = league.get("settings") if isinstance(league.get("settings"), dict) else {}
    positions = league.get("roster_positions") or []
    if not isinstance(positions, list):
        positions = []
    size = _int(
        league.get("total_rosters")
        or settings.get("num_teams")
        or settings.get("teams")
    )
    return {
        "name": str(league.get("name") or "").strip(),
        "size": size,
        "roster_positions": positions,
        "settings": settings,
        "is_sf": is_sf_from_league(positions, settings),
        "has_format": has_format_signal(positions, settings),
    }


def build_league_chrome(
    *,
    name: str = "",
    size: int = 0,
    roster_positions: Optional[list] = None,
    week: int = 0,
    season_type: str = "",
    offseason: bool = False,
    is_sf: Optional[bool] = None,
    format_known: Optional[bool] = None,
    settings: Optional[dict] = None,
) -> Dict[str, Any]:
    """Dict the nav chip and ``__brctx`` both consume.

    When slot data is missing, ``format`` stays empty instead of inventing
    ``1QB`` — a Superflex league must not flash the 1QB fallback.
    """
    positions = roster_positions or []
    if is_sf is None:
        is_sf = is_sf_from_league(positions, settings)
    if format_known is None:
        format_known = has_format_signal(positions, settings) or is_sf is True
    size_n = _int(size)
    week_n = _int(week)
    fmt = format_label(size_n, bool(is_sf)) if format_known else ""
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


def merge_chrome_sources(
    *,
    ctx: Optional[dict] = None,
    saved_name: str = "",
    provider_league: Optional[dict] = None,
    week: int = 0,
    season_type: str = "",
    offseason: bool = False,
) -> Dict[str, Any]:
    """Combine dashboard cache, a live provider league, and a saved name."""
    ctx = ctx if isinstance(ctx, dict) else {}
    live = fields_from_provider_league(provider_league)
    ctx_league = ctx.get("league") if isinstance(ctx.get("league"), dict) else {}
    ctx_settings = ctx.get("settings") if isinstance(ctx.get("settings"), dict) else {}
    if not ctx_settings and isinstance(ctx_league.get("settings"), dict):
        ctx_settings = ctx_league["settings"]
    ctx_positions = ctx.get("roster_positions") or ctx_league.get("roster_positions") or []
    if not isinstance(ctx_positions, list):
        ctx_positions = []
    name = (
        str(ctx_league.get("name") or "").strip()
        or live["name"]
        or str(saved_name or "").strip()
    )
    size = _int(
        ctx.get("total_rosters")
        or ctx_league.get("total_rosters")
        or (len(ctx.get("rosters") or []) if ctx.get("rosters") else 0)
        or live["size"]
    )
    positions = ctx_positions or live["roster_positions"]
    settings = ctx_settings or live["settings"]
    known = has_format_signal(positions, settings)
    is_sf = is_sf_from_league(positions, settings) if known else False
    return build_league_chrome(
        name=name,
        size=size,
        roster_positions=positions,
        week=week,
        season_type=season_type,
        offseason=offseason,
        is_sf=is_sf if known else False,
        format_known=known,
        settings=settings,
    )
