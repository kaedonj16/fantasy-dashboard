"""Authoritative NFL season/week context and cache dimensions.

The provider state is authoritative.  Calendar inference exists only as an
observable availability fallback; importantly, January and February still
belong to the season which began in the previous calendar year.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional


VALID_PHASES = frozenset({"off", "pre", "reg", "post"})


def calendar_nfl_season(on_date: Optional[date] = None) -> int:
    """Return the NFL season containing *on_date* when provider state is absent."""
    day = on_date or datetime.now(timezone.utc).date()
    return day.year - 1 if day.month <= 2 else day.year


def _calendar_phase(day: date) -> str:
    if day.month <= 2:
        return "post"
    if day.month <= 7:
        return "off"
    if day.month == 8:
        return "pre"
    return "reg"


def normalize_nfl_state(
    state: Optional[Mapping[str, Any]], *, on_date: Optional[date] = None,
    provider: str = "sleeper",
) -> dict:
    """Normalize provider state and attach lightweight freshness provenance."""
    raw = dict(state or {})
    day = on_date or datetime.now(timezone.utc).date()
    try:
        season = int(raw.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    fallback_reason = None
    if season < 2000:
        season = calendar_nfl_season(day)
        fallback_reason = "provider season unavailable"
    try:
        week = max(0, int(raw.get("week") or raw.get("display_week") or 0))
    except (TypeError, ValueError):
        week = 0
    phase = str(raw.get("season_type") or "").strip().lower()
    if phase not in VALID_PHASES:
        phase = _calendar_phase(day)
        fallback_reason = fallback_reason or "provider phase unavailable"
    raw.update({"season": season, "week": week, "season_type": phase})
    raw["freshness"] = {
        "source_season": season,
        "source_week": week,
        "provider": provider,
        "classification": "live" if fallback_reason is None else "fallback",
        "fallback_reason": fallback_reason,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    return raw


def season_cache_key(namespace: str, *, season: int, week: Optional[int] = None,
                     league_id: Optional[str] = None, scoring: Optional[str] = None,
                     provider: Optional[str] = None) -> str:
    """Build a stable key which cannot accidentally cross season boundaries."""
    parts = [namespace, f"s{int(season)}"]
    if week is not None:
        parts.append(f"w{int(week)}")
    if provider:
        parts.append(f"p:{provider}")
    if league_id:
        parts.append(f"l:{league_id}")
    if scoring:
        parts.append(f"sc:{scoring}")
    return "|".join(parts)


def current_sample_weight(games: int, *, full_weight_at: int = 6) -> float:
    """Intentional early-season blend weight for observed current-year data."""
    return min(1.0, max(0.0, float(games) / max(1, int(full_weight_at))))
