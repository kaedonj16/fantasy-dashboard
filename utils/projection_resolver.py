"""Canonical projected-PPG resolution and provenance.

This module is the boundary between projection providers and application
features.  Consumers must not choose a provider or annualize a weekly number;
they ask for an explicit projection context and receive both the value and its
provenance.  Sleeper is authoritative.  A caller-supplied secondary value is
used only when Sleeper has no row, which keeps legacy/offline imports usable
without letting them outrank Sleeper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from statistics import median
from typing import Any, Mapping, Optional

from utils.fantasy_scoring import projection_points
from utils.proj_variant import pick_proj_variant

PROJECTION_CACHE_VERSION = "canonical-projection-v1"
SEASON_AVERAGE = "season_average"
WEEKLY = "weekly"


def scoring_fingerprint(settings: Optional[Mapping[str, Any]]) -> str:
    """Stable cache discriminator; explicit zeros and custom rates are retained."""
    normalized = {str(k): settings[k] for k in sorted(settings or {})}
    body = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(body.encode("utf-8")).hexdigest()[:16]


def projection_cache_key(player_id: str, season: int, scoring_settings=None,
                         projection_type: str = SEASON_AVERAGE,
                         week: Optional[int] = None, source_version: str = "sleeper") -> str:
    """Context-complete key; scoring formats and weekly/season values cannot collide."""
    return ":".join((PROJECTION_CACHE_VERSION, source_version, str(season),
                     projection_type, str(week or "season"),
                     scoring_fingerprint(scoring_settings), str(player_id)))


@dataclass(frozen=True)
class ProjectionResult:
    ppg: Optional[float]
    source: Optional[str]
    projection_type: str
    scoring_variant: str
    scoring_fingerprint: str
    season: int
    week: Optional[int]
    fallback_used: bool
    cache_version: str = PROJECTION_CACHE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _positive(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 2) if value > 0 else None


def _sleeper_week_value(entry, settings, pos="") -> Optional[float]:
    if entry is None:
        return None
    return _positive(projection_points(entry, dict(settings or {}), pos))


def resolve_projected_ppg(player_id: str, scoring_settings: Optional[Mapping] = None,
                          season: int = 2026, week: Optional[int] = None,
                          projection_type: str = SEASON_AVERAGE, *,
                          weekly_maps: Optional[Mapping[int, Mapping]] = None,
                          position: str = "", secondary_ppg=None,
                          conservative_ppg=None, sleeper_season_ppg=None) -> dict:
    """Resolve one explicit projection context using a uniform fallback order.

    ``weekly_maps`` is injectable to keep the kernel pure and testable.  Without
    it, cached Sleeper week files are loaded.  Season average is the median of
    positive active-week projections (byes do not become zero-point games).
    """
    if projection_type not in (SEASON_AVERAGE, WEEKLY):
        raise ValueError("projection_type must be 'season_average' or 'weekly'")
    if projection_type == WEEKLY and week is None:
        raise ValueError("week is required for a weekly projection")
    settings = dict(scoring_settings or {})
    pid = str(player_id)
    variant = pick_proj_variant(settings)
    if weekly_maps is None:
        from utils.utils import load_week_projection
        weeks = [int(week)] if projection_type == WEEKLY else list(range(1, 19))
        weekly_maps = {w: load_week_projection(int(season), w) or {} for w in weeks}
    weeks = [int(week)] if projection_type == WEEKLY else sorted(weekly_maps)
    values = [_sleeper_week_value((weekly_maps.get(w) or {}).get(pid), settings, position)
              for w in weeks]
    values = [v for v in values if v is not None]
    if values:
        ppg, source, fallback = (values[0] if projection_type == WEEKLY else round(median(values), 2)), "sleeper", False
    else:
        # Sleeper's season feed covers players absent from preseason week files.
        # It is still primary, not a provider fallback.
        ppg = _positive(sleeper_season_ppg) if projection_type == SEASON_AVERAGE else None
        source, fallback = ("sleeper", False) if ppg is not None else (None, True)
    if ppg is None:
        ppg = _positive(secondary_ppg)
        source, fallback = ("secondary", True) if ppg is not None else (None, True)
        if ppg is None:
            ppg = _positive(conservative_ppg)
            if ppg is not None:
                source = "conservative"
    return ProjectionResult(ppg, source, projection_type, variant,
                            scoring_fingerprint(settings), int(season),
                            int(week) if week is not None else None, fallback).to_dict()


def resolve_projected_ppg_many(player_ids, scoring_settings=None, season=2026,
                               week=None, projection_type=SEASON_AVERAGE, *,
                               weekly_maps=None, positions=None, secondary=None,
                               conservative=None) -> dict[str, dict]:
    """Bulk facade that loads Sleeper data once and delegates to the same kernel."""
    if weekly_maps is None:
        from utils.utils import load_week_projection
        weeks = [int(week)] if projection_type == WEEKLY else list(range(1, 19))
        weekly_maps = {w: load_week_projection(int(season), w) or {} for w in weeks}
    sleeper_season = {}
    if projection_type == SEASON_AVERAGE:
        # This compatibility fill is provider data, not a different authority.
        from data_building.fetch_projections import fetch_sleeper_season_ppg_variants
        by_player = fetch_sleeper_season_ppg_variants(int(season)) or {}
        variant = pick_proj_variant(dict(scoring_settings or {}))
        sleeper_season = {str(pid): (row or {}).get(variant) or (row or {}).get("ppr")
                          for pid, row in by_player.items()}
    return {str(pid): resolve_projected_ppg(
        str(pid), scoring_settings, season, week, projection_type,
        weekly_maps=weekly_maps, position=(positions or {}).get(str(pid), ""),
        secondary_ppg=(secondary or {}).get(str(pid)),
        conservative_ppg=(conservative or {}).get(str(pid)),
        sleeper_season_ppg=sleeper_season.get(str(pid))) for pid in player_ids}
