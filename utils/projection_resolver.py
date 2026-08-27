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
import logging
from statistics import median
from typing import Any, Mapping, Optional

from utils.fantasy_scoring import projection_points
from utils.proj_variant import pick_proj_variant

PROJECTION_CACHE_VERSION = "canonical-projection-v2"
SEASON_AVERAGE = "season_average"
WEEKLY = "weekly"
POINTS_PER_GAME = "points_per_game"
POINTS = "points"
_LOG = logging.getLogger(__name__)

# These are corruption detectors, not fantasy-performance caps. Even extreme
# historical K/DST weeks remain far below this value; a 50-150 value in a PPG
# field is overwhelmingly a season total. Skill-position projections use a
# wider guard solely to reject obvious unit/category corruption.
_MAX_PLAUSIBLE_PPG = {"K": 30.0, "DEF": 40.0}
_DEFAULT_MAX_PLAUSIBLE_PPG = 80.0


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
    source_projection_type: Optional[str] = None
    unit: str = POINTS_PER_GAME
    position: str = ""
    season_points: Optional[float] = None
    projected_games: Optional[float] = None
    cache_version: str = PROJECTION_CACHE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _positive(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 2) if value > 0 else None


def _valid_ppg(value, pos="", *, player_id="", origin="") -> Optional[float]:
    value = _positive(value)
    if value is None:
        return None
    normalized_pos = str(pos or "").upper()
    ceiling = _MAX_PLAUSIBLE_PPG.get(normalized_pos, _DEFAULT_MAX_PLAUSIBLE_PPG)
    if value > ceiling:
        _LOG.warning("Rejected implausible projected PPG (possible season-total unit): "
                     "player=%s pos=%s value=%s origin=%s", player_id,
                     normalized_pos or "unknown", value, origin or "unknown")
        return None
    return value


def _sleeper_week_value(entry, settings, pos="", player_id="") -> Optional[float]:
    if entry is None:
        return None
    return _valid_ppg(projection_points(entry, dict(settings or {}), pos), pos,
                      player_id=player_id, origin="sleeper_week")


def _season_total_projection(entry, settings, pos="", player_id=""):
    """Return ``(ppg, season_points, projected_games)`` from explicit totals.

    Sleeper occasionally reports ``gp=18`` with the bye included. Reuse the
    fetch pipeline's active-game semantics rather than blindly dividing by 17.
    """
    if not isinstance(entry, Mapping):
        return None, None, None
    from data_building.fetch_projections import season_games_for_ppg
    # projection_points uses Sleeper's published pts_* total for standard
    # formats and centrally scores preserved raw season stats for custom rules.
    raw_stats = entry.get("raw_stats")
    if not isinstance(raw_stats, Mapping) and any(entry.get(k) is not None for k in
                                                  ("pts_ppr", "pts_half_ppr", "pts_std")):
        raw_stats = entry
    score_entry = {"raw_stats": raw_stats} if isinstance(raw_stats, Mapping) else entry
    points = _positive(projection_points(score_entry, dict(settings or {}), pos))
    if points is None:
        return None, None, None
    games = season_games_for_ppg(entry.get("gp"))
    ppg = _valid_ppg(points / games, pos, player_id=player_id,
                     origin=f"sleeper_season/{games:g}")
    return ppg, round(points, 2), games


def resolve_projected_ppg(player_id: str, scoring_settings: Optional[Mapping] = None,
                          season: int = 2026, week: Optional[int] = None,
                          projection_type: str = SEASON_AVERAGE, *,
                          weekly_maps: Optional[Mapping[int, Mapping]] = None,
                          position: str = "", secondary_ppg=None,
                          conservative_ppg=None, sleeper_season_ppg=None,
                          sleeper_season_entry=None) -> dict:
    """Resolve one explicit projection context using a uniform fallback order.

    ``weekly_maps`` is injectable to keep the kernel pure and testable.  Without
    it, cached Sleeper week files are loaded. Season average uses the Sleeper
    season product first; a positive-week median is only its explicit fallback.
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
    values = [_sleeper_week_value((weekly_maps.get(w) or {}).get(pid), settings, position, pid)
              for w in weeks]
    values = [v for v in values if v is not None]
    season_points = projected_games = None
    ppg = source = source_projection_type = None
    fallback = True
    if projection_type == SEASON_AVERAGE:
        # Strict authority: Sleeper's season product outranks any aggregation of
        # weekly products. Weekly-derived PPG exists only as a source fallback.
        ppg, season_points, projected_games = _season_total_projection(
            sleeper_season_entry, settings, position, pid)
        if ppg is not None:
            source, source_projection_type, fallback = "sleeper", "sleeper_season", False
        elif sleeper_season_entry is None and sleeper_season_ppg is not None:
            # Unit-safe compatibility input from older callers, never preferred
            # over an explicit season stat line.
            ppg = _valid_ppg(sleeper_season_ppg, position, player_id=pid,
                             origin="sleeper_season_ppg")
            if ppg is not None:
                source, source_projection_type, fallback = "sleeper", "sleeper_season", False
        if ppg is None and values:
            ppg = round(median(values), 2)
            source, source_projection_type, fallback = "sleeper", "sleeper_weekly_derived", True
    elif values:
        ppg = values[0]
        source, source_projection_type, fallback = "sleeper", "sleeper_week", False
    if ppg is None:
        ppg = _valid_ppg(secondary_ppg, position, player_id=pid, origin="secondary_ppg")
        source, source_projection_type, fallback = (("secondary", projection_type, True)
                                                    if ppg is not None else (None, None, True))
        if ppg is None:
            ppg = _valid_ppg(conservative_ppg, position, player_id=pid,
                             origin="conservative_ppg")
            if ppg is not None:
                source, source_projection_type = "conservative", projection_type
    return ProjectionResult(ppg, source, projection_type, variant,
                            scoring_fingerprint(settings), int(season),
                            int(week) if week is not None else None, fallback,
                            source_projection_type=source_projection_type,
                            position=str(position or "").upper(),
                            season_points=season_points,
                            projected_games=projected_games).to_dict()


def resolve_projected_ppg_many(player_ids, scoring_settings=None, season=2026,
                               week=None, projection_type=SEASON_AVERAGE, *,
                               weekly_maps=None, positions=None, secondary=None,
                               conservative=None) -> dict[str, dict]:
    """Bulk facade that loads Sleeper data once and delegates to the same kernel."""
    if weekly_maps is None:
        from utils.utils import load_week_projection
        weeks = [int(week)] if projection_type == WEEKLY else list(range(1, 19))
        weekly_maps = {w: load_week_projection(int(season), w) or {} for w in weeks}
    sleeper_season_lines = {}
    if projection_type == SEASON_AVERAGE:
        # This compatibility fill is provider data, not a different authority.
        from data_building.fetch_projections import load_sleeper_season_stat_lines
        sleeper_season_lines = load_sleeper_season_stat_lines(int(season)) or {}
    return {str(pid): resolve_projected_ppg(
        str(pid), scoring_settings, season, week, projection_type,
        weekly_maps=weekly_maps, position=(positions or {}).get(str(pid), ""),
        secondary_ppg=(secondary or {}).get(str(pid)),
        conservative_ppg=(conservative or {}).get(str(pid)),
        sleeper_season_entry=sleeper_season_lines.get(str(pid))) for pid in player_ids}
