"""Consolidated utils module: projections.

fantasy scoring, projection variants/resolution, qualification

Merged from: utils/fantasy_scoring.py, utils/proj_variant.py, utils/projection_resolver.py, utils/week_proj.py, utils/model_confidence.py, utils/evaluation_metrics.py, utils/season_qualification.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING
from dashboard_services.api import (
    _fetch_league,
    get_nfl_games_for_week_raw,
    get_transactions,
    get_rosters,
    get_users,
    get_traded_picks,
    get_nfl_state,
    get_nfl_players, fetch_team_game_logs_html, fetch_tank_boxscore, get_matchups,
)
from dashboard_services.display_names import team_label_from_user, username_from_user


# ======================================================================
# From utils/fantasy_scoring.py
# ======================================================================

"""Pure fantasy-points scoring from a Sleeper stats line.

Extracted from app.py so the points calculation can be unit-tested without the
pandas/DB stack. Uses Sleeper's stat/scoring key names; every scoring value
falls back to a standard PPR-ish default when the league omits it.
"""


_DEFAULT_RATES = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rec": 0.0,
    "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}


def completed_points_summary(points: list[float] | tuple[float, ...]) -> dict | None:
    """Aggregate actual appearances without rounding drift.

    An empty sequence means no appearance (missing/N/A), while ``[0.0]`` is a
    genuine score and remains distinguishable from missing data.
    """
    if not points:
        return None
    total = sum(float(p) for p in points)
    return {"games": len(points), "total": total, "ppg": total / len(points)}


def _rate(settings: dict, key: str) -> float:
    """Respect explicit zero scoring; only use defaults when a key is absent."""
    value = settings[key] if key in settings else _DEFAULT_RATES.get(key, 0.0)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def score_stats(s: dict, ss: dict, pos: str = "") -> float:
    """Compute points from a projected Sleeper stat line and league settings.

    In addition to the common defaults, every projected stat with an exact
    scoring-settings key is included (first downs, two-point conversions,
    returns, completions, sacks, etc.). Milestone bonuses are applied once.
    """
    s = s or {}
    ss = ss or {}
    p = 0.0
    handled = set(_DEFAULT_RATES)
    for key in handled:
        p += float(s.get(key) or 0) * _rate(ss, key)
    # Sleeper uses matching names for most custom stat/rate pairs.
    for key, value in s.items():
        if key in handled or key.startswith("bonus_") or key not in ss:
            continue
        try:
            p += float(value or 0) * float(ss.get(key) or 0)
        except (TypeError, ValueError):
            continue
    if str(pos).upper() == "TE":
        p += float(s.get("rec") or 0) * _rate(ss, "bonus_rec_te")
    py = s.get("pass_yd") or 0
    ry = s.get("rush_yd") or 0
    ey = s.get("rec_yd") or 0
    rr = ry + ey
    if py >= 400: p += (ss.get("bonus_pass_yd_400") or 0)
    elif py >= 300: p += (ss.get("bonus_pass_yd_300") or 0)
    if ry >= 200: p += (ss.get("bonus_rush_yd_200") or 0)
    elif ry >= 100: p += (ss.get("bonus_rush_yd_100") or 0)
    if ey >= 200: p += (ss.get("bonus_rec_yd_200") or 0)
    elif ey >= 100: p += (ss.get("bonus_rec_yd_100") or 0)
    if rr >= 200: p += (ss.get("bonus_rush_rec_yd_200") or 0)
    elif rr >= 100: p += (ss.get("bonus_rush_rec_yd_100") or 0)
    return p


# week_stats box-score lines use plural yardage keys and ``int`` for
# interceptions; score_stats speaks Sleeper's names. Remap before scoring so the
# league's own rates, bonuses, and TE premium stay consistent.
_WEEK_STATS_TO_SLEEPER = {
    "pass_yds": "pass_yd",
    "rush_yds": "rush_yd",
    "rec_yds": "rec_yd",
    "int": "pass_int",
}


def week_stats_line_points(entry: dict, ss: dict, pos: str = "") -> float | None:
    """Fantasy points implied by a ``week_stats`` box-score line.

    The matchup slide shows two numbers from two feeds: the authoritative live
    points (Sleeper ``players_points``) and a human-readable box-score line
    (Footballguys, Tank01-overlaid). When Footballguys is still republishing a
    prior week/season, the box-score line can contradict the points. Scoring the
    line lets callers detect that contradiction and hide the stale line.

    Returns None when the entry has no numeric stats to score.
    """
    if not isinstance(entry, dict):
        return None
    remapped: dict = {}
    for key, value in entry.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        remapped[_WEEK_STATS_TO_SLEEPER.get(key, key)] = value
    if not remapped:
        return None
    return score_stats(remapped, ss or {}, pos)


def _sleeper_standard_points(raw: dict, ss: dict):
    """Sleeper's own projected total for a *standard* PPR/half/std league.

    Sleeper's projections payload carries its own computed totals
    (``pts_ppr`` / ``pts_half_ppr`` / ``pts_std``) alongside the raw stat line,
    so for a plain-scoring league we display those verbatim and match the
    Sleeper app exactly instead of recomputing (which can drift on interception
    rates, rounding, and category coverage).

    Returns None — so the caller recomputes — whenever the league's scoring has
    anything Sleeper's standard totals don't reflect: a non-standard reception
    value, a passing-TD value other than 4, a TE premium, or any yardage-
    milestone / first-down bonus. Those leagues are genuinely custom and must be
    scored from the raw line.
    """
    ss = ss or {}
    if "rec" not in ss:
        return None
    try:
        rec = float(ss.get("rec"))
    except (TypeError, ValueError):
        return None
    pts_key = {1.0: "pts_ppr", 0.5: "pts_half_ppr", 0.0: "pts_std"}.get(rec)
    if pts_key is None:
        return None  # custom reception value (e.g. 0.75) → recompute
    try:
        if float(ss.get("pass_td", 4.0) or 0) != 4.0:
            return None  # 6pt (or other) passing TD → recompute
    except (TypeError, ValueError):
        return None
    # Compare league rates on settings, not just stats present on this player.
    # Yahoo/Flea/MFL extras used to keep pts_ppr whenever the WR line omitted
    # the custom category (INT, fumbles, 6-pt TDs already handled above).
    for stat_key, standard_rate in _DEFAULT_RATES.items():
        if stat_key == "rec" or stat_key not in ss:
            continue
        try:
            if float(ss.get(stat_key) or 0) != float(standard_rate):
                return None
        except (TypeError, ValueError):
            return None
    # Any active yardage-milestone / first-down / TE-premium bonus makes
    # Sleeper's standard total wrong for this league.
    for k, v in ss.items():
        if not (k.startswith("bonus_") or k in ("pass_fd", "rush_fd", "rec_fd")):
            continue
        try:
            if float(v or 0) != 0:
                return None
        except (TypeError, ValueError):
            continue
    val = raw.get(pts_key)
    if val is None:
        return None  # payload lacks the precomputed total → recompute
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def projection_points(entry: dict, scoring_settings: dict, pos: str = "") -> float:
    """Select exact league scoring from a cached multi-variant projection."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, dict):
        return 0.0
    raw = entry.get("raw_stats")
    if isinstance(raw, dict):
        # Standard PPR/half/std leagues: show Sleeper's own projected total so the
        # number matches the Sleeper app exactly. Custom scoring falls through.
        sleeper_pts = _sleeper_standard_points(raw, scoring_settings or {})
        if sleeper_pts is not None:
            return round(sleeper_pts, 2)
        if scoring_settings:
            return round(score_stats(raw, scoring_settings, pos), 2)
    variant = pick_proj_variant(scoring_settings or {})
    return float(entry.get(variant) or entry.get("ppr") or 0.0)


def week_stat_points(stats, scoring_settings=None, pos: str = "") -> float:
    """Fantasy points for a played (or projected) stat line under league settings.

    Start/Sit L4 / season PPG used to read ``pts_ppr`` for every league, so an
    ESPN standard or half-PPR roster showed PPR form next to league-scored
    projections. Reuse the same selection as ``projection_points``.
    """
    if not isinstance(stats, dict):
        return 0.0
    try:
        return float(projection_points({"raw_stats": stats}, scoring_settings or {}, pos) or 0)
    except (TypeError, ValueError):
        return 0.0


def weekly_projection_points(week_map, pid, scoring_settings=None, pos: str = ""):
    """Points for one player from a cached weekly projection map.

    Honors Sleeper's own published totals for plain PPR/half/std leagues and
    recomputes from the raw stat line for custom scoring (see projection_points).
    Returns None when the player is absent from the map, so callers can tell
    "no projection" apart from a real zero (bye / inactive).
    """
    if not isinstance(week_map, dict):
        return None
    entry = week_map.get(str(pid))
    if entry is None:
        entry = week_map.get(pid)
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        return projection_points(entry, scoring_settings or {}, pos)
    return None


# ======================================================================
# From utils/proj_variant.py
# ======================================================================

"""Projection-variant selection.

Extracted from utils/utils.py so this pure logic can be unit-tested without
importing that module's heavier dependencies (requests / bs4 / dashboard_services).
Given a league's raw Sleeper scoring settings, returns the key of the projection
set that matches its scoring — reception points, TE premium, and passing-TD value.
"""


def pick_proj_variant(raw_sleeper_settings: dict) -> str:
    """
    Return the projection variant key that matches a league's scoring settings.
    Keys: ppr | half_ppr | std | tep | 6pt_ppr | 6pt_half | 6pt_tep
    """
    s = raw_sleeper_settings or {}
    rec      = float(s.get("rec", 1.0))
    te_bonus = float(s.get("bonus_rec_te", 0.0))
    pass_td  = float(s.get("pass_td", 4.0))

    tep   = te_bonus >= 0.25
    six   = pass_td >= 5.5

    if rec >= 1.0:
        base = "ppr"
    elif rec >= 0.4:
        base = "half_ppr"
    else:
        base = "std"

    if six and tep and base == "ppr":
        return "6pt_tep"
    if six and base == "ppr":
        return "6pt_ppr"
    if six and base == "half_ppr":
        return "6pt_half"
    if tep and base == "ppr":
        return "tep"
    return base


def pick_proj_variant_from_draft_scoring(scoring: dict | None) -> str:
    """Map draft-room scoring ``{ppr, tep, passTd}`` onto pick_proj_variant keys."""
    s = scoring or {}
    return pick_proj_variant({
        "rec": s.get("ppr", s.get("rec", 1.0)),
        "bonus_rec_te": s.get("tep", s.get("bonus_rec_te", 0.0)),
        "pass_td": s.get("passTd", s.get("pass_td", 4.0)),
    })


# ======================================================================
# From utils/projection_resolver.py
# ======================================================================

"""Canonical projected-PPG resolution and provenance.

This module is the boundary between projection providers and application
features.  Consumers must not choose a provider or annualize a weekly number;
they ask for an explicit projection context and receive both the value and its
provenance.  Sleeper is authoritative.  A caller-supplied secondary value is
used only when Sleeper has no row, which keeps legacy/offline imports usable
without letting them outrank Sleeper.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import logging
from statistics import median
from typing import Mapping


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
        from utils.data_cache import load_week_projection
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
        from utils.data_cache import load_week_projection
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


# ======================================================================
# From utils/week_proj.py
# ======================================================================

"""Week-projection map helpers (Flask-free).

Matchups and Scout both need to unwrap ``proj_by_week[week]`` into a pid →
value map. Keep this module free of ``dashboard_services.api`` / Flask so unit
jobs that only install ruff + pytest can still exercise Scout.
"""



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


# ======================================================================
# From utils/model_confidence.py
# ======================================================================

"""Shared, explainable confidence and rank-stability helpers."""

import math


def confidence_from_inputs(available: int, expected: int, sample_size: int = 0) -> dict:
    """Confidence derived from completeness plus optional historical sample size."""
    completeness = min(1.0, max(0.0, available / max(expected, 1)))
    sample = 1.0 - math.exp(-max(sample_size, 0) / 100.0) if sample_size else completeness
    score = round(100 * (0.7 * completeness + 0.3 * sample))
    label = "High" if score >= 80 else "Medium" if score >= 55 else "Low"
    return {"score": score, "label": label, "completeness": round(completeness, 3)}


def rank_interval(rank: int, confidence_score: float, field_size: int) -> tuple[int, int]:
    """Explainable likely-rank range that narrows as confidence increases."""
    uncertainty = max(0.0, min(1.0, 1.0 - float(confidence_score) / 100.0))
    spread = max(1, round(max(field_size, 1) * uncertainty * 0.25))
    return max(1, rank - spread), min(max(field_size, 1), rank + spread)


# ======================================================================
# From utils/evaluation_metrics.py
# ======================================================================

"""Decision-specific evaluation metrics shared by model backtests."""



def brier_score(predictions, outcomes) -> float:
    pairs = list(zip(predictions, outcomes))
    return sum((float(p) - float(y)) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def log_loss(predictions, outcomes) -> float:
    pairs, eps = list(zip(predictions, outcomes)), 1e-6
    if not pairs:
        return 0.0
    return -sum(float(y) * math.log(min(1-eps, max(eps, float(p))))
                + (1-float(y)) * math.log(min(1-eps, max(eps, 1-float(p))))
                for p, y in pairs) / len(pairs)


def precision_at_k(scores, outcomes, k: int) -> float:
    pairs = list(zip(scores, outcomes))
    chosen = sorted(pairs, key=lambda pair: pair[0], reverse=True)[:max(0, k)]
    return sum(bool(y) for _, y in chosen) / len(chosen) if chosen else 0.0


def decision_regret(recommended_value: float, optimal_value: float) -> float:
    """Lost realized utility versus the best legal hindsight decision."""
    return max(0.0, float(optimal_value) - float(recommended_value))


# ======================================================================
# From utils/season_qualification.py
# ======================================================================
"""Shared, schedule-backed qualification rules for in-season statistics.

Qualification is deliberately based on *fully completed NFL rounds*.  A
Thursday game does not advance the sample while the rest of that week's slate
is still being played, and a bye never counts against an individual player.
"""

from dataclasses import dataclass
FULL_GAMES_MIN = 4
FULL_VOLUME_MINS = {"games": 4, "total_pass_att": 50, "total_carries": 20,
                    "total_targets": 15, "total_receptions": 10,
                    "total_touches": 20}


def _is_regular(game: dict) -> bool:
    value = str(game.get("seasonType") or game.get("season_type") or "").lower()
    return value in ("", "2", "reg", "regular", "regular season") or "regular" in value


def _is_final(game: dict) -> bool:
    """Return whether a game is safely known to be complete.

    Provider completion flags are authoritative.  The schedule cache can,
    however, retain its preseason ``Scheduled`` status after a game has been
    played.  Once the game's *calendar date* is in the past, it is also safe to
    consider it complete.  This deliberately does not use kickoff timestamps,
    so a round cannot qualify while games on its final calendar day are live.
    """
    if game.get("completed") is True or game.get("is_complete") is True:
        return True
    code = str(game.get("gameStatusCode") or game.get("status_code") or "").strip()
    status = str(game.get("gameStatus") or game.get("status") or "").lower()
    if code in ("2", "3") or "final" in status or "completed" in status:
        return True
    # A date fallback must not convert an explicitly postponed/cancelled game
    # into a final merely because its original date has passed.
    if any(word in status for word in ("postpon", "cancel", "suspend")):
        return False

    game_date = str(game.get("gameDate") or game.get("game_date") or "")[:10]
    compact = game_date.replace("-", "")
    if len(compact) != 8 or not compact.isdigit():
        return False
    try:
        scheduled_date = date.fromisoformat(
            f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        )
    except ValueError:
        return False
    return scheduled_date < date.today()


def completed_regular_season_rounds(
    season: int,
    *,
    week_start: Optional[int] = None,
    week_end: Optional[int] = None,
    load_week: Optional[Callable[[int, int], Iterable[dict]]] = None,
    max_week: int = 18,
) -> list[int]:
    """Return rounds whose complete regular-season slate is provider-final."""
    if load_week is None:
        from utils.utils import load_week_schedule
        load_week = load_week_schedule
    lo = max(1, int(week_start or 1))
    hi = min(max_week, int(week_end or max_week))
    completed = []
    for week in range(lo, hi + 1):
        games = [g for g in (load_week(int(season), week) or [])
                 if isinstance(g, dict) and _is_regular(g)]
        if games and all(_is_final(g) for g in games):
            completed.append(week)
    return completed


def scaled_minimum(full_minimum: int, completed_rounds: int) -> int:
    """Scale cumulative gates linearly through the normal four-game sample."""
    full = max(1, int(full_minimum))
    rounds = max(0, int(completed_rounds))
    if rounds == 0:
        return 1
    return max(1, min(full, (full * min(rounds, FULL_GAMES_MIN) + FULL_GAMES_MIN - 1)
                            // FULL_GAMES_MIN))


def player_completed_weeks(
    player_id: str,
    season: int,
    *,
    load_week: Optional[Callable[[int, int], Iterable[dict]]] = None,
) -> list[int]:
    """Weeks where the player's own team game is final.

    Unlike :func:`completed_regular_season_rounds` (which needs the whole NFL
    slate final), a week counts here as soon as THAT PLAYER's team has
    finished playing.  The modal's sample note therefore updates the moment
    their game goes final, even mid-week.  Bye weeks never count against the
    player.  Mid-season trades are handled via the per-week team map.
    """
    if load_week is None:
        from utils.utils import load_week_schedule
        load_week = load_week_schedule
    season = int(season)

    # Player's team per week (handles mid-season trades).
    try:
        from data_building.external_data.player_team_history import (
            teams_in_season,
        )
        stints = teams_in_season(str(player_id), season) or []
    except Exception:
        stints = []
    if not stints:
        return []
    week_team: dict[int, str] = {}
    for stint in stints:
        team = str(stint.get("team") or "").strip().upper()
        if not team:
            continue
        weeks = stint.get("weeks") or []
        if weeks:
            for w in weeks:
                week_team.setdefault(int(w), team)
        else:
            # Season-granularity fallback: team unknown per week.
            for w in range(1, 19):
                week_team.setdefault(w, team)

    # Bound the scan: the current season stops at the current NFL week; past
    # seasons scan the full schedule (the date fallback marks old games final).
    try:
        from dashboard_services.api import get_nfl_state
        current = get_nfl_state() or {}
        cur_season = int(current.get("season") or 0)
        cur_week = int(current.get("week") or 0)
    except Exception:
        cur_season, cur_week = 0, 0
    if cur_season and season == cur_season and cur_week:
        max_week = min(18, cur_week)
    elif cur_season and season > cur_season:
        return []
    else:
        max_week = 18

    completed = []
    for week in range(1, max_week + 1):
        team = week_team.get(week)
        if not team:
            continue  # bye week or no team data; never counts against them
        try:
            games = [g for g in (load_week(season, week) or [])
                     if isinstance(g, dict)]
        except Exception:
            continue
        game = next(
            (g for g in games
             if str(g.get("away") or "").upper() == team
             or str(g.get("home") or "").upper() == team),
            None,
        )
        if game is None:
            continue
        if _is_final(game):
            completed.append(week)
            continue
        status = str(game.get("gameStatus") or game.get("status") or "").lower()
        if any(word in status for word in ("postpon", "cancel", "suspend")):
            continue  # odd scheduling; keep scanning later weeks
        # Weeks are chronological: a non-final game means later weeks have
        # not been played yet.
        break
    return completed


def bulk_player_completed_weeks(
    player_ids: Iterable[str],
    season: int,
) -> dict[str, list[int]]:
    """{player_id: completed weeks} for many players, efficiently.

    Same per-player finality semantics as :func:`player_completed_weeks`
    (a week counts when THAT player's team game is final; byes never count;
    chronological break on first non-final scheduled game), but hoists the
    shared work: one get_nfl_state() call, one schedule load per week, and
    one team-map load. Use for bulk consumers like the rankings page where
    calling player_completed_weeks() per player would repeat that work
    thousands of times. Never raises; players with no team data map to [].
    """
    season = int(season)
    pids = [str(p) for p in (player_ids or [])]
    result: dict[str, list[int]] = {pid: [] for pid in pids}
    if not pids:
        return result

    # Bound the scan (mirrors player_completed_weeks).
    try:
        from dashboard_services.api import get_nfl_state
        current = get_nfl_state() or {}
        cur_season = int(current.get("season") or 0)
        cur_week = int(current.get("week") or 0)
    except Exception:
        cur_season, cur_week = 0, 0
    if cur_season and season > cur_season:
        return result
    if cur_season and season == cur_season and cur_week:
        max_week = min(18, cur_week)
    else:
        max_week = 18

    # Team per player per week (handles mid-season trades).
    try:
        from data_building.external_data.player_team_history import (
            load_weekly_team_map,
            season_team_map,
        )
        weekly_map = load_weekly_team_map(season) or {}
        season_map = season_team_map(season) or {}
    except Exception:
        weekly_map, season_map = {}, {}

    # Finality per (team, week), loaded once.
    try:
        from utils.utils import load_week_schedule
    except Exception:
        return result
    final_by_team_week: dict[tuple[str, int], bool] = {}
    has_game_by_team_week: dict[tuple[str, int], bool] = {}
    postponed_by_team_week: dict[tuple[str, int], bool] = {}
    for week in range(1, max_week + 1):
        try:
            games = [g for g in (load_week_schedule(season, week) or [])
                      if isinstance(g, dict)]
        except Exception:
            continue
        for g in games:
            for side in ("away", "home"):
                team = str(g.get(side) or "").strip().upper()
                if not team:
                    continue
                key = (team, week)
                has_game_by_team_week[key] = True
                if _is_final(g):
                    final_by_team_week[key] = True
                status = str(g.get("gameStatus") or g.get("status") or "").lower()
                if any(w in status for w in ("postpon", "cancel", "suspend")):
                    postponed_by_team_week[key] = True

    for pid in pids:
        completed: list[int] = []
        for week in range(1, max_week + 1):
            team = (weekly_map.get(pid) or {}).get(week)
            if not team:
                team = season_map.get(pid)
            team = str(team or "").strip().upper()
            if not team:
                continue  # bye week or no team data; never counts against them
            key = (team, week)
            if not has_game_by_team_week.get(key):
                continue
            if final_by_team_week.get(key):
                completed.append(week)
                continue
            if postponed_by_team_week.get(key):
                continue  # odd scheduling; keep scanning later weeks
            # Weeks are chronological: a non-final game means later weeks
            # have not been played yet.
            break
        result[pid] = completed
    return result


def player_sample_note(player_id: str, season: int) -> Optional[str]:
    """Per-player small-sample note, e.g. "Small sample · 3 games".

    Counts weeks where the player's own team game is final, so the note
    updates as soon as their game finishes.  Returns None once the sample is
    no longer small (4+ completed games) or the player has no team data.
    """
    n = len(player_completed_weeks(player_id, season))
    if 0 < n < FULL_GAMES_MIN:
        return f"Small sample · {n} game{'s' if n != 1 else ''}"
    return None


def player_qualification_note(
    player_id: str,
    season: int,
    *,
    fallback_note: Optional[str] = None,
    fallback_provisional: bool = False,
) -> tuple[Optional[str], bool]:
    """Per-player (note, provisional), falling back to league-wide values.

    The note counts weeks where the player's own team game is final, so it
    updates the moment their game finishes, even mid-week.  When per-player
    team data is unavailable (or no games played yet), returns the provided
    fallbacks (typically the league-wide policy's note/provisional).
    """
    try:
        n = len(player_completed_weeks(player_id, season))
    except Exception:
        n = 0
    if n == 0:
        return fallback_note, fallback_provisional
    if n < FULL_GAMES_MIN:
        return f"Small sample · {n} game{'s' if n != 1 else ''}", True
    return None, False


@dataclass(frozen=True)
class QualificationPolicy:
    season: int
    completed_weeks: tuple[int, ...]
    games_min: int

    @property
    def provisional(self) -> bool:
        return 0 < len(self.completed_weeks) < FULL_GAMES_MIN

    def minimum(self, volume_column: str, full_minimum: Optional[int] = None) -> int:
        normal = int(full_minimum or FULL_VOLUME_MINS.get(volume_column, 1))
        return scaled_minimum(normal, len(self.completed_weeks))

    def note(self) -> Optional[str]:
        if not self.provisional:
            return None
        n = len(self.completed_weeks)
        # Counts fully completed NFL rounds, not the player's games. Word it
        # as weeks so it can't be misread as a games-played total (the stats
        # tab may show an in-progress week the rankings don't count yet).
        return f"Small sample · {n} week{'s' if n != 1 else ''} final"


def qualification_policy(season: int, *, week_start: Optional[int] = None,
                         week_end: Optional[int] = None, load_week=None) -> QualificationPolicy:
    weeks = completed_regular_season_rounds(
        season, week_start=week_start, week_end=week_end, load_week=load_week)
    progress = len(weeks)
    # Missing old schedule files must not turn a completed historical season
    # into a one-game sample. The current season is obtained from provider state,
    # not the wall clock.
    if not weeks and load_week is None:
        try:
            from dashboard_services.api import get_nfl_state
            current = int((get_nfl_state() or {}).get("season") or season)
            if int(season) < current:
                lo = max(1, int(week_start or 1))
                hi = min(18, int(week_end or 18))
                weeks = list(range(lo, hi + 1))
                progress = FULL_GAMES_MIN
        except Exception:
            pass
    return QualificationPolicy(int(season), tuple(weeks),
                               scaled_minimum(FULL_GAMES_MIN, progress))


# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L221 ---
TANK01_HOST = "disabled.invalid"

# --- utils/utils.py L222 ---
BASE = f"https://{TANK01_HOST}"

# --- utils/utils.py L223 ---
SCHEDULE_CACHE: dict[tuple[int, int], dict] = {}

# --- utils/utils.py L224 ---
SCHEDULE_TTL = 60 * 10  # seconds

# --- utils/utils.py L226 ---
TANK01_API_HOST = "disabled.invalid"

# --- utils/utils.py L227 ---
TANK01_API_KEY = os.environ.get("TANK01_API_KEY", "")  # RapidAPI key — set via env

# --- utils/utils.py L229 ---
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
    "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

# --- utils/utils.py L240 ---
def _headers(api_key: str) -> dict:
    return {
        "x-rapidapi-host": TANK01_HOST,
        "x-rapidapi-key": api_key,
    }

# --- utils/utils.py L1034 ---
def fetch_week_from_tank01(season: int, week: int, raw_scoring_settings: dict = None) -> dict:
    """Compatibility shim: paid weekly projections are unavailable.

    Callers continue through their existing non-paid projection hierarchy; no
    season totals or play-by-play estimates are substituted.
    """
    return {}

# --- utils/utils.py L1042 ---
def _sleeper_stats_to_variants(st: dict, pos: str, raw_scoring_settings: dict = None) -> Optional[dict]:
    """
    Compute all scoring variants from a Sleeper raw-stats dict.

    Sleeper's projections endpoint returns projected stat lines
    (pass_yd, pass_td, rec, rec_yd, ...) — not pre-computed pts_ppr.
    When raw_scoring_settings is supplied the league-specific values are
    used, otherwise standard defaults apply. All seven variants are
    computed so pick_proj_variant() can select the right one later.
    """
    ss = raw_scoring_settings or {}
    TEP_BONUS = 0.5

    pass_yd  = float(st.get("pass_yd")  or 0)
    pass_td  = float(st.get("pass_td")  or 0)
    pass_int = float(st.get("pass_int") or 0)
    rush_yd  = float(st.get("rush_yd")  or 0)
    rush_td  = float(st.get("rush_td")  or 0)
    rec      = float(st.get("rec")      or 0)
    rec_yd   = float(st.get("rec_yd")   or 0)
    rec_td   = float(st.get("rec_td")   or 0)
    fum_lost = float(st.get("fum_lost") or 0)

    # Do not discard kickers, defenses, returners, or IDP projections merely
    # because they have no offensive yardage. Keep any numeric projected stat
    # that the league can score. ADP-only rows (``adp_*`` / ``pos_adp_*``)
    # are not projections — Sleeper pads the feed with those before weekly
    # lines publish, and treating them as 0.0 projections hid real misses.
    def _is_proj_stat(key: str, value: Any) -> bool:
        if not isinstance(value, (int, float)) or value == 0:
            return False
        k = str(key).lower()
        if k.startswith("adp") or k.startswith("pos_adp"):
            return False
        return True

    if not any(_is_proj_stat(k, v) for k, v in st.items()):
        return None

    # League-specific scoring rates (with standard defaults)
    pass_yd_rate  = float(ss.get("pass_yd",        ss.get("passYards",         0.04)))
    pass_td_rate  = float(ss.get("pass_td",        ss.get("passTD",            4.0)))
    pass_int_rate = float(ss.get("pass_int",       ss.get("passInterceptions", -2.0)))
    rush_yd_rate  = float(ss.get("rush_yd",        ss.get("rushYards",         0.1)))
    rush_td_rate  = float(ss.get("rush_td",        ss.get("rushTD",            6.0)))
    rec_yd_rate   = float(ss.get("rec_yd",         ss.get("receivingYards",    0.1)))
    rec_td_rate   = float(ss.get("rec_td",         ss.get("receivingTD",       6.0)))
    fum_rate      = float(ss.get("fum_lost",       ss.get("fumbles",           -2.0)))
    te_bonus_rate = float(ss.get("bonus_rec_te",   0.0))

    base = (
        pass_yd  * pass_yd_rate
        + pass_td  * pass_td_rate
        + pass_int * pass_int_rate
        + rush_yd  * rush_yd_rate
        + rush_td  * rush_td_rate
        + rec_yd   * rec_yd_rate
        + rec_td   * rec_td_rate
        + fum_lost * fum_rate
    )

    ppr  = base + rec * 1.0
    half = base + rec * 0.5
    std  = base

    # TE premium: use league's bonus_rec_te if set, else standard 0.5
    tep_rate = te_bonus_rate if te_bonus_rate > 0 else TEP_BONUS
    tep = ppr + (rec * tep_rate if pos == "TE" else 0.0)

    # 6pt passing TD: difference vs the league's actual pass_td rate
    td6 = pass_td * max(0.0, 6.0 - pass_td_rate)

    return {
        # Preserve the source stat line in the shared cache. Each league can then
        # apply its complete scoring settings at read time without cache pollution
        # from whichever league happened to fetch this week first.
        "raw_stats": dict(st),
        "ppr":      round(ppr, 2),
        "half_ppr": round(half, 2),
        "std":      round(std, 2),
        "tep":      round(tep, 2),
        "6pt_ppr":  round(ppr  + td6, 2),
        "6pt_half": round(half + td6, 2),
        "6pt_tep":  round(tep  + td6, 2),
    }

# --- utils/utils.py L1129 ---
def fetch_week_from_sleeper(season: int, week: int, raw_scoring_settings: dict = None) -> dict:
    """
    Fetch Sleeper's own weekly projections, keyed by Sleeper player_id.
    The API returns projected stat lines; fantasy points are computed here
    from those raw stats. The raw line is cached so each league applies its
    complete scoring settings at read time.
    Returns {} on any failure so the caller can fall back to Tank01.
    """
    from utils.data_cache import load_players_index
    url = f"https://api.sleeper.app/v1/projections/nfl/regular/{season}/{week}"
    try:
        print(f"📡 Fetching Sleeper projections for {season} Week {week}...")
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"⚠️ Sleeper projections error {resp.status_code}: {resp.text[:160]}")
            return {}
        data = resp.json()
    except Exception as e:
        print(f"⚠️ Sleeper projections fetch failed: {e}")
        return {}

    # Response is {player_id: {stats: {...}}} or {player_id: {...flat stats...}}
    rows = []
    if isinstance(data, dict):
        for pid, entry in data.items():
            if isinstance(entry, dict):
                rows.append((str(pid), entry))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                pid = str(entry.get("player_id") or "")
                if pid:
                    rows.append((pid, entry))

    players_index = load_players_index() or {}
    out: dict = {}
    for pid, entry in rows:
        if isinstance(entry.get("stats"), dict):
            st = dict(entry["stats"])
            # Depending on the projection feed/version, Sleeper's displayed
            # totals live beside ``stats`` rather than inside it.  Keep them
            # with the cached raw line so projection_points() can use the exact
            # number shown by Sleeper instead of reconstructing it.
            for key in ("pts_ppr", "pts_half_ppr", "pts_std"):
                if key in entry and key not in st:
                    st[key] = entry[key]
        else:
            st = entry
        pos = players_index.get(pid, {}).get("pos", "")
        variants = _sleeper_stats_to_variants(st, pos, raw_scoring_settings)
        if variants:
            out[pid] = variants

    print(f"✅ Retrieved {len(out)} Sleeper projections for Week {week}")
    return out

# --- utils/utils.py L1185 ---
def fetch_week_projections(season: int, week: int, raw_scoring_settings: dict = None) -> dict:
    """Fetch weekly projections from Sleeper (sole source for all projection data)."""
    return fetch_week_from_sleeper(season, week, raw_scoring_settings)

# --- utils/utils.py L1190 ---
def map_weekly_projections_to_sleeper(
        weekly_rows: List[dict],
        idx_sleeper: Dict[str, dict],
) -> dict:
    """
    Convert Tank01 rows -> multi-variant projection dict.
    { sleeper_id: {"ppr": X, "half_ppr": Y, "std": Z,
                   "tep": A, "6pt_ppr": B, "6pt_half": C, "6pt_tep": D} }

    Variants:
      ppr       — 1pt/rec, 4pt passing TD
      half_ppr  — 0.5pt/rec, 4pt passing TD
      std       — 0pt/rec, 4pt passing TD
      tep       — PPR + 0.5pt/rec bonus for TEs, 4pt passing TD
      6pt_ppr   — PPR, 6pt passing TD
      6pt_half  — half-PPR, 6pt passing TD
      6pt_tep   — PPR + TEP, 6pt passing TD
    """
    from utils.data_cache import load_teams_index
    from utils.data_cache import load_players_index
    out: dict = {}

    teams_index = load_teams_index() or {}
    players_index = load_players_index() or {}

    TEP_BONUS = 0.5  # standard TE premium per reception

    for group in weekly_rows:
        if not isinstance(group, list):
            continue
        for row in group:
            if not isinstance(row, dict):
                continue

            team_id_raw = row.get("teamID")
            tank_id = row.get("playerID")

            if team_id_raw and not tank_id:
                # DEF / team row — same value for all variants
                proj = row.get("fantasyPointsDefault")
                if proj is None:
                    continue
                team_key = next(
                    (k for k, v in teams_index.items() if v.get("teamId") == str(team_id_raw)),
                    None,
                )
                if team_key:
                    v = float(proj)
                    out[str(team_key)] = {
                        "ppr": v, "half_ppr": v, "std": v,
                        "tep": v, "6pt_ppr": v, "6pt_half": v, "6pt_tep": v,
                    }

            elif tank_id:
                fp = row.get("fantasyPointsDefault") or {}
                if isinstance(fp, dict):
                    ppr  = float(fp.get("PPR")     or fp.get("ppr")     or 0)
                    half = float(fp.get("halfPPR")  or fp.get("half")    or 0)
                    std  = float(fp.get("standard") or fp.get("std")     or 0)
                else:
                    ppr = half = std = float(fp or 0)

                if ppr == 0 and half == 0 and std == 0:
                    continue

                pid = next(
                    (k for k, v in players_index.items() if v.get("tankId") == str(tank_id)),
                    None,
                )
                if not pid:
                    continue

                pos = players_index.get(pid, {}).get("pos", "")

                # Projected passing TDs (for 6pt TD adjustment: +2 per pass TD vs 4pt base)
                passing = row.get("passing") or {}
                proj_pass_td = float(passing.get("passTD") or passing.get("passIng_td") or 0)
                td_bonus = proj_pass_td * 2  # difference between 6pt and 4pt TD

                # TEP: extra 0.5/rec for TEs only
                tep_bonus = 0.0
                if pos == "TE":
                    rec_stats = row.get("receiving") or row.get("stats") or {}
                    proj_rec = float(rec_stats.get("rec") or rec_stats.get("receptions") or 0)
                    tep_bonus = proj_rec * TEP_BONUS

                out[str(pid)] = {
                    "ppr":      round(ppr, 2),
                    "half_ppr": round(half, 2),
                    "std":      round(std, 2),
                    "tep":      round(ppr + tep_bonus, 2),
                    "6pt_ppr":  round(ppr + td_bonus, 2),
                    "6pt_half": round(half + td_bonus, 2),
                    "6pt_tep":  round(ppr + tep_bonus + td_bonus, 2),
                }

    return out
