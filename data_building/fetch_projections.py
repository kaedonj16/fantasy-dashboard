"""Sleeper-only projection helpers.

Weekly and season-level projection context comes exclusively from Sleeper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / "cache"

# Matches utils.proj_variant.pick_proj_variant keys and the weekly projection
# payload (ppr / half_ppr / std plus TE-premium and 6-point passing-TD layers).
PROJ_VARIANTS = ("ppr", "half_ppr", "std", "tep", "6pt_ppr", "6pt_half", "6pt_tep")


def weekly_variant_values(weekly_maps) -> dict[str, dict[str, list[float]]]:
    """Collect positive weekly projection values per player and scoring variant.

    ``weekly_maps`` is an iterable of week dicts as stored on disk:
    ``{pid: {ppr, half_ppr, std, tep, 6pt_ppr, 6pt_half, 6pt_tep}}``.
    A missing variant falls back to that week's ``ppr`` value so a sparse week
    does not drop the player from a scoring-specific season average.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for week_map in weekly_maps or []:
        if not isinstance(week_map, dict):
            continue
        for pid, row in week_map.items():
            pid = str(pid)
            if isinstance(row, dict):
                ppr_fallback = row.get("ppr")
                for key in PROJ_VARIANTS:
                    value = row.get(key)
                    if value is None:
                        value = ppr_fallback
                    try:
                        value = float(value or 0)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        out.setdefault(pid, {}).setdefault(key, []).append(value)
            else:
                try:
                    value = float(row or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    out.setdefault(pid, {}).setdefault("ppr", []).append(value)
    return out


def season_ppg_from_weekly(weekly_maps) -> dict[str, dict[str, float]]:
    """Median-of-weeks PPG for every scoring variant: ``{pid: {variant: ppg}}``."""
    from statistics import median

    result: dict[str, dict[str, float]] = {}
    for pid, by_var in weekly_variant_values(weekly_maps).items():
        result[pid] = {
            key: round(float(median(vals)), 2)
            for key, vals in by_var.items()
            if vals
        }
    return result


def _year_weekly_values(year: int) -> dict[str, dict[str, list[float]]]:
    """Load weeks 1-18 and collect positive weekly values per scoring variant."""
    from utils.utils import load_week_projection

    weeks = [load_week_projection(year, week) or {} for week in range(1, 19)]
    return weekly_variant_values(weeks)


def fetch_sleeper_season_ppg_variants(
    year: int,
    players_index: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """Season PPG for every scoring variant, keyed by Sleeper player id.

    Shape: ``{pid: {ppr: 18.5, half_ppr: 16.2, 6pt_ppr: 20.1, ...}}``.
    PPG is the median of positive weekly values so byes do not dilute it.
    ``players_index`` is accepted so callers can share kwargs with the PPR helper.
    """
    collected = _year_weekly_values(year)
    from statistics import median
    result: dict[str, dict[str, float]] = {}
    for pid, by_var in collected.items():
        result[pid] = {
            key: round(float(median(vals)), 2)
            for key, vals in by_var.items()
            if vals
        }
    return result


def fetch_sleeper_season_projections(
    year: int,
    scoring: str = "ppr",
    players_index: Optional[dict] = None,
) -> dict[str, dict]:
    """Aggregate Sleeper weekly projections into a season PPG baseline.

    The median positive weekly projection is used for PPG so bye weeks and
    missing weekly rows do not dilute a player's expected active-week output.
    The returned shape matches the old season-projection helper, which lets all
    consumers use one Sleeper-only source without maintaining parallel models.
    """
    from statistics import median
    from utils.utils import load_players_index

    players_index = players_index or load_players_index() or {}
    variant = scoring if scoring in PROJ_VARIANTS else "ppr"
    collected = _year_weekly_values(year)

    result = {}
    for pid, by_var in collected.items():
        weekly = by_var.get(variant) or by_var.get("ppr") or []
        if not weekly:
            continue
        result[pid] = {
            "pos": str((players_index.get(pid) or {}).get("pos") or "").upper(),
            "season_pts": round(sum(weekly), 1),
            "ppg": round(float(median(weekly)), 2),
        }
    return result


# ---------------------------------------------------------------------------
# Sleeper weekly projections (in-season)
# ---------------------------------------------------------------------------

def _sleeper_proj_cache_path(year: int, week: int) -> Path:
    return _CACHE_DIR / f"sleeper_proj_{year}_w{week:02d}.json"


def _raw_to_pts(st: dict, pos: str, scoring: str) -> float:
    """Compute fantasy points for one player from Sleeper raw projected stats."""
    pass_yd  = float(st.get("pass_yd")  or 0)
    pass_td  = float(st.get("pass_td")  or 0)
    pass_int = float(st.get("pass_int") or 0)
    rush_yd  = float(st.get("rush_yd")  or 0)
    rush_td  = float(st.get("rush_td")  or 0)
    rec      = float(st.get("rec")      or 0)
    rec_yd   = float(st.get("rec_yd")   or 0)
    rec_td   = float(st.get("rec_td")   or 0)
    fum_lost = float(st.get("fum_lost") or 0)

    if (pass_yd + pass_td + rush_yd + rush_td + rec + rec_yd + rec_td) == 0:
        return 0.0

    base = (
        pass_yd  * 0.04  + pass_td  * 4.0  + pass_int * -2.0
        + rush_yd * 0.1  + rush_td  * 6.0
        + rec_yd  * 0.1  + rec_td   * 6.0
        + fum_lost * -2.0
    )
    rec_pts  = 1.0 if scoring in ("ppr", "tep", "6pt_ppr", "6pt_tep") else (
        0.5 if scoring in ("half_ppr", "6pt_half") else 0.0)
    tep_pts  = 0.5 if (scoring in ("tep", "6pt_tep") and pos == "TE") else 0.0
    td6_pts  = pass_td * 2.0 if scoring in ("6pt_ppr", "6pt_half", "6pt_tep") else 0.0
    return round(base + rec * rec_pts + rec * tep_pts + td6_pts, 2)
