"""Sleeper-only projection helpers.

Weekly and season-level projection context comes exclusively from Sleeper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / "cache"

# Sleeper drops the weekly stat line (proj PPG 0) once a player is out for the
# year. Displayed point projections must not refill those players from any
# other source (FantasyPros preseason files, last-season actuals, etc.).
_SEASON_ENDING_INJURY = frozenset({"IR", "PUP", "NFI"})


def unprojected_season_injury(injury_status: Optional[str], sleeper_ppg) -> bool:
    """True when Sleeper has zeroed/omitted a player who is out for the year.

    PUP/IR players Sleeper still projects (expected return) keep that number.
    """
    status = str(injury_status or "").strip().upper()
    if status not in _SEASON_ENDING_INJURY:
        return False
    try:
        ppg = float(sleeper_ppg or 0)
    except (TypeError, ValueError):
        ppg = 0.0
    return ppg <= 0


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
    from utils.utils import load_players_index, load_week_projection

    players_index = players_index or load_players_index() or {}
    variant = {"half_ppr": "half_ppr", "std": "std"}.get(scoring, "ppr")
    values: dict[str, list[float]] = {}
    for week in range(1, 19):
        for pid, row in (load_week_projection(year, week) or {}).items():
            if isinstance(row, dict):
                value = row.get(variant)
                if value is None:
                    value = row.get("ppr")
            else:
                value = row
            try:
                value = float(value or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.setdefault(str(pid), []).append(value)

    result = {}
    for pid, weekly in values.items():
        ppg = round(float(median(weekly)), 2)
        result[pid] = {
            "pos": str((players_index.get(pid) or {}).get("pos") or "").upper(),
            "season_pts": round(sum(weekly), 1),
            "ppg": ppg,
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
