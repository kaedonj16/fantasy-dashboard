from __future__ import annotations

from typing import Any, Dict, Optional


CONFERENCE_SOS = {
    "sec": 1.00,
    "big ten": 0.97,
    "big 12": 0.93,
    "acc": 0.90,
    "pac-12": 0.89,
    "american": 0.82,
    "sun belt": 0.80,
    "mountain west": 0.79,
    "conference usa": 0.75,
    "mac": 0.74,
}


def _get_num(stats: Dict[str, Any], key: str) -> Optional[float]:
    val = stats.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def derive_explosive_run_rate(stats: Dict[str, Any]) -> Optional[float]:
    """
    Derive explosive_run_rate when play-level data is unavailable.

    Assumption: we proxy explosive propensity from yards-per-carry slope above a
    replacement baseline of 3.0 YPC and cap at 20%.
    Formula: max(0, min(0.20, (ypc - 3.0) / 20)).
    """
    ypc = _get_num(stats, "yds_per_carry")
    attempts = _get_num(stats, "rush_attempts")
    if ypc is None or attempts is None or attempts < 20:
        return None
    return round(max(0.0, min(0.20, (ypc - 3.0) / 20.0)), 4)


def derive_player_level_sos(stats: Dict[str, Any]) -> Optional[float]:
    """Conference-based SOS proxy in [0, 1] with deterministic map."""
    conference = (stats.get("conference") or "").strip().lower()
    if not conference:
        return None
    return CONFERENCE_SOS.get(conference, 0.72)


def derive_performance_vs_top_defenses(stats: Dict[str, Any]) -> Optional[float]:
    """
    Context-aware proxy score in [0, 100].

    Formula mixes player production efficiency with SOS:
      base = 0.6 * dominator_rating + 0.4 * market_share_yards
      score = base * (0.75 + 0.25 * sos) * 100
    """
    dom = _get_num(stats, "dominator_rating")
    msy = _get_num(stats, "market_share_yards")
    sos = derive_player_level_sos(stats)
    if dom is None or msy is None or sos is None:
        return None
    base = (0.6 * dom) + (0.4 * msy)
    return round(max(0.0, min(100.0, base * (0.75 + 0.25 * sos) * 100.0)), 2)


def derive_true_early_declare(player: Dict[str, Any]) -> Optional[bool]:
    """True if explicitly early declare and non-senior class year when available."""
    early = player.get("early_declare")
    if early is None:
        return None
    class_year = str(player.get("class_year") or player.get("experience") or "").upper()
    if class_year.startswith("SR"):
        return False
    return bool(early)
