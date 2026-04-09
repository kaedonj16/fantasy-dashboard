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

# Assumed target-per-route baselines by position for route-run proxies.
# These are empirical averages from college data; adjust as better data arrives.
_TPRR_BASELINE: Dict[str, float] = {
    "WR": 0.28,
    "TE": 0.35,
    "RB": 0.22,
}

# Fallback reception-per-route baselines for when targets are unavailable.
# CFBD does not expose target counts; receptions are the best public substitute.
# Derived from: typical college catch rate (~62% WR, ~66% TE, ~72% RB) × tprr baseline.
_RPRR_BASELINE: Dict[str, float] = {
    "WR": 0.17,   # ~1 catch per 5.9 routes
    "TE": 0.23,   # ~1 catch per 4.3 routes
    "RB": 0.16,   # ~1 catch per 6.3 routes
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


# ---------------------------------------------------------------------------
# Proxy derivations (added to cover more metrics when PFF/SIS unavailable)
# ---------------------------------------------------------------------------


def derive_routes_run_proxy(stats: Dict[str, Any], position: str) -> Optional[float]:
    """
    Estimate total routes run for a season when play-level route data is unavailable.

    Primary path (when targets available):
      routes_run ≈ targets / tprr_baseline
      WR ≈ 0.28, TE ≈ 0.35, RB ≈ 0.22 targets/route

    Fallback path (CFBD does not expose targets; receptions used instead):
      routes_run ≈ receptions / rprr_baseline
      WR ≈ 0.17, TE ≈ 0.23, RB ≈ 0.16 receptions/route
      Confidence drops to 0.30 (extra indirection through catch rate assumption).

    Requires: games_played ≥ 4, and either targets > 0 or receptions > 0.
    """
    pos = (position or "").upper()
    games = _get_num(stats, "games_played")
    if games is None or games < 4:
        return None

    # Primary: targets (future-proof for any source that provides them)
    tprr_base = _TPRR_BASELINE.get(pos)
    targets = _get_num(stats, "targets")
    if tprr_base is not None and targets is not None and targets > 0:
        return round(targets / tprr_base, 1)

    # Fallback: receptions (always available from CFBD)
    rprr_base = _RPRR_BASELINE.get(pos)
    receptions = _get_num(stats, "receptions")
    if rprr_base is not None and receptions is not None and receptions > 0:
        return round(receptions / rprr_base, 1)

    return None


def derive_yprr_proxy(stats: Dict[str, Any], position: str) -> Optional[float]:
    """
    Estimate yards per route run (YPRR) using receiving yards and estimated routes.

    Formula: yprr ≈ receiving_yards / derive_routes_run_proxy(stats, position)
    Routes are estimated from targets when available, or receptions as a fallback
    (see derive_routes_run_proxy). Compound proxy — uncertainty compounds.
    Requires: receiving_yards, and same constraints as derive_routes_run_proxy.
    Confidence: 0.35 targets-path / 0.28 receptions-path — directional only.
    """
    routes = derive_routes_run_proxy(stats, position)
    if routes is None or routes <= 0:
        return None

    rec_yards = _get_num(stats, "receiving_yards")
    if rec_yards is None:
        return None

    return round(rec_yards / routes, 3)


def derive_tprr_proxy(stats: Dict[str, Any], position: str) -> Optional[float]:
    """
    Estimate targets per route run (TPRR) using targets and estimated routes.

    Formula: tprr ≈ targets / derive_routes_run_proxy(stats, position)
    For WR/TE/RB. The result should be close to the position baseline used to
    derive routes_run_proxy, so this mainly validates internal consistency and
    catches outlier targets shares.
    Confidence: 0.35 — compound proxy.
    """
    routes = derive_routes_run_proxy(stats, position)
    if routes is None or routes <= 0:
        return None

    targets = _get_num(stats, "targets")
    if targets is None:
        return None

    return round(targets / routes, 4)


def derive_yac_per_att_proxy(stats: Dict[str, Any]) -> Optional[float]:
    """
    Proxy for yards after contact per rushing attempt (RBs only).

    PFF's true YAC requires contact-point tracking. As a college proxy we map
    yards-per-carry (YPC) above a threshold carry quality baseline onto the YAC
    range [0, 5.0].

    Formula: max(0, min(5.0, (ypc - 2.5) * 0.65))
    Calibration: elite YAC≈3.5–4.5 corresponds to YPC≈7–9, baseline YAC≈1.5
    corresponds to YPC≈4.8.
    Requires: yds_per_carry, rush_attempts ≥ 20.
    Confidence: 0.40 — systematic proxy; treats all YPC gains as equally contact-driven.
    """
    ypc = _get_num(stats, "yds_per_carry")
    attempts = _get_num(stats, "rush_attempts")
    if ypc is None or attempts is None or attempts < 20:
        return None

    return round(max(0.0, min(5.0, (ypc - 2.5) * 0.65)), 3)


def derive_mtf_per_att_proxy(stats: Dict[str, Any]) -> Optional[float]:
    """
    Proxy for missed tackles forced per rushing attempt (RBs only).

    PFF's mtf/att requires per-play contact data. We approximate it from YPC
    using a similar slope to explosive_run_rate but scaled to missed-tackle
    rate shape (typical elite: 0.20–0.30, average: 0.12–0.18).

    Formula: max(0, min(0.30, (ypc - 3.0) / 12))
    Requires: yds_per_carry, rush_attempts ≥ 20.
    Confidence: 0.30 — very rough; use only as relative ordinal proxy.
    """
    ypc = _get_num(stats, "yds_per_carry")
    attempts = _get_num(stats, "rush_attempts")
    if ypc is None or attempts is None or attempts < 20:
        return None

    return round(max(0.0, min(0.30, (ypc - 3.0) / 12.0)), 4)


def derive_adjusted_comp_pct_proxy(stats: Dict[str, Any]) -> Optional[float]:
    """
    Proxy for PFF's adjusted completion percentage (QBs only).

    PFF's metric strips drops and adjusts for difficulty of attempt (aDOT).
    As a proxy we start with raw completion_pct and apply a light TD/INT ratio
    quality multiplier: better decision-making inflates slightly, poor INT rate
    deflates slightly.

    Formula:
      multiplier = min(1.12, max(0.88, 1 + (td_int_ratio - 2.0) * 0.03))
      adjusted_comp_pct = completion_pct * multiplier

    Where td_int_ratio = 2.0 is treated as average (no adjustment).
    Requires: completion_pct, pass_attempts ≥ 50.
    Confidence: 0.60 — raw completion_pct is reliable; multiplier adds minor signal.
    """
    comp_pct = _get_num(stats, "completion_pct")
    pass_att = _get_num(stats, "pass_attempts")
    if comp_pct is None or pass_att is None or pass_att < 50:
        return None

    td_int_ratio = _get_num(stats, "td_int_ratio")
    if td_int_ratio is not None:
        multiplier = min(1.12, max(0.88, 1.0 + (td_int_ratio - 2.0) * 0.03))
    else:
        multiplier = 1.0

    return round(comp_pct * multiplier, 2)


def derive_twp_rate_proxy(stats: Dict[str, Any]) -> Optional[float]:
    """
    Proxy for turnover-worthy play rate (QBs only).

    PFF's TWP rate counts all plays that should have resulted in a turnover
    (including near-interceptions). As a direct proxy we use raw interception
    rate (INT / pass_attempts * 100), which is the most accessible public
    equivalent.

    Formula: (interceptions / pass_attempts) * 100
    Requires: interceptions, pass_attempts ≥ 50.
    Confidence: 0.65 — direct measurable stat, not PFF-adjusted but strongly correlated.
    """
    interceptions = _get_num(stats, "interceptions")
    pass_att = _get_num(stats, "pass_attempts")
    if interceptions is None or pass_att is None or pass_att < 50:
        return None

    return round((interceptions / pass_att) * 100.0, 3)
