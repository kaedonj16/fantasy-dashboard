"""
Historical NFL signal calibration.

Downloads nflverse player stats (GitHub releases) and joins with CFBD college
production to identify which college metrics most strongly predict NFL fantasy
success.  Outputs correlation statistics and calibrated weight recommendations
that can inform adjustments to prospect_model.py.

Usage:
    from data_building.rookie_pipeline.historical_calibration import (
        run_calibration, get_calibrated_weights
    )

    results = run_calibration(draft_years=range(2016, 2025))
    print(results["weight_recommendations"])
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import statistics
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────────────────────────────────────

# nflverse releases player stats as CSV on GitHub
# File was renamed: player_stats_{year}.csv → stats_player_reg_{year}.csv
_NFLVERSE_BASE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/stats_player_reg_{year}.csv"
)

# nflverse roster data (includes draft info: draft_year, draft_pick, position)
_NFLVERSE_ROSTER = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "rosters/roster_{year}.csv"
)

# nflverse combine data
_NFLVERSE_COMBINE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "combine/combine.csv"
)

# CFBD API — reuse the helper from ingestion.py
try:
    from .ingestion import _cfbd_get
except ImportError:
    _cfbd_get = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_csv(url: str) -> List[Dict[str, str]]:
    """Download a CSV URL and return list of row dicts."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    except Exception as exc:
        log.warning("[calibration] Failed to fetch %s: %s", url, exc)
        return []


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "", "NA", "NULL") else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# NFL fantasy points calculation (PPR)
# ─────────────────────────────────────────────────────────────────────────────

def _calc_ppr_points(row: Dict[str, str]) -> float:
    """Compute PPR fantasy points from a nflverse player_stats row."""
    pts = 0.0
    pts += _safe_float(row.get("rushing_yards"))   * 0.1
    pts += _safe_float(row.get("rushing_tds"))     * 6.0
    pts += _safe_float(row.get("receiving_yards")) * 0.1
    pts += _safe_float(row.get("receiving_tds"))   * 6.0
    pts += _safe_float(row.get("receptions"))      * 1.0   # PPR
    pts += _safe_float(row.get("passing_yards"))   * 0.04
    pts += _safe_float(row.get("passing_tds"))     * 4.0
    pts -= _safe_float(row.get("interceptions"))   * 2.0
    pts -= _safe_float(row.get("sack_fumbles_lost")) * 2.0
    pts -= _safe_float(row.get("rushing_fumbles_lost")) * 2.0
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build NFL outcome data per drafted player
# ─────────────────────────────────────────────────────────────────────────────

def _build_nfl_outcomes(
    draft_years: List[int],
    nfl_data_years: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    For each draft class, compute per-player NFL fantasy outcomes.

    Fetches rosters to get draft metadata, then stats for N years post-draft.

    Returns: {player_name_lower: {
        "draft_year": int, "draft_pick": int, "position": str,
        "nfl_ppr_y1": float, "nfl_ppr_y2": float, "nfl_ppr_peak": float,
        "nfl_ppr_4yr_avg": float, "nfl_games_y1": int,
    }}
    """
    outcomes: Dict[str, Dict[str, Any]] = {}

    for draft_year in draft_years:
        log.info("[calibration] Loading draft class %d roster", draft_year)
        roster_rows = _fetch_csv(_NFLVERSE_ROSTER.format(year=draft_year))

        # Build player → draft metadata from the rookie class
        draft_class: Dict[str, Dict[str, Any]] = {}
        for row in roster_rows:
            if _safe_float(row.get("draft_year")) != draft_year:
                continue
            player = (row.get("full_name") or row.get("player_name") or "").strip().lower()
            if not player:
                continue
            pos = (row.get("position") or "").upper()
            if pos not in ("QB", "RB", "WR", "TE"):
                continue
            pick = _safe_float(row.get("draft_number"), 300)
            draft_class[player] = {
                "draft_year": draft_year,
                "draft_pick": int(pick),
                "position": pos,
                "gsis_id": row.get("gsis_id") or "",
            }

        # Accumulate NFL stats for nfl_data_years following draft
        season_stats: Dict[str, List[float]] = {p: [] for p in draft_class}
        season_games: Dict[str, List[int]]   = {p: [] for p in draft_class}

        for nfl_yr in range(draft_year, draft_year + nfl_data_years):
            stat_rows = _fetch_csv(_NFLVERSE_BASE.format(year=nfl_yr))
            gid_to_pts:   Dict[str, float] = {}
            gid_to_games: Dict[str, int]   = {}
            for sr in stat_rows:
                gid = sr.get("player_id") or sr.get("player_name", "").lower()
                if not gid:
                    continue
                season_pts = _calc_ppr_points(sr)
                gid_to_pts[gid]   = gid_to_pts.get(gid, 0.0) + season_pts
                gid_to_games[gid] = gid_to_games.get(gid, 0) + int(_safe_float(sr.get("games"), 1))

            for player, meta in draft_class.items():
                gid = meta["gsis_id"]
                # Try gsis_id first, then name fallback
                pts   = gid_to_pts.get(gid)
                games = gid_to_games.get(gid)
                if pts is None:
                    # name-based fallback
                    pts   = gid_to_pts.get(player, 0.0)
                    games = gid_to_games.get(player, 0)
                season_stats[player].append(pts or 0.0)
                season_games[player].append(games or 0)

        # Aggregate into outcomes
        for player, meta in draft_class.items():
            sseries = season_stats[player]
            gseries = season_games[player]
            if not sseries:
                continue
            entry = {**meta}
            entry["nfl_ppr_y1"]      = sseries[0] if len(sseries) >= 1 else 0.0
            entry["nfl_ppr_y2"]      = sseries[1] if len(sseries) >= 2 else 0.0
            entry["nfl_ppr_y3"]      = sseries[2] if len(sseries) >= 3 else 0.0
            entry["nfl_ppr_peak"]    = max(sseries) if sseries else 0.0
            entry["nfl_ppr_4yr_avg"] = statistics.mean(sseries) if sseries else 0.0
            entry["nfl_games_y1"]    = gseries[0] if gseries else 0
            outcomes[player]         = entry

    log.info("[calibration] Built NFL outcomes for %d players", len(outcomes))
    return outcomes


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build college predictor features per player
# ─────────────────────────────────────────────────────────────────────────────

def _build_college_features(
    draft_years: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Fetch CFBD stats for each draft class and compute predictor features.

    Returns: {player_name_lower: {
        "dominator_rating": float, "rec_yds_pg": float, "tds_pg": float,
        "breakout_age": float, "team_pass_rate": float, "draft_capital": float,
        "conf_quality": float, ...
    }}
    """
    if _cfbd_get is None:
        log.warning("[calibration] _cfbd_get unavailable — CFBD features will be empty")
        return {}

    from .prospect_model import _conf_quality, pick_to_draft_capital_score  # type: ignore
    from .mock_draft_consensus import pick_to_draft_capital_score  # noqa: F811

    features: Dict[str, Dict[str, float]] = {}

    for draft_year in draft_years:
        log.info("[calibration] Fetching CFBD features for draft class %d", draft_year)
        for yr in [draft_year - 1, draft_year - 2]:
            try:
                rows = _cfbd_get("/stats/player/season", {"year": yr, "seasonType": "regular"})
            except Exception as exc:
                log.warning("[calibration] CFBD /stats/player/season %d failed: %s", yr, exc)
                continue

            for row in (rows or []):
                player = (row.get("player") or "").strip().lower()
                if not player:
                    continue

                stat_type = row.get("statType", "")
                val = _safe_float(row.get("stat"))
                if player not in features:
                    features[player] = {"draft_year": float(draft_year)}

                # Flatten key stats
                if stat_type == "REC YDS":
                    features[player]["rec_yds"] = features[player].get("rec_yds", 0) + val
                elif stat_type == "REC TD":
                    features[player]["rec_tds"] = features[player].get("rec_tds", 0) + val
                elif stat_type == "RUSH YDS":
                    features[player]["rush_yds"] = features[player].get("rush_yds", 0) + val
                elif stat_type == "RUSH TD":
                    features[player]["rush_tds"] = features[player].get("rush_tds", 0) + val
                elif stat_type == "PASS YDS":
                    features[player]["pass_yds"] = features[player].get("pass_yds", 0) + val
                elif stat_type == "PASS TD":
                    features[player]["pass_tds"] = features[player].get("pass_tds", 0) + val

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov    = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x  = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    std_y  = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y * n)


def _correlate_predictors(
    outcomes: Dict[str, Dict[str, Any]],
    features: Dict[str, Dict[str, float]],
    target: str = "nfl_ppr_4yr_avg",
    positions: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Pearson r between each college feature and the NFL target.

    Returns: {feature_name: pearson_r}
    """
    from .mock_draft_consensus import pick_to_draft_capital_score

    correlations: Dict[str, List[Tuple[float, float]]] = {}

    for player, out in outcomes.items():
        if positions and out.get("position") not in positions:
            continue
        y_val = _safe_float(out.get(target))
        feat  = features.get(player, {})
        if not feat:
            # Still use draft capital from outcome data
            pick = out.get("draft_pick", 300)
            dc   = pick_to_draft_capital_score(pick)
            feat = {"draft_capital_score": dc}

        for fname, fval in feat.items():
            if fname in ("draft_year",):
                continue
            correlations.setdefault(fname, []).append((_safe_float(fval), y_val))

    # Also add draft capital score to correlations using outcomes directly
    dc_pairs: List[Tuple[float, float]] = []
    for player, out in outcomes.items():
        if positions and out.get("position") not in positions:
            continue
        y_val = _safe_float(out.get(target))
        pick  = out.get("draft_pick", 300)
        dc    = pick_to_draft_capital_score(pick)
        dc_pairs.append((dc, y_val))
    if dc_pairs:
        correlations["draft_capital_score"] = dc_pairs

    result: Dict[str, float] = {}
    for fname, pairs in correlations.items():
        if len(pairs) < 10:
            continue
        xs, ys = zip(*pairs)
        result[fname] = round(_pearson_r(list(xs), list(ys)), 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Weight recommendation
# ─────────────────────────────────────────────────────────────────────────────

# Current model weights for reference
REFINED_POSITION_WEIGHTS = {
    "QB": {
        "draft_capital": 0.26,
        "production": 0.13,
        "utilization": 0.05,
        "efficiency": 0.18,
        "age": 0.08,
        "breakout": 0.04,
        "athleticism": 0.10,
        "competition": 0.07,
        "environment": 0.02,
        "durability": 0.00,
        "experience": 0.07,
    },
    "RB": {
        "draft_capital": 0.24,
        "production": 0.22,
        "utilization": 0.09,
        "efficiency": 0.07,
        "age": 0.09,
        "breakout": 0.12,
        "athleticism": 0.10,
        "competition": 0.05,
        "environment": 0.01,
        "durability": 0.01,
    },
    "WR": {
        "draft_capital": 0.29,
        "production": 0.22,
        "utilization": 0.04,
        "efficiency": 0.10,
        "age": 0.08,
        "breakout": 0.12,
        "athleticism": 0.07,
        "competition": 0.07,
        "environment": 0.01,
        "durability": 0.00,
    },
    "TE": {
        "draft_capital": 0.26,
        "production": 0.18,
        "utilization": 0.05,
        "efficiency": 0.12,
        "age": 0.13,
        "breakout": 0.05,
        "athleticism": 0.12,
        "competition": 0.07,
        "environment": 0.01,
        "durability": 0.01,
    },
}

# Feature → component mapping (CFBD features → prospect_model components)
_FEATURE_TO_COMPONENT = {
    "draft_capital_score":  "draft_capital",
    "rec_yds":              "production",
    "rush_yds":             "production",
    "pass_yds":             "production",
    "rec_tds":              "production",
    "rush_tds":             "production",
    "pass_tds":             "production",
    "dominator_rating":     "production",
    "breakout_age":         "breakout",
    "conf_quality":         "competition",
    "team_pass_rate":       "environment",
}


def _recommend_weights(
    correlations_by_pos: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Given per-position correlations, suggest component weight adjustments.

    Logic: aggregate |r| by component, normalise to sum=1, compare to current weights.
    Return delta recommendations.
    """
    recommendations: Dict[str, Dict[str, float]] = {}

    for pos, corr in correlations_by_pos.items():
        # Use position-specific current weights as the baseline for blending
        curr_weights = REFINED_POSITION_WEIGHTS.get(pos, next(iter(REFINED_POSITION_WEIGHTS.values())))

        comp_r: Dict[str, List[float]] = {}
        for feat, r in corr.items():
            comp = _FEATURE_TO_COMPONENT.get(feat)
            if comp:
                comp_r.setdefault(comp, []).append(abs(r))

        # Average |r| per component
        avg_r = {comp: statistics.mean(rs) for comp, rs in comp_r.items() if rs}

        # Fill in missing components with their current weights (no data = keep current)
        for comp in curr_weights:
            if comp not in avg_r:
                avg_r[comp] = curr_weights[comp]

        total = sum(avg_r.values()) or 1.0
        calibrated = {comp: round(r / total, 4) for comp, r in avg_r.items()}

        # Blend 50/50 current vs data-calibrated (don't over-weight sparse signals)
        blended = {
            comp: round(0.50 * curr_weights.get(comp, 0.05) + 0.50 * calibrated.get(comp, 0.05), 4)
            for comp in set(list(curr_weights) + list(calibrated))
        }
        # Normalise to sum = 1
        total_blended = sum(blended.values()) or 1.0
        blended = {comp: round(w / total_blended, 4) for comp, w in blended.items()}

        recommendations[pos] = {
            "calibrated_from_data": calibrated,
            "blended_recommendation": blended,
            "current": dict(curr_weights),
            "sample_size": len([v for v in correlations_by_pos[pos].values() if v != 0]),
        }

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    draft_years: Optional[List[int]] = None,
    nfl_data_years: int = 4,
    positions: Optional[List[str]] = None,
    target: str = "nfl_ppr_4yr_avg",
) -> Dict[str, Any]:
    """
    Run the full historical calibration pipeline.

    Args:
        draft_years:    Draft classes to analyse (default: 2016-2022)
        nfl_data_years: How many NFL seasons to include per player (default: 4)
        positions:      Positions to analyse (default: WR, RB, TE, QB)
        target:         NFL outcome metric to predict (default: 4-year avg PPR)

    Returns dict with:
        correlations:       {pos: {feature: pearson_r}}
        weight_recommendations: {pos: {blended_recommendation: {...}}}
        summary:            human-readable summary string
    """
    if draft_years is None:
        draft_years = list(range(2016, 2023))
    if positions is None:
        positions = ["WR", "RB", "TE", "QB"]

    log.info("[calibration] Starting calibration for draft years %s", draft_years)

    # Step 1: NFL outcomes
    outcomes = _build_nfl_outcomes(draft_years, nfl_data_years)
    if not outcomes:
        log.warning("[calibration] No NFL outcome data — calibration cannot proceed")
        return {"error": "No NFL data available", "correlations": {}, "weight_recommendations": {}}

    # Step 2: College features
    features = _build_college_features(draft_years)

    # Step 3: Correlations by position
    correlations: Dict[str, Dict[str, float]] = {}
    for pos in positions:
        correlations[pos] = _correlate_predictors(outcomes, features, target, [pos])
        log.info("[calibration] %s correlations: %s", pos, correlations[pos])

    # Step 4: Weight recommendations
    weight_recommendations = _recommend_weights(correlations)

    # Build summary
    summary_lines = ["=== Historical Calibration Results ===", f"Target: {target}", ""]
    for pos in positions:
        corr = correlations.get(pos, {})
        if not corr:
            continue
        summary_lines.append(f"── {pos} ──")
        top = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, r in top:
            summary_lines.append(f"  {feat:<30s} r={r:+.3f}")
        rec = weight_recommendations.get(pos, {}).get("blended_recommendation", {})
        if rec:
            summary_lines.append(f"  Recommended weights:")
            for comp, w in sorted(rec.items(), key=lambda x: -x[1]):
                curr = _CURRENT_WEIGHTS.get(comp, 0)
                delta = w - curr
                arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "=")
                summary_lines.append(f"    {comp:<20s} {w:.3f}  ({arrow} {delta:+.3f} vs current)")
        summary_lines.append("")

    return {
        "draft_years":           draft_years,
        "target":                target,
        "n_players":             len(outcomes),
        "correlations":          correlations,
        "weight_recommendations": weight_recommendations,
        "summary":               "\n".join(summary_lines),
    }


def get_calibrated_weights(
    positions: Optional[List[str]] = None,
    draft_years: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run calibration and return just the blended weight recommendations per position.

    Returns: {"WR": {"draft_capital": 0.30, "production": 0.15, ...}, "RB": {...}, ...}
    """
    result = run_calibration(draft_years=draft_years, positions=positions)
    out: Dict[str, Dict[str, float]] = {}
    for pos, rec in result.get("weight_recommendations", {}).items():
        out[pos] = rec.get("blended_recommendation", dict(REFINED_POSITION_WEIGHTS.get(pos, {})))
    return out
