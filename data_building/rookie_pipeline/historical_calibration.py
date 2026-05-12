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
import datetime as dt
import io
import json
import logging
import math
import statistics
import urllib.request
from urllib.error import HTTPError
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

# CFBD API - reuse the helper from ingestion.py
try:
    from .ingestion import _cfbd_get
except ImportError:
    _cfbd_get = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_csv(url: str, *, quiet_404: bool = False) -> List[Dict[str, str]]:
    """Download a CSV URL and return list of row dicts.

    Args:
        url: CSV URL to fetch.
        quiet_404: If True, suppress warning logs for HTTP 404 responses.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    except HTTPError as exc:
        if not (quiet_404 and exc.code == 404):
            log.warning("[calibration] Failed to fetch %s: %s", url, exc)
        return []
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
    latest_completed_regular_season = dt.datetime.utcnow().year - 1

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

        last_eval_year = min(draft_year + nfl_data_years - 1, latest_completed_regular_season)
        for nfl_yr in range(draft_year, last_eval_year + 1):
            stat_rows = _fetch_csv(
                _NFLVERSE_BASE.format(year=nfl_yr),
                quiet_404=(nfl_yr >= latest_completed_regular_season),
            )

            if not stat_rows:
                # If a recent season file is unavailable yet (or temporarily missing),
                # stop extending the label window for this class to avoid noisy 404s.
                if nfl_yr >= latest_completed_regular_season:
                    log.info(
                        "[calibration] No NFL player_stats for season %d yet; "
                        "using seasons through %d for draft class %d",
                        nfl_yr,
                        nfl_yr - 1,
                        draft_year,
                    )
                    break
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

def _build_features_from_db(
    draft_years: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Pull rich college features from rookie_prospect_source_data for any draft
    classes that have data stored in the DB.  Returns a name-keyed dict in the
    same format as _build_college_features().

    The DB has advanced metrics (YAC, aDOT, contested_catch_rate, PFF grades,
    elusive_rating, etc.) that CFBD's public API does not expose, making these
    more informative than a pure CFBD fetch.
    """
    try:
        from dashboard_services.db import get_conn
    except ImportError:
        return {}

    features: Dict[str, Dict[str, float]] = {}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        rp.name,
                        rp.position,
                        rp.draft_class_year,
                        rp.age,
                        sd.season,
                        sd.games_played,
                        -- Passing
                        sd.pass_yards, sd.pass_tds, sd.pass_attempts, sd.completions,
                        sd.interceptions,
                        -- Rushing
                        sd.rush_attempts, sd.rush_yards, sd.rush_tds,
                        -- Receiving
                        sd.receptions, sd.targets, sd.receiving_yards, sd.receiving_tds,
                        -- Derived
                        sd.dominator_rating, sd.market_share_yards,
                        sd.yds_per_carry, sd.yds_per_reception, sd.yds_per_attempt,
                        sd.completion_pct, sd.td_int_ratio,
                        -- Team context
                        sd.conference, sd.team_pass_rate,
                        -- Advanced metrics (migration 014)
                        sd.yards_after_catch_per_reception,
                        sd.avg_depth_of_target,
                        sd.contested_catch_rate,
                        sd.drop_rate,
                        sd.grades_offense,
                        sd.breakaway_percentage,
                        sd.elusive_rating,
                        sd.pff_passing_grade,
                        sd.big_time_throw_rate,
                        sd.adjusted_completion_rate
                    FROM rookie_prospects rp
                    JOIN rookie_prospect_source_data sd ON sd.player_id = rp.player_id
                    WHERE rp.draft_class_year = ANY(%s)
                    ORDER BY rp.name, sd.season DESC
                    """,
                    (list(draft_years),)
                )
                rows = cur.fetchall()
    except Exception as exc:
        log.warning("[calibration] DB feature fetch failed: %s", exc)
        return {}

    if not rows:
        return {}

    # Group by player name - use the latest season's advanced metrics, accumulate totals
    player_seasons: Dict[str, List[Any]] = {}
    for row in rows:
        name = str(row["name"] or "").strip().lower()
        if name:
            player_seasons.setdefault(name, []).append(dict(row))

    try:
        from .prospect_model import _conf_quality  # type: ignore
    except ImportError:
        _conf_quality = lambda conf, pos: 0.65  # type: ignore

    for name, seasons in player_seasons.items():
        # Use most recent season for advanced metrics / per-game features
        latest = seasons[0]  # already sorted DESC by season

        gp = max(_safe_float(latest.get("games_played")), 1.0)
        rec_yds   = _safe_float(latest.get("receiving_yards"))
        rec_tds   = _safe_float(latest.get("receiving_tds"))
        rush_yds  = _safe_float(latest.get("rush_yards"))
        rush_tds  = _safe_float(latest.get("rush_tds"))
        pass_yds  = _safe_float(latest.get("pass_yards"))
        pass_tds  = _safe_float(latest.get("pass_tds"))
        rush_att  = _safe_float(latest.get("rush_attempts"))
        pass_att  = _safe_float(latest.get("pass_attempts"))
        recs      = _safe_float(latest.get("receptions"))

        all_yds = rec_yds + rush_yds + pass_yds

        feat: Dict[str, float] = {
            "draft_year": float(latest.get("draft_class_year") or 0),
            # Raw totals
            "rec_yds":   rec_yds,
            "rec_tds":   rec_tds,
            "rush_yds":  rush_yds,
            "pass_yds":  pass_yds,
            # Per-game rates
            "rec_yds_pg":       rec_yds / gp,
            "rush_yds_pg":      rush_yds / gp,
            "pass_yds_pg":      pass_yds / gp,
            "rec_tds_pg":       rec_tds / gp,
            "receptions_pg":    recs / gp,
            "rush_attempts_pg": rush_att / gp,
            "pass_attempts_pg": pass_att / gp,
            "all_yds_pg":       all_yds / gp,
            "games_played":     gp,
        }

        # Efficiency from stored derived fields
        for k in ("dominator_rating", "market_share_yards", "yds_per_carry",
                  "yds_per_reception", "yds_per_attempt", "completion_pct", "td_int_ratio"):
            v = latest.get(k)
            if v is not None:
                feat[k] = _safe_float(v)

        # Competition / environment
        conf = latest.get("conference") or ""
        pos  = str(latest.get("position") or "WR").upper()
        feat["conf_quality"] = _conf_quality(conf, pos)
        if latest.get("team_pass_rate") is not None:
            feat["team_pass_rate"] = _safe_float(latest["team_pass_rate"])

        # Advanced metrics (migration 014 - may be NULL for older classes)
        for k, fk in [
            ("yards_after_catch_per_reception", "yac_per_rec"),
            ("avg_depth_of_target",             "avg_depth_of_target"),
            ("contested_catch_rate",             "contested_catch_rate"),
            ("drop_rate",                        "drop_rate"),
            ("grades_offense",                   "pff_grade"),
            ("breakaway_percentage",             "breakaway_pct"),
            ("elusive_rating",                   "elusive_rating"),
            ("pff_passing_grade",                "pff_passing_grade"),
            ("big_time_throw_rate",              "big_time_throw_rate"),
            ("adjusted_completion_rate",         "adjusted_completion_pct"),
        ]:
            v = latest.get(k)
            if v is not None:
                feat[fk] = _safe_float(v)

        # Age at draft
        if latest.get("age") is not None:
            feat["age_at_draft"] = _safe_float(latest["age"])

        features[name] = feat

    log.info("[calibration] Built DB features for %d players", len(features))
    return features


def _build_college_features(
    draft_years: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Fetch CFBD stats for each draft class and compute predictor features.
    First tries the DB (richer features), then falls back to CFBD API.

    Returns: {player_name_lower: {
        "rec_yds_pg": float, "rec_tds_pg": float, "dominator_rating": float,
        "yac_per_rec": float, "avg_depth_of_target": float,
        "conf_quality": float, "team_pass_rate": float, ...
    }}
    """
    # Start with any DB-stored features (richer; covers the active class)
    db_features = _build_features_from_db(draft_years)

    if _cfbd_get is None:
        log.warning("[calibration] _cfbd_get unavailable - CFBD features will be empty")
        return db_features

    try:
        from .prospect_model import _conf_quality  # type: ignore
    except ImportError:
        _conf_quality = lambda conf, pos: 0.65  # type: ignore

    # CFBD API fetch for historical classes not in DB
    cfbd_features: Dict[str, Dict[str, float]] = {}

    # Per-season accumulator: player → year → stat_type → value
    _season_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    _season_gp:    Dict[str, int] = {}  # player → games played (most recent)

    for draft_year in draft_years:
        log.info("[calibration] Fetching CFBD features for draft class %d", draft_year)

        # Use up to 4 seasons before draft (full college career)
        for yr in [draft_year - 1, draft_year - 2, draft_year - 3, draft_year - 4]:
            try:
                stat_rows = _cfbd_get("/stats/player/season", {"year": yr, "seasonType": "regular"})
            except Exception as exc:
                log.warning("[calibration] CFBD /stats/player/season %d failed: %s", yr, exc)
                continue

            for row in (stat_rows or []):
                player    = (row.get("player") or "").strip().lower()
                team      = (row.get("team")   or "").strip()
                conf      = (row.get("conference") or "").strip()
                stat_type = row.get("statType", "")
                val       = _safe_float(row.get("stat"))
                if not player:
                    continue

                if player not in _season_stats:
                    _season_stats[player] = {}
                if yr not in _season_stats[player]:
                    _season_stats[player][yr] = {
                        "_draft_year": float(draft_year),
                        "_team": team,
                        "_conf": conf,
                    }

                s = _season_stats[player][yr]
                if stat_type == "REC YDS":
                    s["rec_yds"] = s.get("rec_yds", 0) + val
                elif stat_type == "REC TD":
                    s["rec_tds"] = s.get("rec_tds", 0) + val
                elif stat_type == "RUSH YDS":
                    s["rush_yds"] = s.get("rush_yds", 0) + val
                elif stat_type == "RUSH TD":
                    s["rush_tds"] = s.get("rush_tds", 0) + val
                elif stat_type == "PASS YDS":
                    s["pass_yds"] = s.get("pass_yds", 0) + val
                elif stat_type == "PASS TD":
                    s["pass_tds"] = s.get("pass_tds", 0) + val
                elif stat_type == "REC":
                    s["receptions"] = s.get("receptions", 0) + val
                elif stat_type == "RUSH ATT":
                    s["rush_attempts"] = s.get("rush_attempts", 0) + val
                elif stat_type == "PASS ATT":
                    s["pass_attempts"] = s.get("pass_attempts", 0) + val
                elif stat_type == "PASS COMP":
                    s["completions"] = s.get("completions", 0) + val
                elif stat_type == "INT":
                    s["interceptions"] = s.get("interceptions", 0) + val
                elif stat_type == "GP":
                    s["games_played"] = val

    # Collapse _season_stats into cfbd_features using the most recent season
    for player, seasons in _season_stats.items():
        latest_yr = max(seasons.keys())
        s = seasons[latest_yr]
        gp = max(_safe_float(s.get("games_played")), 1.0)

        rec_yds  = _safe_float(s.get("rec_yds"))
        rec_tds  = _safe_float(s.get("rec_tds"))
        rush_yds = _safe_float(s.get("rush_yds"))
        rush_tds = _safe_float(s.get("rush_tds"))
        pass_yds = _safe_float(s.get("pass_yds"))
        pass_tds = _safe_float(s.get("pass_tds"))
        rush_att = _safe_float(s.get("rush_attempts"))
        pass_att = _safe_float(s.get("pass_attempts"))
        recs     = _safe_float(s.get("receptions"))
        comps    = _safe_float(s.get("completions"))
        ints     = _safe_float(s.get("interceptions"))
        all_yds  = rec_yds + rush_yds + pass_yds

        feat: Dict[str, float] = {
            "draft_year":       s.get("_draft_year", 0.0),
            # Totals
            "rec_yds":          rec_yds,
            "rec_tds":          rec_tds,
            "rush_yds":         rush_yds,
            "pass_yds":         pass_yds,
            # Per-game
            "rec_yds_pg":       rec_yds / gp,
            "rush_yds_pg":      rush_yds / gp,
            "pass_yds_pg":      pass_yds / gp,
            "rec_tds_pg":       rec_tds / gp,
            "receptions_pg":    recs / gp,
            "rush_attempts_pg": rush_att / gp,
            "pass_attempts_pg": pass_att / gp,
            "all_yds_pg":       all_yds / gp,
            "games_played":     gp,
        }

        # Efficiency
        if recs > 0:
            feat["yds_per_reception"] = rec_yds / recs
        if rush_att > 0:
            feat["yds_per_carry"] = rush_yds / rush_att
        if pass_att > 0:
            feat["yds_per_attempt"] = pass_yds / pass_att
            feat["completion_pct"]  = comps / pass_att * 100
        if ints > 0 and pass_tds > 0:
            feat["td_int_ratio"] = pass_tds / ints

        # Competition
        conf = s.get("_conf", "")
        pos  = "WR"  # default; CFBD stats don't reliably include position
        feat["conf_quality"] = _conf_quality(conf, pos)

        cfbd_features[player] = feat

    # Merge: DB features win over CFBD (DB has richer metrics)
    merged = {**cfbd_features, **db_features}
    log.info("[calibration] Built college features: %d CFBD + %d DB = %d total",
             len(cfbd_features), len(db_features), len(merged))
    return merged


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
        "draft_capital": 0.22,
        "production": 0.17,
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

# Feature → component mapping (CFBD/DB features → prospect_model components)
_FEATURE_TO_COMPONENT = {
    # Draft
    "draft_capital_score":          "draft_capital",
    # Production - volume
    "rec_yds":                      "production",
    "rec_yds_pg":                   "production",
    "rush_yds":                     "production",
    "rush_yds_pg":                  "production",
    "pass_yds":                     "production",
    "pass_yds_pg":                  "production",
    "rec_tds":                      "production",
    "rec_tds_pg":                   "production",
    "rush_tds":                     "production",
    "pass_tds":                     "production",
    "all_yds_pg":                   "production",
    # Production - share / dominator
    "dominator_rating":             "production",
    "market_share_yards":           "production",
    "pass_share":                   "production",
    # Efficiency
    "yds_per_reception":            "efficiency",
    "yds_per_carry":                "efficiency",
    "yds_per_attempt":              "efficiency",
    "completion_pct":               "efficiency",
    "td_int_ratio":                 "efficiency",
    "yac_per_rec":                  "efficiency",
    "avg_depth_of_target":          "efficiency",
    "contested_catch_rate":         "efficiency",
    "drop_rate":                    "efficiency",
    "pff_grade":                    "efficiency",
    "adjusted_completion_pct":      "efficiency",
    "big_time_throw_rate":          "efficiency",
    # Utilization
    "receptions_pg":                "utilization",
    "targets_pg":                   "utilization",
    "rush_attempts_pg":             "utilization",
    "pass_attempts_pg":             "utilization",
    # Breakout / age
    "breakout_age":                 "breakout",
    "age_at_draft":                 "age",
    "games_played":                 "durability",
    # Athleticism
    "ras_score":                    "athleticism",
    "forty_yard":                   "athleticism",
    "breakaway_pct":                "athleticism",
    "elusive_rating":               "athleticism",
    # Competition / environment
    "conf_quality":                 "competition",
    "team_pass_rate":               "environment",
    "sagarin_team_rating":          "environment",
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
        log.warning("[calibration] No NFL outcome data - calibration cannot proceed")
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

    # Build summary (fix: use per-position current weights, not undefined global)
    summary_lines = ["=== Historical Calibration Results ===", f"Target: {target}", ""]
    for pos in positions:
        corr = correlations.get(pos, {})
        if not corr:
            continue
        curr_weights = REFINED_POSITION_WEIGHTS.get(pos, {})
        summary_lines.append(f"── {pos} ──")
        top = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        for feat, r in top:
            summary_lines.append(f"  {feat:<35s} r={r:+.3f}")
        rec = weight_recommendations.get(pos, {}).get("blended_recommendation", {})
        if rec:
            summary_lines.append(f"  Recommended component weights:")
            for comp, w in sorted(rec.items(), key=lambda x: -x[1]):
                curr = curr_weights.get(comp, 0)
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
