"""
Prospect evaluation model — position-aware, multi-factor scoring.

Each component score is 0-100.  Final prospect_score is a weighted sum.

Component weights:
    projected_draft_capital_score 30 %   NFL draft position (position-weighted)
                                         RB/WR/TE: 1.20-1.25x (early picks are gold)
                                         QB: 0.65x (top picks common, less predictive)
    production_score             15 %   college production volume (elite producers translate)
    athleticism_score            12 %   combine / speed score / RAS
    breakout_profile_score       10 %   early-career dominance trajectory
    utilization_score             5 %   opportunity share (targets/carries per game)
    efficiency_score              8 %   per-attempt / per-target efficiency
    competition_score             8 %   conference + opponent quality (Notre Dame: 0.94)
    age_score                     6 %   age-adjusted production; youth premium
    environment_adjustment        3 %   team scheme / usage context
    durability_score              3 %   games missed, injury history
    ──────────────────────────────────
    Total                       100 %

Position-weighted draft capital + day 3 penalty (pick > 64: 0.80x)

Position-specific adjustments are baked into each scorer to handle the
different stat profiles of QB / RB / WR / TE.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _scale(raw: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] → [0, 100], clipped."""
    if hi <= lo:
        return 50.0
    return _clip((raw - lo) / (hi - lo) * 100.0)


def _sigmoid_score(x: float, midpoint: float, steepness: float = 1.0) -> float:
    """Logistic curve scaled to 0-100, centred at `midpoint`."""
    z = steepness * (x - midpoint)
    return round(100.0 / (1.0 + math.exp(-z)), 2)


def _best_season(seasons: List[Dict], key: str) -> float:
    vals = [_safe(s.get(key)) for s in seasons if s.get(key) is not None]
    return max(vals) if vals else 0.0


def _latest_season(seasons: List[Dict]) -> Optional[Dict]:
    if not seasons:
        return None
    return max(seasons, key=lambda s: _safe(s.get("season"), 0))


def _career_seasons(seasons: List[Dict]) -> int:
    return len(seasons)


# ─────────────────────────────────────────────────────────────────────────────
# Conference quality table
# Power 5 / big non-conf → higher competition score
# ─────────────────────────────────────────────────────────────────────────────

CONF_QUALITY: Dict[str, float] = {
    # Power 2
    "SEC":               1.00,
    "Big Ten":           1.00,

    # Upper Power Tier
    "Big 12":            0.90,
    "ACC":               0.88,

    # Legacy Pac (if still used)
    "Pac-12":            0.89,   # slightly above ACC historically

    # Elite Independent
    "Notre Dame":        0.94,

    # Top G5
    "American":          0.78,

    # Mid G5
    "Mountain West":     0.70,
    "Sun Belt":          0.66,

    # Lower G5
    "MAC":               0.60,
    "CUSA":              0.56,

    # Independents (non-ND)
    "BYU":               0.84,
    "Army":              0.68,
    "Liberty":           0.66,
    "UMass":             0.60,
    "New Mexico State":  0.60,

    # Fallback bucket
    "FBS Independents":  0.70,

    # FCS
    "FCS":               0.48,
}

DEFAULT_CONF_QUALITY = 0.62


def _conf_quality(conference: Optional[str]) -> float:
    if not conference:
        return DEFAULT_CONF_QUALITY
    for k, v in CONF_QUALITY.items():
        if k.lower() in conference.lower():
            return v
    return DEFAULT_CONF_QUALITY


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_production_season(season: Dict, pos: str) -> float:
    """
    Compute the raw production score (pre-transfer-penalty) for a single season.
    Returns the weighted component score (0-100).
    """
    gp = max(_safe(season.get("games_played"), 12), 1)

    if pos == "WR":
        rec_yds_pg = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg = _safe(season.get("receiving_tds"))   / gp
        dom        = _safe(season.get("dominator_rating"))
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        prod = (
            _scale(rec_yds_pg, 40,  120) * 0.45 +
            _scale(rec_tds_pg, 0.3, 1.0) * 0.30 +
            _scale(dom,        0.10, 0.45) * 0.25
        )
        # Red zone proxy: TDs per 100 receiving yards — rewards goal-line separators
        if total_rec_yds >= 200:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 8.0:   prod = _clip(prod * 1.06)
            elif rz_rate >= 5.5: prod = _clip(prod * 1.03)
        return prod

    elif pos == "RB":
        rush_yds_pg = _safe(season.get("rush_yards"))     / gp
        rec_yds_pg  = _safe(season.get("receiving_yards")) / gp
        all_yds_pg  = (
            _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        ) / gp
        tds_pg      = (
            _safe(season.get("rush_tds")) + _safe(season.get("receiving_tds"))
        ) / gp
        dom         = _safe(season.get("dominator_rating"))
        ypc         = _safe(season.get("yds_per_carry"))

        prod = (
            _scale(rush_yds_pg, 40,  160) * 0.35 +
            _scale(all_yds_pg,  50,  180) * 0.25 +
            _scale(tds_pg,      0.5,  2.0) * 0.25 +
            _scale(dom,         0.15, 0.70) * 0.15
        )
        if rec_yds_pg >= 20:
            prod = _clip(prod * 1.10)
        if ypc >= 6.5:
            prod = _clip(prod * 1.08)
        if dom >= 0.30:
            prod = _clip(prod * 1.12)
        # Red zone proxy: TDs per 100 total yards
        total_yds = _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        total_tds = _safe(season.get("rush_tds"))   + _safe(season.get("receiving_tds"))
        if total_yds >= 300:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 6.0:   prod = _clip(prod * 1.05)
            elif rz_rate >= 4.0: prod = _clip(prod * 1.02)
        return prod

    elif pos == "QB":
        pass_yds_pg = _safe(season.get("pass_yards")) / gp
        tds_pg      = _safe(season.get("pass_tds"))   / gp
        comp_pct    = _safe(season.get("completion_pct"), 60.0)
        ypa         = _safe(season.get("yds_per_attempt"), 7.0)
        td_int      = _safe(season.get("td_int_ratio"),    2.0)
        prod = (
            _scale(pass_yds_pg, 180, 380) * 0.30 +
            _scale(tds_pg,        1.5,  3.5) * 0.25 +
            _scale(comp_pct,     60.0, 76.0) * 0.20 +
            _scale(ypa,           6.5,  10.5) * 0.15 +
            _scale(td_int,        1.5,   6.0) * 0.10
        )
        # Mobile QB bonus: rushing production adds significant fantasy value
        rush_yds_pg    = _safe(season.get("rush_yards")) / gp
        rush_tds_season = _safe(season.get("rush_tds"))
        if rush_yds_pg >= 30:
            prod = _clip(prod * 1.08)   # 8% for QB with meaningful rushing
        if rush_yds_pg >= 50:
            prod = _clip(prod * 1.05)   # additional 5% for elite rushing QB
        if rush_tds_season >= 5:
            prod = _clip(prod * 1.04)   # bonus for multi-TD rushing QBs
        return prod

    elif pos == "TE":
        rec_yds_pg    = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg    = _safe(season.get("receiving_tds"))   / gp
        dom           = _safe(season.get("dominator_rating"))
        rec_pg        = _safe(season.get("receptions"))       / gp
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        prod = (
            _scale(rec_yds_pg, 30,  95)  * 0.40 +
            _scale(rec_tds_pg, 0.2, 0.8) * 0.30 +
            _scale(dom,        0.08, 0.30) * 0.15 +
            _scale(rec_pg,     2.0,  7.0) * 0.15
        )
        # Red zone proxy: TE goal-line usage is extremely valuable in NFL
        if total_rec_yds >= 150:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 9.0:   prod = _clip(prod * 1.07)
            elif rz_rate >= 6.0: prod = _clip(prod * 1.04)
        return prod

    return 40.0


def calc_production_score(seasons: List[Dict], position: str) -> float:
    """
    Per-game production vs position-specific elite thresholds.
    Uses a blend of the best season and latest season to capture both
    peak value and recent performance.

    Transfer penalty: Players who transfer to weaker conferences get
    production discounted, as stats may be inflated by weaker competition.
    """
    if not seasons:
        return 40.0  # neutral when no data

    pos = position.upper()
    ls  = _latest_season(seasons) or {}
    gp  = max(_safe(ls.get("games_played"), 12), 1)

    # Check for conference downgrade (transfer to weaker competition)
    transfer_penalty = 1.0
    if len(seasons) >= 2:
        sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0))
        conf_qualities = []
        for season in sorted_seasons:
            conf = season.get("conference", "")
            team = season.get("team", "") or ""
            if "notre dame" in team.lower():
                conf_qualities.append(0.94)
            else:
                conf_qualities.append(_conf_quality(conf))

        # Check for conference quality change across transfers
        for i in range(1, len(conf_qualities)):
            prev_quality = conf_qualities[i-1]
            curr_quality = conf_qualities[i]
            delta = curr_quality - prev_quality  # positive = upgrade, negative = downgrade

            if delta <= -0.20:
                # Downward transfer: discount stats that may be inflated by weaker competition
                drop_magnitude = -delta / prev_quality
                penalty = min(0.35, drop_magnitude * 0.75)   # 15–35% penalty
                transfer_penalty *= (1.0 - penalty)
                break
            elif delta >= 0.15:
                # Upward transfer: producing in a stronger conference is harder —
                # reward the player (e.g. FCS → SEC and still put up numbers)
                upgrade_bonus = min(0.12, (delta / prev_quality) * 0.60)  # up to +12%
                transfer_penalty *= (1.0 + upgrade_bonus)
                break

    if pos not in ("WR", "RB", "QB", "TE"):
        return 40.0

    # With only one season there is nothing to blend — just score it directly.
    if len(seasons) == 1:
        return _clip(_score_production_season(ls, pos) * transfer_penalty)

    # Score latest season and best individual season; blend to reward peak while
    # still weighting recent output (NFL analysts evaluate both)
    latest_score = _score_production_season(ls, pos)

    # Find best season by primary volume metric per position
    _PEAK_KEY = {"WR": "receiving_yards", "RB": "rush_yards", "QB": "pass_yards", "TE": "receiving_yards"}
    peak_key = _PEAK_KEY[pos]
    peak_season = max(seasons, key=lambda s: _safe(s.get(peak_key)), default=ls)
    best_score = _score_production_season(peak_season, pos)

    # 85% weight on whichever is higher (recent or peak), 15% on the other
    prod = max(latest_score, best_score) * 0.85 + min(latest_score, best_score) * 0.15

    # Apply transfer penalty to discourage stat inflation from weak competition
    return _clip(prod * transfer_penalty)


def calc_efficiency_score(seasons: List[Dict], position: str) -> float:
    """
    Per-attempt / per-target efficiency.  Rewards quality over quantity.

    Uses the latest season as the primary signal.  When multiple seasons are
    available a consistency bonus (±5) is applied: sustained high efficiency
    across seasons is more predictive than a single-year peak.
    """
    if not seasons:
        return 45.0

    pos = position.upper()
    ls  = _latest_season(seasons) or {}

    if pos == "WR":
        ypr = _safe(ls.get("yds_per_reception"), 10.0)
        ms  = _safe(ls.get("market_share_yards"))
        eff = _scale(ypr, 9.0, 18.0) * 0.60 + _scale(ms, 0.10, 0.45) * 0.40

    elif pos == "RB":
        ypc   = _safe(ls.get("yds_per_carry"), 4.5)
        ms    = _safe(ls.get("market_share_yards"))
        ypr   = _safe(ls.get("yds_per_reception"), 7.0)
        eff   = (
            _scale(ypc,  3.5,  7.5)  * 0.55 +
            _scale(ms,   0.20, 0.75) * 0.25 +
            _scale(ypr,  5.0, 12.0)  * 0.20
        )

    elif pos == "QB":
        ypa   = _safe(ls.get("yds_per_attempt"), 7.0)
        cpct  = _safe(ls.get("completion_pct"),  62.0)
        td_int= _safe(ls.get("td_int_ratio"),     2.0)
        eff   = (
            _scale(ypa,   6.5, 10.5) * 0.45 +
            _scale(cpct, 60.0, 76.0) * 0.30 +
            _scale(td_int, 1.5,  7.0) * 0.25
        )

    elif pos == "TE":
        ypr = _safe(ls.get("yds_per_reception"), 9.0)
        ms  = _safe(ls.get("market_share_yards"))
        # Catch rate: receptions / targets measures hands + separation reliability.
        # Only computed when targets is populated (not always available from CFBD).
        targets = _safe(ls.get("targets"))
        recs    = _safe(ls.get("receptions"))
        catch_rate_score = 50.0  # neutral default when targets unknown
        if targets > 0:
            catch_rate = recs / targets
            catch_rate_score = _scale(catch_rate, 0.55, 0.82)  # 55% → 0, 82%+ → 100
        eff = (
            _scale(ypr, 8.0, 16.0) * 0.50 +
            _scale(ms,  0.05, 0.30) * 0.30 +
            catch_rate_score         * 0.20
        )

    else:
        return 45.0

    # Multi-season consistency bonus: ±5 points based on whether the key
    # efficiency metric held up across seasons (sustained efficiency > one-year wonder)
    if len(seasons) >= 2:
        _KEY = {"WR": "yds_per_reception", "RB": "yds_per_carry",
                "QB": "yds_per_attempt",   "TE": "yds_per_reception"}
        key = _KEY.get(pos)
        if key:
            vals = [_safe(s.get(key)) for s in seasons if s.get(key) is not None]
            if len(vals) >= 2:
                avg_val = sum(vals) / len(vals)
                latest_val = _safe(ls.get(key))
                # Bonus if latest ≈ or exceeds multi-year average; penalty if big drop
                consistency = (latest_val - avg_val) / max(avg_val, 0.1)
                eff = _clip(eff + _clip(consistency * 10, -5.0, 5.0))

    return _clip(eff)


# Typical draft-class age by position (age at start of NFL rookie year).
# Updated to reflect modern college football (COVID year, grad transfers, etc.)
_TYPICAL_AGE = {"QB": 23.5, "RB": 22.0, "WR": 22.5, "TE": 23.0}
_AGE_ELITE   = {"QB": 22.0, "RB": 20.5, "WR": 21.0, "TE": 21.5}
_AGE_WORST   = {"QB": 27.5, "RB": 25.0, "WR": 25.5, "TE": 26.0}   # QB more lenient — development timelines vary widely


def calc_age_score(age: Optional[float], draft_year: int, position: str) -> float:
    """
    Younger prospects earn a premium.  Age is evaluated relative to draft class.
    A 20-year-old RB producing at a high level is worth more than a 23-year-old.
    """
    if age is None:
        return 50.0  # neutral default
    # Convert decimal to float if needed
    age = float(age)
    pos = position.upper()
    elite = _AGE_ELITE.get(pos, 21.5)
    worst = _AGE_WORST.get(pos, 26.0)
    # Invert: lower age → higher score.  _scale maps [elite, worst] → [100, 0].
    score = _scale(worst - age, 0.0, worst - elite)
    # Floor at 20 so age never completely destroys a player's score; cap at 95.
    return _clip(score, 20, 95)


def calc_breakout_score(seasons: List[Dict], age: Optional[float], position: str) -> float:
    """
    Rewards early-career dominance and upward trajectory.

    Key signals:
    - Was the best season achieved at ≤20 (WR/RB/TE) or ≤21 (QB)?
    - Did production increase year-over-year?
    - Is dominator_rating above the breakout threshold?
    """
    if not seasons or len(seasons) < 1:
        return 40.0

    pos = position.upper()
    sorted_s = sorted(seasons, key=lambda s: _safe(s.get("season"), 0))
    ls = sorted_s[-1]  # most recent season

    # Dominator breakout threshold by position.
    # QB dominator_rating is receiving-based and not meaningful for QBs — use neutral.
    dom_thresh = {"WR": 0.20, "RB": 0.35, "TE": 0.12}
    dom = _safe(ls.get("dominator_rating"))
    thresh = dom_thresh.get(pos)

    if pos == "QB":
        # QB dominator proxy: pass yards as share of team total yards
        # A QB generating ≥60% of team total yards is an alpha
        pass_yds = _safe(ls.get("pass_yards"))
        team_yds = _safe(ls.get("team_total_yards"))
        if team_yds > 0:
            qb_share = pass_yds / team_yds
            dom_score = _scale(qb_share, 0.40, 0.70)
        else:
            # Fallback: raw pass yards as proxy for workload dominance
            gp = max(_safe(ls.get("games_played"), 12), 1)
            pass_yds_pg = pass_yds / gp
            dom_score = _scale(pass_yds_pg, 200, 380)
    elif thresh is None:
        dom_score = 50.0  # unknown position
    elif dom > 0:
        dom_score = min(100, max(0, (dom / thresh - 1) * 50 + 60))
    else:
        dom_score = 30.0

    # Trajectory: did production grow?
    traj_score = 50.0
    if len(sorted_s) >= 2:
        prev = sorted_s[-2]
        curr_yds = _safe(ls.get("receiving_yards", 0)) + _safe(ls.get("rush_yards", 0)) + _safe(ls.get("pass_yards", 0)) * 0.5
        prev_yds = _safe(prev.get("receiving_yards", 0)) + _safe(prev.get("rush_yards", 0)) + _safe(prev.get("pass_yards", 0)) * 0.5
        if prev_yds > 0:
            growth = (curr_yds - prev_yds) / prev_yds
            traj_score = _clip(_scale(growth, -0.20, 0.60))

    # Adjusted breakout age — continuous scale calibrated by position.
    # Uses the age the player was in the FIRST season they hit the dom threshold,
    # not their current age.  A WR who dominated at 18 is a generational rarity;
    # a WR who first broke out at 22 as a senior is ordinary.
    #
    # Scale anchors (WR/RB/TE):  ≤19 → 30 pts,  20 → 20,  21 → 10,  22 → 0,  23+ → -10
    # Scale anchors (QB):        ≤19 → 25 pts,  20 → 18,  21 → 10,  22 → 0,  23+ → -8
    adj_breakout_score = 0.0
    if age is not None:
        # Find age at first breakout season
        breakout_age = age  # default to current age
        if thresh is not None and sorted_s:
            current_year = _safe(sorted_s[-1].get("season"), 0)
            for s in sorted_s:
                s_year = _safe(s.get("season"), 0)
                if _safe(s.get("dominator_rating")) >= thresh and s_year > 0 and current_year > 0:
                    breakout_age = age - (current_year - s_year)
                    break

        if pos == "QB":
            # QBs develop later: age 18.5→100, age 23.5→0
            # Inverted: lower age = higher score, so compute (max_age - age) / range
            raw = (23.5 - breakout_age) / (23.5 - 18.5) * 100.0
            adj_breakout_score = _clip(raw, 0.0, 100.0)
        else:
            # Skill positions (WR/RB/TE): age 18.5→100, age 23.0→0
            raw = (23.0 - breakout_age) / (23.0 - 18.5) * 100.0
            adj_breakout_score = _clip(raw, 0.0, 100.0)

    score = dom_score * 0.45 + traj_score * 0.35 + adj_breakout_score * 0.20
    return _clip(score)


# Position-specific weights for athleticism metrics.
# Reflects NFL scouting priorities: speed matters most for WR, explosiveness for RB,
# overall athleticism (RAS) for QB, catching radius (vertical) for TE.
# three_cone / shuttle = agility/change-of-direction, important for WR routes and RB open-field.
_ATH_WEIGHTS: Dict[str, Dict[str, float]] = {
    "WR": {"forty": 0.35, "vertical": 0.20, "broad": 0.15, "ras": 0.15,
           "speed_score": 0.25, "three_cone": 0.10, "shuttle": 0.05},
    "RB": {"speed_score": 0.30, "broad": 0.25, "forty": 0.15, "vertical": 0.10,
           "ras": 0.15, "three_cone": 0.05},
    "QB": {"ras": 0.40, "forty": 0.35, "vertical": 0.15, "broad": 0.10},
    "TE": {"vertical": 0.30, "ras": 0.25, "broad": 0.20, "forty": 0.15,
           "three_cone": 0.10},
}

# Maximum athleticism score when metric coverage is sparse.
# Prevents a single elite metric (e.g. one 4.28 40-time) from yielding a top score
# when we have no idea about the rest of the athletic profile.
_ATH_DATA_CAPS = {1: 72, 2: 84, 3: 93}   # n_metrics_present → cap


def calc_athleticism_score(athleticism: Dict[str, Any], position: str) -> float:
    """
    Combine / pro-day metrics.  Falls back gracefully to positional median (55)
    when data is missing.

    RAS (Relative Athletic Score) is the most reliable single signal (0-10).
    Uses position-specific metric weights (e.g., 40-time matters more for WR,
    vertical for TE, speed score for RB).
    """
    if not athleticism:
        return 55.0

    pos = position.upper()

    # Compute individual metric scores (each 0-100)
    metric_scores: Dict[str, float] = {}

    # RAS (0-10 → scale to 0-100)
    ras = athleticism.get("ras_score")
    if ras is not None:
        metric_scores["ras"] = _scale(_safe(ras), 4.0, 10.0)

    # 40-yard dash (position-specific thresholds, inverted: faster = higher score)
    forty_raw = athleticism.get("forty_yard")
    if forty_raw:
        forty = _safe(forty_raw)
        thresholds = {
            "WR":  (4.25, 4.65),
            "RB":  (4.30, 4.65),
            "QB":  (4.40, 5.00),
            "TE":  (4.45, 4.90),
        }
        lo, hi = thresholds.get(pos, (4.30, 4.90))
        metric_scores["forty"] = _scale(hi - forty, 0.0, hi - lo)

    # Vertical jump
    vert = athleticism.get("vertical_inches")
    if vert:
        metric_scores["vertical"] = _scale(_safe(vert), 28.0, 44.0)

    # Broad jump
    broad = athleticism.get("broad_jump_in")
    if broad:
        metric_scores["broad"] = _scale(_safe(broad), 100, 140)

    # Speed score = weight * (40^4)^-1 * normalisation constant
    weight_lbs = athleticism.get("weight_lbs")
    if forty_raw and weight_lbs:
        ss_raw = (_safe(weight_lbs) * 200) / (_safe(forty_raw) ** 4)
        # Elite ~115+, average ~100
        metric_scores["speed_score"] = _scale(ss_raw, 80.0, 130.0)

    # Agility / change-of-direction (three_cone and short_shuttle).
    # Inverted: lower time = higher score.
    three_cone = athleticism.get("three_cone")
    if three_cone:
        # Elite ~6.45s, average ~7.0s, poor ~7.4s
        metric_scores["three_cone"] = _scale(_safe(three_cone), 7.40, 6.45)

    shuttle = athleticism.get("short_shuttle")
    if shuttle:
        # Elite ~3.95s, average ~4.25s, poor ~4.55s
        metric_scores["shuttle"] = _scale(_safe(shuttle), 4.55, 3.95)

    if not metric_scores:
        return 55.0

    # Weighted average using position-specific weights
    pos_weights = _ATH_WEIGHTS.get(pos, {})
    if pos_weights:
        weighted_sum = 0.0
        total_weight = 0.0
        for metric, score in metric_scores.items():
            w = pos_weights.get(metric, 0.10)  # small default weight for unlisted metrics
            weighted_sum += score * w
            total_weight += w
        if total_weight > 0:
            raw = _clip(weighted_sum / total_weight)
        else:
            raw = _clip(sum(metric_scores.values()) / len(metric_scores))
    else:
        raw = _clip(sum(metric_scores.values()) / len(metric_scores))

    # Apply data completeness cap: a single outstanding metric should not yield
    # an elite athleticism score when the rest of the profile is unknown.
    n = len(metric_scores)
    cap = _ATH_DATA_CAPS.get(n, 100)   # 4+ metrics → no cap
    return _clip(raw, 0.0, float(cap))


def calc_utilization_score(seasons: List[Dict], position: str) -> float:
    """
    Volume of opportunity — how much of the team's offence ran through this player.

    Distinct from production (yards/TDs) and efficiency (per-touch quality):
    utilization measures raw opportunity share, which is highly predictive of
    early NFL role (a high-target WR translates to a high-target role).

    WR / TE: targets per game  (if targets unavailable, receptions per game as proxy)
    RB:      rush attempts per game + targets per game
    QB:      pass attempts per game (workload / usage signal)
    """
    if not seasons:
        return 50.0

    pos = position.upper()
    ls  = _latest_season(seasons) or {}
    gp  = max(_safe(ls.get("games_played"), 12), 1)

    if pos == "WR":
        targets = _safe(ls.get("targets"))
        if targets > 0:
            tpg = targets / gp
        else:
            # Fallback: receptions as proxy (underestimates slightly)
            tpg = _safe(ls.get("receptions")) / gp
        # Elite WR: 9+ targets/game; strong: 6+; average: 4
        return _clip(_scale(tpg, 3.0, 10.0))

    elif pos == "TE":
        targets = _safe(ls.get("targets"))
        if targets > 0:
            tpg = targets / gp
        else:
            tpg = _safe(ls.get("receptions")) / gp
        # Elite TE: 7+ targets/game; strong: 4+; average: 2.5
        return _clip(_scale(tpg, 2.0, 8.0))

    elif pos == "RB":
        carries  = _safe(ls.get("rush_attempts")) / gp
        rec_tgts = _safe(ls.get("targets"))
        rec_pg   = (rec_tgts / gp) if rec_tgts > 0 else (_safe(ls.get("receptions")) / gp)
        # Combines rush volume + receiving involvement
        # Elite: 20+ carries + 4+ targets; average: 12 carries + 2 targets
        rush_util = _scale(carries, 8.0, 22.0)
        recv_util = _scale(rec_pg,  1.0,  5.0)
        return _clip(rush_util * 0.70 + recv_util * 0.30)

    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        # Elite: 35+ attempts/game; average: 25; game-manager: 18
        return _clip(_scale(att_pg, 16.0, 40.0))

    return 50.0


def calc_competition_score(seasons: List[Dict]) -> float:
    """
    Conference quality + implied opponent strength.
    Recent seasons are weighted more heavily to reflect current competition level.
    Transfer players who upgraded conferences get credit for their most recent context.
    """
    if not seasons:
        return 55.0

    # Sort seasons by year (most recent first)
    sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0), reverse=True)

    total_weight = 0.0
    weighted_quality = 0.0

    for i, season in enumerate(sorted_seasons):
        conf = season.get("conference", "")
        team = (season.get("team", "") or "").lower()

        # Notre Dame gets a flat quality bonus regardless of conference label
        if "notre dame" in team:
            quality = 0.94
        else:
            quality = _conf_quality(conf)

        # Recency weights: 1.0, 0.8, 0.6, 0.4, 0.2 for up to 5 seasons
        weight = max(0.2, 1.0 - i * 0.2)
        weighted_quality += quality * weight
        total_weight += weight

    avg_quality = weighted_quality / total_weight if total_weight > 0 else DEFAULT_CONF_QUALITY
    return _clip(_scale(avg_quality, 0.45, 1.00))


def calc_environment_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Adjusts for team usage patterns.

    - High pass rate for skill players (WR/TE) is better context (more targets)
    - For RBs, high rush rate inflates volume — slight discount applied
    - Uses a recency-weighted average of team pass rate across all seasons
      (same decay as competition score) so transferred players aren't locked
      to their latest team's scheme.
    """
    if not seasons:
        return 50.0

    pos = position.upper()

    # Recency-weighted pass rate: most recent season weight 1.0, decaying by 0.2
    sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0), reverse=True)
    weighted_sum = 0.0
    total_weight = 0.0
    for i, s in enumerate(sorted_seasons):
        pr = _safe(s.get("team_pass_rate"), 0.52)
        w  = max(0.2, 1.0 - i * 0.2)
        weighted_sum += pr * w
        total_weight += w
    pass_rate = weighted_sum / total_weight if total_weight > 0 else 0.52

    if pos in ("WR", "TE"):
        # More passing = more opportunities; 55-65% pass rate is ideal
        base = _scale(pass_rate, 0.42, 0.68)
    elif pos == "RB":
        # RBs benefit from balanced / run-heavy but it's easier to produce in bad offences
        # Slight penalty for very pass-heavy (≥65%) teams since it reduces rush attempts
        if pass_rate > 0.62:
            base = _scale(1.0 - pass_rate, 0.30, 0.55)
        else:
            base = _scale(pass_rate, 0.35, 0.62) * 0.5 + 50 * 0.5
    else:
        base = 50.0

    return _clip(base)


def calc_durability_score(seasons: List[Dict]) -> float:
    """
    Games played vs expected, with recency weighting.

    - A missed-game season 3 years ago matters less than last year's injury
    - Most recent season carries the most weight
    - Extra penalty when the most recent season was short (≤8 games)
    - 12-game base; 14+ excellent; 8 or fewer triggers meaningful penalty
    """
    if not seasons:
        return 60.0

    sorted_s = sorted(seasons, key=lambda s: s.get("season", 0))
    gp_entries = [
        (i, _safe(s.get("games_played")))
        for i, s in enumerate(sorted_s)
        if s.get("games_played") is not None
    ]
    if not gp_entries:
        return 60.0

    n = len(sorted_s)
    # Recency weights: most recent season gets highest weight
    weighted_sum = 0.0
    total_weight = 0.0
    for rank, gp in gp_entries:
        # rank 0 = oldest; rank n-1 = most recent
        w = 0.4 + 0.6 * (rank / max(n - 1, 1))
        weighted_sum += gp * w
        total_weight += w

    avg_gp = weighted_sum / total_weight

    # Most-recent season penalty: if the player just had a short season it's a red flag
    recent_gp = gp_entries[-1][1]  # last entry is most recent with gp data
    recent_penalty = max(0.0, (9.0 - recent_gp) * 6.0) if recent_gp < 9 else 0.0

    # Career floor: any season with very few games is concerning
    min_gp = min(gp for _, gp in gp_entries)
    floor_pen = max(0.0, (7.0 - min_gp) * 4.0)

    avg_score = _scale(avg_gp, 7.0, 14.0)
    return _clip(avg_score - recent_penalty - floor_pen)


# ─────────────────────────────────────────────────────────────────────────────
# Fantasy translation bonus
# ─────────────────────────────────────────────────────────────────────────────

# Position-level adjustment reflecting long-run fantasy value scarcity.
# TE premium is negative because TE translation to fantasy is hardest.
POSITION_FANTASY_MULT: Dict[str, float] = {
    "WR": 1.05,
    "RB": 1.00,
    "QB": 0.90,   # QBs are less valued in 1QB dynasty
    "TE": 0.92,
}
POSITION_FANTASY_MULT_SF: Dict[str, float] = {
    "WR": 1.00,
    "RB": 0.98,
    "QB": 1.20,   # QBs get big SF premium
    "TE": 0.90,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "production":      0.15,    # College production volume (elite producers translate)
    "utilization":     0.05,    # Opportunity share (targets/game, carries/game)
    "efficiency":      0.08,    # Per-attempt quality (less than volume but additive)
    "age":             0.06,    # Age-adjusted production; youth premium
    "breakout":        0.10,    # Early-career dominance trajectory
    "athleticism":     0.12,    # Combine / speed score / RAS
    "competition":     0.08,    # Conference + opponent quality
    "environment":     0.03,    # Team scheme / usage context
    "durability":      0.03,    # Games missed, injury history
    "draft_capital":   0.30,    # NFL draft position is king (position-weighted)
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


def score_prospect(
    prospect: Dict[str, Any],
    draft_capital: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run all component scorers and produce a final prospect_score.

    Args:
        prospect:      Normalised prospect dict from ingestion.py
        draft_capital: Consensus dict from mock_draft_consensus.py

    Returns:
        Dict with all component scores + final prospect_score + reasons.
    """
    pos      = (prospect.get("position") or "WR").upper()
    seasons  = prospect.get("seasons") or []
    age      = prospect.get("age")
    ath      = prospect.get("athleticism") or {}
    dy       = int(prospect.get("draft_class_year") or 2026)

    production_score    = calc_production_score(seasons, pos)
    utilization_score   = calc_utilization_score(seasons, pos)
    efficiency_score    = calc_efficiency_score(seasons, pos)
    age_score           = calc_age_score(age, dy, pos)
    breakout_score      = calc_breakout_score(seasons, age, pos)
    athleticism_score   = calc_athleticism_score(ath, pos)
    competition_score   = calc_competition_score(seasons)
    environment_score   = calc_environment_adjustment(seasons, pos)
    durability_score    = calc_durability_score(seasons)

    if draft_capital:
        dc_score = _safe(draft_capital.get("projected_draft_capital_score"), 40.0)
    else:
        # Default to mid-round 5 pick (~150) when no mock data available
        # Players without any mock buzz are typically day 3 picks or UDFAs
        # Pick 150 → ~6 draft capital score (late day 3)
        from data_building.rookie_pipeline.mock_draft_consensus import pick_to_draft_capital_score
        dc_score = pick_to_draft_capital_score(150)  # ~6 score for late day 3

    # Position-specific draft capital multipliers
    # Key insight: QBs go early often, but RB/WR/TE going top-10 is HUGE
    # WR taken in top 10 = elite prospect, QB taken top 10 = happens every year
    dc_multiplier = {
        "WR": 1.25,   # Early WR picks are gold (rare and predictive)
        "RB": 1.20,   # Early RB picks are premium (high opportunity)
        "TE": 1.15,   # Early TE picks are valuable (rare to go early)
        "QB": 0.65,   # QB draft capital less predictive for fantasy (deep position)
    }.get(pos, 1.00)

    # Apply Day 3 penalty: picks outside top 2 rounds (pick > 64) are discounted.
    # Position-specific: QBs frequently go late but still start; RBs/WRs drafted late
    # rarely become WR1/RB1s; TEs are more position-scarcity resistant.
    if draft_capital:
        projected_pick = draft_capital.get("projected_pick")
        if projected_pick and projected_pick > 64:
            day3_penalty = {
                "QB": 0.65,   # Late QBs rarely become starters; major upside compression
                "WR": 0.80,   # Day 3 WRs have tough path to targets
                "RB": 0.82,   # Late RBs can hit via workload but odds are worse
                "TE": 0.90,   # TEs develop slowly; Day 3 still viable
            }.get(pos, 0.80)
            dc_multiplier *= day3_penalty

    dc_score_adjusted = _clip(dc_score * dc_multiplier)

    prospect_score = (
        production_score      * WEIGHTS["production"]    +
        utilization_score     * WEIGHTS["utilization"]   +
        efficiency_score      * WEIGHTS["efficiency"]    +
        age_score             * WEIGHTS["age"]           +
        breakout_score        * WEIGHTS["breakout"]      +
        athleticism_score     * WEIGHTS["athleticism"]   +
        competition_score     * WEIGHTS["competition"]   +
        environment_score     * WEIGHTS["environment"]   +
        durability_score      * WEIGHTS["durability"]    +
        dc_score_adjusted     * WEIGHTS["draft_capital"]
    )

    # Graduated generational prospect boost.
    # Tiered by how many elite markers align: production, athleticism, draft capital.
    # Only applies to non-QB skill positions (fantasy value ceiling clearer).
    #
    # Tier 3 (8% boost): all three elite — Ja'Marr Chase / Bijan Robinson tier
    # Tier 2 (5% boost): two of three elite — solid top-10 calibre
    # Tier 1 (3% boost): one elite marker + draft capital — high-upside prospect
    if pos in ("WR", "RB", "TE"):
        elite_prod = production_score >= 72
        elite_ath  = athleticism_score >= 78
        elite_dc   = dc_score >= 85  # Top-10 pick

        elite_count = sum([elite_prod, elite_ath, elite_dc])
        if elite_count >= 3:
            prospect_score *= 1.08
        elif elite_count == 2 and elite_dc:
            prospect_score *= 1.05
        elif elite_count >= 1 and elite_dc:
            prospect_score *= 1.03
        prospect_score = _clip(prospect_score)

    prospect_score = round(prospect_score, 2)

    # Quality-weighted confidence score.
    # Each data source is weighted by how much it improves scoring accuracy.
    # Draft capital + multiple mocks = highest confidence signal.
    ls_check = _latest_season(seasons) or {}
    conf_points = 0.0
    if seasons:
        conf_points += 15.0  # any season data
        if len(seasons) >= 2:
            conf_points += 10.0  # multi-season data (more reliable trends)
        if ls_check.get("dominator_rating"):
            conf_points += 10.0  # dominator rating available (key metric)
        if ls_check.get("games_played"):
            conf_points += 5.0   # exact games played (not default 12)
        if ls_check.get("targets") is not None:
            conf_points += 5.0   # target data (WR/TE evaluation quality)
    if age is not None:
        conf_points += 15.0  # age known (critical for age_score reliability)
    if ath:
        conf_points += 15.0  # any athleticism data
        if len(ath) >= 4:
            conf_points += 10.0  # full combine profile
    if draft_capital:
        conf_points += 10.0  # at least one mock
        n_mocks = draft_capital.get("num_mocks_used", 1)
        if n_mocks >= 5:
            conf_points += 5.0   # consensus from multiple analysts
    confidence_score = round(_clip(conf_points), 1)

    fantasy_translation = round(
        POSITION_FANTASY_MULT.get(pos, 1.0) * prospect_score, 2
    )

    # Human-readable reasons
    reasons = _build_reasons(
        prospect, pos, seasons,
        production_score, utilization_score, efficiency_score, age_score,
        breakout_score, athleticism_score, competition_score,
        environment_score, durability_score, dc_score, draft_capital,
    )

    return {
        "player_id":                    prospect["player_id"],
        "draft_class_year":             dy,
        "production_score":             round(production_score, 2),
        "utilization_score":            round(utilization_score, 2),
        "efficiency_score":             round(efficiency_score, 2),
        "age_score":                    round(age_score, 2),
        "breakout_profile_score":       round(breakout_score, 2),
        "athleticism_score":            round(athleticism_score, 2),
        "competition_score":            round(competition_score, 2),
        "environment_adjustment":       round(environment_score, 2),
        "durability_score":             round(durability_score, 2),
        "projected_draft_capital_score":round(dc_score, 2),
        "fantasy_translation_score":    round(fantasy_translation, 2),
        "confidence_score":             confidence_score,
        "prospect_score":               prospect_score,
        "key_reasons":                  reasons,
    }


def _build_reasons(
    prospect, pos, seasons,
    prod, util, eff, age_sc, break_sc, ath_sc, comp_sc,
    env_sc, dur_sc, dc_sc, dc_dict
) -> str:
    """Build a bullet-point string summarising the prospect's strengths/flags."""
    bullets: List[str] = []
    ls    = _latest_season(seasons) or {}
    age   = prospect.get("age")
    gp    = max(_safe(ls.get("games_played"), 12), 1)

    # ── Production ────────────────────────────────────────────────────────────
    if prod >= 75:
        bullets.append("Elite production profile — dominant volume for their position")
    elif prod >= 55:
        bullets.append("Solid production numbers with room to grow at the NFL level")
    else:
        bullets.append("Limited production volume — may need time to develop")

    # ── Utilization ───────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        tpg = (_safe(ls.get("targets")) or _safe(ls.get("receptions"))) / gp
        if util >= 70:
            bullets.append(f"High target share ({tpg:.1f} tgts/game) — clear featured receiver")
        elif util <= 35 and seasons:
            bullets.append(f"Limited target volume ({tpg:.1f} tgts/game) — role player or committee")
    elif pos == "RB":
        carries = _safe(ls.get("rush_attempts")) / gp
        if util >= 70:
            bullets.append(f"Workhorse RB usage ({carries:.1f} carries/game)")
        elif util <= 35:
            bullets.append(f"Limited rush volume ({carries:.1f} carries/game) — situational role")
    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        if util >= 70:
            bullets.append(f"High-volume passing attack ({att_pg:.0f} att/game)")

    # ── Efficiency ────────────────────────────────────────────────────────────
    if eff >= 75:
        bullets.append("High efficiency metrics (yards per touch, market share) stand out")
    elif eff <= 35:
        bullets.append("Below-average efficiency — volume stats may overstate true impact")

    # ── Dominator rating ──────────────────────────────────────────────────────
    dom = _safe(ls.get("dominator_rating"))
    if dom >= 0.35 and pos in ("WR", "RB"):
        bullets.append(f"Team dominator rating {dom:.0%} — commanded an outsized share of team production")
    elif dom >= 0.20 and pos == "TE":
        bullets.append(f"Strong target share for a TE — {dom:.0%} team dominator")

    # ── Red zone proxy ────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        total_yds = _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("receiving_tds"))
        if total_yds >= 200:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 7.0:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) — dangerous in red zone")
    elif pos == "RB":
        total_yds = _safe(ls.get("rush_yards")) + _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("rush_tds"))   + _safe(ls.get("receiving_tds"))
        if total_yds >= 300:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 5.5:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) — goal-line threat")

    # ── Age / adjusted breakout age ───────────────────────────────────────────
    if age_sc >= 80:
        bullets.append(f"Exceptional age-adjusted production — elite output very young ({age:.1f} yrs)")
    elif age_sc < 40 and age is not None:
        bullets.append(f"Age concern: {age:.1f} yrs is older than typical for this position")

    if break_sc >= 75:
        bullets.append("Clear breakout trajectory — early-career dominance at a young age")
    elif break_sc >= 60:
        bullets.append("Solid breakout profile — consistent improvement season-over-season")

    # ── Athleticism ───────────────────────────────────────────────────────────
    ath      = prospect.get("athleticism", {}) or {}
    ras      = ath.get("ras_score")
    forty    = ath.get("forty_yard")
    tc       = ath.get("three_cone")
    shuttle  = ath.get("short_shuttle")
    ath_msgs = []
    if ras is not None and _safe(ras) >= 9.0:
        ath_msgs.append(f"RAS {_safe(ras):.1f}/10")
    if forty is not None and _safe(forty) <= 4.35 and pos in ("WR", "RB"):
        ath_msgs.append(f"{_safe(forty):.2f}s 40-yd")
    if tc is not None and _safe(tc) <= 6.55 and pos in ("WR", "RB"):
        ath_msgs.append(f"{_safe(tc):.2f}s 3-cone (elite agility)")
    if ath_msgs:
        bullets.append(f"Elite athleticism: {', '.join(ath_msgs)}")
    elif ath_sc <= 35 and ath:
        bullets.append("Below-average athleticism profile — will need to win on scheme/technique")

    # ── Competition ───────────────────────────────────────────────────────────
    conf = ls.get("conference", "")
    if comp_sc >= 70:
        bullets.append(f"Production came against quality competition ({conf})")
    elif comp_sc <= 40:
        bullets.append(f"Played at lower competition level ({conf}) — discount applied")

    # ── Transfer history ──────────────────────────────────────────────────────
    if prospect.get("transfer_history"):
        th = prospect["transfer_history"]
        if isinstance(th, list) and len(th) >= 1:
            bullets.append(f"Transfer: {' → '.join(th)} — context matters for production evaluation")
        elif isinstance(th, str):
            bullets.append(f"Transfer history: {th}")

    # ── Draft capital ─────────────────────────────────────────────────────────
    if dc_dict:
        pick  = dc_dict.get("projected_pick")
        rnd   = dc_dict.get("projected_round")
        n_mocks = dc_dict.get("num_mocks_used", 0)
        conf_pct = dc_dict.get("consensus_confidence", 0)
        if pick and rnd:
            conf_str = f", {conf_pct:.0f}% consensus" if n_mocks >= 3 else ""
            bullets.append(
                f"Projected pick #{pick} (Round {rnd}) across {n_mocks} mock drafts"
                f"{conf_str} — draft capital score {dc_sc:.0f}/100"
            )

    # ── Durability ────────────────────────────────────────────────────────────
    if dur_sc <= 40:
        recent_gp = _safe(ls.get("games_played"))
        if recent_gp and recent_gp < 10:
            bullets.append(f"Durability concern: only {int(recent_gp)} games in most recent season")
        else:
            bullets.append("Durability concern: missed significant games in college career")

    # ── Early declare ─────────────────────────────────────────────────────────
    if prospect.get("early_declare"):
        bullets.append("Early declarant — chose to enter draft before exhausting eligibility")

    return "\n".join(f"• {b}" for b in bullets)


# ─────────────────────────────────────────────────────────────────────────────
# Batch scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_all_prospects(
    prospects: List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Score a list of prospects and add overall_rank / position_rank.

    Args:
        prospects:     List of normalised prospect dicts
        consensus_map: {player_id: consensus_dict}

    Returns:
        List of score dicts, sorted by prospect_score descending.
    """
    if consensus_map is None:
        consensus_map = {}

    scores = []
    for p in prospects:
        dc = consensus_map.get(p["player_id"])
        scores.append(score_prospect(p, dc))

    # Sort overall
    scores.sort(key=lambda x: x["prospect_score"], reverse=True)
    for i, s in enumerate(scores):
        s["overall_rank"] = i + 1

    # Position ranks
    pos_counters: Dict[str, int] = {}
    for s in sorted(scores, key=lambda x: x["prospect_score"], reverse=True):
        # Need position from prospect list
        pid  = s["player_id"]
        pos  = next((p["position"].upper() for p in prospects if p["player_id"] == pid), "UNK")
        pos_counters[pos] = pos_counters.get(pos, 0) + 1
        s["position_rank"] = pos_counters[pos]

    return scores
