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

Position-weighted draft capital, day-tier multipliers (Day 1 R1 > Day 2 R2-3 > Day 3 R4-7)

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
    # Power 2 — abbreviations and full names (CFBD returns full names)
    "SEC":                       1.00,
    "Southeastern":              1.00,   # "Southeastern Conference"
    "Big Ten":                   1.00,
    "Big 10":                    1.00,

    # Upper Power Tier
    "Big 12":                    0.90,
    "Big Twelve":                0.90,
    "ACC":                       0.88,
    "Atlantic Coast":            0.88,   # "Atlantic Coast Conference"

    # Legacy Pac (if still used)
    "Pac-12":                    0.89,
    "Pac 12":                    0.89,
    "Pacific-12":                0.89,

    # Power 4 (2024+ Big 12 expansion / new branding)
    "Big East":                  0.82,

    # Elite Independent
    "Notre Dame":                0.94,

    # Top G5
    "American":                  0.78,
    "American Athletic":         0.78,

    # Mid G5
    "Mountain West":             0.70,
    "Sun Belt":                  0.66,

    # Lower G5
    "MAC":                       0.60,
    "Mid-American":              0.60,
    "CUSA":                      0.56,
    "Conference USA":            0.56,

    # Independents (non-ND)
    "BYU":                       0.84,
    "Army":                      0.68,
    "Liberty":                   0.66,
    "UMass":                     0.60,
    "New Mexico State":          0.60,

    # Fallback bucket
    "FBS Independents":          0.70,
    "FBS Independent":           0.70,

    # FCS
    "FCS":                       0.48,
}

DEFAULT_CONF_QUALITY = 0.72   # unknown ≈ neutral, not penalised like a weak G5

# ─────────────────────────────────────────────────────────────────────────────
# Sagarin team-caliber dominator adjustment
# Applied multiplicatively to the WR/TE pass-share metric.
# Alabama 2020 (predictor ~99) → +6.47% cap.
# Non-D1 / unrated           → -9.3% floor.
# ─────────────────────────────────────────────────────────────────────────────
_SAGARIN_FBS_AVG = 75.0           # approximate mean Sagarin predictor for FBS teams
_SAGARIN_SCALE   = 0.0647 / 24.0  # ≈ 0.00270 per rating point above avg
                                   # (99 − 75) × 0.00270 = 0.0648 → capped at +6.47%

# ─────────────────────────────────────────────────────────────────────────────
# Scheme-inflation penalties
# High-volume / spread systems where college WR stats are poor NFL predictors.
# Applied as a multiplier to production AND efficiency scores for WRs only.
# Tennessee (Heupel air-raid): very high ADOT, schemed targets, historically
#   weak NFL translation for receivers outside the #1 option.
# Ole Miss (Kiffin spread): similar volume inflation at lower severity.
# ─────────────────────────────────────────────────────────────────────────────
SCHEME_INFLATION_SYSTEMS: Dict[str, float] = {
    "tennessee": 0.60,
    "ole miss":  0.80,
}


def _scheme_inflation_discount(team: Optional[str]) -> float:
    """Return the production/efficiency discount multiplier for known stat-inflating WR systems."""
    return SCHEME_INFLATION_SYSTEMS.get((team or "").strip().lower(), 1.0)


def _conf_quality(conference: Optional[str]) -> float:
    if not conference:
        return DEFAULT_CONF_QUALITY
    conf_lower = conference.lower()
    for k, v in CONF_QUALITY.items():
        if k.lower() in conf_lower:
            return v
    return DEFAULT_CONF_QUALITY


def _sagarin_dom_adj(rating: Optional[float], conference: Optional[str] = None) -> float:
    """
    Convert a Sagarin CFB predictor rating into a multiplicative dominator
    adjustment for WR/TE pass-share scoring.

    Bounds (user-calibrated):
        None + FCS/non-D1 conference → -9.3%  floor
        None + FBS conference        →  0.0%  neutral (Sagarin fetch failed)
        FBS average (~75)            →  0.0%
        Alabama 2020 (~99)           → +6.47% cap

    The conference-aware fallback prevents historical backtest years from
    incorrectly penalising every FBS player when the Sagarin page for that
    season year is unavailable.  _conf_quality() returns 0.48 for FCS and
    higher for all FBS conferences, so 0.50 is a clean dividing line.
    """
    if rating is None:
        if _conf_quality(conference) >= 0.50:
            return 0.0    # FBS team — Sagarin unavailable, apply no adjustment
        return -0.093     # Confirmed FCS / non-D1
    raw = (rating - _SAGARIN_FBS_AVG) * _SAGARIN_SCALE
    return max(-0.093, min(0.0647, raw))


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_production_season(season: Dict, pos: str, skip_sagarin: bool = False) -> float:
    """
    Compute the raw production score (pre-transfer-penalty) for a single season.
    Returns the weighted component score (0-100).
    """
    # Guard against bad games-played values (0/1) that can explode per-game rates
    # and incorrectly saturate production to 100.
    gp_raw = _safe(season.get("games_played"), 12)
    gp = gp_raw if gp_raw >= 4 else 12

    if pos == "WR":
        rec_yds_pg    = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg    = _safe(season.get("receiving_tds"))   / gp
        dom           = _safe(season.get("dominator_rating"))
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        team_pass_yds = _safe(season.get("team_pass_yards"))
        sag_adj       = 0.0 if skip_sagarin else _sagarin_dom_adj(
                            season.get("sagarin_team_rating"), season.get("conference"))

        # Pass-share dominator: rec yards as % of team passing yards, adjusted
        # for team caliber via Sagarin predictor rating.
        # Falls back to legacy total-offense dominator if team_pass_yards absent.
        if team_pass_yds > 0:
            pass_share = (total_rec_yds / team_pass_yds) * (1 + sag_adj)
            dom_score  = _scale(pass_share, 0.08, 0.35)
        else:
            dom_score  = _scale(dom * (1 + sag_adj), 0.08, 0.40)

        prod = (
            _scale(rec_yds_pg, 32,  120) * 0.40 +
            _scale(rec_tds_pg, 0.25, 0.9) * 0.30 +
            dom_score                      * 0.30
        )
        # YAC bonus: dynamic after-catch ability is a strong NFL translation signal
        yac = _safe(season.get("yards_after_catch_per_reception"))
        if yac >= 7.0:
            prod = _clip(prod * 1.10)
        elif yac >= 5.5:
            prod = _clip(prod * 1.05)
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
            _scale(rush_yds_pg, 20,  150) * 0.18 +
            _scale(all_yds_pg,  30,  180) * 0.17 +
            _scale(tds_pg,       0.4,  2.0) * 0.25 +
            _scale(dom,          0.12, 0.75) * 0.15 +
            _scale(ypc,          3.5,  9.0) * 0.25
        )
        # Receiving tiers: require meaningful rec share so dedicated pass-catchers
        # (Coleman ~34%) are rewarded differently from incidental receivers (Johnson ~20%).
        # Use an additive multiplier budget (single final clip) to reduce 100-point
        # ceiling effects from stacked multiplicative boosts.
        mult = 1.0
        rec_share = rec_yds_pg / max(all_yds_pg, 1.0)
        if rec_yds_pg >= 20 and rec_share >= 0.28:
            mult += 0.08   # dedicated pass-catcher
        elif rec_yds_pg >= 20 and rec_share >= 0.22:
            mult += 0.06   # strong receiving back
        elif rec_yds_pg >= 15:
            mult += 0.03   # incidental receiver
        if ypc >= 6.0:
            mult += 0.03
        if dom >= 0.30:
            mult += 0.05
        # Red zone proxy: TDs per 100 total yards
        total_yds = _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        total_tds = _safe(season.get("rush_tds"))   + _safe(season.get("receiving_tds"))
        if total_yds >= 300:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 6.0:
                mult += 0.03
            elif rz_rate >= 4.0:
                mult += 0.015

        mult = min(mult, 1.16)  # hard cap so one-season RB production doesn't trivially max
        return _clip(prod * mult)

    elif pos == "QB":
        pass_yds_pg    = _safe(season.get("pass_yards")) / gp
        tds_pg         = _safe(season.get("pass_tds"))   / gp
        comp_pct       = _safe(season.get("completion_pct"), 60.0)
        ypa            = _safe(season.get("yds_per_attempt"), 7.0)
        td_int         = _safe(season.get("td_int_ratio"),    2.0)
        rush_yds_pg    = _safe(season.get("rush_yards")) / gp
        rush_tds_season = _safe(season.get("rush_tds"))

        # Rushing QBs: elite college rushers (Lamar, Kyler, Jalen Hurts) are
        # systematically penalised by pure passing metrics. Rushing is a direct
        # fantasy component in the NFL, not just a tie-breaker.
        # Architecture: 85% passing composite + 15% rushing component so that
        # a pocket passer can still max out (rushing = 0 → no penalty) while
        # a dual-threat who rushes 80+ yd/game adds a full 15 pts on top.
        # Completion% weight reduced (0.25→0.18) because spread/RPO systems
        # inflate college comp% (Mac Jones 77%) while dual-threat systems
        # suppress it without signalling worse NFL potential.
        pass_comp = (
            _scale(pass_yds_pg, 150, 330) * 0.22 +
            _scale(tds_pg,        1.5,  3.5) * 0.22 +
            _scale(comp_pct,     58.0, 76.0) * 0.18 +
            _scale(ypa,           6.5,  10.5) * 0.28 +
            _scale(td_int,        1.5,   6.0) * 0.10
        )
        rush_comp = _scale(rush_yds_pg, 10.0, 85.0)  # 0–100; Lamar ~90+ yd/g → near 100

        prod = _clip(pass_comp * 0.85 + rush_comp * 0.15)

        # Extra multiplier for elite rushing + TDs (dual-threat upside)
        if rush_yds_pg >= 70 and rush_tds_season >= 8:
            prod = _clip(prod * 1.06)
        elif rush_yds_pg >= 50 and rush_tds_season >= 5:
            prod = _clip(prod * 1.03)
        return prod

    elif pos == "TE":
        rec_yds_pg    = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg    = _safe(season.get("receiving_tds"))   / gp
        dom           = _safe(season.get("dominator_rating"))
        rec_pg        = _safe(season.get("receptions"))       / gp
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        team_pass_yds = _safe(season.get("team_pass_yards"))
        sag_adj       = 0.0 if skip_sagarin else _sagarin_dom_adj(
                            season.get("sagarin_team_rating"), season.get("conference"))

        # Pass-share dominator for TEs — same logic as WR, narrower scale
        # (TEs command a smaller share of passing yards than WRs).
        if team_pass_yds > 0:
            pass_share = (total_rec_yds / team_pass_yds) * (1 + sag_adj)
            dom_score  = _scale(pass_share, 0.04, 0.22)
        else:
            dom_score  = _scale(dom * (1 + sag_adj), 0.05, 0.20)

        prod = (
            _scale(rec_yds_pg, 20,  85)  * 0.35 +
            _scale(rec_tds_pg, 0.12, 0.5) * 0.30 +
            dom_score                      * 0.20 +
            _scale(rec_pg,     1.0,  6.5) * 0.15
        )
        # YAC bonus: move TEs who create after the catch are more NFL-translatable
        te_yac = _safe(season.get("yards_after_catch_per_reception"))
        if te_yac >= 5.0:
            prod = _clip(prod * 1.08)
        elif te_yac >= 3.5:
            prod = _clip(prod * 1.04)
        # Red zone proxy: TE goal-line usage is extremely valuable in NFL
        if total_rec_yds >= 200:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 10.0:
                prod = _clip(prod * 1.05)
            elif rz_rate >= 7.0:
                prod = _clip(prod * 1.02)
        return prod

    return 52.0


def _eval_metric_value(eval_metrics: Optional[Dict], name: str, min_confidence: float = 0.0):
    """
    Safely extract a metric value from an eval_metrics dict.

    eval_metrics has the shape: {metric_name: {value: ..., confidence: ..., ...}}
    Returns None if metric absent, None value, or confidence below threshold.
    """
    if not eval_metrics:
        return None
    entry = eval_metrics.get(name)
    if not isinstance(entry, dict):
        return None
    confidence = entry.get("confidence") or 0.0
    if confidence < min_confidence:
        return None
    return entry.get("value")


def _eval_metric_percent(
    eval_metrics: Optional[Dict],
    name: str,
    min_confidence: float = 0.0,
) -> Optional[float]:
    """
    Read an eval metric and normalize to 0-100 percent scale.
    Accepts either [0,1] rates or already-scaled percentages.
    """
    val = _eval_metric_value(eval_metrics, name, min_confidence=min_confidence)
    if val is None:
        return None
    f = _safe(val, default=0.0)
    return f * 100.0 if 0.0 <= f <= 1.0 else f


def calc_production_score(
    seasons: List[Dict],
    position: str,
    eval_metrics: Optional[Dict] = None,
    skip_sagarin: bool = False,
) -> float:
    """
    Per-game production vs position-specific elite thresholds.
    Uses a blend of the best season and latest season to capture both
    peak value and recent performance.

    Transfer penalty: Players who transfer to weaker conferences get
    production discounted, as stats may be inflated by weaker competition.

    eval_metrics (optional): If the evaluation pipeline produced yprr/yac_per_att
    metrics, these are used to apply a confidence-weighted production adjustment:
    - WR/TE: yprr proxy ≥ 1.8 boosts score; < 1.2 deflates slightly (±15% max)
    - RB:    yac_per_att proxy adjusts production ±10%
    The adjustment is capped intentionally to prevent proxies from dominating.
    """
    if not seasons:
        return 52.0  # pre-draft neutral: unknown ≠ bad

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
        return _clip(_score_production_season(ls, pos, skip_sagarin) * transfer_penalty)

    # Score latest season and best individual season; blend to reward peak while
    # still weighting recent output (NFL analysts evaluate both)
    latest_score = _score_production_season(ls, pos, skip_sagarin)

    # Find best season by primary volume metric per position
    _PEAK_KEY = {"WR": "receiving_yards", "RB": "rush_yards", "QB": "pass_yards", "TE": "receiving_yards"}
    peak_key = _PEAK_KEY[pos]
    peak_season = max(seasons, key=lambda s: _safe(s.get(peak_key)), default=ls)
    best_score = _score_production_season(peak_season, pos, skip_sagarin)

    # 85% weight on whichever is higher (recent or peak), 15% on the other
    prod = max(latest_score, best_score) * 0.85 + min(latest_score, best_score) * 0.15

    # Apply transfer penalty to discourage stat inflation from weak competition
    prod = _clip(prod * transfer_penalty)

    # ── Eval-metric blending ─────────────────────────────────────────────────
    # When evaluation pipeline metrics are available, apply a confidence-weighted
    # adjustment.  Each adjustment is intentionally small (±10–15%) so that low-
    # confidence proxies don't override the primary stat-based score.
    pos = position.upper()
    if eval_metrics and pos in ("WR", "TE"):
        yprr = _eval_metric_value(eval_metrics, "yprr", min_confidence=0.30)
        if yprr is not None:
            # 1.8 yprr ≈ strong college producer; 1.2 ≈ below average
            yprr_adj = min(1.15, max(0.85, float(yprr) / 1.8))
            confidence = (eval_metrics.get("yprr") or {}).get("confidence", 0.35)
            blend = 0.80 + 0.20 * confidence  # max 20% weight from this proxy
            prod = _clip(prod * (blend + (1.0 - blend) * yprr_adj))

    elif eval_metrics and pos == "RB":
        yac = _eval_metric_value(eval_metrics, "yac_per_att", min_confidence=0.35)
        if yac is not None:
            # 2.5 yac ≈ average RB; 4.0 ≈ elite contact runner
            yac_adj = min(1.10, max(0.90, float(yac) / 2.5))
            confidence = (eval_metrics.get("yac_per_att") or {}).get("confidence", 0.40)
            blend = 0.85 + 0.15 * confidence
            prod = _clip(prod * (blend + (1.0 - blend) * yac_adj))

    # Position-specific advanced metric bonuses from rookie source data.
    if eval_metrics and pos in ("WR", "TE"):
        ccr = _eval_metric_percent(eval_metrics, "contested_catch_rate", min_confidence=0.45)
        if ccr is not None:
            prod = _clip(prod + _clip((ccr - 45.0) * 0.10, -4.0, 4.0))

        adot = _eval_metric_value(eval_metrics, "avg_depth_of_target", min_confidence=0.45)
        if adot is not None:
            adot_bonus = _clip((float(adot) - 8.0) * 0.6, -3.0, 4.0)
            # High ADOT in an air-raid system with poor contested-catch ability is
            # scheme-generated volume, not a separation skill signal.
            if float(adot) > 14.0 and ccr is not None and ccr < 50.0:
                adot_bonus = min(adot_bonus, 0.0)
            prod = _clip(prod + adot_bonus)

    elif eval_metrics and pos == "RB":
        elusive = _eval_metric_value(eval_metrics, "elusive_rating", min_confidence=0.45)
        if elusive is not None:
            # Scale-based and symmetric: 0 at elusive=90 (roughly average), up to
            # +5 at 130 (elite), down to -4 at 55 (poor). Symmetry reduces ceiling lock.
            elusive_delta = _clip((float(elusive) - 90.0) / 40.0, -0.8, 1.0)
            prod = _clip(prod + (elusive_delta * 5.0))

        breakaway = _eval_metric_percent(eval_metrics, "explosive_run_rate", min_confidence=0.40)
        if breakaway is not None:
            # Scale-based and symmetric: 0 at 20% (average), +4 at 40% elite, -3 at 8%.
            breakaway_delta = _clip((breakaway - 20.0) / 20.0, -1.0, 1.0)
            prod = _clip(prod + (breakaway_delta * 4.0))

    elif eval_metrics and pos == "QB":
        pff_pass = _eval_metric_value(eval_metrics, "pff_passing_grade", min_confidence=0.45)
        if pff_pass is not None:
            prod = _clip(prod + _clip((float(pff_pass) - 70.0) * 0.20, -4.0, 7.0))

        btt = _eval_metric_percent(eval_metrics, "big_time_throw_rate", min_confidence=0.45)
        if btt is None:
            btt = _eval_metric_percent(eval_metrics, "btt_rate", min_confidence=0.45)
        if btt is not None:
            prod = _clip(prod + _clip((btt - 4.0) * 0.9, -3.0, 5.0))

    # Scheme-inflation discount: WR stats from high-volume spread systems translate
    # poorly to the NFL.  Apply a multiplier before returning.
    if pos == "WR":
        scheme_discount = _scheme_inflation_discount(ls.get("team"))
        if scheme_discount < 1.0:
            prod = _clip(prod * scheme_discount)

    return prod


def calc_efficiency_score(
    seasons: List[Dict],
    position: str,
    eval_metrics: Optional[Dict] = None,
) -> float:
    """
    Per-attempt / per-target efficiency.  Rewards quality over quantity.

    Uses the latest season as the primary signal.  When multiple seasons are
    available a consistency bonus (±5) is applied: sustained high efficiency
    across seasons is more predictive than a single-year peak.

    eval_metrics (optional): When available and confidence is sufficient:
    - QB: adjusted_comp_pct replaces raw completion_pct; twp_rate supplements td_int
    - WR/TE: tprr proxy supplements yds_per_reception via a soft blend
    """
    if not seasons:
        return 52.0  # pre-draft neutral: unknown ≠ bad

    pos = position.upper()
    ls  = _latest_season(seasons) or {}

    if pos == "WR":
        ypr = _safe(ls.get("yds_per_reception"), 10.0)
        ms  = _safe(ls.get("market_share_yards"))
        eff = _scale(ypr, 8.5, 17.0) * 0.55 + _scale(ms, 0.08, 0.40) * 0.35

        # Real PFF YPRR — centered adjustment: 2.0 = neutral, 2.8 = +8, 1.2 = -8
        yprr = _eval_metric_value(eval_metrics, "yprr", min_confidence=0.75)
        if yprr is not None:
            yprr_score = _scale(float(yprr), 1.2, 2.8)
            eff = _clip(eff + (yprr_score - 50.0) * 0.16)

        # PFF route running grade — strong predictor of NFL separation ability
        route_grade = _eval_metric_value(eval_metrics, "grades_pass_route", min_confidence=0.75)
        if route_grade is not None:
            route_score = _scale(float(route_grade), 58.0, 90.0)
            eff = _clip(eff + (route_score - 50.0) * 0.10)

        # Success rate vs. press — key NFL readiness indicator
        press_sr = _eval_metric_value(eval_metrics, "success_rate_vs_press", min_confidence=0.70)
        if press_sr is not None:
            press_score = _scale(float(press_sr), 55.0, 85.0)
            eff = _clip(eff + (press_score - 50.0) * 0.06)

        # Route target rate from RP — centered adjustment: 30% = neutral, 42% = +6, 18% = -6
        rtr = _eval_metric_value(eval_metrics, "route_target_rate", min_confidence=0.75)
        if rtr is not None:
            tprr_equiv = float(rtr) / 100.0
            rtr_score  = _scale(tprr_equiv, 0.18, 0.42)
            eff = _clip(eff + (rtr_score - 50.0) * 0.12)
        else:
            # Tprr proxy — lower priority fallback when RP data absent
            tprr = _eval_metric_value(eval_metrics, "tprr", min_confidence=0.30)
            if tprr is not None:
                tprr_score = _scale(float(tprr), 0.18, 0.42)
                tprr_conf  = (eval_metrics.get("tprr") or {}).get("confidence", 0.35)
                eff = eff * (1.0 - 0.08 * tprr_conf) + tprr_score * (0.08 * tprr_conf)

    elif pos == "RB":
        ypc   = _safe(ls.get("yds_per_carry"), 4.25)
        ms    = _safe(ls.get("market_share_yards"))
        ypr   = _safe(ls.get("yds_per_reception"), 7.0)
        eff   = (
            _scale(ypc,  3.5,  7.0)  * 0.60 +
            _scale(ms,   0.10, 0.45) * 0.15 +
            _scale(ypr,  5.0, 12.0)  * 0.25   # increased: receiving efficiency matters for dynasty RBs
        )
        # PFF pass route grade — predicts pass-game role and long-term dynasty value
        route_grade = _eval_metric_value(eval_metrics, "grades_pass_route", min_confidence=0.70)
        if route_grade is not None:
            route_score = _scale(float(route_grade), 55.0, 85.0)
            eff = _clip(eff + (route_score - 50.0) * 0.08)

    elif pos == "QB":
        ypa   = _safe(ls.get("yds_per_attempt"), 7.0)
        td_int= _safe(ls.get("td_int_ratio"),     2.0)

        # Use adjusted_comp_pct from eval pipeline when available and confident;
        # otherwise fall back to raw completion_pct from college stats.
        adj_cpct = _eval_metric_percent(eval_metrics, "adjusted_comp_pct", min_confidence=0.55)
        if adj_cpct is not None:
            cpct = _safe(adj_cpct, 62.0)
        else:
            cpct = _safe(ls.get("completion_pct"), 62.0)

        # twp_rate proxy (interception rate) can supplement td_int signal.
        # Lower twp_rate → better decision-making → slightly boost td_int weight.
        twp = _eval_metric_value(eval_metrics, "twp_rate", min_confidence=0.55)
        if twp is not None:
            # twp_rate in [0, 5]; lower is better.  Treat as mild modifier on td_int score.
            twp_penalty = _clip(float(twp) / 5.0, 0.0, 1.0)  # 0 = great, 1 = bad
            td_int_mod = td_int * (1.0 + 0.15 * (1.0 - twp_penalty))
        else:
            td_int_mod = td_int

        eff   = (
            _scale(ypa,      6.5, 10.5) * 0.45 +
            _scale(cpct,    60.0, 76.0) * 0.30 +
            _scale(td_int_mod, 1.5, 7.0) * 0.25
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
            catch_rate_score = _scale(catch_rate, 0.60, 0.85)  # 55% → 0, 82%+ → 100
        eff = (
            _scale(ypr, 7.5, 14.0) * 0.50 +
            _scale(ms,  0.04, 0.25) * 0.30 +
            catch_rate_score         * 0.20
        )
        # Supplement with tprr proxy when available
        tprr = _eval_metric_value(eval_metrics, "tprr", min_confidence=0.30)
        if tprr is not None:
            tprr_score = _scale(float(tprr), 0.22, 0.48)
            tprr_conf  = (eval_metrics.get("tprr") or {}).get("confidence", 0.35)
            eff = eff * (1.0 - 0.08 * tprr_conf) + tprr_score * (0.08 * tprr_conf)

    else:
        return 52.0

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

    # Advanced efficiency adjustments from evaluation metrics.
    if eval_metrics and pos in ("WR", "TE"):
        drop_rate = _eval_metric_percent(eval_metrics, "drop_rate", min_confidence=0.45)
        if drop_rate is not None:
            # Lower drop rate is better.
            eff = _clip(eff + _clip((7.0 - drop_rate) * 1.1, -6.0, 6.0))

        yac_rec = _eval_metric_value(eval_metrics, "yac_per_att", min_confidence=0.45)
        if yac_rec is not None:
            eff = _clip(eff + _clip((float(yac_rec) - 5.0) * 1.6, -4.0, 6.0))

    elif eval_metrics and pos == "RB":
        pff_rush = _eval_metric_value(eval_metrics, "pff_rushing_grade", min_confidence=0.45)
        if pff_rush is not None:
            eff = _clip(eff + _clip((float(pff_rush) - 68.0) * 0.22, -5.0, 7.0))

    elif eval_metrics and pos == "QB":
        adj_cpct = _eval_metric_percent(eval_metrics, "adjusted_comp_pct", min_confidence=0.45)
        if adj_cpct is not None:
            eff = _clip(eff + _clip((adj_cpct - 65.0) * 0.35, -5.0, 7.0))

        psr = _eval_metric_value(eval_metrics, "nfl_passer_rating", min_confidence=0.45)
        if psr is not None:
            eff = _clip(eff + _clip((float(psr) - 85.0) * 0.18, -4.0, 6.0))

        p2s = _eval_metric_percent(eval_metrics, "pressure_to_sack_rate", min_confidence=0.45)
        if p2s is not None:
            # Lower pressure-to-sack conversion is better QB pocket behavior.
            eff = _clip(eff + _clip((20.0 - p2s) * 0.35, -5.0, 5.0))

    # Scheme-inflation discount: WR efficiency metrics in high-volume spread
    # systems are scheme-inflated (e.g. yds/rec boosted by deep ADOT).
    if pos == "WR":
        scheme_discount = _scheme_inflation_discount(ls.get("team"))
        if scheme_discount < 1.0:
            eff = _clip(eff * scheme_discount)

    return _clip(eff)


# Typical draft-class age by position (age at start of NFL rookie year).
# Updated to reflect modern college football (COVID year, grad transfers, etc.)
_TYPICAL_AGE = {"QB": 23.0, "RB": 22.5, "WR": 22.5, "TE": 23.0}
_AGE_ELITE   = {"QB": 21.0, "RB": 20.5, "WR": 21.0, "TE": 20.5}
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
        return 52.0

    pos = position.upper()
    sorted_s = sorted(seasons, key=lambda s: _safe(s.get("season"), 0))
    ls = sorted_s[-1]  # most recent season

    # Dominator breakout threshold by position.
    # QB dominator_rating is receiving-based and not meaningful for QBs — use neutral.
    dom_thresh = {"WR": 0.25, "RB": 0.275, "TE": 0.12}
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

        # Dominance quality gate: the early-age bonus should reflect HOW dominant
        # the player was, not merely that they cleared the threshold.  A player
        # barely above threshold (dom ≈ thresh) should not receive the same age
        # credit as a player who doubled the threshold.
        # Scale: dom = thresh → 0% credit; dom = 2×thresh → 100% credit.
        if thresh is not None and thresh > 0 and pos != "QB":
            if dom > thresh:
                dominance_factor = min(1.0, (dom / thresh - 1.0))
                adj_breakout_score *= dominance_factor
            else:
                adj_breakout_score = 0.0

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
_ATH_DATA_CAPS = {1: 74, 2: 82, 3: 92}   # n_metrics_present → cap


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
        return _clip(_scale(tpg, 2.5, 10.0))

    elif pos == "TE":
        targets = _safe(ls.get("targets"))
        if targets > 0:
            tpg = targets / gp
        else:
            tpg = _safe(ls.get("receptions")) / gp
        # Elite TE: 7+ targets/game; strong: 4+; average: 2.5
        return _clip(_scale(tpg, 1.5, 8.0))

    elif pos == "RB":
        carries  = _safe(ls.get("rush_attempts")) / gp
        rec_tgts = _safe(ls.get("targets"))
        rec_pg   = (rec_tgts / gp) if rec_tgts > 0 else (_safe(ls.get("receptions")) / gp)
        # Combines rush volume + receiving involvement
        # Elite: 20+ carries + 4+ targets; average: 12 carries + 2 targets
        rush_util = _scale(carries, 8.0, 22.0)
        recv_util = _scale(rec_pg,  1.0,  5.0)
        # Dynasty: pass-catching usage is more stable and more valued than rush volume
        return _clip(rush_util * 0.55 + recv_util * 0.45)

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

    - WR/TE: yards-based pass share (team_pass_yards / team_total_yards).
      Captures how much of the offense's actual production came through the air,
      not just how often they called pass plays. Falls back to attempt-based
      team_pass_rate when yard totals are unavailable.
    - For RBs, high rush rate inflates volume — slight discount applied.
    - Uses a recency-weighted average across all seasons so transferred players
      aren't locked to their latest team's scheme.
    """
    if not seasons:
        return 50.0

    pos = position.upper()

    # Recency-weighted pass share: most recent season weight 1.0, decaying by 0.2
    sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0), reverse=True)
    weighted_sum = 0.0
    total_weight = 0.0
    for i, s in enumerate(sorted_seasons):
        team_pass_yds  = _safe(s.get("team_pass_yards"))
        team_total_yds = _safe(s.get("team_total_yards"))
        if team_pass_yds > 0 and team_total_yds > 0:
            pr = team_pass_yds / team_total_yds   # yards-based pass share
        else:
            pr = _safe(s.get("team_pass_rate"), 0.55)  # fallback: attempt-based rate
        w  = max(0.2, 1.0 - i * 0.2)
        weighted_sum += pr * w
        total_weight += w
    pass_rate = weighted_sum / total_weight if total_weight > 0 else 0.55

    if pos in ("WR", "TE"):
        # Yards-based pass share: 40% = run-heavy floor, 72% = Air Raid ceiling
        base = _scale(pass_rate, 0.40, 0.72)
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
    "WR": 1.00,
    "RB": 1.00,
    "QB": 0.90,   # QBs are less valued in 1QB dynasty
    "TE": 0.85,   # TEs face a 2-3 year development delay; lower per-game ceiling vs WRs
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

# Position-specific weights for more realistic evaluation
POSITION_WEIGHTS = {
    "QB": {
        "draft_capital": 0.26,
        "production": 0.15,
        "utilization": 0.03,
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
        # Draft capital is the single strongest RB predictor (1st-round RBs hit at 83%,
        # the highest hit rate of any position/tier — higher than 1st-round WRs at 64%).
        # Raised to 0.29 to match the data. Breakout and competition reduced to compensate.
        "draft_capital": 0.29,
        "production": 0.18,
        "utilization": 0.08,
        "efficiency": 0.10,
        "age": 0.09,
        "breakout": 0.11,
        "athleticism": 0.10,
        "competition": 0.04,
        "environment": 0.00,
        "durability": 0.01,
    },
    "WR": {
        # WR calibration - reduce overgrading by lowering weights on most influential components
        # Keep draft capital emphasis but reduce production/age to prevent inflation
        "draft_capital": 0.29,
        "production": 0.20,
        "utilization": 0.04,
        "efficiency": 0.10,
        "age": 0.08,
        "breakout": 0.09,
        "athleticism": 0.07,
        "competition": 0.07,
        "environment": 0.06,
        "durability": 0.00,
    },
    "TE": {
        # Draft capital (r=0.72) and age (TEs develop late; young elite TEs are rare) lead.
        # Athleticism defines generational TEs; efficiency (YPR + catch rate) is predictive.
        # College TE production and utilization are less reliable signals due to blocking roles.
        "draft_capital": 0.26,
        "production": 0.19,
        "utilization": 0.07,
        "efficiency": 0.12,
        "age": 0.10,
        "breakout": 0.06,
        "athleticism": 0.10,
        "competition": 0.08,
        "environment": 0.01,
        "durability": 0.01,
    },
}

# Validate that all position weights sum to 1.0
for pos, weights in POSITION_WEIGHTS.items():
    assert abs(sum(weights.values()) - 1.0) < 0.001, f"{pos} weights must sum to 1.0, got {sum(weights.values())}"


# Enhanced evaluation functions
# 

from typing import Dict

def calc_loaded_roster_adjustment(
    team: str,
    position: str,
    season: int,
    production_score: float,
    market_share: float,
    ypc: float = 0.0,
) -> float:
    loaded_rosters: Dict[str, Dict[str, Dict[int, int]]] = {
        "Ohio State": {
            "WR": {
                2021: 2,
                2022: 2,
                2023: 2,
                2024: 2,
                2025: 3,
            },
        },
        "Alabama": {
            "WR": {
                2020: 3,
                2021: 2,
                2022: 2,
                2025: 3,
            },
            "RB": {
                2020: 2,
                2021: 2,
            },
        },
        "Georgia": {
            "TE": {
                2021: 2,
                2022: 2,
            },
            "WR": {
                2024: 2,
                2025: 2,
            },
        },
        "USC": {
            "WR": {
                2022: 2,
                2024: 2,
                2025: 2,
            },
        },
        "LSU": {
            "WR": {
                2019: 3,
                2022: 2,
                2024: 2,
                2025: 2,
            },
        },
        "Texas": {
            "WR": {
                2023: 2,
                2024: 2,
                2025: 3,
            },
        },
        "Oregon": {
            "WR": {
                2024: 2,
                2025: 3,
            },
            "RB": {
                2025: 2,
            },
        },
        "Notre Dame": {
            "RB": {
                2024: 2,   # Price + Jeremiyah Love; blocked-by-generational-back scenario
                2025: 2,
            },
        },
    }

    team = (team or "").strip()
    position = (position or "").strip().upper()

    if not team or not position:
        return 1.0

    try:
        production_score = float(production_score)
    except (TypeError, ValueError):
        production_score = 0.0

    try:
        market_share = float(market_share)
    except (TypeError, ValueError):
        market_share = 0.0

    production_score = max(0.0, min(production_score, 100.0))
    market_share = max(0.0, min(market_share, 1.0))

    room_size = (
        loaded_rosters.get(team, {})
        .get(position, {})
        .get(season, 0)
    )

    if room_size < 2:
        return 1.0

    if room_size >= 4:
        base_bonus = 0.12
    elif room_size == 3:
        base_bonus = 0.10
    elif position == "RB":
        base_bonus = 0.10   # committee RBs are more opportunity-limited than shared WR rooms
    else:
        base_bonus = 0.06

    if market_share >= 0.30:
        ms_factor = 1.00
    elif market_share >= 0.24:
        ms_factor = 0.85
    elif market_share >= 0.18:
        ms_factor = 0.60
    elif market_share >= 0.12:
        ms_factor = 0.35
    else:
        ms_factor = 0.10

    if production_score >= 85:
        prod_factor = 1.00
    elif production_score >= 75:
        prod_factor = 0.80
    elif production_score >= 65:
        prod_factor = 0.55
    elif production_score >= 50:
        prod_factor = 0.30
    elif position == "RB" and ypc >= 5.5:
        # Committee RB with elite YPC: volume suppressed by opportunity, not talent.
        # Per-carry efficiency is the quality signal — use it instead of production_score.
        prod_factor = 0.65
    else:
        prod_factor = 0.10

    realized_bonus = base_bonus * ((ms_factor * 0.6) + (prod_factor * 0.4))
    multiplier = 1.0 + realized_bonus

    return min(multiplier, 1.18)


def calc_depth_chart_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Data-driven crowded-room bonus based on CFBD-derived depth chart rank.

    A WR2/RB2 who still puts up meaningful volume despite playing behind an
    alpha shows elite underlying talent suppressed by opportunity, not a talent
    deficit.  This generalises calc_loaded_roster_adjustment() to all teams
    automatically, rather than requiring a manually curated list.

    depth_rank=1  → no adjustment (face-value production)
    depth_rank=2+ → bonus proportional to production volume and group size

    Returns a multiplier in [1.0, 1.20].  Returns 1.0 when depth data is absent
    (e.g. seed-only prospects), allowing the manual loaded_roster fallback to apply.
    """
    if not seasons:
        return 1.0

    ls = _latest_season(seasons)
    if not ls:
        return 1.0

    depth_rank = ls.get("depth_rank")
    group_size = ls.get("position_group_size")

    if depth_rank is None or group_size is None:
        return 1.0
    if depth_rank == 1 or group_size < 2:
        return 1.0

    pos = position.upper()
    gp = max(_safe(ls.get("games_played"), 12), 1)

    # Thresholds (yds/game) that represent "still producing meaningfully"
    # despite not being the WR1/RB1.
    if pos in ("WR", "TE"):
        vol = _safe(ls.get("receiving_yards")) / gp
        tiers = {2: [(70, 0.12), (50, 0.08), (35, 0.05)],
                 3: [(60, 0.15), (40, 0.10)]}
    elif pos == "RB":
        vol = _safe(ls.get("rush_yards")) / gp
        tiers = {2: [(80, 0.12), (60, 0.08), (40, 0.05)],
                 3: [(60, 0.15), (40, 0.10)]}
    else:
        return 1.0

    rank_tiers = tiers.get(min(depth_rank, 3), [])
    base_bonus = 0.0
    for threshold, bonus in rank_tiers:
        if vol >= threshold:
            base_bonus = bonus
            break

    if base_bonus == 0.0:
        return 1.0

    # Scale up slightly for larger rooms — harder to carve out volume with more competition
    group_factor = min(1.0, group_size / 4.0)
    realized_bonus = base_bonus * (0.60 + 0.40 * group_factor)
    return min(1.0 + realized_bonus, 1.20)


def draft_capital_multiplier(round_selected: int) -> float:
    """
    Draft capital multiplier aligned with the NFL's 3-day structure.

    Day 1  — Round 1 (picks 1-32):    highest signal, team commits first resource
    Day 2  — Rounds 2-3 (picks 33-96): solid investment, starter likelihood meaningful
    Day 3  — Rounds 4-7 (picks 97+):  developmental, far less predictive for dynasty
    """
    if round_selected == 1:        # Day 1
        return 1.15
    elif round_selected == 2:      # Day 2 early
        return 1.08
    elif round_selected == 3:      # Day 2 late
        return 1.00
    elif round_selected == 4:      # Day 3 early
        return 0.88
    elif round_selected == 5:      # Day 3 mid
        return 0.78
    else:                          # Day 3 late (rounds 6-7)
        return 0.65


def calc_experience_score(seasons: List[Dict], position: str) -> float:
    """
    Experience metric for quarterbacks to mitigate misses like Trey Lance.
    
    Args:
        seasons: List of season data
        position: Player position
    
    Returns:
        Experience score (0-100)
    """
    if position != "QB":
        return 0.0
    
    # Calculate total games started across all seasons.
    # Fall back to games_played when games_started is unavailable (common for
    # pre-draft profiles that only have box-score game counts, not start data).
    games_started = sum(_safe(s.get("games_started", 0)) for s in seasons)
    if games_started == 0:
        games_started = sum(_safe(s.get("games_played", 0)) for s in seasons)

    # Normalize to 0-100 scale (40 games = full experience)
    experience_ratio = min(games_started / 40.0, 1.0)
    return experience_ratio * 100.0


def calc_late_round_upside(draft_capital: Optional[Dict], seasons: List[Dict], position: str) -> float:
    """
    Late-round breakout indicator for undervalued players with elite underlying metrics.
    
    Args:
        draft_capital: Draft capital data
        seasons: List of season data
        position: Player position
    
    Returns:
        Late-round upside score (0-100)
    """
    if not draft_capital:
        return 0.0
    
    projected_round = draft_capital.get("projected_round")
    if projected_round is None or projected_round < 4:
        return 0.0
    
    # Check for elite underlying metrics
    if not seasons:
        return 0.0
    
    best_dominator = max([_safe(s.get("dominator_rating", 0)) for s in seasons])
    dominator_rating = best_dominator
    
    yprr = 0.0
    if position in ("WR", "TE"):
        best_yprr = max([_safe(s.get("yards_per_route_run", 0)) for s in seasons])
        yprr = best_yprr
    
    # Late-round upside criteria
    has_elite_dominator = dominator_rating >= 0.35
    has_elite_yprr = yprr >= 2.5
    
    if has_elite_dominator or has_elite_yprr:
        # Base upside score for late-round prospects with elite metrics
        base_score = 75.0
        
        # Bonus for having both metrics
        if has_elite_dominator and has_elite_yprr:
            base_score = 90.0
        
        # Position-specific adjustments
        if position == "WR":
            base_score += 5.0  # WRs benefit more from elite efficiency
        elif position == "RB":
            base_score += 3.0  # RBs benefit from dominator rating
        
        return min(base_score, 100.0)
    
    return 0.0


def calc_interaction_features(production_score: float, efficiency_score: float, 
                             athleticism_score: float, draft_capital_score: float) -> Dict[str, float]:
    """
    Interaction features between key metrics to capture nuanced signals.
    
    Args:
        production_score: Production component score
        efficiency_score: Efficiency component score
        athleticism_score: Athleticism component score
        draft_capital_score: Draft capital component score
    
    Returns:
        Dictionary of interaction feature scores
    """
    # Production-efficiency interaction (high production + high efficiency)
    prod_eff_interaction = (production_score * efficiency_score) / 100.0
    
    # Athleticism-draft capital interaction (elite athlete + high draft capital)
    ath_dc_interaction = (athleticism_score * draft_capital_score) / 100.0
    
    # Production-athleticism interaction (elite producer + elite athlete)
    prod_ath_interaction = (production_score * athleticism_score) / 100.0
    
    # Triple interaction (all three elite)
    triple_interaction = (production_score * efficiency_score * athleticism_score) / 10000.0
    
    return {
        "production_efficiency_interaction": prod_eff_interaction,
        "athleticism_draft_capital_interaction": ath_dc_interaction,
        "production_athleticism_interaction": prod_ath_interaction,
        "triple_interaction": triple_interaction,
    }


def calc_translation_adjustment(
    prospect: Dict[str, Any],
    position: str,
    draft_capital: Optional[Dict[str, Any]],
    production_score: float,
    efficiency_score: float,
    age_score: float,
) -> float:
    """
    Position-specific post-model adjustment (in points) to reduce common misses:
      - WR false positives on low-translation profiles (hands/YAC + weak efficiency)
      - Day-2/Day-3 WR/RB underrates with strong underlying profiles
      - TE volatility when receiving-usage profile is weak
    """
    if not prospect.get("seasons"):
        return 0.0

    latest = _latest_season(prospect["seasons"]) or {}
    adj = 0.0
    projected_pick = _safe((draft_capital or {}).get("projected_pick"), 300.0)

    if position == "WR":
        drop_rate = _safe(latest.get("drop_rate"), 0.0)
        contested = _safe(latest.get("contested_catch_rate"), 0.0)
        yac = _safe(latest.get("yards_after_catch_per_reception"), 0.0)
        yprr = _safe(latest.get("yards_per_route_run"), 0.0)
        market_share = _safe(latest.get("market_share_yards"), 0.0)
        gp = max(_safe(latest.get("games_played"), 12.0), 1.0)
        rec_yds_pg = _safe(latest.get("rec_yds_pg"), _safe(latest.get("receiving_yards"), 0.0) / gp)
        rec_tds_pg = _safe(latest.get("rec_tds_pg"), _safe(latest.get("receiving_tds"), 0.0) / gp)

        # Penalize classic WR false-positive profiles (high volume, poor translation traits)
        if drop_rate >= 10.0:
            adj -= 2.0
        if contested > 0 and contested < 45.0:
            adj -= 1.5
        if yac > 0 and yac < 2.8:
            adj -= 1.0
        # Early-pick WR guardrail: weak production+efficiency profile should not sit
        # near elite tier solely via draft capital.
        if projected_pick <= 64 and production_score < 62 and efficiency_score < 58:
            adj -= 3.0
        if projected_pick <= 50 and rec_yds_pg > 0 and rec_yds_pg < 55 and rec_tds_pg < 0.50:
            adj -= 1.5

        # Boost strong skill indicators for non-elite draft capital WRs (Kupp/Puka archetype)
        if projected_pick > 40 and (yprr >= 2.6 or market_share >= 0.30):
            adj += 2.5
        if projected_pick > 75 and production_score >= 72 and efficiency_score >= 68:
            adj += 1.5
        if projected_pick > 90 and yprr >= 2.8 and market_share >= 0.28 and age_score >= 60:
            adj += 2.0
        if projected_pick > 80 and (rec_yds_pg >= 80 or rec_tds_pg >= 0.90):
            adj += 1.5

    elif position == "RB":
        gp = max(_safe(latest.get("games_played"), 12.0), 1.0)
        rec_yds_pg = _safe(latest.get("rec_yds_pg"), _safe(latest.get("receiving_yards"), 0.0) / gp)
        dominator = _safe(latest.get("dominator_rating"), 0.0)
        ypc = _safe(latest.get("yds_per_carry"), 0.0)

        # Raise dual-threat and high-dominator RBs drafted outside top tiers
        if projected_pick > 50 and rec_yds_pg >= 20:
            adj += 2.0
        if projected_pick > 75 and dominator >= 0.30 and ypc >= 5.2:
            adj += 2.0
        if projected_pick > 100 and rec_yds_pg >= 25 and dominator >= 0.28:
            adj += 1.5

    elif position == "TE":
        rec_yds_pg = _safe(latest.get("rec_yds_pg"), 0.0)
        target_share = _safe(latest.get("target_share"), 0.0)
        draft_age = _safe(prospect.get("age"), 0.0)

        # Baseline dynasty development cost: even elite TEs rarely contribute
        # meaningfully until year 2-3 of their NFL career.  Applied universally
        # regardless of profile to anchor TE scores below equivalent-capital WRs.
        adj -= 4.0

        # Profile-based TE shrinkage for weak receiving-usage profiles.
        if rec_yds_pg > 0 and rec_yds_pg < 35:
            adj -= 3.0
        if target_share > 0 and target_share < 0.14:
            adj -= 2.0
        if draft_age and draft_age > 23.0 and age_score < 50:
            adj -= 1.0
        if projected_pick <= 64 and rec_yds_pg > 0 and rec_yds_pg < 42:
            adj -= 2.0

    # Keep the adjustment bounded; this is a corrective signal, not the main model.
    return max(-14.0, min(6.0, adj))


def score_prospect(
    prospect: Dict[str, Any],
    draft_capital: Optional[Dict[str, Any]] = None,
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
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
    raw_age  = prospect.get("age")
    age      = float(raw_age) if raw_age is not None else None
    ath      = prospect.get("athleticism") or {}
    dy       = int(prospect.get("draft_class_year") or 2026)

    # _eval_metrics is attached by the pipeline orchestration (pipeline.py) when
    # the evaluation pipeline has run first.  Shape: {metric_name: metric_payload}.
    eval_metrics: Optional[Dict] = prospect.get("_eval_metrics") or None

    production_score    = calc_production_score(seasons, pos, eval_metrics=eval_metrics,
                                                skip_sagarin=skip_sagarin)
    
    # Apply crowded-room bonus: prefer data-driven depth rank when available,
    # fall back to manually curated loaded_roster list when depth data is absent.
    ls = _latest_season(seasons) or {}
    team = ls.get("team", "")
    season = ls.get("season", 0)
    market_share = _safe(ls.get("market_share_yards"), 0.15)

    depth_chart_adjustment = calc_depth_chart_adjustment(seasons, pos)

    if depth_chart_adjustment != 1.0:
        # CFBD depth rank data available — use data-driven adjustment
        loaded_roster_adjustment = depth_chart_adjustment
    else:
        # No depth data (seed-only) — fall back to manually curated list
        loaded_roster_adjustment = calc_loaded_roster_adjustment(
            team, pos, season, production_score, market_share,
            ypc=_safe(ls.get("yds_per_carry")),
        )
    production_score = _clip(production_score * loaded_roster_adjustment)
    
    utilization_score   = calc_utilization_score(seasons, pos)
    efficiency_score    = calc_efficiency_score(seasons, pos, eval_metrics=eval_metrics)
    age_score           = calc_age_score(age, dy, pos)
    breakout_score      = calc_breakout_score(seasons, age, pos)
    athleticism_score   = calc_athleticism_score(ath, pos)
    competition_score   = calc_competition_score(seasons)
    environment_score   = calc_environment_adjustment(seasons, pos)
    durability_score    = calc_durability_score(seasons)

    if draft_capital:
        dc_score = _safe(draft_capital.get("projected_draft_capital_score"), 40.0)
        projected_round = draft_capital.get("projected_round")
    else:
        # Default: no mock coverage -> treat like a late day-3 pick for this position
        from data_building.rookie_pipeline.mock_draft_consensus import pick_to_draft_capital_score
        dc_score = pick_to_draft_capital_score(150, pos)
        projected_round = 6  # Default to late round for no mock coverage

    #  Enhanced draft capital modeling with nonlinear tiered bonuses
    # Apply round-specific multipliers to capture nonlinear value
    if projected_round and draft_capital:
        round_multiplier = draft_capital_multiplier(projected_round)
        dc_score = _clip(dc_score * round_multiplier)

    # ── Position dc multiplier ──────────────────────────────────────────────
    # TE draft capital is less predictive than RB/WR (harder college-to-NFL
    # translation: contested catches, blocking duties, late development curves).
    # QBs are NOT discounted here — pick #1 QB correctly scores 100/100.
    # The lower dynasty value of QB capital vs WR/RB capital is captured
    # entirely through the QB draft_capital WEIGHT (0.22 vs WR 0.29).
    # TE draft capital is significantly less predictive for dynasty: even a Round 1
    # TE typically contributes minimally for 2-3 years (development curve + TE scarcity
    # doesn't translate to immediate fantasy points).  0.72 means a Round 1 Day 1 TE
    # gets 1.15 × 0.72 = 0.83× vs 1.15× for a Round 1 WR.
    dc_multiplier = {"TE": 0.72}.get(pos, 1.00)

    # ── Day-3 penalty ───────────────────────────────────────────────────────
    # Applied only to true Day 3 picks (Round 4+, pick ≥ 97).
    # Day 2 picks (rounds 2-3, picks 33-96) are not penalised here — their
    # reduced value is already captured by draft_capital_multiplier.
    # Tiered within Day 3: late Day 3 (round 6-7, pick > 141) hit harder.
    if draft_capital:
        projected_pick = draft_capital.get("projected_pick")
        if projected_pick and projected_pick > 96:   # Day 3 starts at Round 4
            if projected_pick > 140:                 # Late Day 3 (rounds 6-7)
                day3_penalty = {
                    "QB": 0.55,
                    "WR": 0.65,
                    "RB": 0.68,
                    "TE": 0.75,
                }.get(pos, 0.65)
            else:                                    # Early Day 3 (rounds 4-5, pick 97-140)
                day3_penalty = {
                    "QB": 0.70,
                    "WR": 0.82,
                    "RB": 0.84,
                    "TE": 0.90,
                }.get(pos, 0.82)
            dc_multiplier *= day3_penalty

    dc_score_adjusted = _clip(dc_score * dc_multiplier)

    #  Calculate new enhancement features
    experience_score = calc_experience_score(seasons, pos)
    late_round_upside = calc_late_round_upside(draft_capital, seasons, pos)
    interaction_features = calc_interaction_features(
        production_score, efficiency_score, athleticism_score, dc_score_adjusted
    )

    # Get position-specific weights (optionally overridden by calibrated weights)
    weights_source = position_weights_override or POSITION_WEIGHTS
    pos_weights = dict(weights_source.get(pos, POSITION_WEIGHTS["WR"]))

    # Post-draft: actual pick is certain — increase draft capital weight, spread
    # the reduction proportionally across all other components so sum stays 1.0.
    is_actual_pick = (draft_capital or {}).get("is_actual_pick", False)
    if is_actual_pick:
        boost = 0.06
        pos_weights["draft_capital"] = min(0.50, pos_weights["draft_capital"] + boost)
        other_keys = [k for k in pos_weights if k != "draft_capital"]
        other_sum  = sum(pos_weights[k] for k in other_keys)
        target_sum = 1.0 - pos_weights["draft_capital"]
        if other_sum > 0:
            ratio = target_sum / other_sum
            for k in other_keys:
                pos_weights[k] *= ratio

    # Base prospect score with position-specific weights
    prospect_score = (
        production_score      * pos_weights["production"]    +
        utilization_score     * pos_weights["utilization"]   +
        efficiency_score      * pos_weights["efficiency"]    +
        age_score             * pos_weights["age"]           +
        breakout_score        * pos_weights["breakout"]      +
        athleticism_score     * pos_weights["athleticism"]   +
        competition_score     * pos_weights["competition"]   +
        environment_score     * pos_weights["environment"]   +
        durability_score      * pos_weights["durability"]    +
        dc_score_adjusted     * pos_weights["draft_capital"]
    )

    # Add experience score for QBs (uses the experience weight in QB weights)
    if pos == "QB" and "experience" in pos_weights:
        prospect_score += experience_score * pos_weights["experience"]

    # Add late-round upside bonus for players with elite underlying metrics
    if late_round_upside > 0:
        upside_bonus = late_round_upside * 0.05  # 5% of upside score as bonus
        prospect_score += upside_bonus

    # Apply benchmark boost system for NFL success predictors
    from benchmark_boosts import calc_benchmark_boost, apply_benchmark_boost
    
    # Get draft pick for benchmark calculations
    draft_pick = None
    if draft_capital:
        draft_pick = draft_capital.get("projected_pick")
    
    # Calculate benchmark boosts
    benchmark_boosts = calc_benchmark_boost(
        position=pos,
        age=age,
        draft_pick=draft_pick,
        seasons=seasons,
        athleticism=ath,
        production_score=production_score,
        utilization_score=utilization_score,
        efficiency_score=efficiency_score,
        competition_score=competition_score,
        breakout_score=breakout_score
    )
    
    # Apply benchmark boosts to final score.
    # The benchmark boost is the only post-sum modifier: a small signal (max 3%)
    # for prospects who meet multiple elite criteria.  Interaction bonuses and
    # the generational multiplier have been removed — the weighted component sum
    # is already calibrated on an absolute historical scale, so layering extra
    # boosts produces class-relative inflation rather than a stable absolute grade.
    prospect_score = apply_benchmark_boost(prospect_score, benchmark_boosts)

    # Absolute-scale calibration curve. The weighted-component sum accurately
    # ranks prospects but compresses the distribution: true generational talent
    # (Chase, Nabers, Robinson) naturally scores ~86 after the weighted sum.
    # This quadratic boost proportionally amplifies high-scoring profiles so
    # the grade scale matches the intended tiers: 94+ generational, 85+ top tier.
    # Formula: score + (score / 100)² × 11
    # Effect at key breakpoints:
    #   86 raw → 94.4  (generational)
    #   82 raw → 89.4  (upper top tier)
    #   80 raw → 87.0  (top tier)
    #   75 raw → 81.2  (solid starter)
    #   65 raw → 69.6  (developmental)
    prospect_score = min(100.0, prospect_score + (prospect_score / 100.0) ** 2 * 11.0)

    # Position-specific translation adjustment from historical miss archetypes.
    # Applied late (after global scaling) so the correction magnitude is preserved.
    translation_adjustment = calc_translation_adjustment(
        prospect=prospect,
        position=pos,
        draft_capital=draft_capital,
        production_score=production_score,
        efficiency_score=efficiency_score,
        age_score=age_score,
    )
    prospect_score += translation_adjustment

    # Thin-sample volume gate: when both production AND utilization fall below
    # position-typical thresholds, efficiency/athleticism/breakout signals are
    # based on sparse college evidence and are less reliable.  Penalise the
    # final score proportionally to the combined deficit.  The gate only fires
    # when BOTH metrics are below par simultaneously — a player with high
    # utilization but modest production (e.g. a committee RB) is not penalised.
    # Maximum penalty: 22% (both metrics at zero).  Example: Sadiq (prod=52.5,
    # util=43.6) → gate≈0.64 → ~8% penalty, dropping him ~5-6 points.
    _PROD_GATE = 65.0
    _UTIL_GATE = 55.0
    # RB bypass: a back with elite per-carry efficiency (YPC ≥ 5.5) is
    # opportunity-constrained, not talent-constrained (e.g. blocked by a generational
    # back). The gate was designed for genuinely sparse evidence, not committee backs
    # who produce efficiently in limited carries.
    _rb_ypc = _safe((_latest_season(seasons) or {}).get("yds_per_carry")) if pos == "RB" else 0.0
    _rb_efficiency_bypass = (pos == "RB" and _rb_ypc >= 5.5)
    if production_score < _PROD_GATE and utilization_score < _UTIL_GATE and not _rb_efficiency_bypass:
        gate = (production_score / _PROD_GATE) * (utilization_score / _UTIL_GATE)
        prospect_score *= 1.0 - (1.0 - gate) * 0.22

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
        environment_score, durability_score, dc_score_adjusted, draft_capital,
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
        "projected_draft_capital_score":round(dc_score_adjusted, 2),
        "fantasy_translation_score":    round(fantasy_translation, 2),
        "confidence_score":             confidence_score,
        "prospect_score":               prospect_score,
        "key_reasons":                  reasons,
        "experience_score":             round(experience_score, 2),
        "late_round_upside":             round(late_round_upside, 2),
        "translation_adjustment":        round(translation_adjustment, 2),
        "loaded_roster_adjustment":      round(loaded_roster_adjustment, 3),
        "depth_chart_adjustment":        round(depth_chart_adjustment, 3),
        "depth_rank":                    ls.get("depth_rank"),
        "position_group_size":           ls.get("position_group_size"),
        "production_efficiency_interaction": round(interaction_features["production_efficiency_interaction"], 2),
        "athleticism_draft_capital_interaction": round(interaction_features["athleticism_draft_capital_interaction"], 2),
        "production_athleticism_interaction": round(interaction_features["production_athleticism_interaction"], 2),
        "benchmark_boosts":             benchmark_boosts,
        "total_benchmark_boost":        round(benchmark_boosts.get("total_boost", 0.0), 3),
        "triple_interaction":           round(interaction_features["triple_interaction"], 2),
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
        bullets.append("High efficiency metrics stand out")
    elif eff <= 35:
        bullets.append("Below-average efficiency — volume stats may overstate true impact")

    # ── Advanced metrics ──────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        adv: List[str] = []

        ccr = ls.get("contested_catch_rate")
        if ccr is not None:
            pct = float(ccr)
            if pct >= 80:
                adv.append(f"{pct:.0f}% contested catch rate — elite ball-winner in traffic")
            elif pct >= 65:
                adv.append(f"{pct:.0f}% contested catch rate — reliable in contested situations")
            elif pct < 40:
                adv.append(f"{pct:.0f}% contested catch rate — struggles in jump-ball situations")

        dr = ls.get("drop_rate")
        if dr is not None:
            dpct = float(dr)
            if dpct <= 3.0:
                adv.append(f"{dpct:.1f}% drop rate — elite ball security")
            elif dpct >= 10.0:
                adv.append(f"{dpct:.0f}% drop rate — ball security concern")

        yac = ls.get("yards_after_catch_per_reception")
        if yac is not None:
            yac = float(yac)
            thresh     = 5.5 if pos == "WR" else 4.0
            low_thresh = 2.5 if pos == "WR" else 1.8
            if yac >= thresh:
                adv.append(f"{yac:.1f} YAC/reception — dynamic after the catch")
            elif yac <= low_thresh:
                adv.append(f"{yac:.1f} YAC/reception — limited after-catch production")

        adot = ls.get("avg_depth_of_target")
        if adot is not None and pos == "WR":
            adot = float(adot)
            if adot >= 14.0:
                adv.append(f"{adot:.1f}-yd avg depth of target — true deep threat")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT — short-area route specialist")

        pff_off = ls.get("grades_offense")
        if pff_off is not None:
            pff_off = float(pff_off)
            if pff_off >= 85.0:
                adv.append(f"PFF offensive grade {pff_off:.1f} — elite overall grade")
            elif pff_off >= 75.0:
                adv.append(f"PFF offensive grade {pff_off:.1f} — above-average")

        if pos == "WR":
            sr = ls.get("slot_rate")
            if sr is not None and float(sr) >= 0.65:
                adv.append(f"{float(sr):.0f}% slot rate — primary slot receiver")

        if pos == "TE":
            ir = ls.get("inline_rate")
            if ir is not None:
                ir = float(ir)
                if ir >= 0.60:
                    adv.append(f"{ir:.0f}% inline rate — traditional in-line TE")
                elif ir <= 0.20:
                    adv.append(f"{ir:.0f}% inline rate — move TE / receives in space")

        bullets.extend(adv[:3])

    elif pos == "RB":
        adv = []

        pff_off = ls.get("grades_offense")
        if pff_off is not None:
            pff_off = float(pff_off)
            if pff_off >= 80.0:
                adv.append(f"PFF offensive grade {pff_off:.1f} — elite overall grade")
            elif pff_off >= 70.0:
                adv.append(f"PFF offensive grade {pff_off:.1f} — above-average")

        elusive = ls.get("elusive_rating")
        if elusive is not None:
            elusive = float(elusive)
            if elusive >= 90.0:
                adv.append(f"Elusive rating {elusive:.1f} — exceptional open-field threat")
            elif elusive >= 70.0:
                adv.append(f"Elusive rating {elusive:.1f} — above-average evasion ability")

        bp = ls.get("breakaway_percentage")
        if bp is not None:
            bpct = float(bp)
            if bpct >= 18.0:
                adv.append(f"{bpct:.0f}% breakaway run rate — consistent big-play threat")
            elif bpct >= 12.0:
                adv.append(f"{bpct:.0f}% breakaway run rate — shows explosive burst")

        bullets.extend(adv[:3])

    elif pos == "QB":
        adv = []

        pff_pass = ls.get("pff_passing_grade")
        if pff_pass is not None:
            pff_pass = float(pff_pass)
            if pff_pass >= 85.0:
                adv.append(f"PFF passing grade {pff_pass:.1f} — elite passer grade")
            elif pff_pass >= 75.0:
                adv.append(f"PFF passing grade {pff_pass:.1f} — above-average")

        acr = ls.get("adjusted_completion_rate")
        if acr is not None:
            apct = float(acr)
            if apct >= 75.0:
                adv.append(f"{apct:.0f}% adjusted completion rate — highly accurate")
            elif apct <= 58.0:
                adv.append(f"{apct:.0f}% adjusted completion rate — accuracy concern")

        btt = ls.get("big_time_throw_rate")
        if btt is not None:
            bpct = float(btt)
            if bpct >= 8.0:
                adv.append(f"{bpct:.1f}% big-time throw rate — attacks deep coverage effectively")
            elif bpct >= 5.0:
                adv.append(f"{bpct:.1f}% big-time throw rate — willing to push ball downfield")

        adot = ls.get("avg_depth_of_target")
        if adot is not None:
            adot = float(adot)
            if adot >= 10.0:
                adv.append(f"{adot:.1f}-yd avg depth of target — attacks full field vertically")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT — relies heavily on short/underneath game")

        bullets.extend(adv[:3])

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
        bullets.append(f"Production came against quality competition")
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
                f"{conf_str}"
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
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
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
        scores.append(
            score_prospect(
                p,
                dc,
                skip_sagarin=skip_sagarin,
                position_weights_override=position_weights_override,
            )
        )

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
