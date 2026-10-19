"""
Prospect evaluation model - position-aware, multi-factor scoring.

Each component score is 0-100.  Final prospect_score is a weighted sum.

Model version: 3.4 (see MODEL_VERSION)
  v1.x: baseline components below.
  v2.0: adds recruiting-pedigree component (247 Composite via CFBD),
        WEPA opponent-adjusted efficiency blend (QB/RB), and SP+/Team Talent
        inputs to the competition adjustment.  Per Kaedon's rule, grading
        changes are version-bumped, never silently tweaked: the version is
        persisted per row in rookie_rankings.model_version.
  v3.4: two-mode grading. COMPLETE mode is v3.3. IN-SEASON mode (the run
        supplies the active season year and the player's latest season is
        that year at 1-11 stored games) grades on prior complete seasons
        plus the partial season at per-game rates with paced counting
        gates; untested athleticism is missing data (weight redistributed),
        never the projected 67.0 floor; confidence is discounted by the
        partial sample. Also: durability only counts seasons where the
        player held a real role (usage thresholds), so backup seasons no
        longer read as injury seasons in either mode; the WR/TE red-zone
        proxy thresholds are recalibrated to the recorded TD-rate
        distribution (they could never fire at the old 5.5-10.0 levels).

Component weights (v2.0 - recruiting takes 5%, others rescaled proportionally):
    projected_draft_capital_score 27-28%  NFL draft position (position-weighted)
    production_score             ~14-19%  college production volume
    athleticism_score            ~7-10%   combine / speed score / RAS
    breakout_profile_score        ~4-11%  early-career dominance trajectory
    utilization_score             ~3-8%   opportunity share
    efficiency_score              ~8-17%  per-attempt / per-target efficiency
    competition_score             ~4-7%   conference + opponent quality (+SP+ SOS)
    age_score                     ~8-13%  age-adjusted production; youth premium
    environment_adjustment        ~0-6%   team scheme / usage context
    durability_score              ~0-1%   games missed, injury history
    recruiting_score               5 %   247 Composite pedigree (stars/rating)
    ──────────────────────────────────
    Total                        100 %

Position-weighted draft capital, day-tier multipliers (Day 1 R1 > Day 2 R2-3 > Day 3 R4-7)

Position-specific adjustments are baked into each scorer to handle the
different stat profiles of QB / RB / WR / TE.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# Grading-model version. Bump on ANY change to scoring logic or weights.
MODEL_VERSION = "3.4"


def active_season_year_for_run(today=None) -> Optional[int]:
    """The college season in progress on the run date, else None.

    The season runs from Aug 25 (the durability function's own season-
    start convention) through mid-January (the national championship
    window): Aug 25 to Dec 31 of year Y is season Y, and Jan 1 to Jan 20
    of year Y is still season Y-1. Outside that window no season is in
    progress and every profile grades in COMPLETE mode.

    Callers pass the result to score_prospect / score_all_prospects as
    the active season year. It must come from the RUN CALENDAR, never
    from a player's draft class: deriving it per player would flip
    historical final seasons of 11 or fewer games into in-season mode
    and silently move the backtest (round 14 of the iteration log).
    """
    from datetime import date as _d
    today = today or _d.today()
    if today.month >= 8:
        return today.year if today >= _d(today.year, 8, 25) else None
    if today.month == 1 and today.day <= 20:
        return today.year - 1
    return None


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


def _latest_season(seasons: List[Dict]) -> Optional[Dict]:
    if not seasons:
        return None
    # Prefer the latest season with actual stat data; the eval pipeline writes
    # placeholder rows for the draft year (e.g. 2027) that must not shadow the
    # real latest college season (e.g. 2026).
    def _has_data(s: Dict) -> bool:
        return any(_safe(s.get(k)) > 0 for k in (
            "receiving_yards", "rush_yards", "pass_yards",
            "receptions", "rush_attempts", "pass_attempts",
        ))
    with_data = [s for s in seasons if _has_data(s)]
    pool = with_data or seasons
    return max(pool, key=lambda s: _safe(s.get("season"), 0))


# ─────────────────────────────────────────────────────────────────────────────
# Conference quality table
# Power 5 / big non-conf → higher competition score
# ─────────────────────────────────────────────────────────────────────────────

CONF_QUALITY: Dict[str, float] = {
    # Power 2 - abbreviations and full names (CFBD returns full names)
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
            return 0.0    # FBS team - Sagarin unavailable, apply no adjustment
        return -0.093     # Confirmed FCS / non-D1
    raw = (rating - _SAGARIN_FBS_AVG) * _SAGARIN_SCALE
    return max(-0.093, min(0.0647, raw))


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_production_season(season: Dict, pos: str, skip_sagarin: bool = False,
                             _partial: bool = False) -> float:
    """
    Compute the raw production score (pre-transfer-penalty) for a single season.
    Returns the weighted component score (0-100).

    v3.4 (round 14): _partial marks the in-progress season in in-season
    mode. Counting-stat gates inside the season score (red-zone yardage
    gates, QB dual-threat TD gates) are paced by min(1, GP/12) so a partial
    season is not held to full-season counting thresholds. All rate legs
    are untouched.
    """
    # Guard against bad games-played values (0/1) that can explode per-game rates
    # and incorrectly saturate production to 100.
    gp_raw = _safe(season.get("games_played"), 12)
    gp = gp_raw if gp_raw >= 4 else 12
    _pace = min(1.0, gp / 12.0) if _partial else 1.0

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
            _scale(rec_yds_pg, 32,  100) * 0.40 +
            _scale(rec_tds_pg, 0.25, 0.9) * 0.30 +
            dom_score                      * 0.30
        )
        # YAC bonus: dynamic after-catch ability is a strong NFL translation signal
        yac = _safe(season.get("yards_after_catch_per_reception"))
        if yac >= 7.0:
            prod = _clip(prod * 1.10)
        elif yac >= 5.5:
            prod = _clip(prod * 1.05)
        # Red zone proxy: TDs per 100 receiving yards - rewards goal-line
        # separators. v3.4 recalibration: the old thresholds (8.0 / 5.5)
        # sat 3-6x above the best recorded mark, so the bonus could NEVER
        # fire. Across the recorded fixture + panel seasons at 200+
        # receiving yards, the rate runs 0.00-1.62 with a median near
        # 0.93; the new tiers sit at the recorded distribution's top
        # quarter (1.15) and elite edge (1.5: Pitts 2020 scored 1.56,
        # Green 2025 1.62).
        if total_rec_yds >= 200 * _pace:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 1.5:   prod = _clip(prod * 1.06)
            elif rz_rate >= 1.15: prod = _clip(prod * 1.03)
        # v3.0: G5 production discount REMOVED. Competition quality is already
        # captured by calc_competition_score, so the additional production cut
        # (up to -25%) was double-jeopardy. One mechanism, not two.
        pass
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
            _scale(rush_yds_pg, 20,  120) * 0.18 +
            _scale(all_yds_pg,  30,  150) * 0.17 +
            _scale(tds_pg,       0.4,  1.6) * 0.25 +
            _scale(dom,          0.12, 0.50) * 0.15 +  # ceiling lowered: 0.45+ dom is elite
            _scale(ypc,          3.5,  7.0) * 0.25     # ceiling lowered: 7.0 YPC is generational
        )
        # v3.2 V8: genuine pass-catching profiles move 0.05 of weight from
        # the rush anchor to the all-purpose anchor.
        _rec_share_v8 = rec_yds_pg / max(all_yds_pg, 1.0)
        if _rec_share_v8 >= 0.20:
            prod += (_scale(all_yds_pg, 30, 150) - _scale(rush_yds_pg, 20, 120)) * 0.05
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
        # Red zone proxy: TDs per 100 total yards.
        # Realistic benchmarks for RBs: 1.0–2.5 range for elite scorers.
        total_yds = _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        total_tds = _safe(season.get("rush_tds"))   + _safe(season.get("receiving_tds"))
        if total_yds >= 300 * _pace:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 2.0:
                mult += 0.03
            elif rz_rate >= 1.0:
                mult += 0.015

        mult = min(mult, 1.20)  # slightly higher cap to allow elite all-around profiles
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
            _scale(pass_yds_pg, 150, 285) * 0.22 +  # v3.2 V5
            _scale(tds_pg,        1.5,  2.8) * 0.22 +  # v3.2 V5
            _scale(comp_pct,     58.0, 72.0) * 0.18 +
            _scale(ypa,           6.5,   9.5) * 0.28 +
            _scale(td_int,        1.5,   6.0) * 0.10
        )
        rush_comp = _scale(rush_yds_pg, 10.0, 85.0)  # 0–100; Lamar ~90+ yd/g → near 100

        # v3.2 R7-A1: max(pass, blend). A perfect passing season is a
        # 100 production season whether or not the QB also ran.
        prod = _clip(max(pass_comp, pass_comp * 0.85 + rush_comp * 0.15))

        # Extra multiplier for elite rushing + TDs (dual-threat upside)
        # v3.4: in-season, the season TD total is paced to a 12-game
        # equivalent before it is compared to the counting thresholds.
        _rush_tds_cmp = rush_tds_season / _pace if _pace > 0 else rush_tds_season
        if rush_yds_pg >= 70 and _rush_tds_cmp >= 8:
            prod = _clip(prod * 1.06)
        elif rush_yds_pg >= 50 and _rush_tds_cmp >= 5:
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

        # Pass-share dominator for TEs - same logic as WR, adjusted scale for elite TEs
        # (Elite TEs like Bowers can command WR-level share of passing yards).
        if team_pass_yds > 0:
            pass_share = (total_rec_yds / team_pass_yds) * (1 + sag_adj)
            dom_score  = _scale(pass_share, 0.04, 0.40)  # v3.2 R7-B: beyond McBride's 0.36 record share
        else:
            dom_score  = _scale(dom * (1 + sag_adj), 0.05, 0.25)  # Increased upper bound from 0.22

        prod = (
            _scale(rec_yds_pg, 20,  100) * 0.27 +  # v3.2 R7-B: beyond Pitts' 96.3/g record pace
            _scale(rec_tds_pg, 0.12, 0.5) * 0.25 +  # saturation anchor, weight trimmed
            dom_score                      * 0.35 +  # pass share: the stable TE signal
            _scale(rec_pg,     1.0,  8.0) * 0.13    # v3.2 R7-B: beyond McBride's 7.5/g
        )
        # Elite TE bonus: players like Brock Bowers who produce at WR levels get extra credit
        if rec_yds_pg >= 55 and dom_score >= 47:  # v3.2 R7-B: same physical share, rescaled
            prod = _clip(prod * 1.18)  # 18% bonus for WR-level production (lowered thresholds)
        
        # Special bonus for historically elite TEs (multiple major award winners)
        # This captures players like Brock Bowers who won best college TE twice
        if rec_yds_pg >= 50 and dom_score >= 42:  # v3.2 R7-B: same physical share, rescaled
            prod = _clip(prod * 1.08)  # Additional 8% bonus for elite production
        # YAC bonus: move TEs who create after the catch are more NFL-translatable
        te_yac = _safe(season.get("yards_after_catch_per_reception"))
        if te_yac >= 5.0:
            prod = _clip(prod * 1.08)
        elif te_yac >= 3.5:
            prod = _clip(prod * 1.04)
        # Red zone proxy: TE goal-line usage is extremely valuable in
        # NFL. v3.4 recalibration: same defect and same recorded
        # distribution as the WR proxy (old thresholds 10.0 / 7.0 could
        # never fire; recorded rates top out at 1.62).
        if total_rec_yds >= 200 * _pace:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 1.5:
                prod = _clip(prod * 1.05)
            elif rz_rate >= 1.15:
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
    _inseason_year: Optional[int] = None,
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
                # Upward transfer: producing in a stronger conference is harder -
                # reward the player (e.g. FCS → SEC and still put up numbers)
                upgrade_bonus = min(0.12, (delta / prev_quality) * 0.60)  # up to +12%
                transfer_penalty *= (1.0 + upgrade_bonus)
                break

    if pos not in ("WR", "RB", "QB", "TE"):
        return 40.0

    # With only one season there is nothing to blend - just score it directly.
    if len(seasons) == 1:
        return _clip(_score_production_season(
            ls, pos, skip_sagarin,
            _partial=(_inseason_year is not None
                      and ls.get("season") == _inseason_year)) * transfer_penalty)

    # Score latest season and best individual season; blend to reward peak while
    # still weighting recent output (NFL analysts evaluate both)
    latest_score = _score_production_season(
        ls, pos, skip_sagarin,
        _partial=(_inseason_year is not None
                  and ls.get("season") == _inseason_year))

    # Find best season by primary volume metric per position
    _PEAK_KEY = {"WR": "receiving_yards", "RB": "rush_yards", "QB": "pass_yards", "TE": "receiving_yards"}
    peak_key = _PEAK_KEY[pos]
    if _inseason_year is not None:
        # v3.4: in-season peak selection uses the 12-game-equivalent
        # rate (totals / max(GP, 12)) instead of raw totals, so a partial
        # season's calendar-bound totals cannot lose the peak slot to a
        # longer complete season on volume alone, and vice versa.
        def _peak_key_fn(s):
            _g = _safe(s.get("games_played"), 12)
            _g = _g if _g >= 4 else 12
            return _safe(s.get(peak_key)) / max(_g, 12)
        peak_season = max(seasons, key=_peak_key_fn, default=ls)
    else:
        peak_season = max(seasons, key=lambda s: _safe(s.get(peak_key)), default=ls)
    best_score = _score_production_season(
        peak_season, pos, skip_sagarin,
        _partial=(_inseason_year is not None
                  and peak_season.get("season") == _inseason_year))

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
        if ccr is None:
            # Fallback to Reception Perception when PFF contested_catch_rate is absent.
            # RP uses a 38-70 scale vs PFF's 0-100; map RP onto the PFF scale.
            ccr_rp = _eval_metric_value(eval_metrics, "contested_catch_rate_rp", min_confidence=0.65)
            if ccr_rp is not None:
                # Linear map: RP 38 -> PFF 30, RP 70 -> PFF 85
                ccr = 30.0 + (float(ccr_rp) - 38.0) * (55.0 / 32.0)
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
        # elusive_rating dropped: was PFF-only, no free replacement.

        breakaway = _eval_metric_percent(eval_metrics, "explosive_run_rate", min_confidence=0.40)
        if breakaway is not None:
            # Scale-based and symmetric: 0 at 20% (average), +2 at 40% elite, -1.5 at 8%.
            breakaway_delta = _clip((breakaway - 20.0) / 20.0, -1.0, 1.0)
            prod = _clip(prod + (breakaway_delta * 2.0))

    elif eval_metrics and pos == "QB":
        # pff_passing_grade is now sourced from CFBD PPA (free replacement for PFF).
        # See rookie_sources.py for the PPA-to-grade mapping.
        pff_pass = _eval_metric_value(eval_metrics, "pff_passing_grade", min_confidence=0.45)
        if pff_pass is not None:
            prod = _clip(prod + _clip((float(pff_pass) - 70.0) * 0.20, -4.0, 7.0))

        # big_time_throw_rate dropped: was PFF-only, no free replacement.

    # Scheme-inflation discount: high-volume spread systems inflate skill-position
    # production stats.  Applied to both WR and TE - air-raid volume pumps TE
    # target counts and yardage totals as much as WR numbers.
    if pos in ("WR", "TE"):
        scheme_discount = _scheme_inflation_discount(ls.get("team"))
        if scheme_discount < 1.0:
            prod = _clip(prod * scheme_discount)

    return prod


def calc_efficiency_score(
    seasons: List[Dict],
    position: str,
    eval_metrics: Optional[Dict] = None,
    wepa: Optional[Dict] = None,
    _inseason_year: Optional[int] = None,
) -> float:
    """
    Per-attempt / per-target efficiency.  Rewards quality over quantity.

    Uses the latest season as the primary signal.  When multiple seasons are
    available a consistency bonus (±5) is applied: sustained high efficiency
    across seasons is more predictive than a single-year peak.

    eval_metrics (optional): When available and confidence is sufficient:
    - QB: nfl_passer_rating (computed from box score) supplements efficiency
    - WR/TE: tprr proxy supplements yds_per_reception via a soft blend

    wepa (optional, model v2.0): {(season, wepa_type): {"adj_efficiency_score": 0-100}}.
    CFBD's opponent-adjusted efficiency percentile for the latest season is
    blended 25/75 into the box-score efficiency.  Only QB (passing) and RB
    (rushing) have WEPA coverage - WR/TE skip silently.
    """
    if not seasons:
        return 52.0  # pre-draft neutral: unknown ≠ bad

    pos = position.upper()
    ls  = _latest_season(seasons) or {}

    if pos == "WR":
        ypr = _safe(ls.get("yds_per_reception"), 10.0)
        ms  = _safe(ls.get("market_share_yards"))
        eff = _scale(ypr, 8.5, 17.0) * 0.50 + _scale(ms, 0.08, 0.30) * 0.50  # v3.3 F2: MS anchor 0.30 (0.40 sat ~52% beyond the best recorded WR season, DeVonta Smith 2020 at 0.264)

        # Real PFF YPRR - centered adjustment: 2.0 = neutral, 2.8 = +8, 1.2 = -8
        yprr = _eval_metric_value(eval_metrics, "yprr", min_confidence=0.75)
        if yprr is not None:
            yprr_score = _scale(float(yprr), 1.2, 2.8)
            eff = _clip(eff + (yprr_score - 50.0) * 0.16)

        # grades_pass_route dropped: was PFF-only, no free replacement.

        # Success rate vs. press coverage
        press_sr = _eval_metric_value(eval_metrics, "success_rate_vs_press", min_confidence=0.70)
        if press_sr is not None:
            press_score = _scale(float(press_sr), 55.0, 85.0)
            eff = _clip(eff + (press_score - 50.0) * 0.06)

        # Success rate vs. man coverage - strongest coverage-type predictor for NFL transition
        man_sr = _eval_metric_value(eval_metrics, "success_rate_vs_man", min_confidence=0.70)
        if man_sr is not None:
            man_score = _scale(float(man_sr), 45.0, 75.0)
            eff = _clip(eff + (man_score - 50.0) * 0.10)

        # Success rate vs. zone coverage
        zone_sr = _eval_metric_value(eval_metrics, "success_rate_vs_zone", min_confidence=0.70)
        if zone_sr is not None:
            zone_score = _scale(float(zone_sr), 52.0, 82.0)
            eff = _clip(eff + (zone_score - 50.0) * 0.08)

        # Contested catch rate - hands + body catching in traffic; physical ceiling signal
        contested_rp = _eval_metric_value(eval_metrics, "contested_catch_rate_rp", min_confidence=0.65)
        if contested_rp is not None:
            contested_score = _scale(float(contested_rp), 38.0, 70.0)
            eff = _clip(eff + (contested_score - 50.0) * 0.08)

        # Tackle break rate - YAC/elusiveness after the catch
        tbr = _eval_metric_value(eval_metrics, "tackle_break_rate", min_confidence=0.65)
        if tbr is not None:
            tbr_score = _scale(float(tbr), 8.0, 28.0)
            eff = _clip(eff + (tbr_score - 50.0) * 0.06)

        # Route target rate from RP - centered adjustment: 30% = neutral, 42% = +6, 18% = -6
        rtr = _eval_metric_value(eval_metrics, "route_target_rate", min_confidence=0.75)
        if rtr is not None:
            tprr_equiv = float(rtr) / 100.0
            rtr_score  = _scale(tprr_equiv, 0.18, 0.42)
            eff = _clip(eff + (rtr_score - 50.0) * 0.12)
        else:
            # Tprr proxy - lower priority fallback when RP data absent
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
        # grades_pass_route dropped: was PFF-only, no free replacement.

    elif pos == "QB":
        ypa   = _safe(ls.get("yds_per_attempt"), 7.0)
        td_int= _safe(ls.get("td_int_ratio"),     2.0)

        # adjusted_comp_pct dropped (was PFF-only). Use raw completion_pct.
        cpct = _safe(ls.get("completion_pct"), 62.0)

        # twp_rate dropped: was PFF-only, no free replacement.
        td_int_mod = td_int

        eff   = (
            _scale(ypa,      6.5,  9.5) * 0.45 +
            _scale(cpct,    58.0, 72.0) * 0.30 +
            _scale(td_int_mod, 1.5, 6.0) * 0.25  # v3.2 V5
        )

        # v3.2 Q4: one bad partial season should not overwrite a full elite
        # season. Blend 70% latest with 30% best (by pass yards).
        if len(seasons) >= 2:
            if _inseason_year is not None:
                # v3.4: same 12-game-equivalent rate key as production,
                # so the peak slot is not decided by raw totals mid-season.
                def _qb_peak_fn(s):
                    _g = _safe(s.get("games_played"), 12)
                    _g = _g if _g >= 4 else 12
                    return _safe(s.get("pass_yards")) / max(_g, 12)
                _best_qb = max(seasons, key=_qb_peak_fn)
            else:
                _best_qb = max(seasons, key=lambda s: _safe(s.get("pass_yards")))
            if _best_qb is not ls:
                _b_eff = (
                    _scale(_safe(_best_qb.get("yds_per_attempt"), 7.0), 6.5, 9.5) * 0.45 +
                    _scale(_safe(_best_qb.get("completion_pct"), 62.0), 58.0, 72.0) * 0.30 +
                    _scale(_safe(_best_qb.get("td_int_ratio"), 2.0), 1.5, 6.0) * 0.25
                )
                eff = eff * 0.15 + _b_eff * 0.85  # v3.2 R7-A2: peak blend matches production (85% peak / 15% latest)

    elif pos == "TE":
        ypr = _safe(ls.get("yds_per_reception"), 9.0)
        ms  = _safe(ls.get("market_share_yards"))
        # Catch rate: receptions / targets measures hands + separation reliability.
        # Only computed when targets is populated (not always available from CFBD).
        targets = _safe(ls.get("targets"))
        recs    = _safe(ls.get("receptions"))
        # v3.2: targets below receptions is a pipeline failure, not data
        # (audit section 0). Treat as missing so catch rate can never be
        # computed from a garbage ratio above 100 percent.
        if targets < recs:
            targets = 0.0
        catch_rate_score = 50.0  # neutral default when targets unknown
        if targets > 0:
            catch_rate = recs / targets
            catch_rate_score = _scale(catch_rate, 0.60, 0.85)  # 55% → 0, 82%+ → 100
        if targets > 0:
            eff = (
                _scale(ypr, 7.5, 14.0) * 0.50 +
                _scale(ms,  0.04, 0.25) * 0.30 +
                catch_rate_score         * 0.20
            )
        else:
            # v3.2: targets missing -> catch rate is missing data, not neutral.
            eff = (
                _scale(ypr, 7.5, 14.0) * 0.625 +
                _scale(ms,  0.04, 0.25) * 0.375
            )
        # Supplement with tprr proxy when available
        tprr = _eval_metric_value(eval_metrics, "tprr", min_confidence=0.30)
        if tprr is not None:
            tprr_score = _scale(float(tprr), 0.22, 0.48)
            tprr_conf  = (eval_metrics.get("tprr") or {}).get("confidence", 0.35)
            eff = eff * (1.0 - 0.08 * tprr_conf) + tprr_score * (0.08 * tprr_conf)

        # grades_pass_route dropped: was PFF-only, no free replacement.

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
        # drop_rate dropped: was PFF-only, no free replacement.

        yac_rec = _eval_metric_value(eval_metrics, "yac_per_att", min_confidence=0.45)
        if yac_rec is not None:
            eff = _clip(eff + _clip((float(yac_rec) - 5.0) * 1.6, -4.0, 6.0))

    elif eval_metrics and pos == "RB":
        # pff_rushing_grade is now sourced from CFBD PPA (free replacement for PFF).
        pff_rush = _eval_metric_value(eval_metrics, "pff_rushing_grade", min_confidence=0.45)
        if pff_rush is not None:
            eff = _clip(eff + _clip((float(pff_rush) - 68.0) * 0.22, -5.0, 7.0))

    elif eval_metrics and pos == "QB":
        # adjusted_comp_pct dropped: was PFF-only, no free replacement.
        # nfl_passer_rating is now computed from box-score data (free replacement).
        psr = _eval_metric_value(eval_metrics, "nfl_passer_rating", min_confidence=0.45)
        if psr is not None:
            eff = _clip(eff + _clip((float(psr) - 85.0) * 0.18, -4.0, 6.0))

        # pressure_to_sack_rate dropped: was PFF-only, no free replacement.

    # Scheme-inflation discount: high-volume spread systems inflate efficiency
    # metrics (yds/rec, yds/target) for WRs and TEs alike.
    if pos in ("WR", "TE"):
        scheme_discount = _scheme_inflation_discount(ls.get("team"))
        if scheme_discount < 1.0:
            eff = _clip(eff * scheme_discount)

    # Model v2.0: opponent-adjusted WEPA blend.  wepa_score is a 0-100
    # percentile within the season/position cohort - a conservative 25% blend
    # so box-score efficiency stays primary and WEPA only nudges.
    if wepa:
        _WEPA_TYPE = {"QB": "passing", "RB": "rushing"}
        wtype = _WEPA_TYPE.get(pos)
        if wtype:
            rec = wepa.get((ls.get("season", 0), wtype)) or {}
            wscore = rec.get("adj_efficiency_score")
            if wscore is not None:
                try:
                    eff = _clip(eff * 0.75 + float(wscore) * 0.25)
                except (TypeError, ValueError):
                    pass

    return _clip(eff)


# Typical draft-class age by position (age at start of NFL rookie year).
# Updated to reflect modern college football (COVID year, grad transfers, etc.)
_TYPICAL_AGE = {"QB": 23.0, "RB": 22.5, "WR": 22.5, "TE": 23.0}
_AGE_ELITE   = {"QB": 21.0, "RB": 20.5, "WR": 21.0, "TE": 21.0}  # Raised from 20.5 - elite TEs can be slightly older
_AGE_WORST   = {"QB": 27.5, "RB": 25.0, "WR": 25.5, "TE": 26.0}   # QB more lenient - development timelines vary widely


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
    # Floor at 20 so age never completely destroys a player's score.
    # v3.2 R7-A6: cap 100 (was 95): the elite anchor IS the 100 anchor.
    return _clip(score, 20, 100)


def calc_breakout_score(seasons: List[Dict], age: Optional[float], position: str,
                        _inseason_year: Optional[int] = None) -> float:
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
    # QB dominator_rating is receiving-based and not meaningful for QBs - use neutral.
    dom_thresh = {"WR": 0.25, "RB": 0.275, "TE": 0.12}
    # v3.2 R7-A4: peak-season dominator (TE: pass share, per T5).
    def _dom_of(_s):
        if pos == "TE":
            _tpy = _safe(_s.get("team_pass_yards"))
            if _tpy > 0:
                return _safe(_s.get("receiving_yards")) / _tpy
        return _safe(_s.get("dominator_rating"))
    _peak_s = max(sorted_s, key=_dom_of) if sorted_s else ls
    dom = _dom_of(_peak_s)
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
        _ratio = dom / thresh
        if _ratio > 1:
            # v3.3 F3: above-threshold saturation. WR saturates at 1.4x the
            # threshold (a 0.35 dominator, just beyond recorded WR seasons);
            # RB/TE keep 1.8x, which is algebraically identical to the old
            # linear curve (40 / 0.8 = 50 points per unit of ratio).
            _sat = 1.4 if pos == "WR" else 1.8
            dom_score = min(100, 60 + 40 * min((_ratio - 1) / (_sat - 1), 1))
        else:
            dom_score = min(100, max(0, (_ratio - 1) * 50 + 60))
    else:
        dom_score = 30.0

    # Trajectory INTO the breakout season (per-game rates, so an
    # injury-shortened final season cannot erase the breakout).
    traj_score = 50.0
    if len(sorted_s) >= 2:
        def _pg(_s):
            _y = _safe(_s.get("receiving_yards", 0)) + _safe(_s.get("rush_yards", 0)) + _safe(_s.get("pass_yards", 0)) * 0.5
            return _y / max(_safe(_s.get("games_played"), 1), 1)
        if _peak_s is sorted_s[0]:
            # Broke out immediately: trajectory is whether he SUSTAINED it.
            _a, _b = _peak_s, sorted_s[-1]
        else:
            _a, _b = sorted_s[0], _peak_s
        if _pg(_a) > 0:
            # v3.3 V-A: when the peak season's per-game rate is at least 90%
            # of the first season's rate, the rate was maintained, so score
            # trajectory on TOTAL production growth into the peak season
            # instead of per-game growth (a short first season otherwise
            # reads as decline when only the workload grew).
            def _tot(_s):
                return _safe(_s.get("receiving_yards", 0)) + _safe(_s.get("rush_yards", 0)) + _safe(_s.get("pass_yards", 0)) * 0.5
            # v3.4: the totals-growth switch must not fire when the peak
            # season is the in-progress one; a partial season's totals are
            # calendar-bound, so trajectory stays on per-game rates.
            _peak_partial = (
                _inseason_year is not None
                and _b.get("season") == _inseason_year
            )
            if _pg(_b) >= 0.9 * _pg(_a) and _tot(_a) > 0 and not _peak_partial:
                _growth = (_tot(_b) - _tot(_a)) / _tot(_a)
            else:
                _growth = (_pg(_b) - _pg(_a)) / _pg(_a)
            traj_score = _clip(_scale(_growth, -0.20, 0.60))

    # Adjusted breakout age - continuous scale calibrated by position.
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
            _crossed = False
            for s in sorted_s:
                s_year = _safe(s.get("season"), 0)
                if _dom_of(s) >= thresh and s_year > 0 and current_year > 0:
                    breakout_age = age - (current_year - s_year)
                    _crossed = True
                    break
            if not _crossed and current_year > 0 and _safe(_peak_s.get("season"), 0) > 0:
                breakout_age = age - (current_year - _safe(_peak_s.get("season"), 0))

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
        # Improved scale: dom = thresh → 50% credit; dom = 1.5×thresh → 100% credit.
        # The old scale (0% at threshold) was too harsh, especially for in-progress
        # seasons where DOM can be volatile with small samples.
        if thresh is not None and thresh > 0 and pos != "QB":
            ratio = dom / thresh
            if ratio >= 1.0:
                # v3.2 V6b: 1.0x -> 0.65, 1.4x -> 1.0
                dominance_factor = min(1.0, 0.65 + (ratio - 1.0) / 0.4 * 0.35)
            else:
                # v3.2 V6b: smooth ramp below threshold, no knife edge
                dominance_factor = 0.65 * ratio * ratio
            adj_breakout_score *= dominance_factor

    score = dom_score * 0.45 + traj_score * 0.35 + adj_breakout_score * 0.20

    # Elite underclassman floor: a freshman/sophomore (age <= 20 in that
    # season) with elite raw production has proven breakout ability regardless
    # of what the weighted components say. Don't let trajectory noise from
    # in-progress seasons or DOM thresholds on stacked teams drag down a
    # genuinely elite early-career season.
    # WR/TE: 1,200+ receiving yards; RB: 1,200+ rush yards; QB: 3,500+ pass yards
    if age is not None and sorted_s:
        current_year = _safe(sorted_s[-1].get("season"), 0)
        for s in sorted_s:
            s_year = _safe(s.get("season"), 0)
            if s_year <= 0 or current_year <= 0:
                continue
            s_age = age - (current_year - s_year)
            if s_age > 20:
                continue
            # v3.4: for the in-progress season, the yardage gates are
            # counting thresholds, so the season's yards are paced to a
            # 12-game equivalent before they are compared (and before the
            # floor value is computed from them).
            _gp_r = _safe(s.get("games_played"), 0)
            _paced = (
                _inseason_year is not None and s_year == _inseason_year
                and 0 < _gp_r < 12
            )
            def _y_of(key, _s=s):
                _v = _safe(_s.get(key, 0))
                return _v * (12.0 / _gp_r) if _paced else _v
            # v3.2 V6a: ramp the floor (70 at the low end, 85 at the old cliff)
            _floor = None
            if pos in ("WR", "TE"):
                _y = _y_of("receiving_yards")
                if _y >= 1000:
                    _floor = 70.0 + min(1.0, (_y - 1000.0) / 200.0) * 15.0
            elif pos == "RB":
                _y = _y_of("rush_yards")
                if _y >= 1000:
                    _floor = 70.0 + min(1.0, (_y - 1000.0) / 200.0) * 15.0
            elif pos == "QB":
                _y = _y_of("pass_yards")
                if _y >= 3000:
                    _floor = 70.0 + min(1.0, (_y - 3000.0) / 500.0) * 15.0
            if _floor is not None:
                score = max(score, _floor)
                break

    # v3.1: Ascending trajectory bonus. A player whose most recent season's
    # per-game production exceeds the prior season's is developing, not
    # stagnating. The base traj_score rewards growth, but players who never
    # hit the dominator threshold get crushed by dom_score=30 and
    # adj_breakout_score=0. This bonus recognizes real improvement.
    # Up to +15, scaled by growth rate.
    if len(sorted_s) >= 2:
        _ls = sorted_s[-1]
        _pv = sorted_s[-2]
        _ls_yds = _safe(_ls.get("receiving_yards", 0)) + _safe(_ls.get("rush_yards", 0)) + _safe(_ls.get("pass_yards", 0)) * 0.5
        _pv_yds = _safe(_pv.get("receiving_yards", 0)) + _safe(_pv.get("rush_yards", 0)) + _safe(_pv.get("pass_yards", 0)) * 0.5
        # Use per-game rates to avoid penalizing in-progress seasons
        _ls_gp = max(_safe(_ls.get("games_played"), 1), 1)
        _pv_gp = max(_safe(_pv.get("games_played"), 1), 1)
        _ls_pg = _ls_yds / _ls_gp
        _pv_pg = _pv_yds / _pv_gp
        if _pv_pg > 0 and _ls_pg > _pv_pg:
            _growth = (_ls_pg - _pv_pg) / _pv_pg
            # +15 at 50%+ growth, scaled linearly, 0 at no growth
            _traj_bonus = min(15.0, max(0.0, _growth / 0.50 * 15.0))
            score = _clip(score + _traj_bonus)

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
_ATH_DATA_CAPS = {1: 74, 2: 82, 3: 96}   # n_metrics_present → cap


# ── Knot-anchored combine subscales (v3.3) ─────────────────────────────────
# Every combine leg except RAS is scored against recorded marks at
# performance tiers of drafted players at the position (nflverse combine
# data, drafted only, classes 2000-2025): piecewise-linear through the tier
# marks below, listed in tier order worst / p1 / p5 / p10 / p25 / p50 / p75 /
# p90 / p97 / p99 / best and priced by _KNOT_SCORES. The marks are fixed
# recorded constants, so the scale stays absolute across years. RAS has no
# drafted-tester distribution in that source and stays on its linear
# 4.0-10.0 scale.
_KNOT_SCORES = [0, 5, 12, 22, 38, 56, 77, 88, 93, 96, 100]

_KNOT_MARKS = {
    "QB": {
        "forty": [5.37, 5.266, 5.102, 5.004, 4.9, 4.81, 4.69, 4.58, 4.511, 4.411, 4.33],
        "vertical": [21.5, 24.51, 26.5, 27.05, 29.5, 31.5, 33.875, 36.0, 37.5, 38.5, 40.5],
        "broad": [96.0, 99.0, 100.0, 103.0, 107.0, 112.0, 116.0, 121.0, 125.0, 127.0, 129.0],
        "speed": [53.39, 57.64, 67.61, 70.65, 77.27, 84.53, 92.7, 99.65, 110.59, 118.15, 126.88],
        "cone": [7.8, 7.713, 7.517, 7.408, 7.22, 7.11, 6.96, 6.856, 6.762, 6.729, 6.65],
        "shuttle": [4.78, 4.654, 4.55, 4.52, 4.42, 4.295, 4.18, 4.09, 4.026, 3.952, 3.87],
    },
    "RB": {
        "forty": [4.83, 4.758, 4.67, 4.644, 4.58, 4.51, 4.45, 4.39, 4.355, 4.33, 4.24],
        "vertical": [26.5, 28.325, 30.0, 30.5, 33.0, 35.0, 37.0, 39.5, 41.0, 41.5, 43.0],
        "broad": [103.0, 106.53, 111.0, 113.0, 117.0, 120.0, 123.75, 127.0, 130.0, 132.94, 137.0],
        "speed": [82.9, 83.92, 88.18, 91.64, 97.16, 103.49, 109.23, 116.11, 121.84, 124.46, 129.55],
        "cone": [7.68, 7.556, 7.39, 7.27, 7.15, 7.01, 6.89, 6.81, 6.7, 6.654, 6.5],
        "shuttle": [4.67, 4.62, 4.508, 4.44, 4.348, 4.235, 4.143, 4.07, 3.995, 3.935, 3.93],
    },
    "WR": {
        "forty": [4.79, 4.711, 4.62, 4.59, 4.54, 4.47, 4.41, 4.35, 4.31, 4.279, 4.21],
        "vertical": [26.0, 28.5, 31.0, 32.5, 34.0, 36.0, 38.0, 39.5, 41.5, 42.575, 45.0],
        "broad": [105.0, 110.0, 114.0, 115.0, 118.0, 122.0, 126.0, 130.0, 134.0, 136.0, 140.0],
        "speed": [73.34, 78.76, 84.92, 87.59, 94.17, 100.27, 106.84, 112.51, 118.97, 122.78, 133.5],
        "cone": [7.54, 7.409, 7.31, 7.19, 7.07, 6.94, 6.8, 6.675, 6.612, 6.561, 6.5],
        "shuttle": [4.77, 4.57, 4.45, 4.39, 4.31, 4.2, 4.11, 4.02, 3.946, 3.886, 3.73],
    },
    "TE": {
        "forty": [5.09, 5.023, 4.93, 4.885, 4.81, 4.73, 4.64, 4.545, 4.481, 4.417, 4.37],
        "vertical": [26.5, 27.22, 29.0, 30.0, 31.5, 33.5, 35.5, 37.5, 39.5, 41.78, 43.5],
        "broad": [102.0, 104.68, 109.0, 110.0, 113.0, 116.0, 120.5, 124.0, 129.0, 131.66, 134.0],
        "speed": [78.07, 83.05, 87.04, 89.81, 94.91, 102.1, 108.03, 116.0, 122.7, 127.54, 138.03],
        "cone": [7.95, 7.652, 7.511, 7.44, 7.302, 7.12, 6.99, 6.88, 6.81, 6.76, 6.72],
        "shuttle": [4.82, 4.698, 4.57, 4.53, 4.45, 4.34, 4.25, 4.145, 4.09, 4.011, 3.98],
    },
}


def _knot_score(value, pos, leg):
    """Piecewise-linear score for one combine mark against the recorded
    drafted-tester tier marks at the position. Marks pair with scores in
    performance order, so lower-is-better legs (forty, cone, shuttle) price
    correctly once the pairs are sorted by mark value. Returns None when
    the position/leg has no recorded tier table."""
    marks = _KNOT_MARKS.get(pos, {}).get(leg)
    if not marks:
        return None
    pts = sorted(zip(marks, _KNOT_SCORES))
    if value <= pts[0][0]:
        return float(pts[0][1])
    if value >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return float(pts[-1][1])


def calc_athleticism_score(athleticism: Dict[str, Any], position: str) -> float:
    """
    Combine / pro-day metrics.  Falls back gracefully to positional median (55)
    when data is missing.

    RAS (Relative Athletic Score) is the most reliable single signal (0-10).
    Uses position-specific metric weights (e.g., 40-time matters more for WR,
    vertical for TE, speed score for RB).
    """
    if not athleticism:
        return None

    pos = position.upper()

    # Compute individual metric scores (each 0-100)
    metric_scores: Dict[str, float] = {}

    # RAS (0-10 → scale to 0-100)
    ras = athleticism.get("ras_score")
    if ras is not None:
        metric_scores["ras"] = _scale(_safe(ras), 4.0, 10.0)

    # 40-yard dash: knot-anchored to recorded drafted-tester tiers at the
    # position (v3.3). Faster = higher score via the tier pairing.
    forty_raw = athleticism.get("forty_yard")
    if forty_raw:
        _s = _knot_score(_safe(forty_raw), pos, "forty")
        if _s is not None:
            metric_scores["forty"] = _s

    # Vertical jump
    vert = athleticism.get("vertical_inches")
    if vert:
        _s = _knot_score(_safe(vert), pos, "vertical")
        if _s is not None:
            metric_scores["vertical"] = _s

    # Broad jump
    broad = athleticism.get("broad_jump_in")
    if broad:
        _s = _knot_score(_safe(broad), pos, "broad")
        if _s is not None:
            metric_scores["broad"] = _s

    # Speed score = weight * (40^4)^-1 * normalisation constant
    weight_lbs = athleticism.get("weight_lbs")
    if forty_raw and weight_lbs:
        ss_raw = (_safe(weight_lbs) * 200) / (_safe(forty_raw) ** 4)
        _s = _knot_score(ss_raw, pos, "speed")
        if _s is not None:
            metric_scores["speed_score"] = _s

    # Agility / change-of-direction (three_cone and short_shuttle).
    # Lower time = higher score. v3.3 F0: the old _scale calls passed
    # (hi, lo) inverted, which made _scale return a constant 50.0 for every
    # player; the knot tiers price these legs correctly.
    three_cone = athleticism.get("three_cone")
    if three_cone:
        _s = _knot_score(_safe(three_cone), pos, "cone")
        if _s is not None:
            metric_scores["three_cone"] = _s

    shuttle = athleticism.get("short_shuttle")
    if shuttle:
        _s = _knot_score(_safe(shuttle), pos, "shuttle")
        if _s is not None:
            metric_scores["shuttle"] = _s

    if not metric_scores:
        return None

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
    Volume of opportunity - how much of the team's offence ran through this player.

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
        _tgt_present = rec_tgts > 0
        rec_pg   = (rec_tgts / gp) if _tgt_present else (_safe(ls.get("receptions")) / gp)
        # Combines rush volume + receiving involvement
        # Elite: 20+ carries + 4+ targets; average: 12 carries + 2 targets
        rush_util = _scale(carries, 8.0, 22.0)
        # v3.3 V-C: when targets are missing, receptions per game is read on
        # a receptions scale (0.5 to 3.5), not the targets scale (1.0 to
        # 5.0): 1.6 receptions/g is not 1.6 targets/g, and the old read
        # under-credited every back graded without target data.
        recv_util = _scale(rec_pg,  1.0,  5.0) if _tgt_present else _scale(rec_pg, 0.5, 3.5)
        # Dynasty: pass-catching usage is more stable and more valued than rush volume
        return _clip(rush_util * 0.55 + recv_util * 0.45)

    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        # Elite: 35+ attempts/game; average: 25; game-manager: 18
        return _clip(_scale(att_pg, 16.0, 40.0))

    return 50.0


def calc_recruiting_score(recruiting: Optional[Dict[str, Any]]) -> float:
    """
    Recruiting-pedigree signal (model v2.0).

    247 Composite rating → 0-100.  Recruiting pedigree is one of the strongest
    known predictors of NFL success, largely orthogonal to college production
    (it measures talent before the college scheme touches it).

    Missing data is neutral (50.0), never punitive: older classes, JUCO and
    international prospects often have no composite entry.
    """
    if not recruiting:
        return 50.0
    rating = recruiting.get("composite_rating")
    if rating is not None:
        try:
            # Composite runs ~0.78 (2-star) to ~1.00 (elite 5-star).  The
            # median draft prospect (~0.87, high 3-star) maps to neutral 50,
            # so known-average pedigree never scores below unknown.
            return _clip(_scale(float(rating), 0.74, 1.00))
        except (TypeError, ValueError):
            pass
    stars = recruiting.get("stars")
    if stars is not None:
        try:
            # Aligned to the composite scale: 5 (typical 0.985) -> 94,
            # 4 (0.93) -> 73, 3 (0.87) -> 50, 2 (0.80) -> 23.
            return {5: 94.0, 4: 73.0, 3: 50.0, 2: 23.0}.get(int(stars), 50.0)
        except (TypeError, ValueError):
            pass
    return 50.0


def calc_competition_score(
    seasons: List[Dict],
    team_context: Optional[Dict] = None,
) -> float:
    """
    Conference quality + implied opponent strength.
    Recent seasons are weighted more heavily to reflect current competition level.
    Transfer players who upgraded conferences get credit for their most recent context.

    team_context (optional, model v2.0): {(school_lower, season): {sp_sos, ...}}.
    When SP+ strength-of-schedule is available for the latest season's team it
    is blended 50/50 with the conference-quality estimate - a direct measure
    of who they actually played rather than who shares their conference.
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

    # Model v2.0: blend in SP+ SOS when available for the latest season's team.
    if team_context and sorted_seasons:
        latest = sorted_seasons[0]
        team_key = ((latest.get("team") or "").lower().strip(), latest.get("season", 0))
        ctx = team_context.get(team_key) or {}
        sos = ctx.get("sp_sos")
        if sos is not None:
            try:
                # SP+ SOS typically spans roughly -12 (cupcake) to +12 (gauntlet)
                sos_quality = _scale(float(sos), -12.0, 12.0) / 100.0
                avg_quality = avg_quality * 0.5 + sos_quality * 0.5
            except (TypeError, ValueError):
                pass

    return _clip(_scale(avg_quality, 0.45, 1.00))


def calc_environment_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Adjusts for team usage patterns.

    - WR/TE: yards-based pass share (team_pass_yards / team_total_yards).
      Captures how much of the offense's actual production came through the air,
      not just how often they called pass plays. Falls back to attempt-based
      team_pass_rate when yard totals are unavailable.
    - For RBs, high rush rate inflates volume - slight discount applied.
    - Uses a recency-weighted average across all seasons so transferred players
      aren't locked to their latest team's scheme.
    """
    if not seasons:
        return 50.0

    pos = position.upper()

    # v3.2 R7-A5: no environment inputs (or QB, whose environment the
    # function does not model) -> missing data, weight redistributes.
    _has_env = any(
        (_safe(s.get("team_pass_yards")) > 0 and _safe(s.get("team_total_yards")) > 0)
        or s.get("team_pass_rate") is not None
        for s in seasons
    )
    if not _has_env or pos == "QB":
        return None

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


def _season_has_role(season: Dict, position: Optional[str]) -> bool:
    """
    v3.4: did the player hold a real role in this season?

    Stored games_played counts APPEARANCES, so a backup's 2-game season
    and an injured starter's 2-game season look identical on GP alone.
    Usage separates them: a season only counts as missed time when the
    player was actually in the rotation. Thresholds sit in the empty
    bands of the recorded distributions (panel + fixtures):
    - QB pass attempts >= 75: recorded backup seasons top out at 41
      (Mestemaker 2024; Sayin 2024 is 12, Moore 2024 is 8); the lightest
      recorded starter season is 99 (Moore's 4-game 2026). ~3 games of
      starting volume.
    - RB touches (carries + receptions) >= 60: recorded mop-up/rotation
      partials top out at 43 (Lacy 2026); recorded starter partials run
      110+ (Baugh 2026: 117). A starter lost at midseason runs ~90-120.
    - WR targets >= 20 or receptions >= 12; TE targets >= 15 or
      receptions >= 10: recorded non-role seasons top out at 9
      receptions (Evans 2024, Reynolds 2024); the lightest recorded role
      seasons are 13 (Green 2024). Targets are OR-ed with receptions
      because the targets feed is a separate step that can lag.
    With no position given, any position's test passing is enough (the
    usage fields are position-specific in practice).
    """
    att = _safe(season.get("pass_attempts"))
    touches = _safe(season.get("rush_attempts")) + _safe(season.get("receptions"))
    tgt = season.get("targets")
    tgt = _safe(tgt) if tgt is not None else 0.0
    rec = _safe(season.get("receptions"))
    pos = (position or "").upper()
    if pos == "QB":
        return att >= 75
    if pos == "RB":
        return touches >= 60
    if pos == "WR":
        return tgt >= 20 or rec >= 12
    if pos == "TE":
        return tgt >= 15 or rec >= 10
    return att >= 75 or touches >= 60 or tgt >= 20 or rec >= 12


def calc_durability_score(seasons: List[Dict],
                          position: Optional[str] = None,
                          _inseason_year: Optional[int] = None) -> Optional[float]:
    """
    Durability: how often the player is available when the team plays.

    Core principle: games_played / games_available.
    - 12+ games in a completed season = full season (100%). The extra games
      beyond 12 are just playoff runs, not extra durability credit.
    - In-progress season: compare against current week. If the player has
      played every game so far, that's 100% - don't discount for the season
      being unfinished.

    Sophistication beyond raw percentage:
    - Recency weighting: recent seasons predict future availability better
    - Missed-games penalty: a completed season under 9 games suggests injury
      risk. Scaled: 8 games = small ding, 4 games = major red flag.
    - In-progress seasons never trigger penalties, only contribute positively
      to the average if the player has been available.

    v3.4 role rule (both modes): a season under the 12-game full-season
    bar counts only when the player held a real role in it (see
    _season_has_role). Backup and mop-up seasons are excluded entirely,
    neither as availability nor as missed time; before this, Sayin's
    4-game backup season priced 44.07 and Mestemaker's 2-game backup
    season priced 6.01. A 12+ game season is availability-proven by
    itself and always counts, whatever the usage (a blocking TE who
    dressed 16 times was available 16 times). If no counted season has
    GP data, durability is missing and its weight redistributes. (A
    season-ending injury in a player's first 1-2 starts is
    indistinguishable from backup work in stored data and is forgiven
    with it; that is the accepted cost of the rule.)

    v3.4 in-season variant: the in-progress season is NOT blanket-
    excluded. It counts when it is a role season, with absences measured
    against a bye-adjusted calendar floor (current week minus 2: FBS
    teams carry at most 2 byes, so a shortfall beyond that is certainly
    real missed time, while a starter who has played every team game is
    never charged for byes). Non-role partial seasons stay excluded,
    like any non-role season. The partial season never triggers the
    completed-season missed-games penalty.

    Returns None if no GP data (weight redistributed, like athleticism).
    """
    if not seasons:
        return None

    from datetime import date as _d
    today = _d.today()
    current_year = today.year
    season_start = _d(current_year, 8, 25)
    current_week = 1 if today < season_start else min((today - season_start).days // 7 + 1, 16)

    FULL_SEASON_GAMES = 12  # 12+ = full season, no bonus for playoff extras

    sorted_s = sorted(seasons, key=lambda s: s.get("season", 0))
    scored = []  # (pct, is_completed, gp)

    for s in sorted_s:
        gp = s.get("games_played")
        if gp is None:
            continue
        gp = _safe(gp)
        if gp < FULL_SEASON_GAMES and not _season_has_role(s, position):
            continue
        season_year = s.get("season", 0)
        is_partial = (_inseason_year is not None
                      and season_year == _inseason_year)

        if season_year == current_year:
            # In-progress: available = weeks played so far. In-season
            # mode measures absences against the bye-adjusted floor so
            # a healthy starter is not charged for his team's byes.
            if is_partial:
                denom = max(current_week - 2, 1)
            else:
                denom = current_week
            pct = min(gp / denom, 1.0) if denom > 0 else 1.0
            scored.append((pct, False, gp))
        elif season_year < current_year:
            # Completed: 12+ games = 100%, scale below that
            pct = min(gp / FULL_SEASON_GAMES, 1.0)
            scored.append((pct, not is_partial, gp))

    if not scored:
        return None

    # Weighted average: recent seasons count more
    n = len(scored)
    weighted_sum = 0.0
    total_weight = 0.0
    for rank, (pct, _, _) in enumerate(scored):
        w = 0.5 + 0.5 * (rank / max(n - 1, 1))  # 0.5 to 1.0
        weighted_sum += pct * w * 100.0
        total_weight += w
    base = weighted_sum / total_weight

    # Penalty for missed games in completed seasons only.
    # 9+ games: no penalty. Below that, increasingly harsh.
    # 8 games -> -5, 6 games -> -15, 4 games -> -30, 2 games -> -45
    penalty = 0.0
    for pct, is_completed, gp in scored:
        if not is_completed:
            continue
        if gp < 9:
            # Quadratic penalty: missing more games hurts disproportionately
            missed = 9 - gp
            penalty = max(penalty, missed * missed * 1.2)

    return _clip(base - penalty)



# ─────────────────────────────────────────────────────────────────────────────
# Fantasy translation bonus
# ─────────────────────────────────────────────────────────────────────────────

# Position-level adjustment reflecting long-run fantasy value scarcity.
# TE premium is negative because TE translation to fantasy is hardest.
POSITION_FANTASY_MULT: Dict[str, float] = {
    "WR": 1.00,
    "RB": 1.00,
    "QB": 0.90,   # QBs are less valued in 1QB dynasty
    "TE": 0.95,   # Increased from 0.85 - elite TEs should be much closer to WR level
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
        # the highest hit rate of any position/tier - higher than 1st-round WRs at 64%).
        # Raised to 0.29 to match the data. Breakout and competition reduced to compensate.
        "draft_capital": 0.26,
        "production": 0.19,
        "utilization": 0.08,
        "efficiency": 0.10,
        "age": 0.09,
        "breakout": 0.12,
        "athleticism": 0.11,
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
        # Draft capital raised: pick position is the strongest single signal for TEs
        # once scheme inflation and profile penalties are properly applied.
        # Age stays meaningful: early declarers (≤22) have far better dynasty hit rates.
        # Production given extra credit now that scheme inflation is applied to TEs.
        "draft_capital": 0.24,
        "production": 0.20,
        "utilization": 0.07,
        "efficiency": 0.11,
        "age": 0.14,
        "breakout": 0.08,
        "athleticism": 0.10,
        "competition": 0.05,
        "environment": 0.00,
        "durability": 0.01,
    },
}

# ── Model v2.0: recruiting-pedigree component ─────────────────────────────────
# 5% weight carved proportionally out of every other component so the blend
# stays at 1.0.  Calibrated overrides (historical_calibration) predate v2.0
# and lack this key - score_prospect reads it via .get("recruiting", 0.0).
_RECRUITING_WEIGHT = 0.05
for _pos in POSITION_WEIGHTS:
    _w = POSITION_WEIGHTS[_pos]
    for _k in list(_w.keys()):
        _w[_k] *= (1.0 - _RECRUITING_WEIGHT)
    _w["recruiting"] = _RECRUITING_WEIGHT

# Validate that all position weights sum to 1.0
for pos, weights in POSITION_WEIGHTS.items():
    assert abs(sum(weights.values()) - 1.0) < 0.001, f"{pos} weights must sum to 1.0, got {sum(weights.values())}"


# Enhanced evaluation functions
# 

from typing import Tuple

def calc_loaded_roster_adjustment(
    team: str,
    position: str,
    season: int,
    production_score: float,
    market_share: float,
    ypc: float = 0.0,
) -> float:
    # Values are (room_size, talent_tier).
    # talent_tier 1 = Day-2/3 starter above the prospect (minimal suppression signal)
    # talent_tier 2 = clear first-round prospect, established alpha role
    # talent_tier 3 = elite/top-15 pick prospect with dominant alpha role
    loaded_rosters: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {
        "Ohio State": {
            "WR": {
                2021: (2, 3),  # Olave (#10) + Wilson (#20) - co-equal first-round alphas
                2022: (2, 3),  # JSN (Biletnikoff, consensus WR1 prospect) + Egbuka (first-round)
                2023: (2, 3),  # MHJ (consensus WR1, top-5 overall) + Egbuka (first-round)
                2024: (3, 3),  # Smith (generational) + Egbuka (#19) + Tate (#4) - three first-rounders
                2025: (3, 3),  # Smith (projected #1 WR overall) + Tate (#4 overall pick)
            },
        },
        "Alabama": {
            "WR": {
                2020: (3, 3),  # Smith (Heisman/#10) + Waddle (#4) + Metchie (first-round)
                2021: (2, 3),  # Jameson Williams (#12 overall, dominant alpha 1,572 yds) + Metchie (#38)
                2022: (2, 1),  # Burton (6th round) + Holden (UDFA) - no first-round grade
                2025: (2, 2),  # Ryan Williams (first-round grade, WR2-3 in class, clear alpha)
            },
            "RB": {
                2020: (2, 3),  # Najee Harris (Doak Walker, #24 overall, dominant alpha)
                2021: (2, 1),  # Brian Robinson (3rd round/#98, clear alpha but Day 2-3 grade)
            },
        },
        "Georgia": {
            "TE": {
                2021: (2, 3),  # Brock Bowers (92.1 PFF freshman record, #13 overall eventual pick)
                2022: (2, 3),  # Bowers (established #1 TE in class) + Washington (Day 2/#93)
            },
            "WR": {
                2024: (2, 1),  # Lovett (Day 2-3) + Bell (undrafted) - no first-round grade
                2025: (2, 1),  # Bell (undrafted) + Young (4th round) - below first-round threshold
            },
        },
        "USC": {
            "WR": {
                2022: (2, 2),  # Jordan Addison (Biletnikoff, #23 overall, unambiguous alpha)
                2024: (2, 1),  # Branch (Day 2, no clear alpha) + Lemon (emerging, not yet first-round)
                2025: (2, 2),  # Lemon (#20 overall, Biletnikoff, clear alpha) + Lane (Day 2)
            },
        },
        "LSU": {
            "WR": {
                2019: (3, 3),  # Chase (#5) + Jefferson (#22) + Marshall (2nd round) - legendary room
                2022: (2, 1),  # Boutte (grade cratered Day 3 due to injuries) + Nabers/Thomas emerging sophs
                2023: (2, 3),  # Nabers (#6 overall) + Brian Thomas (#23 overall) - co-equal elite alphas
                2024: (2, 1),  # Kyren Lacy (Day 2) + Anderson - post-elite-era room
                2025: (2, 1),  # Aaron Anderson (Day 2) - no first-round grade alpha
            },
        },
        "Texas": {
            "WR": {
                2023: (2, 2),  # Worthy (#28) + Mitchell (#52) - two first/2nd-round picks, three-way split
                2024: (2, 2),  # Golden (#23, clear alpha by season end) + Bond (Day 2)
                2025: (2, 1),  # Wingo (unproven) + Moore (Day 3) - no first-round grade
            },
        },
        "Oregon": {
            "WR": {
                2024: (2, 1),  # Johnson (Day 2-3) + Stewart (not yet first-round grade) - three-way split
                2025: (2, 1),  # Stewart injured entire season; Moore (5-star freshman) unproven
            }
        },
        "Notre Dame": {
            "RB": {
                2024: (2, 3),  # Love (#3 overall, dominant alpha) + Price (#32 despite never starting)
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

    entry = (
        loaded_rosters.get(team, {})
        .get(position, {})
        .get(season)
    )

    if not entry:
        return 1.0

    room_size, talent_tier = entry

    if room_size < 2:
        return 1.0

    # Only backups benefit from the crowded-room bonus - a player who dominates
    # their team's market share is the alpha, not the one being suppressed.
    # Threshold: >= 0.28 market share means they're the clear lead back/receiver;
    # their stats are face-value, not opportunity-constrained.
    if market_share >= 0.28:
        return 1.0

    if room_size >= 4:
        base_bonus = 0.12
    elif room_size == 3:
        base_bonus = 0.10
    elif position == "RB":
        base_bonus = 0.10   # committee RBs are more opportunity-limited than shared WR rooms
    else:
        base_bonus = 0.06

    # ms_factor: inverted so lower market share (more suppressed) = bigger bonus.
    # A backup with ms=0.10 is heavily blocked; ms=0.25 is getting decent volume
    # despite a loaded room - both deserve a bonus, but the more blocked player more so.
    if market_share <= 0.12:
        ms_factor = 1.00   # heavily suppressed backup
    elif market_share <= 0.18:
        ms_factor = 0.80   # clear backup role
    elif market_share <= 0.22:
        ms_factor = 0.60   # committee / split role
    else:
        ms_factor = 0.40   # decent volume but still not the alpha

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
        # Per-carry efficiency is the quality signal - use it instead of production_score.
        prod_factor = 0.65
    else:
        prod_factor = 0.10

    # Scale base_bonus by how elite the blocking talent was.
    # Tier 3 (generational alpha above) warrants a 55% larger bonus than the baseline.
    tier_scale = {1: 1.00, 2: 1.25, 3: 1.55}.get(talent_tier, 1.00)
    base_bonus = base_bonus * tier_scale

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

    # Scale up slightly for larger rooms - harder to carve out volume with more competition
    group_factor = min(1.0, group_size / 4.0)
    realized_bonus = base_bonus * (0.60 + 0.40 * group_factor)
    return min(1.0 + realized_bonus, 1.20)


def draft_capital_multiplier(round_selected: int) -> float:
    """
    Draft capital multiplier aligned with the NFL's 3-day structure.

    Day 1  - Round 1 (picks 1-32):    no bonus; dc_score already encodes pick value
    Day 2  - Rounds 2-3 (picks 33-96): solid investment, starter likelihood meaningful
    Day 3  - Rounds 4-7 (picks 97+):  developmental, far less predictive for dynasty
    """
    if round_selected == 1:        # Day 1 - no multiplier; pick value captured in dc_score
        return 1.00
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
    # v3.2: no pick from any source means NO pick, not a pick-300 default.
    # NaN makes every pick-gated comparison below False, so stat-only
    # branches still run and pick-gated bonuses AND penalties stay off.
    projected_pick = _safe((draft_capital or {}).get("projected_pick"), float("nan"))

    if position == "WR":
        contested = _safe(latest.get("contested_catch_rate"), 0.0)
        yac = _safe(latest.get("yards_after_catch_per_reception"), 0.0)
        yprr = _safe(latest.get("yards_per_route_run"), 0.0)
        market_share = _safe(latest.get("market_share_yards"), 0.0)
        gp = max(_safe(latest.get("games_played"), 12.0), 1.0)
        rec_yds_pg = _safe(latest.get("rec_yds_pg"), _safe(latest.get("receiving_yards"), 0.0) / gp)
        rec_tds_pg = _safe(latest.get("rec_tds_pg"), _safe(latest.get("receiving_tds"), 0.0) / gp)

        # Penalize classic WR false-positive profiles (high volume, poor translation traits)
        # drop_rate check removed: was PFF-only, no free replacement.
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

    elif position == "QB":
        gp = max(_safe(latest.get("games_played"), 12.0), 1.0)
        rush_yds_pg = _safe(latest.get("rush_yards"), 0.0) / gp
        # Dual-threat QBs are systematically undervalued: poor college completion%
        # penalises the production score, but rushing is a direct fantasy component
        # that the model underweights.  Josh Allen (98 yd/g), Lamar Jackson (63 yd/g),
        # and Jalen Hurts (76 yd/g) were all underrated by the model without this bonus.
        if rush_yds_pg >= 35:
            adj += 3.5  # v3.2 Q3: thresholds lowered from 40/25 to 35/20
        elif rush_yds_pg >= 20:
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
        seasons_te = prospect.get("seasons") or []
        target_share = _safe(latest.get("target_share"), 0.0)
        draft_age = _safe(prospect.get("age"), 0.0)

        # Use latest season for penalty checks (reflects current role)
        rec_yds_pg = _safe(latest.get("rec_yds_pg"), 0.0)
        if rec_yds_pg == 0.0:
            gp_l = max(_safe(latest.get("games_played"), 12.0), 1.0)
            rec_yds_pg = _safe(latest.get("receiving_yards"), 0.0) / gp_l

        # Use peak season for upside bonus - an injury-shortened year should not
        # mask a player whose best season was historically dominant (e.g. Bowers 2022).
        def _rec_ydpg(s: dict) -> float:
            v = _safe(s.get("rec_yds_pg"), 0.0)
            if v:
                return v
            gp = max(_safe(s.get("games_played"), 12.0), 1.0)
            return _safe(s.get("receiving_yards"), 0.0) / gp
        best_rec_yds_pg = max((_rec_ydpg(s) for s in seasons_te), default=rec_yds_pg)

        # v3.0: Baseline TE development delay is now CONDITIONAL, not universal.
        # Previously every TE started at -5.0. Now only applies to TEs with
        # genuinely weak receiving usage (blocking archetypes).
        # v3.2 R5-TE: low RAW volume is already charged in production and
        # utilization. When the TE owns a real share of his own passing
        # offense, the low raw volume was the offense's, not the player's:
        # do not charge the same volume again in translation.
        _best_share = max(
            ((_safe(s.get("receiving_yards")) / _safe(s.get("team_pass_yards")))
             if _safe(s.get("team_pass_yards")) > 0 else 0.0 for s in seasons_te),
            default=0.0)
        if best_rec_yds_pg < 50.0 and _best_share < 0.17:
            adj -= 5.0

        # Elite receiving TE upside: uses peak season so injury years don't suppress it.
        # Tiers calibrated so generational prospects (Pitts #4 → ~91, Bowers #13 → ~88).
        if projected_pick <= 10 and best_rec_yds_pg >= 60:
            adj += 9.0
        elif projected_pick <= 15 and best_rec_yds_pg >= 75:
            adj += 8.0
        elif projected_pick <= 20 and best_rec_yds_pg >= 65:
            adj += 5.0
        elif projected_pick <= 25 and best_rec_yds_pg >= 60:
            adj += 3.0
        elif projected_pick <= 40 and best_rec_yds_pg >= 60:
            adj += 2.0  # v3.2 T3: TE1s usually land 25-45, not top 25

        # Profile-based shrinkage for weak receiving-usage TEs (blocking archetypes).
        if rec_yds_pg > 0 and rec_yds_pg < 35:
            adj -= 3.0
        if target_share > 0 and target_share < 0.10:
            adj -= 2.0
        if draft_age and draft_age > 23.0 and age_score < 50:
            adj -= 1.0
        if projected_pick <= 64 and rec_yds_pg > 0 and rec_yds_pg < 42 and _best_share < 0.17:
            adj -= 2.0

    # Keep the adjustment bounded; this is a corrective signal, not the main model.
    return max(-14.0, min(9.0, adj))  # v3.2 V9: bound matches the +9 TE tier


def score_prospect(
    prospect: Dict[str, Any],
    draft_capital: Optional[Dict[str, Any]] = None,
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
    _exclude_capital: bool = False,
    _active_season_year: Optional[int] = None,
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

    # v3.4: two-mode grading (round 14 of the iteration log). A player is
    # IN-SEASON iff the run supplies an active season year, his latest
    # season IS that year, and its stored games played show a partial
    # season (1 to 11 games; 12 is the model's own FULL_SEASON_GAMES
    # constant, and a missing GP is not evidence of partiality). Otherwise
    # COMPLETE mode: every in-season branch below is gated on _inseason,
    # and with no active year supplied the grading is the complete-mode
    # path end to end (this is what keeps the backtest on final profiles
    # reproducible).
    _ls_mode = _latest_season(seasons) or {}
    _gp_mode = _ls_mode.get("games_played")
    _inseason = (
        _active_season_year is not None
        and _ls_mode.get("season") == _active_season_year
        and _gp_mode is not None
        and 1 <= _safe(_gp_mode) <= 11
    )
    _inseason_year = int(_active_season_year) if _inseason else None

    production_score    = calc_production_score(seasons, pos, eval_metrics=eval_metrics,
                                                skip_sagarin=skip_sagarin,
                                                _inseason_year=_inseason_year)
    
    # Apply crowded-room bonus: prefer data-driven depth rank when available,
    # fall back to manually curated loaded_roster list when depth data is absent.
    ls = _latest_season(seasons) or {}
    team = ls.get("team", "")
    season = ls.get("season", 0)
    market_share = _safe(ls.get("market_share_yards"), 0.15)

    depth_chart_adjustment = calc_depth_chart_adjustment(seasons, pos)

    if depth_chart_adjustment != 1.0:
        # CFBD depth rank data available - use data-driven adjustment
        loaded_roster_adjustment = depth_chart_adjustment
    else:
        # No depth data (seed-only) - fall back to manually curated list
        loaded_roster_adjustment = calc_loaded_roster_adjustment(
            team, pos, season, production_score, market_share,
            ypc=_safe(ls.get("yds_per_carry")),
        )
    production_score = _clip(production_score * loaded_roster_adjustment)

    utilization_score   = calc_utilization_score(seasons, pos)
    # Utilization is pure opportunity share (targets/carries/game) - fully suppressed
    # when splitting reps with multiple NFL-caliber teammates. Apply the full multiplier.
    utilization_score = _clip(utilization_score * loaded_roster_adjustment)

    efficiency_score    = calc_efficiency_score(
        seasons, pos, eval_metrics=eval_metrics,
        wepa=prospect.get("wepa"),
        _inseason_year=_inseason_year,
    )
    # Efficiency blends per-touch quality (ypr, ypc - unaffected by depth) with
    # market_share_yards (~35% of WR/TE base, ~15% of RB base - suppressed by depth).
    # Apply ~40% of the bonus to credit the market-share fraction without distorting
    # the per-touch quality signal.
    if loaded_roster_adjustment > 1.0:
        eff_adj = 1.0 + (loaded_roster_adjustment - 1.0) * 0.40
        efficiency_score = _clip(efficiency_score * eff_adj)

    age_score           = calc_age_score(age, dy, pos)
    breakout_score      = calc_breakout_score(seasons, age, pos,
                                              _inseason_year=_inseason_year)
    # Dominator rating (primary breakout signal) is directly suppressed in loaded rooms.
    # Apply ~35% of the bonus so backed-up players who still show early-breakout patterns
    # get appropriate credit.
    if loaded_roster_adjustment > 1.0:
        bk_adj = 1.0 + (loaded_roster_adjustment - 1.0) * 0.35
        breakout_score = _clip(breakout_score * bk_adj)

    athleticism_score   = calc_athleticism_score(ath, pos)
    # v3.3 three-zone athleticism transform (class-wide; supersedes the
    # v3.2 R7-A3 tested-QB-only compression, which is removed so nothing
    # double-applies). Testing is a red-flag check, not a gate: the raw
    # component stands at full strength at and below 60, the competent-to-
    # very-good range is compressed (slope 0.70 from 60 to 85), and
    # exceptional testing keeps a bounded toe above 85 (slope 0.90 from the
    # 77.5 intercept, so a perfect raw 100 lands at 91.0).
    if athleticism_score is not None:
        _r = athleticism_score
        if _r <= 60.0:
            pass
        elif _r <= 85.0:
            athleticism_score = 60.0 + (_r - 60.0) * 0.70
        else:
            athleticism_score = 77.5 + (_r - 85.0) * 0.90
    # v3.0 R6: Production-implied athleticism floor. Elite producers are
    # almost certainly functional athletes; don't leave the athleticism
    # weight on the table pre-combine. The v3.2 constant (70.0) is passed
    # through the v3.3 transform, which re-denominates it to 67.0.
    _athleticism_projected = False
    # v3.4: in-season, untested athleticism is MISSING data, not a
    # projected 67.0. The floor is suppressed and the existing missing-
    # athleticism redistribution below carries the weight. Tested players
    # are unaffected (the floor only ever applies when score is None).
    if (not _inseason
            and athleticism_score is None and production_score >= 90.0
            and breakout_score >= 75.0):
        athleticism_score = 67.0
        _athleticism_projected = True
    competition_score   = calc_competition_score(
        seasons, team_context=prospect.get("team_context"),
    )
    recruiting_score    = calc_recruiting_score(prospect.get("recruiting"))
    environment_score   = calc_environment_adjustment(seasons, pos)
    durability_score    = calc_durability_score(seasons, position=pos,
                                                _inseason_year=_inseason_year)

    # v3.2: draft capital is present only when a projected pick exists from
    # some source (mock consensus, actual pick, or the seed fallback). No
    # pick anywhere means the capital component is MISSING data: it is
    # excluded below and its weight redistributes over present components.
    # The old pick-150 default was a tax on uncovered prospects (audit
    # section 2, round 6 of the v3.2 iteration log), not a neutral prior.
    _capital_missing = (
        _exclude_capital
        or not draft_capital
        or draft_capital.get("projected_pick") is None
    )
    if draft_capital and not _capital_missing:
        dc_score = _safe(draft_capital.get("projected_draft_capital_score"), 40.0)
        projected_round = draft_capital.get("projected_round")
    else:
        dc_score = 0.0
        projected_round = None

    #  Enhanced draft capital modeling with nonlinear tiered bonuses
    # Apply round-specific multipliers to capture nonlinear value
    if projected_round and draft_capital:
        round_multiplier = draft_capital_multiplier(projected_round)
        dc_score = _clip(dc_score * round_multiplier)

    # ── Position dc multiplier ──────────────────────────────────────────────
    # TE draft capital is less predictive than RB/WR (harder college-to-NFL
    # translation: contested catches, blocking duties, late development curves).
    # QBs are NOT discounted here - pick #1 QB correctly scores 100/100.
    # The lower dynasty value of QB capital vs WR/RB capital is captured
    # entirely through the QB draft_capital WEIGHT (0.22 vs WR 0.29).
    # TE draft capital is significantly less predictive for dynasty: even a Round 1
    # TE typically contributes minimally for 2-3 years (development curve + TE scarcity
    # doesn't translate to immediate fantasy points).  0.92 means a Round 1 TE
    # gets 0.92× vs 1.0× for a Round 1 WR - narrow gap; elite TEs deserve near-WR
    # draft capital credit once profile penalties handle the development discount.
    dc_multiplier = {}.get(pos, 1.00)  # v3.2 T1: TE 0.92 multiplier removed

    # ── Day-3 penalty ───────────────────────────────────────────────────────
    # Applied only to true Day 3 picks (Round 4+, pick ≥ 97).
    # Day 2 picks (rounds 2-3, picks 33-96) are not penalised here - their
    # reduced value is already captured by draft_capital_multiplier.
    # Tiered within Day 3: late Day 3 (round 6-7, pick > 141) hit harder.
    if draft_capital and not _capital_missing:
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
    late_round_upside = calc_late_round_upside(
        None if _exclude_capital else draft_capital, seasons, pos)
    interaction_features = calc_interaction_features(
        production_score, efficiency_score, athleticism_score or 0.0, dc_score_adjusted
    )

    # Get position-specific weights (optionally overridden by calibrated weights)
    weights_source = position_weights_override or POSITION_WEIGHTS
    pos_weights = dict(weights_source.get(pos, POSITION_WEIGHTS["WR"]))

    # Post-draft: actual pick is certain - increase draft capital weight, spread
    # the reduction proportionally across all other components so sum stays 1.0.
    is_actual_pick = (draft_capital or {}).get("is_actual_pick", False)
    if is_actual_pick and not _exclude_capital:
        boost = 0.06
        pos_weights["draft_capital"] = min(0.50, pos_weights["draft_capital"] + boost)
        other_keys = [k for k in pos_weights if k != "draft_capital"]
        other_sum  = sum(pos_weights[k] for k in other_keys)
        target_sum = 1.0 - pos_weights["draft_capital"]
        if other_sum > 0:
            ratio = target_sum / other_sum
            for k in other_keys:
                pos_weights[k] *= ratio

    # Neutralize missing athleticism: redistribute its weight to other
    # components so a lack of combine data doesn't drag the score down.
    # Keep as None (not 0.0) so the DB stores NULL and the UI hides it.
    _weights = dict(pos_weights)
    _athleticism_missing = athleticism_score is None
    if _athleticism_missing:
        ath_w = _weights.pop("athleticism", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and ath_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + ath_w / remaining)
        # athleticism_score stays None -> DB NULL -> UI hides the bar

    # v3.2 V6c: QB breakout carries no signal; exclude and redistribute.
    if pos == "QB" and "breakout" in _weights:
        bk_w = _weights.pop("breakout", 0.0)
        if bk_w > 0:
            _weights["production"] = _weights.get("production", 0.0) + bk_w / 2.0
            _weights["efficiency"] = _weights.get("efficiency", 0.0) + bk_w / 2.0

    # v3.2 V2/V7: missing utilization and missing age are missing data,
    # not neutral scores. A stored targets value below receptions is a
    # pipeline failure (audit section 0), so it counts as missing too.
    _ls_v32 = _latest_season(seasons) or {}
    _tgt_v32 = _safe(_ls_v32.get("targets"))
    _rec_v32 = _safe(_ls_v32.get("receptions"))
    _util_missing = (
        pos in ("WR", "TE")
        and (_tgt_v32 <= 0 or _tgt_v32 < _rec_v32)
    )
    if _util_missing:
        util_w = _weights.pop("utilization", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and util_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + util_w / remaining)
    _age_missing_v32 = age is None
    if _age_missing_v32:
        age_w = _weights.pop("age", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and age_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + age_w / remaining)

    # v3.1: Neutralize missing durability the same way. A fake 60.0 is
    # worse than honest NULL - it drags down durable players and inflates
    # injury-prone ones. Redistribute the weight when GP data is absent.
    _durability_missing = durability_score is None
    if _durability_missing:
        dur_w = _weights.pop("durability", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and dur_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + dur_w / remaining)
        # durability_score stays None -> DB NULL -> UI hides the bar

    # v3.2 R7-A5: missing environment redistributes like durability.
    if environment_score is None:
        env_w = _weights.pop("environment", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and env_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + env_w / remaining)

    # v3.2: missing draft capital redistributes the same way. The remaining
    # weights renormalize to the total they carried with capital included,
    # so the grade stays on the same scale with capital simply absent.
    if _capital_missing:
        dc_w = _weights.pop("draft_capital", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and dc_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + dc_w / remaining)

    # Base prospect score with position-specific weights.
    # v3.0 R6: Use _weights (athleticism redistribution applied) so the
    # redistributed weight actually takes effect. Previously the sum used
    # pos_weights, silently dropping the redistribution and capping the
    # max score when athleticism was NULL.
    prospect_score = (
        production_score      * _weights["production"]    +
        utilization_score     * _weights.get("utilization", 0.0)   +
        efficiency_score      * _weights["efficiency"]    +
        age_score             * _weights.get("age", 0.0)           +
        breakout_score        * _weights.get("breakout", 0.0)      +
        (athleticism_score or 0.0) * _weights.get("athleticism", 0.0)   +
        competition_score     * _weights["competition"]   +
        (environment_score or 0.0) * _weights.get("environment", 0.0) +
        (durability_score or 0.0) * _weights.get("durability", 0.0) +
        dc_score_adjusted     * _weights.get("draft_capital", 0.0) +
        # v2.0: recruiting pedigree. .get() because calibrated overrides
        # (historical_calibration) predate the recruiting component.
        recruiting_score      * _weights.get("recruiting", 0.0)
    )

    # Add experience score for QBs (uses the experience weight in QB weights)
    if pos == "QB" and "experience" in pos_weights:
        prospect_score += experience_score * pos_weights["experience"]

    # v3.2 R7: no late-round upside add-on. The final grade is the
    # weighted component sum plus the bounded translation adjustment only.

    # v3.2 R7: benchmark boosts are NOT applied. They double-count inputs
    # already inside the components (draft capital, dominator, breakout,
    # production, age, market share, competition, athleticism) and then add
    # a flat +6 benchmark layer on top. An absolute scale cannot carry a
    # second, multiplicative opinion about the same facts.
    benchmark_boosts = {"total_boost": 0.0}

    # Position-specific translation adjustment from historical miss archetypes.
    # Applied late (after global scaling) so the correction magnitude is preserved.
    translation_adjustment = calc_translation_adjustment(
        prospect=prospect,
        position=pos,
        # v3.2: with no pick (or in the talent-only variant), translation
        # receives NO pick: NaN disables every pick-gated bonus and penalty
        # while stat-only branches still run.
        draft_capital=(
            {"projected_pick": float("nan")} if _capital_missing else draft_capital
        ),
        production_score=production_score,
        efficiency_score=efficiency_score,
        age_score=age_score,
    )
    prospect_score += translation_adjustment

    # v3.2 R7: no thin-sample final multiplier. Sample reliability is
    # reported by confidence_score; it does not rescale the grade.

    # Final clamp to the documented 0-100 grade scale. The quadratic boost and
    # benchmark boost each cap at 100, but the additive translation_adjustment
    # (up to +8, e.g. the +3.5 dual-threat-QB bonus) and late-round upside are
    # applied afterwards, so an elite profile could otherwise finish above 100
    # (e.g. a top dual-threat QB reaching 103.5) — breaking the grade scale and
    # the fantasy-scaled overall_score derived from it.
    prospect_score = _clip(prospect_score, 0.0, 99.9)  # v3.2 V9: no 100s

    prospect_score = round(prospect_score, 2)

    # v3.2 transparent capital: talent_score is this same profile graded
    # with draft capital excluded entirely (capital component removed,
    # remaining weights renormalized to their pre-removal total, actual
    # pick weight boost off, translation pick-neutral), through the same
    # final stage. capital_delta = prospect_score - talent_score.
    # Contract: when no projected pick exists from any source, the grade
    # already excludes capital, so talent_score == prospect_score and
    # capital_delta is None (no capital information, not a zero effect).
    if _capital_missing:
        talent_score = prospect_score
        capital_delta = None
    else:
        _talent = score_prospect(
            prospect,
            draft_capital,
            skip_sagarin=skip_sagarin,
            position_weights_override=position_weights_override,
            _exclude_capital=True,
            _active_season_year=_active_season_year,
        )
        talent_score = _talent["prospect_score"]
        capital_delta = round(prospect_score - talent_score, 2)

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
    # v3.4: in-season confidence reflects the thinner evidence. The
    # partial season's sample discount is 20 points at the limit, scaled
    # by the same min(1, GP/12) idiom the model uses elsewhere: a 6-game
    # latest season costs 10 points, an 11-game season costs about 2.
    if _inseason:
        _gp_conf = _safe(_gp_mode, 12)
        confidence_score = round(_clip(
            confidence_score
            - round(20.0 * (1.0 - min(1.0, _gp_conf / 12.0)))), 1)

    fantasy_translation = round(
        POSITION_FANTASY_MULT.get(pos, 1.0) * prospect_score, 2
    )

    # v3.2 archetype: scored NFL-outlook label from the isolated archetype
    # module (v2 rule set). Label only; it never feeds back into the grade.
    try:
        from data_building.rookie_pipeline.prospect_archetypes import classify_archetype
        _arch_profile = {
            "weight_lbs": ath.get("weight_lbs") or prospect.get("weight_lbs"),
            "forty_yard": ath.get("forty_yard"),
            "projected_pick": (draft_capital or {}).get("projected_pick"),
            "age": age,
        }
        _archetype = classify_archetype(
            pos, seasons, production_score=production_score,
            profile=_arch_profile)
    except Exception:
        _archetype = {"archetype": None, "archetype_receipt": None,
                      "archetype_confidence": None}

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
        "athleticism_score":            None if athleticism_score is None else round(athleticism_score, 2),
        "athleticism_projected":        _athleticism_projected,
        "competition_score":            round(competition_score, 2),
        "environment_adjustment":       None if environment_score is None else round(environment_score, 2),
        "durability_score":             None if durability_score is None else round(durability_score, 2),
        "recruiting_score":             round(recruiting_score, 2),
        "model_version":                MODEL_VERSION,
        "in_season":                    _inseason,
        "grading_mode":                 "in_season" if _inseason else "complete",
        "projected_draft_capital_score":None if _capital_missing else round(dc_score_adjusted, 2),
        "talent_score":                 talent_score,
        "capital_delta":                capital_delta,
        "archetype":                    _archetype.get("archetype"),
        "archetype_receipt":            _archetype.get("archetype_receipt"),
        "archetype_confidence":         _archetype.get("archetype_confidence"),
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
        bullets.append("Elite production profile - dominant volume for their position")
    elif prod >= 55:
        bullets.append("Solid production numbers with room to grow at the NFL level")
    else:
        bullets.append("Limited production volume - may need time to develop")

    # ── Utilization ───────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        tpg = (_safe(ls.get("targets")) or _safe(ls.get("receptions"))) / gp
        if util >= 70:
            bullets.append(f"High target share ({tpg:.1f} tgts/game) - clear featured receiver")
        elif util <= 35 and seasons:
            bullets.append(f"Limited target volume ({tpg:.1f} tgts/game) - role player or committee")
    elif pos == "RB":
        carries = _safe(ls.get("rush_attempts")) / gp
        if util >= 70:
            bullets.append(f"Workhorse RB usage ({carries:.1f} carries/game)")
        elif util <= 35:
            bullets.append(f"Limited rush volume ({carries:.1f} carries/game) - situational role")
    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        if util >= 70:
            bullets.append(f"High-volume passing attack ({att_pg:.0f} att/game)")

    # ── Efficiency ────────────────────────────────────────────────────────────
    if eff >= 75:
        bullets.append("High efficiency metrics stand out")
    elif eff <= 35:
        bullets.append("Below-average efficiency - volume stats may overstate true impact")

    # ── Advanced metrics ──────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        adv: List[str] = []

        ccr = ls.get("contested_catch_rate")
        if ccr is not None:
            pct = float(ccr)
            if pct >= 80:
                adv.append(f"{pct:.0f}% contested catch rate - elite ball-winner in traffic")
            elif pct >= 65:
                adv.append(f"{pct:.0f}% contested catch rate - reliable in contested situations")
            elif pct < 40:
                adv.append(f"{pct:.0f}% contested catch rate - struggles in jump-ball situations")

        # drop_rate bullets removed: was PFF-only, no free replacement.

        yac = ls.get("yards_after_catch_per_reception")
        if yac is not None:
            yac = float(yac)
            thresh     = 5.5 if pos == "WR" else 4.0
            low_thresh = 2.5 if pos == "WR" else 1.8
            if yac >= thresh:
                adv.append(f"{yac:.1f} YAC/reception - dynamic after the catch")
            elif yac <= low_thresh:
                adv.append(f"{yac:.1f} YAC/reception - limited after-catch production")

        adot = ls.get("avg_depth_of_target")
        if adot is not None and pos == "WR":
            adot = float(adot)
            if adot >= 14.0:
                adv.append(f"{adot:.1f}-yd avg depth of target - true deep threat")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT - short-area route specialist")

        # grades_offense bullets removed: was PFF-only, no free replacement.

        if pos == "WR":
            sr = ls.get("slot_rate")
            if sr is not None and float(sr) >= 0.65:
                adv.append(f"{float(sr):.0f}% slot rate - primary slot receiver")

        if pos == "TE":
            ir = ls.get("inline_rate")
            if ir is not None:
                ir = float(ir)
                if ir >= 0.60:
                    adv.append(f"{ir:.0f}% inline rate - traditional in-line TE")
                elif ir <= 0.20:
                    adv.append(f"{ir:.0f}% inline rate - move TE / receives in space")

        bullets.extend(adv[:3])

    elif pos == "RB":
        adv = []

        # grades_offense bullets removed: was PFF-only, no free replacement.
        # elusive_rating bullets removed: was PFF-only, no free replacement.

        bp = ls.get("breakaway_percentage")
        if bp is not None:
            bpct = float(bp)
            if bpct >= 18.0:
                adv.append(f"{bpct:.0f}% breakaway run rate - consistent big-play threat")
            elif bpct >= 12.0:
                adv.append(f"{bpct:.0f}% breakaway run rate - shows explosive burst")

        bullets.extend(adv[:3])

    elif pos == "QB":
        adv = []

        # pff_passing_grade is now sourced from CFBD PPA; reword bullet accordingly.
        pff_pass = ls.get("pff_passing_grade")
        if pff_pass is not None:
            pff_pass = float(pff_pass)
            if pff_pass >= 85.0:
                adv.append(f"Passing efficiency grade {pff_pass:.1f} - elite passer grade")
            elif pff_pass >= 75.0:
                adv.append(f"Passing efficiency grade {pff_pass:.1f} - above-average")

        # adjusted_completion_rate bullets removed: was PFF-only, no free replacement.
        # big_time_throw_rate bullets removed: was PFF-only, no free replacement.

        adot = ls.get("avg_depth_of_target")
        if adot is not None:
            adot = float(adot)
            if adot >= 10.0:
                adv.append(f"{adot:.1f}-yd avg depth of target - attacks full field vertically")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT - relies heavily on short/underneath game")

        bullets.extend(adv[:3])

    # ── Dominator rating ──────────────────────────────────────────────────────
    dom = _safe(ls.get("dominator_rating"))
    if dom >= 0.35 and pos in ("WR", "RB"):
        bullets.append(f"Team dominator rating {dom:.0%} - commanded an outsized share of team production")
    elif dom >= 0.20 and pos == "TE":
        bullets.append(f"Strong target share for a TE - {dom:.0%} team dominator")

    # ── Red zone proxy ────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        total_yds = _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("receiving_tds"))
        if total_yds >= 200:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 7.0:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) - dangerous in red zone")
    elif pos == "RB":
        total_yds = _safe(ls.get("rush_yards")) + _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("rush_tds"))   + _safe(ls.get("receiving_tds"))
        if total_yds >= 300:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 5.5:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) - goal-line threat")

    # ── Age / adjusted breakout age ───────────────────────────────────────────
    if age_sc >= 80:
        bullets.append(f"Exceptional age-adjusted production - elite output very young ({age:.1f} yrs)")
    elif age_sc < 40 and age is not None:
        bullets.append(f"Age concern: {age:.1f} yrs is older than typical for this position")

    if break_sc >= 75:
        bullets.append("Clear breakout trajectory - early-career dominance at a young age")
    elif break_sc >= 60:
        bullets.append("Solid breakout profile - consistent improvement season-over-season")

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
    elif ath_sc is not None and ath_sc <= 35 and ath:
        bullets.append("Below-average athleticism profile - will need to win on scheme/technique")

    # ── Competition ───────────────────────────────────────────────────────────
    conf = ls.get("conference", "")
    if comp_sc >= 70:
        bullets.append(f"Production came against quality competition")
    elif comp_sc <= 40:
        bullets.append(f"Played at lower competition level ({conf}) - discount applied")

    # ── Transfer history ──────────────────────────────────────────────────────
    if prospect.get("transfer_history"):
        th = prospect["transfer_history"]
        if isinstance(th, list) and len(th) >= 1:
            bullets.append(f"Transfer: {' → '.join(th)} - context matters for production evaluation")
        elif isinstance(th, str):
            bullets.append(f"Transfer history: {th}")

    # ── Draft capital ─────────────────────────────────────────────────────────
    if dc_dict:
        pick  = dc_dict.get("projected_pick")
        rnd   = dc_dict.get("projected_round")
        n_mocks = dc_dict.get("num_mocks_used", 0)
        conf_pct = dc_dict.get("consensus_confidence", 0)
        is_actual = dc_dict.get("is_actual_pick", False)
        
        if pick and rnd:
            if is_actual:
                # Draft is complete - show actual pick
                bullets.append(f"Drafted: Pick #{pick} (Round {rnd})")
            else:
                # Draft not complete - show mock draft projection
                conf_str = f", {conf_pct:.0f}% consensus" if n_mocks >= 3 else ""
                bullets.append(
                    f"Projected pick #{pick} (Round {rnd}) across {n_mocks} mock drafts"
                    f"{conf_str}"
                )

    # ── Durability ────────────────────────────────────────────────────────────
    if dur_sc is not None and dur_sc <= 40:
        recent_gp = _safe(ls.get("games_played"))
        if recent_gp and recent_gp < 10:
            bullets.append(f"Durability concern: only {int(recent_gp)} games in most recent season")
        else:
            bullets.append("Durability concern: missed significant games in college career")

    # ── Early declare ─────────────────────────────────────────────────────────
    if prospect.get("early_declare"):
        bullets.append("Early declarant - chose to enter draft before exhausting eligibility")

    return "\n".join(f"• {b}" for b in bullets)


# ─────────────────────────────────────────────────────────────────────────────
# Batch scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_all_prospects(
    prospects: List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
    active_season_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Score a list of prospects and add overall_rank / position_rank.

    Args:
        prospects:          List of normalised prospect dicts
        consensus_map:      {player_id: consensus_dict}
        active_season_year: The college season in progress on the run
            date (see active_season_year_for_run), enabling v3.4
            in-season mode per player. None (the default, and what the
            backtest passes) grades every profile in COMPLETE mode.

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
                _active_season_year=active_season_year,
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

    # Calculate tiers based on prospect_score
    for s in scores:
        score = s["prospect_score"]
        if score >= 85:
            s["tier"] = 1
            s["tier_label"] = "Elite Prospect"
        elif score >= 72:
            s["tier"] = 2
            s["tier_label"] = "Top Prospect"
        elif score >= 60:
            s["tier"] = 3
            s["tier_label"] = "Day-2 Upside"
        elif score >= 44:
            s["tier"] = 4
            s["tier_label"] = "Developmental"
        elif score >= 33:
            s["tier"] = 5
            s["tier_label"] = "Deep Flier"
        else:
            s["tier"] = 6
            s["tier_label"] = "Low Priority"

    return scores
