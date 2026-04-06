"""
Prospect evaluation model — position-aware, multi-factor scoring.

Each component score is 0-100.  Final prospect_score is a weighted sum.

Component weights:
    production_score             22 %   core stat volume relative to peers
    efficiency_score             15 %   per-attempt / per-target efficiency
    age_score                    12 %   age-adjusted production; youth premium
    breakout_profile_score       10 %   early-career dominance trajectory
    athleticism_score            10 %   combine / speed score / RAS
    competition_score             8 %   conference + opponent quality
    environment_adjustment        5 %   team scheme / usage context
    durability_score              5 %   games missed, injury history
    projected_draft_capital_score 13 %   from mock_draft_consensus; important
                                         supplement but not the sole driver
    ──────────────────────────────────
    Total                       100 %

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
    "SEC":               1.00,
    "Big Ten":           0.96,
    "Big 12":            0.88,
    "ACC":               0.85,
    "Pac-12":            0.83,
    "Mountain West":     0.73,   # was 0.60 — legitimate mid-major that produces NFL talent
    "American":          0.68,   # was 0.58
    "Sun Belt":          0.62,   # was 0.52
    "MAC":               0.58,   # was 0.50
    "CUSA":              0.54,   # was 0.48
    "FBS Independents":  0.75,   # Notre Dame / BYU tier
}

DEFAULT_CONF_QUALITY = 0.62     # was 0.55


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

def calc_production_score(seasons: List[Dict], position: str) -> float:
    """
    Per-game production vs position-specific elite thresholds.
    Uses the best single season to capture peak value.
    """
    if not seasons:
        return 40.0  # neutral when no data

    pos = position.upper()
    ls  = _latest_season(seasons) or {}
    gp  = max(_safe(ls.get("games_played"), 12), 1)

    if pos == "WR":
        rec_yds_pg = _safe(ls.get("receiving_yards")) / gp
        rec_tds_pg = _safe(ls.get("receiving_tds"))   / gp
        dom        = _safe(ls.get("dominator_rating"))
        # Elite thresholds: ~90 rec-yds/g, 0.7 td/g, 0.30+ dominator
        prod = (
            _scale(rec_yds_pg, 40,  120) * 0.45 +
            _scale(rec_tds_pg, 0.3, 1.0) * 0.30 +
            _scale(dom,        0.10, 0.45) * 0.25
        )

    elif pos == "RB":
        rush_yds_pg = _safe(ls.get("rush_yards"))     / gp
        all_yds_pg  = (
            _safe(ls.get("rush_yards")) + _safe(ls.get("receiving_yards"))
        ) / gp
        tds_pg      = (
            _safe(ls.get("rush_tds")) + _safe(ls.get("receiving_tds"))
        ) / gp
        dom         = _safe(ls.get("dominator_rating"))
        prod = (
            _scale(rush_yds_pg, 40,  160) * 0.35 +
            _scale(all_yds_pg,  50,  180) * 0.25 +
            _scale(tds_pg,      0.5,  2.0) * 0.25 +
            _scale(dom,         0.15, 0.70) * 0.15
        )

    elif pos == "QB":
        pass_yds_pg = _safe(ls.get("pass_yards")) / gp
        tds_pg      = _safe(ls.get("pass_tds"))   / gp
        comp_pct    = _safe(ls.get("completion_pct"), 60.0)
        ypa         = _safe(ls.get("yds_per_attempt"), 7.0)
        td_int      = _safe(ls.get("td_int_ratio"),    2.0)
        prod = (
            _scale(pass_yds_pg, 180, 380) * 0.30 +
            _scale(tds_pg,        1.5,  3.5) * 0.25 +
            _scale(comp_pct,     60.0, 76.0) * 0.20 +
            _scale(ypa,           6.5,  10.5) * 0.15 +
            _scale(td_int,        1.5,   6.0) * 0.10
        )

    elif pos == "TE":
        rec_yds_pg = _safe(ls.get("receiving_yards")) / gp
        rec_tds_pg = _safe(ls.get("receiving_tds"))   / gp
        dom        = _safe(ls.get("dominator_rating"))
        rec_pg     = _safe(ls.get("receptions"))       / gp
        prod = (
            _scale(rec_yds_pg, 30,  95)  * 0.40 +
            _scale(rec_tds_pg, 0.2, 0.8) * 0.30 +
            _scale(dom,        0.08, 0.30) * 0.15 +
            _scale(rec_pg,     2.0,  7.0) * 0.15
        )

    else:
        return 40.0

    return _clip(prod)


def calc_efficiency_score(seasons: List[Dict], position: str) -> float:
    """
    Per-attempt / per-target efficiency.  Rewards quality over quantity.
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
        ypr   = _safe(ls.get("yds_per_reception"), 9.0)
        ms    = _safe(ls.get("market_share_yards"))
        eff   = _scale(ypr, 8.0, 16.0) * 0.65 + _scale(ms, 0.05, 0.30) * 0.35

    else:
        return 45.0

    return _clip(eff)


# Typical draft-class age by position (age at start of NFL rookie year).
# Updated to reflect modern college football (COVID year, grad transfers, etc.)
_TYPICAL_AGE = {"QB": 23.5, "RB": 22.0, "WR": 22.5, "TE": 23.0}
_AGE_ELITE   = {"QB": 22.0, "RB": 20.5, "WR": 21.0, "TE": 21.5}
_AGE_WORST   = {"QB": 26.5, "RB": 25.0, "WR": 25.5, "TE": 26.0}   # widened — old ranges were too tight


def calc_age_score(age: Optional[float], draft_year: int, position: str) -> float:
    """
    Younger prospects earn a premium.  Age is evaluated relative to draft class.
    A 20-year-old RB producing at a high level is worth more than a 23-year-old.
    """
    if age is None:
        return 50.0  # neutral default
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

    if pos == "QB" or thresh is None:
        dom_score = 50.0  # dominator_rating not applicable for QBs
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

    # Youth at breakout
    youth_bonus = 0.0
    if age is not None:
        # Exceptional: playing dominant college ball at ≤20 (≤21 for QB)
        young_thresh = 21 if pos == "QB" else 20
        if age <= young_thresh:
            youth_bonus = 15.0
        elif age <= young_thresh + 1:
            youth_bonus = 7.0

    score = dom_score * 0.50 + traj_score * 0.35 + youth_bonus
    return _clip(score)


def calc_athleticism_score(athleticism: Dict[str, Any], position: str) -> float:
    """
    Combine / pro-day metrics.  Falls back gracefully to positional median (55)
    when data is missing.

    RAS (Relative Athletic Score) is the most reliable single signal (0-10).
    We also compute a speed score = weight * (40-time^4)^-1 (normalized).
    """
    if not athleticism:
        return 55.0

    pos = position.upper()
    scores: List[float] = []

    # RAS (0-10 → scale to 0-100)
    ras = athleticism.get("ras_score")
    if ras is not None:
        scores.append(_scale(_safe(ras), 4.0, 10.0))

    # 40-yard dash (position-specific thresholds)
    forty = athleticism.get("forty_yard")
    if forty:
        forty = _safe(forty)
        thresholds = {
            "WR":  (4.25, 4.65),
            "RB":  (4.30, 4.65),
            "QB":  (4.40, 5.00),
            "TE":  (4.45, 4.90),
        }
        lo, hi = thresholds.get(pos, (4.30, 4.90))
        # Invert: faster (lower) → higher score
        scores.append(_scale(hi - forty, hi - hi, hi - lo))

    # Vertical jump
    vert = athleticism.get("vertical_inches")
    if vert:
        scores.append(_scale(_safe(vert), 28.0, 44.0))

    # Broad jump
    broad = athleticism.get("broad_jump_in")
    if broad:
        scores.append(_scale(_safe(broad), 100, 140))

    # Speed score = weight * (40^4)^-1 * normalisation constant
    forty = athleticism.get("forty_yard")
    weight = athleticism.get("weight_lbs")
    if forty and weight:
        ss_raw = (_safe(weight) * 200) / (_safe(forty) ** 4)
        # Elite ~115+, average ~100
        scores.append(_scale(ss_raw, 80.0, 130.0))

    if not scores:
        return 55.0

    return _clip(sum(scores) / len(scores))


def calc_competition_score(seasons: List[Dict]) -> float:
    """
    Conference quality + implied opponent strength.
    Uses the best-conference season to avoid penalising transfers.
    """
    if not seasons:
        return 55.0
    best_quality = max(_conf_quality(s.get("conference")) for s in seasons)
    return _clip(_scale(best_quality, 0.45, 1.00))


def calc_environment_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Adjusts for team usage patterns.

    - High pass rate for skill players (WR/TE) is better context (more targets)
    - For RBs, high rush rate inflates volume — slight discount applied
    - Bonus for high team_total_yards (pass-heavy offence enables big numbers)
    """
    if not seasons:
        return 50.0

    pos = position.upper()
    ls  = _latest_season(seasons) or {}
    pass_rate = _safe(ls.get("team_pass_rate"), 0.52)

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
    Games played vs expected.  12-game seasons are typical; 14+ is excellent.
    Reward players who stayed healthy across multiple seasons.
    """
    if not seasons:
        return 60.0

    gp_list = [_safe(s.get("games_played")) for s in seasons if s.get("games_played") is not None]
    if not gp_list:
        return 60.0

    avg_gp  = sum(gp_list) / len(gp_list)
    min_gp  = min(gp_list)

    # Average games score (12 → ~75, 14 → 100, 8 → 35)
    avg_score = _scale(avg_gp, 7.0, 14.0)
    # Penalise if any season was very short
    floor_pen = max(0.0, (8.0 - min_gp) * 5.0)

    return _clip(avg_score - floor_pen)


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
    "production":      0.22,
    "efficiency":      0.15,
    "age":             0.12,
    "breakout":        0.10,
    "athleticism":     0.10,
    "competition":     0.08,
    "environment":     0.05,
    "durability":      0.05,
    "draft_capital":   0.13,
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
        dc_score = 40.0  # neutral fallback — no mocks yet

    prospect_score = (
        production_score  * WEIGHTS["production"]  +
        efficiency_score  * WEIGHTS["efficiency"]  +
        age_score         * WEIGHTS["age"]         +
        breakout_score    * WEIGHTS["breakout"]    +
        athleticism_score * WEIGHTS["athleticism"] +
        competition_score * WEIGHTS["competition"] +
        environment_score * WEIGHTS["environment"] +
        durability_score  * WEIGHTS["durability"]  +
        dc_score          * WEIGHTS["draft_capital"]
    )
    prospect_score = round(_clip(prospect_score), 2)

    # Confidence: based on data availability
    data_fields_present = sum([
        bool(seasons),
        age is not None,
        bool(ath),
        draft_capital is not None,
        bool(_latest_season(seasons) and _latest_season(seasons).get("dominator_rating")),
    ])
    confidence_score = round(_scale(data_fields_present, 0, 5), 1)

    fantasy_translation = round(
        POSITION_FANTASY_MULT.get(pos, 1.0) * prospect_score, 2
    )

    # Human-readable reasons
    reasons = _build_reasons(
        prospect, pos, seasons,
        production_score, efficiency_score, age_score,
        breakout_score, athleticism_score, competition_score,
        dc_score, draft_capital,
    )

    return {
        "player_id":                    prospect["player_id"],
        "draft_class_year":             dy,
        "production_score":             round(production_score, 2),
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
    prod, eff, age_sc, break_sc, ath_sc, comp_sc, dc_sc, dc_dict
) -> str:
    """Build a bullet-point string summarising the prospect's strengths/flags."""
    bullets: List[str] = []
    ls = _latest_season(seasons) or {}
    name = prospect.get("name", "Prospect")
    age  = prospect.get("age")

    if prod >= 75:
        bullets.append(f"Elite production profile — dominant volume for their position")
    elif prod >= 55:
        bullets.append(f"Solid production numbers with room to grow at the NFL level")
    else:
        bullets.append(f"Limited production volume — may need time to develop")

    if eff >= 75:
        bullets.append(f"High efficiency metrics (yards per touch, market share) stand out")

    dom = _safe(ls.get("dominator_rating"))
    if dom >= 0.35 and pos in ("WR", "RB"):
        bullets.append(f"Team dominator rating {dom:.0%} — commanded an outsized share of team production")
    elif dom >= 0.20 and pos == "TE":
        bullets.append(f"Strong target share for a TE — {dom:.0%} team dominator")

    if age_sc >= 80:
        bullets.append(f"Exceptional age-adjusted production — producing at high level very young ({age:.1f} yrs)")
    elif age_sc < 40:
        bullets.append(f"Age concern: {age:.1f} yrs is older than typical for this position")

    if break_sc >= 75:
        bullets.append(f"Clear breakout trajectory — production improved sharply in final season")

    ath_msg = ""
    ras = prospect.get("athleticism", {}).get("ras_score")
    forty = prospect.get("athleticism", {}).get("forty_yard")
    if ras and _safe(ras) >= 9.0:
        ath_msg = f"Elite athleticism (RAS {ras:.1f}/10)"
    elif forty and _safe(forty) <= 4.35 and pos in ("WR", "RB"):
        ath_msg = f"Elite speed ({forty}s 40-yard dash)"
    if ath_msg:
        bullets.append(ath_msg)

    conf = ls.get("conference", "")
    if comp_sc >= 70:
        bullets.append(f"Production came against quality competition ({conf})")
    elif comp_sc <= 40:
        bullets.append(f"Played at lower level of competition ({conf}) — production must be discounted")

    if dc_dict:
        pick  = dc_dict.get("projected_pick")
        rnd   = dc_dict.get("projected_round")
        n     = dc_dict.get("num_mocks_used", 0)
        if pick and rnd:
            bullets.append(
                f"Projected pick #{pick} (Round {rnd}) across {n} mock drafts — "
                f"draft capital score {dc_sc:.0f}/100"
            )

    if prospect.get("early_declare"):
        bullets.append("Early declarant — chose to enter draft before exhausting eligibility")

    if prospect.get("transfer_history"):
        bullets.append(f"Transfer history: {prospect['transfer_history']}")

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
