"""
Value translation layer.

Maps prospect_score (0-100) to dynasty dollar values that sit naturally
beside veterans and picks on the existing 0-999 valuation scale.

Calibration anchors (1QB, 10-team, PPR):
    prospect_score  dynasty_value   comparable
    ──────────────  ─────────────   ──────────────────────────────────
    ≥ 80            400 – 680       Elite T1 (Bijan, Chase, J Love type)
    70 – 79         270 – 399       Strong T1 (Hunter, Waddle type)
    60 – 69         160 – 269       T2 solid day-1 pick
    50 – 59         85  – 159       T2/T3 day-2 upside
    40 – 49         40  – 84        T3/T4 developmental
    < 40             10 – 39        T5/T6 deep flier / UDFA

SF adjustments:
    - QBs  get +80 to +120 (scarcity + 2-QB starting roles)
    - RBs  get –10 to –15  (slight discount in pass-heavy SF)
    - WRs  stay flat / +5
    - TEs  get –10 to –20  (hard translation)

League size adjustments (like the existing value model):
    - 8-team:  multiply by 0.85 (shallower rosters, less dynasty premium)
    - 10-team: base (1.00)
    - 12-team: multiply by 1.08
    - 14-team: multiply by 1.15
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .prospect_model import POSITION_FANTASY_MULT, POSITION_FANTASY_MULT_SF


# ─────────────────────────────────────────────────────────────────────────────
# Core score → value curve (1QB, 10-team base)
# ─────────────────────────────────────────────────────────────────────────────

# Piecewise linear breakpoints: (score_threshold, value_at_threshold)
# Sorted ascending by score.  Values are interpolated between adjacent points.
_CURVE: List[Tuple[float, float]] = [
    (0,   5),
    (30,  20),
    (40,  42),
    (50,  90),
    (58,  155),
    (65,  230),
    (70,  280),
    (75,  340),
    (80,  410),
    (85,  490),
    (88,  540),
    (92,  610),
    (95,  665),
    (100, 700),
]


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation — produces an S-curve between 0 and 1."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0) if edge1 > edge0 else 0.0))
    return t * t * (3.0 - 2.0 * t)


def _piecewise_value(score: float) -> float:
    """Interpolate prospect_score → dynasty value using the calibrated curve.

    For scores ≥ 85 (elite tier) a smoothstep replaces linear interpolation
    so the value curve has a natural S-shape rather than a straight ramp —
    matching real dynasty market behaviour where elite prospects command
    exponentially higher prices up to ~95, then gains compress near 100.
    """
    score = max(0.0, min(100.0, score))

    # Elite tier: apply smoothstep blending between curve segments above 85
    ELITE_THRESHOLD = 85.0
    if score >= ELITE_THRESHOLD:
        # Find the piecewise value via linear interpolation first
        linear_val = _piecewise_linear(score)
        # Also compute what a pure smoothstep from 85→100 would give
        v85 = _piecewise_linear(ELITE_THRESHOLD)
        v100 = _CURVE[-1][1]  # 700
        smooth_val = v85 + (v100 - v85) * _smoothstep(ELITE_THRESHOLD, 100.0, score)
        # Blend: 40% linear (preserves calibration anchors) + 60% smooth
        return 0.40 * linear_val + 0.60 * smooth_val

    return _piecewise_linear(score)


def _piecewise_linear(score: float) -> float:
    """Raw piecewise linear interpolation (no smoothing)."""
    score = max(0.0, min(100.0, score))
    for i in range(1, len(_CURVE)):
        s0, v0 = _CURVE[i - 1]
        s1, v1 = _CURVE[i]
        if score <= s1:
            frac = (score - s0) / (s1 - s0) if s1 > s0 else 0.0
            return v0 + frac * (v1 - v0)
    return _CURVE[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# Position adjustments (1QB vs SF)
# ─────────────────────────────────────────────────────────────────────────────

# Additive adjustments on top of base value
_POS_ADJ_1QB: Dict[str, float] = {
    "QB": -20,   # QBs have lower 1QB dynasty ceiling; -20 keeps QB3/QB4 viable
    "RB":  0,
    "WR": +10,
    "TE": -15,
}
_POS_ADJ_SF: Dict[str, float] = {
    "QB": +90,   # SF QB premium is very real
    "RB": -10,
    "WR":  +5,
    "TE": -15,
}

# Scale multipliers for calibrated league sizes (anchors)
_SIZE_MULT_ANCHORS: List[Tuple[int, float]] = [
    (6,  0.78),
    (8,  0.85),
    (10, 1.00),
    (12, 1.08),
    (14, 1.15),
    (16, 1.20),
    (20, 1.28),
]


def _size_multiplier(league_size: int) -> float:
    """Return value multiplier for any league size via linear interpolation."""
    sizes   = [s for s, _ in _SIZE_MULT_ANCHORS]
    mults   = [m for _, m in _SIZE_MULT_ANCHORS]
    if league_size <= sizes[0]:
        return mults[0]
    if league_size >= sizes[-1]:
        return mults[-1]
    for i in range(1, len(sizes)):
        if league_size <= sizes[i]:
            frac = (league_size - sizes[i - 1]) / (sizes[i] - sizes[i - 1])
            return mults[i - 1] + frac * (mults[i] - mults[i - 1])
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tiers
# ─────────────────────────────────────────────────────────────────────────────

def assign_tier(prospect_score: float) -> Tuple[int, str]:
    """
    Return (tier_number, tier_label) based on overall prospect_score.

    Calibrated so elite skill players (Bijan, J Love, Chase) land Tier 1.
    Tiers are anchored against the full scoring model where elite prospects
    (elite stats + top-10 draft capital) typically score 88–96.

    Tiers:
        1 — Elite Prospect      (≥ 82)   Bijan/Chase/JLove tier; top-8 skill pick + elite stats
        2 — Top Prospect        (68–81)  solid round-1; QBs; early top-20 picks
        3 — Day-2 Upside        (55–67)  round-2 picks; strong developmental floor
        4 — Developmental       (44–54)  round-3/4; high variance, low floor
        5 — Deep Flier          (33–43)  day-3 / UDFA with one standout trait
        6 — Low Priority        (< 33)   minimal dynasty value
    """
    if prospect_score >= 82:
        return 1, "Elite Prospect"
    if prospect_score >= 68:
        return 2, "Top Prospect"
    if prospect_score >= 55:
        return 3, "Day-2 Upside"
    if prospect_score >= 44:
        return 4, "Developmental"
    if prospect_score >= 33:
        return 5, "Deep Flier"
    return 6, "Low Priority"


# ─────────────────────────────────────────────────────────────────────────────
# Draft capital display label
# ─────────────────────────────────────────────────────────────────────────────

def format_draft_capital(
    projected_round: Optional[int],
    projected_pick:  Optional[int],
    low: Optional[int] = None,
    high: Optional[int] = None,
) -> str:
    """Return a human-readable draft capital label, e.g. '1st (Pick 5-8)'."""
    if projected_round is None:
        return "Undrafted?"
    round_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
                    5: "5th", 6: "6th", 7: "7th"}
    rnd = round_labels.get(projected_round, f"{projected_round}th")
    if projected_pick:
        if low and high and low != high:
            return f"{rnd} (#{low}–#{high})"
        return f"{rnd} (#{projected_pick})"
    return rnd


# ─────────────────────────────────────────────────────────────────────────────
# Main translation function
# ─────────────────────────────────────────────────────────────────────────────

def translate_score_to_value(
    score_dict: Dict[str, Any],
    prospect:   Dict[str, Any],
    consensus:  Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Given a prospect score dict (from prospect_model.score_prospect) and the
    raw prospect bio, return a full value dict ready for DB insertion.

    Returns dict with:
        rookie_value, rookie_sf_value,
        rookie_value_8, rookie_value_12, rookie_value_14,
        rookie_sf_value_8, rookie_sf_value_12, rookie_sf_value_14,
        tier, tier_label,
        projected_draft_capital (display string),
        overall_rank, position_rank, prospect_score
    """
    pos     = (prospect.get("position") or "WR").upper()
    ps      = float(score_dict.get("prospect_score") or 0)

    # Base value from calibrated curve
    base = _piecewise_value(ps)

    # 1QB value
    v1qb = base + _POS_ADJ_1QB.get(pos, 0)
    v1qb = max(5, round(v1qb, 1))

    # SF value
    vsf  = base + _POS_ADJ_SF.get(pos, 0)
    vsf  = max(5, round(vsf, 1))

    # League-size variants (interpolated for any size)
    def _sized(base_v: float, size: int) -> float:
        return max(5, round(base_v * _size_multiplier(size), 1))

    tier_num, tier_label = assign_tier(ps)

    # Confidence-based discount: low-confidence prospects get a haircut on value.
    # If we barely have data on a player, their upside is speculative.
    confidence = float(score_dict.get("confidence_score") or 50.0)
    if confidence < 40:
        conf_discount = 0.80  # -20%: very limited data
    elif confidence < 60:
        conf_discount = 0.90  # -10%: partial data
    else:
        conf_discount = 1.00  # full value when data is solid

    v1qb = max(5, round(v1qb * conf_discount, 1))
    vsf  = max(5, round(vsf  * conf_discount, 1))

    # Draft capital display
    dc_label = "Unknown"
    if consensus:
        dc_label = format_draft_capital(
            consensus.get("projected_round"),
            consensus.get("projected_pick"),
            consensus.get("projected_pick_low"),
            consensus.get("projected_pick_high"),
        )

    return {
        "player_id":              prospect["player_id"],
        "draft_class_year":       prospect.get("draft_class_year"),
        "overall_rank":           score_dict.get("overall_rank"),
        "position_rank":          score_dict.get("position_rank"),
        "prospect_score":         ps,
        "confidence_score":       confidence,
        "rookie_value":           v1qb,
        "rookie_sf_value":        vsf,
        "rookie_value_8":         _sized(v1qb, 8),
        "rookie_value_12":        _sized(v1qb, 12),
        "rookie_value_14":        _sized(v1qb, 14),
        "rookie_value_16":        _sized(v1qb, 16),
        "rookie_sf_value_8":      _sized(vsf, 8),
        "rookie_sf_value_12":     _sized(vsf, 12),
        "rookie_sf_value_14":     _sized(vsf, 14),
        "rookie_sf_value_16":     _sized(vsf, 16),
        "tier":                   tier_num,
        "tier_label":             tier_label,
        "projected_draft_capital": dc_label,
    }


def translate_all(
    scores:        List[Dict[str, Any]],
    prospects:     List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Translate a full list of scored prospects to dynasty values."""
    if consensus_map is None:
        consensus_map = {}

    prospect_by_id = {p["player_id"]: p for p in prospects}
    results = []
    for s in scores:
        pid = s["player_id"]
        p   = prospect_by_id.get(pid, {"player_id": pid, "position": "WR"})
        dc  = consensus_map.get(pid)
        results.append(translate_score_to_value(s, p, dc))

    results.sort(key=lambda x: x.get("overall_rank") or 999)
    return results
