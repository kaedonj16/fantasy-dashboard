"""
Value translation layer.

Maps prospect_score (0-100) to dynasty dollar values that sit naturally
beside veterans and picks on the existing 0-999 valuation scale.

Calibration anchors (1QB, 10-team, PPR):
    prospect_score  dynasty_value   comparable
    ──────────────  ─────────────   ──────────────────────────────────
    ≥ 90            550 – 680       Elite #1 overall pick (rare)
    80 – 89         400 – 549       Mid-lottery skill prospect
    70 – 79         270 – 399       Late-1st / early-2nd pick
    60 – 69         160 – 269       Day-2 upside play
    50 – 59         85  – 159       Day-2 depth / raw prospect
    40 – 49         40  – 84        Day-3 developmental
    < 40             10 – 39        UDFA / late flier

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


def _piecewise_value(score: float) -> float:
    """Interpolate prospect_score → dynasty value using the calibrated curve."""
    score = max(0.0, min(100.0, score))
    for i in range(1, len(_CURVE)):
        s0, v0 = _CURVE[i - 1]
        s1, v1 = _CURVE[i]
        if score <= s1:
            frac = (score - s0) / (s1 - s0) if s1 > s0 else 0
            return v0 + frac * (v1 - v0)
    return _CURVE[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# Position adjustments (1QB vs SF)
# ─────────────────────────────────────────────────────────────────────────────

# Additive adjustments on top of base value
_POS_ADJ_1QB: Dict[str, float] = {
    "QB": -30,   # QBs have lower 1QB dynasty ceiling pre-draft
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

# Scale multipliers by league size
_SIZE_MULT: Dict[int, float] = {
    8:  0.85,
    10: 1.00,
    12: 1.08,
    14: 1.15,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tiers
# ─────────────────────────────────────────────────────────────────────────────

def assign_tier(prospect_score: float) -> Tuple[int, str]:
    """
    Return (tier_number, tier_label) based on overall prospect_score.

    Tiers:
        1 — Elite Prospect      (≥ 85)
        2 — Top Prospect        (75–84)
        3 — Day-1 Pick          (65–74)
        4 — Day-2 Upside        (55–64)
        5 — Developmental       (42–54)
        6 — Late Flier          (< 42)
    """
    if prospect_score >= 85:
        return 1, "Elite Prospect"
    if prospect_score >= 75:
        return 2, "Top Prospect"
    if prospect_score >= 65:
        return 3, "Day-1 Pick"
    if prospect_score >= 55:
        return 4, "Day-2 Upside"
    if prospect_score >= 42:
        return 5, "Developmental"
    return 6, "Late Flier"


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

    # League-size variants
    def _sized(base_v: float, size: int) -> float:
        return max(5, round(base_v * _SIZE_MULT.get(size, 1.0), 1))

    tier_num, tier_label = assign_tier(ps)

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
        "rookie_value":           v1qb,
        "rookie_sf_value":        vsf,
        "rookie_value_8":         _sized(v1qb, 8),
        "rookie_value_12":        _sized(v1qb, 12),
        "rookie_value_14":        _sized(v1qb, 14),
        "rookie_sf_value_8":      _sized(vsf, 8),
        "rookie_sf_value_12":     _sized(vsf, 12),
        "rookie_sf_value_14":     _sized(vsf, 14),
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
