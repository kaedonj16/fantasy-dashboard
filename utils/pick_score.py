"""Pure draft pick-score computation.

Extracted from app.py so the pick-scoring engine can be unit-tested without the
pandas/DB stack. ``compute_pick_score`` blends DB dynasty value, ADP, tier,
positional need, youth, momentum and production into a 0-100 score, mirroring
the client-side Draft Room. Pure — reuses ``clamp01`` from utils.draft_grade.
"""
from __future__ import annotations

import math

from utils.draft_grade import clamp01

# Component weights per draft type (approximately normalized within each row).
PS_WEIGHTS = {
    "rookie":  {"vor": 0.06, "value": 0.18, "adp": 0.29, "tier": 0.12, "need": 0.05, "youth": 0.24, "mom": 0.06, "ppg": 0.05},
    "redraft": {"vor": 0.10, "value": 0.24, "adp": 0.33, "tier": 0.08, "need": 0.07, "youth": 0.00, "mom": 0.03, "ppg": 0.18},
    "startup": {"vor": 0.07, "value": 0.24, "adp": 0.30, "tier": 0.12, "need": 0.09, "youth": 0.10, "mom": 0.03, "ppg": 0.10},
}
PS_AGE_PEAKS = {"RB": 24, "WR": 27, "TE": 27, "QB": 29}


def ps_tier_of(value: float, thresholds: list):
    """Tier number for a dynasty value given gap-significance thresholds (1 = elite)."""
    if not thresholds:
        return None
    for i, thr in enumerate(thresholds):
        if value >= thr:
            return i + 1
    return len(thresholds) + 1


def starter_counts(counts: dict) -> dict:
    """Mirror of static/pick_score.js `starterCounts`; pinned by the parity test.
    Effective starters per position from roster slot counts (SF split half to QB,
    FLEX split half each to RB/WR), so the server's VOR/PPG replacement levels
    match the draft room's computeReplacement instead of a hardcoded guess."""
    c = counts or {}

    def n(k):
        try:
            return float(c.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "QB": n("QB") + n("SF") * 0.5,
        "RB": n("RB") + n("FLEX") * 0.5,
        "WR": n("WR") + n("FLEX") * 0.5,
        "TE": n("TE"),
    }


def compute_pick_score(*, pos, value, vor, tier, age, rank_change_7d,
                       avg_pick, pick_no, max_val, draft_type, is_sf,
                       need_raw, qb_count, total_picks=None, num_teams=None,
                       ppg_norm=None, ppr=1.0, tep=0.0, is_tier_cliff=False,
                       survival_adj=0.0, handcuff=False) -> int:
    """Mirror of static/pick_score.js `computePickScore`; the two are pinned
    identical by tests/test_pick_score_parity.py. ``survival_adj`` and
    ``handcuff`` are the live-draft-only timing terms (default off); the grade
    never passes them, which is why the Teams-page grade and the Draft Room
    grade agree. Do not edit this without editing the JS (and vice versa)."""
    pos = (pos or "").upper()
    # DB-sourced numbers arrive as decimal.Decimal; coerce to float so they mix
    # with the float weights below (Decimal * float raises TypeError).
    value = float(value) if value is not None else 0.0
    vor = float(vor) if vor is not None else None
    age = float(age) if age is not None else None
    rank_change_7d = float(rank_change_7d) if rank_change_7d is not None else None
    avg_pick = float(avg_pick) if avg_pick is not None else None
    max_val = float(max_val) if max_val is not None else 0.0
    need_raw = float(need_raw) if need_raw is not None else 0.0
    total_picks = float(total_picks) if total_picks is not None else 0.0
    db_value_norm = clamp01(value / max_val) if max_val and max_val > 0 else 0.0
    # Blend DB dynasty value with ADP-implied quality so market consensus
    # prevents DB value gaps (especially new rookies) from dragging the score
    # unfairly low when ADP says the player is a legitimate round-2/3 pick.
    if avg_pick is not None and total_picks > 0:
        adp_qual_norm = clamp01(1.0 - avg_pick / total_picks)
        value_norm = db_value_norm * 0.35 + adp_qual_norm * 0.65
    else:
        value_norm = db_value_norm
    vor_norm = clamp01(vor / max(max_val, 1)) if vor is not None else value_norm * 0.8

    # ADP component: proportional gap so a 2-pick fall from ADP 2 == a 10-pick
    # fall from ADP 20, with an elite-ADP floor for top-8 players.
    if avg_pick is not None:
        gap = pick_no - avg_pick
        rel = gap / max(avg_pick, 1.5)
        if rel >= 0.5:
            adp_val = 1.0
        elif rel >= -0.3:
            adp_val = 0.5 + rel
        else:
            adp_val = max(0.0, 0.2 + rel * 0.25)
        if avg_pick <= 8:
            adp_val = max(adp_val, clamp01(0.5 + (8 - avg_pick) / 16))
    else:
        adp_val = 0.5

    tier_score = clamp01((10 - min(tier, 9)) / 9) if tier else value_norm
    # Tier-cliff boost: position scarcity when this player's tier is drying up
    # (<=2 left in the bucket). Mirrors the Draft Room's isTierCliff() bump.
    if is_tier_cliff:
        tier_score = clamp01(tier_score + 0.15)

    need_ramp = clamp01((pick_no - 1) / 12.0)
    need = (1 - need_ramp) * 0.5 + need_ramp * need_raw

    youth = 0.5
    if age is not None and pos in ("RB", "WR", "TE", "QB"):
        peak = PS_AGE_PEAKS.get(pos, 27)
        youth = clamp01((peak - age + 4) / 8)

    mom = clamp01((rank_change_7d or 0) / 20 + 0.5)

    # Production: position-normalized PPG. Missing data falls back to value_norm
    # so a player isn't penalized for absent projections (mirrors the Draft Room).
    ppg_n = ppg_norm if ppg_norm is not None else value_norm

    w = PS_WEIGHTS.get(draft_type, PS_WEIGHTS["startup"])
    s = (w["vor"] * vor_norm + w["value"] * value_norm + w["adp"] * adp_val
         + w["tier"] * tier_score + w["need"] * need + w["youth"] * youth
         + w["mom"] * mom + w.get("ppg", 0.0) * ppg_n)

    # Live-draft survival/opportunity-cost term (0 for grading).
    if survival_adj:
        s += float(survival_adj)

    # QB overfill (1QB only): a second QB only carries real opportunity cost in
    # the early rounds. By the late rounds a backup QB is a normal pick, so the
    # penalty tapers out (mirrors the Draft Room).
    if not is_sf and pos == "QB" and qb_count >= 1:
        _teams = int(num_teams) if num_teams else 12
        _round = (int(pick_no) - 1) // max(_teams, 1) + 1 if pick_no else 1
        if _round <= 3:
            _pen = 0.30
        elif _round <= 6:
            _pen = 0.60
        elif _round <= 9:
            _pen = 0.85
        else:
            _pen = 1.0
        if qb_count >= 2:
            _pen *= 0.7
        s *= _pen

    # Live-draft redraft handcuff term (False for grading).
    if handcuff:
        s = min(1.0, s + 0.15)

    # Scoring-format adjustments: shift toward the build the league's scoring
    # rewards. Mirrors the Draft Room's scoringCfg() multipliers exactly.
    if tep and tep > 0 and pos == "TE":
        s *= (1 + 0.12 * tep)
    if pos in ("WR", "TE"):
        if ppr is not None and ppr >= 1:
            s *= 1.02
    elif pos == "RB" and ppr is not None and ppr <= 0:
        s *= 1.03

    # Depth normalization: re-anchor the 0-100 scale to what's achievable at this
    # pick slot so late-round picks aren't unfairly buried (mirrors the Draft Room).
    if total_picks and total_picks > 1 and pick_no:
        _depth = min(0.98, (float(pick_no) - 1) / float(total_picks))
        _par = max(0.40, 1.0 - _depth * 0.44)
        s = s / _par

    return int(math.floor(clamp01(s) * 100 + 0.5))  # round-half-up, matching JS
