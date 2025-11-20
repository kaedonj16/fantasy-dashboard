from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math


# ---------- 1. Data model ----------

@dataclass
class PlayerFeatures:
    """
    All the inputs your model can use for valuing a player.
    Most of these can be None; the model will degrade gracefully.
    """
    player_id: str
    name: str
    position: str  # "QB", "RB", "WR", "TE"
    age: Optional[float]

    team: Optional[str]  # "KC", "DAL", etc.
    projected_points: float  # season-long or ROS projection in your scoring

    # Usage / utilization metrics (all as 0.0–1.0 fractions when possible)
    snap_share: Optional[float] = None         # % of team offensive snaps
    target_share: Optional[float] = None       # % of team pass attempts
    carry_share: Optional[float] = None        # % of team rush attempts
    route_participation: Optional[float] = None  # % of team dropbacks where player ran a route
    redzone_share: Optional[float] = None      # share of team RZ opportunities

    # Team-level context
    team_offense_tier: Optional[int] = None    # e.g. 1 = elite, 2 = good, 3 = mid, 4 = bad
    team_pass_rate: Optional[float] = None     # dropback rate (0–1)
    team_play_volume: Optional[float] = None   # plays per game

    # Risk flags
    injury_risk_flag: Optional[bool] = None
    suspension_risk_flag: Optional[bool] = None


# ---------- 2. Config / weights ----------

# How important each component is for each position.
# Values per position should roughly sum to 1.0.
POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "QB": {"prod": 0.55, "usage": 0.10, "age": 0.15, "situation": 0.15, "risk": 0.05},
    "RB": {"prod": 0.45, "usage": 0.25, "age": 0.15, "situation": 0.10, "risk": 0.05},
    "WR": {"prod": 0.45, "usage": 0.25, "age": 0.15, "situation": 0.10, "risk": 0.05},
    "TE": {"prod": 0.40, "usage": 0.30, "age": 0.15, "situation": 0.10, "risk": 0.05},
}

# Dynasty age curve per position (very tweakable)
AGE_BANDS: Dict[str, List[Tuple[float, float]]] = {
    # (max_age_inclusive, multiplier_on_0_to_1_scale)
    "RB": [(23, 1.00), (25, 0.95), (27, 0.90), (29, 0.80), (99, 0.65)],
    "WR": [(23, 0.95), (26, 1.00), (29, 0.95), (31, 0.85), (99, 0.75)],
    "TE": [(24, 0.90), (27, 1.00), (30, 0.95), (33, 0.85), (99, 0.75)],
    "QB": [(25, 0.90), (29, 1.00), (33, 0.95), (37, 0.85), (99, 0.70)],
}

# How much a "good" team helps, vs a "bad" one, on a 0–1 scale.
TEAM_TIER_SCORES: Dict[int, float] = {
    1: 1.00,  # elite offense
    2: 0.90,  # good
    3: 0.80,  # meh
    4: 0.70,  # bad
}

# Global score scaling so numbers feel like typical "trade values"
BASE_SCALE_BY_POSITION: Dict[str, float] = {
    "QB": 110.0,
    "RB": 120.0,
    "WR": 115.0,
    "TE": 100.0,
}


# ---------- 3. Helpers ----------

def _safe(value: Optional[float], default: float = 0.0) -> float:
    return default if value is None else float(value)


def _normalize(value: float, low: float, high: float) -> float:
    """
    Normalize value in [low, high] to [0, 1].
    If high == low, returns 0.5 (neutral).
    Clamps to [0, 1].
    """
    if high <= low:
        return 0.5
    x = (value - low) / (high - low)
    return max(0.0, min(1.0, x))


def _age_score(features: PlayerFeatures) -> float:
    if features.age is None:
        return 0.8  # unknown age, slightly discounted
    bands = AGE_BANDS.get(features.position, AGE_BANDS["WR"])
    for max_age, mult in bands:
        if features.age <= max_age:
            return mult
    return bands[-1][1]


def _usage_score(features: PlayerFeatures) -> float:
    """
    Build a composite usage score in 0–1 range based on snap/target/carry/etc.
    Assumes inputs are already 0–1 fractions.
    """
    # Different weights by position
    pos = features.position
    snap = _safe(features.snap_share)
    routes = _safe(features.route_participation)
    targets = _safe(features.target_share)
    carries = _safe(features.carry_share)
    rz = _safe(features.redzone_share)

    if pos == "QB":
        # For QBs, snap share is usually ~1; usage is mostly "are they the starter"
        # so we bump them if they are full-time.
        return _normalize(snap, 0.5, 1.0)

    if pos == "RB":
        # Workhorse RB gets a big bump for carries + snaps + RZ.
        raw = (0.35 * snap +
               0.30 * carries +
               0.20 * rz +
               0.15 * targets)
    elif pos == "WR":
        raw = (0.30 * snap +
               0.40 * routes +
               0.25 * targets +
               0.05 * rz)
    elif pos == "TE":
        raw = (0.30 * snap +
               0.40 * routes +
               0.25 * targets +
               0.05 * rz)
    else:
        raw = snap  # fallback

    # raw should already be roughly between 0 and 1, but clamp anyway
    return max(0.0, min(1.0, raw))


def _situation_score(features: PlayerFeatures) -> float:
    """
    Team context: offense tier + pass rate + play volume.
    Returns 0–1.
    """
    tier_score = TEAM_TIER_SCORES.get(features.team_offense_tier or 3, 0.80)
    pass_rate = _safe(features.team_pass_rate, 0.55)     # typical league-average-ish
    play_volume = _safe(features.team_play_volume, 62)   # plays per game

    # Normalize pass_rate and play_volume to 0–1
    pass_component = _normalize(pass_rate, 0.50, 0.65)
    play_component = _normalize(play_volume, 58, 70)

    # Make tier_score itself roughly 0.7–1.0 already, so just blend:
    return 0.5 * tier_score + 0.25 * pass_component + 0.25 * play_component


def _production_scores_by_position(players: List[PlayerFeatures]) -> Dict[str, Tuple[float, float]]:
    """
    Pre-compute min/max projected points by position to normalize production.
    Returns: {position: (min_points, max_points)}
    """
    by_pos: Dict[str, List[float]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p.projected_points)

    result: Dict[str, Tuple[float, float]] = {}
    for pos, pts in by_pos.items():
        if not pts:
            continue
        low = min(pts)
        high = max(pts)
        # Slightly shrink the range to reduce outlier effects
        if len(pts) > 5:
            # ignore one extreme on each side
            sorted_pts = sorted(pts)
            low = sorted_pts[1]
            high = sorted_pts[-2]
        result[pos] = (low, high)
    return result


def _production_score(features: PlayerFeatures,
                      prod_range_by_pos: Dict[str, Tuple[float, float]]) -> float:
    """
    Normalize projected_points for this player to 0–1 for their position.
    """
    pos = features.position
    low, high = prod_range_by_pos.get(pos, (0.0, max(1.0, features.projected_points)))
    return _normalize(features.projected_points, low, high)


def _risk_score(features: PlayerFeatures) -> float:
    """
    1.0 = no risk, 0.7 = substantial risk.
    You can make this more sophisticated later (games missed, etc.).
    """
    penalty = 0.0
    if features.injury_risk_flag:
        penalty += 0.15
    if features.suspension_risk_flag:
        penalty += 0.10
    return max(0.7, 1.0 - penalty)


# ---------- 4. Public API ----------

def compute_player_value(
    features: PlayerFeatures,
    prod_range_by_pos: Dict[str, Tuple[float, float]],
    *,
    custom_position_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Compute a dynasty-style trade value for a single player.

    - features: PlayerFeatures for that player
    - prod_range_by_pos: precomputed production min/max for normalization
    - custom_position_weights: optional override of POSITION_WEIGHTS

    Returns a numeric value (e.g., 0–150) suitable for trade comparisons.
    """
    pos = features.position
    pos_weights = (custom_position_weights or POSITION_WEIGHTS).get(pos, POSITION_WEIGHTS["WR"])
    scale = BASE_SCALE_BY_POSITION.get(pos, 110.0)

    prod_component = _production_score(features, prod_range_by_pos)   # 0–1
    usage_component = _usage_score(features)                          # 0–1
    age_component = _age_score(features)                              # ~0.7–1.0
    situation_component = _situation_score(features)                  # 0–1
    risk_component = _risk_score(features)                            # 0.7–1.0

    # Weighted linear combo
    score_0_1 = (
        pos_weights["prod"] * prod_component +
        pos_weights["usage"] * usage_component +
        pos_weights["age"] * age_component +
        pos_weights["situation"] * situation_component +
        pos_weights["risk"] * risk_component
    )

    # Clamp just in case and scale to "trade value" range
    score_0_1 = max(0.0, min(1.0, score_0_1))
    return score_0_1 * scale


def build_value_table(players: List[PlayerFeatures]) -> Dict[str, float]:
    """
    Given a list of PlayerFeatures (for your whole player pool or league),
    return a {player_id: value} mapping.
    """
    prod_range_by_pos = _production_scores_by_position(players)
    table: Dict[str, float] = {}
    for p in players:
        table[p.player_id] = compute_player_value(p, prod_range_by_pos)
    return table
