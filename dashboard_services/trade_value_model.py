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
    trade_value: float

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

def _z_score(x: Optional[float], mean: float, std: float) -> float:
    if x is None or std <= 0:
        return 0.0
    return (x - mean) / std


def _logistic(x: float) -> float:
    # squashes roughly -2..+2 into ~0.12..0.88
    return 1.0 / (1.0 + math.exp(-x))


def _age_score(features: "PlayerFeatures") -> float:
    """
    Light positional age curve. Not a giant buff, just realistic
    lifespan differences (WR > RB shelf life).
    """
    age = features.age
    pos = features.position

    if age is None:
        return 1.0

    # You can tweak these, but they're intentionally not huge
    if pos in ("WR", "TE"):
        if age <= 24:
            return 1.00
        elif age <= 27:
            return 0.95
        elif age <= 30:
            return 0.85
        else:
            return 0.75

    if pos == "RB":
        if age <= 23:
            return 1.00
        elif age <= 26:
            return 0.85
        else:
            return 0.70

    # QB / others
    if age <= 25:
        return 0.90
    elif age <= 30:
        return 1.00
    elif age <= 34:
        return 0.90
    else:
        return 0.75


def compute_player_value(
    features: "PlayerFeatures",
    pos_dists: Dict[str, PosDistribution],
    *,
    global_scale: float = 100.0,
) -> float:
    """
    Compute a dynasty trade value for a player based on:
    - Stats normalized vs *same-position* peers (proj, ppr_ppg, snap share)
    - Light age curve by position
    - No big manual positional buff

    Returns a 0–global_scale number.
    """
    pos = features.position
    dist = pos_dists.get(pos)
    if not dist:
        return 0.0

    proj = float(features.projected_points or 0.0)
    ppr_ppg = float(getattr(features, "ppr_ppg", 0.0) or 0.0)
    snap_share = float(features.snap_share or 0.0)

    # --- Z-scores within position ---
    proj_z = _z_score(proj, dist.proj_mean, dist.proj_std)
    ppr_z = _z_score(ppr_ppg, dist.ppr_mean, dist.ppr_std)
    snap_z = _z_score(snap_share, dist.snap_mean, dist.snap_std)

    # --- Combine into production/usage components ---
    # Production “AI-ish” combo: projected points + historical ppr
    prod_z = 0.6 * proj_z + 0.4 * ppr_z
    usage_z = snap_z

    prod_score = _logistic(prod_z)   # 0–1
    usage_score = _logistic(usage_z) # 0–1
    age_score = _age_score(features) # ~0.7–1.0

    # You can tune these weights:
    # - prod_score: how good they are when on the field
    # - usage_score: how locked-in their role is
    # - age_score: how safe their value is long-term
    score_0_1 = (
        0.55 * prod_score +
        0.30 * usage_score +
        0.15 * age_score
    )

    score_0_1 = max(0.0, min(1.0, score_0_1))
    return round(score_0_1 * global_scale, 2)



def build_value_table(
    players: List[PlayerFeatures],
    pos_dists: Dict[str, PosDistribution]
) -> Dict[str, float]:
    """
    Given a list of PlayerFeatures and the computed position distributions,
    return a {player_id: trade_value} mapping.
    """
    table: Dict[str, float] = {}

    for p in players:
        value = compute_player_value(p, pos_dists)
        table[p.player_id] = value

    return table

