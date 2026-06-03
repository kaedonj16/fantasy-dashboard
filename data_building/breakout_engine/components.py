"""
Component score calculators for the unified breakout engine.

Each function calculates one of the 7 component scores:
1. opportunity_opened_score
2. competition_removed_score
3. competition_added_penalty
4. team_environment_score
5. player_readiness_score
6. role_trajectory_score
7. confidence_score

All functions return (score: float, details: Dict) tuples.
"""

import math
from datetime import date, timedelta
from typing import List, Optional, Tuple

from .config import *
from .db_helpers import (
    get_vacated_opportunity,
    get_departures_by_team_position,
    get_arrivals_by_team_position,
    get_team_stats,
    get_player_advanced_metrics,
)


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_to_one(value: float, full_value: float) -> float:
    """
    Normalize to 0-1 against a max / full-confidence value.
    """
    if full_value <= 0:
        return 0.0
    return _clamp(value / full_value, 0.0, 1.0)


def _normalize_range(value: float, low: float, high: float) -> float:
    """
    Normalize to 0-1 inside a bounded range.
    """
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _sample_confidence(
        observed: float,
        full_confidence: float,
        min_confidence: float = 0.35
) -> float:
    """
    Reliability multiplier between min_confidence and 1.0.
    Larger samples approach full confidence.
    """
    if full_confidence <= 0:
        return 1.0
    ratio = _clamp(observed / full_confidence, 0.0, 1.0)
    return min_confidence + (1.0 - min_confidence) * ratio


def _pct_change(curr: float, prev: float, neutral_when_prev_zero: Optional[float] = None) -> Optional[float]:
    if prev == 0:
        return neutral_when_prev_zero
    return ((curr - prev) / prev) * 100.0


def _weighted_average(pairs: List[Tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in pairs if weight > 0)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in pairs if weight > 0) / total_weight


def _position_usage_volume(position: str, usage: Dict) -> float:
    """
    Position-aware role volume for prior usage.
    """
    targets = _safe_float(usage.get("targets", 0))
    carries = _safe_float(usage.get("carries", 0))
    routes = _safe_float(usage.get("routes", 0))
    snap_share = _safe_float(usage.get("snap_share", 0))
    pass_attempts = _safe_float(
        usage.get("pass_attempts", usage.get("attempts", 0)),
        0
    )

    if position in ["WR", "TE"]:
        return (
                targets * 0.50 +
                routes * 0.20 +
                snap_share * 0.30
        )
    if position == "RB":
        total_touches = carries + targets
        return (
                carries * 0.35 +
                targets * 0.20 +
                total_touches * 0.15 +
                snap_share * 0.30
        )
    if position == "QB":
        return (
                pass_attempts * 0.50 +
                carries * 0.10 +
                snap_share * 0.40
        )
    return snap_share


def _position_sample_confidence(position: str, usage: Dict) -> float:
    targets = _safe_float(usage.get("targets", 0))
    carries = _safe_float(usage.get("carries", 0))
    pass_attempts = _safe_float(
        usage.get("pass_attempts", usage.get("attempts", 0)),
        0
    )

    if position in ["WR", "TE"]:
        return _sample_confidence(targets, full_confidence=50, min_confidence=0.40)
    if position == "RB":
        carry_conf = _sample_confidence(carries, full_confidence=100, min_confidence=0.40)
        target_conf = _sample_confidence(targets, full_confidence=30, min_confidence=0.40)
        return _weighted_average([(carry_conf, 0.7), (target_conf, 0.3)])
    if position == "QB":
        return _sample_confidence(pass_attempts, full_confidence=250, min_confidence=0.50)
    return 0.50


def _build_player_baseline(position: str, player_prev_usage: Dict) -> Dict:
    """
    Stabilized baseline for comparing competition and role changes.
    """
    targets = _safe_float(player_prev_usage.get("targets", 0))
    carries = _safe_float(player_prev_usage.get("carries", 0))
    routes = _safe_float(player_prev_usage.get("routes", 0))
    snap_share = _safe_float(player_prev_usage.get("snap_share", 0))
    games = max(_safe_float(player_prev_usage.get("games", 0)), 1.0)
    pass_attempts = _safe_float(
        player_prev_usage.get("pass_attempts", player_prev_usage.get("attempts", 0)),
        0
    )

    raw_role = _position_usage_volume(position, player_prev_usage)
    confidence = _position_sample_confidence(position, player_prev_usage)

    if position in ["WR", "TE"]:
        floor_role = 18.0
    elif position == "RB":
        floor_role = 22.0
    elif position == "QB":
        floor_role = 35.0
    else:
        floor_role = 15.0

    stabilized_role = max(raw_role * confidence, floor_role)

    return {
        "targets": round(targets, 1),
        "carries": round(carries, 1),
        "routes": round(routes, 1),
        "pass_attempts": round(pass_attempts, 1),
        "snap_share": round(snap_share, 3),
        "games": round(games, 1),
        "raw_role_score": round(raw_role, 2),
        "confidence": round(confidence, 3),
        "stabilized_role_score": round(stabilized_role, 2),
    }


def _score_bucket(value: float, thresholds: List[Tuple[float, float]], default: float = 0.0) -> float:
    """
    thresholds should be ordered descending as [(minimum_value, score), ...]
    """
    for minimum, score in thresholds:
        if value >= minimum:
            return score
    return default


# ==============================================================================
# PASS-CATCHER OPPORTUNITY POOLING (WR + TE)
# ==============================================================================
# Receiving opportunity and competition are shared between WRs and TEs on the
# same team: targets vacated (or added) by a WR are contestable by a TE, and
# vice versa (e.g. a WR leaving Indianapolis frees up targets a pass-catching
# TE can absorb). Carries stay position-exact (RB-only); RB/QB are unaffected.

PASS_CATCHER_POSITIONS = ("WR", "TE")


def _opportunity_group_keys(team: str, position: str) -> List[Tuple[str, str]]:
    """(team, position) cache keys whose receiving opportunity this player shares."""
    if position in PASS_CATCHER_POSITIONS:
        return [(team, "WR"), (team, "TE")]
    return [(team, position)]


def _pooled_vacated(vacated_cache, team: str, position: str, season: int):
    """Merge vacated opportunity across the player's opportunity group.

    For RB/QB this is identical to a single-key lookup; for WR/TE it sums the
    receiving opportunity vacated by both WRs and TEs on the team.
    """
    merged = None
    for t, p in _opportunity_group_keys(team, position):
        if vacated_cache is not None:
            v = vacated_cache.get((t, p))
        else:
            v = get_vacated_opportunity(t, p, season)
        if not v:
            continue
        if merged is None:
            merged = {"targets": 0.0, "carries": 0.0, "snap_share": 0.0,
                      "departed_players": []}
        merged["targets"] += _safe_float(v.get("targets", 0))
        merged["carries"] += _safe_float(v.get("carries", 0))
        merged["snap_share"] += _safe_float(v.get("snap_share", 0))
        merged["departed_players"] += list(v.get("departed_players", []) or [])
    return merged


# Expected per-game claim for a player with NO prior usage (rookies), by draft
# round. A 1st-round WR projects to a ~#2/#3-receiver target share; later picks
# and UDFAs command progressively less. RB units are weighted carries+targets.
_DRAFT_CLAIM_WR_TE = {1: 5.0, 2: 3.5, 3: 2.5, 4: 1.5}
_DRAFT_CLAIM_RB = {1: 14.0, 2: 9.0, 3: 6.0, 4: 4.0}


def _competitor_claim(position: str, entry: Dict) -> float:
    """A player's claim on vacated work, based on prior usage PER GAME.

    Vacated targets/carries get redistributed toward the players who were
    already earning volume: someone who was 2nd on the team in targets/game
    inherits far more of a departed teammate's work than someone who saw 2
    targets/game. Per-game (not season totals) so an injury-shortened but
    high-usage season still counts.

    A player with NO prior usage (a drafted rookie) has no usage to measure, so
    their claim falls back to an expected share implied by draft capital — e.g. a
    team that loses its WR1 but drafts a 1st-round WR sees that rookie absorb a
    real chunk of the vacated targets. (Draft capital is only a fallback; an
    established player is always weighted by actual usage, not pedigree.)
    """
    games = _safe_float(entry.get("last_season_games", 0))
    targets = _safe_float(entry.get("last_season_targets", 0))
    carries = _safe_float(entry.get("last_season_carries", 0))
    volume = (carries + targets * 1.5) if position == "RB" else targets

    if volume > 0:
        # Per game when we know games played; else approximate from a full slate.
        return volume / games if games >= 1 else volume / 17.0

    # No measured usage — infer expected role from draft capital. An undrafted
    # player who never produced isn't a real competitor for vacated work, so they
    # claim nothing (and don't dilute or inflate the competitor count).
    draft_round = (entry.get("draft_metadata") or {}).get("round")
    if draft_round is None:
        return 0.0
    if position == "RB":
        return _DRAFT_CLAIM_RB.get(draft_round, 2.0)   # rounds 5-7
    return _DRAFT_CLAIM_WR_TE.get(draft_round, 1.0)     # rounds 5-7


def _opportunity_share(
    player_id: str, team: str, position: str,
    incumbents_cache: Optional[Dict], arrivals_cache: Optional[Dict],
) -> Tuple[float, int]:
    """The player's usage-weighted share of the vacated work, and competitor count.

    Returns (share_fraction, competitor_count). The vacated opportunity is split
    among the players who share the group (returning incumbents + arrivals across
    the pass-catcher group, or the exact position for RB) in proportion to each
    player's prior usage per game — not equally and not by draft capital. Falls
    back to a full share (1.0) when there is no usage context to divide by.
    """
    entries: Dict[str, Dict] = {}
    for key in _opportunity_group_keys(team, position):
        for src in (incumbents_cache, arrivals_cache):
            for e in (src or {}).get(key, []):
                pid = e.get("player_id")
                if pid:
                    entries.setdefault(pid, e)

    claims = {pid: _competitor_claim(position, e) for pid, e in entries.items()}
    claims.setdefault(player_id, 0.0)  # scored player always shares the room

    total = sum(claims.values())
    competitors = max(sum(1 for v in claims.values() if v > 0), 1)
    if total <= 0:
        return 1.0, competitors  # no one had prior usage — don't dilute
    return claims[player_id] / total, competitors


def _pooled_list(cache, db_fn, team: str, position: str, season: int) -> List[Dict]:
    """Concatenate a list-valued cache (departures/arrivals) across the group.

    De-dupes by player_id for safety (a player only appears under their own
    position, so concatenation normally has no overlap).
    """
    out: List[Dict] = []
    seen = set()
    for t, p in _opportunity_group_keys(team, position):
        items = cache.get((t, p), []) if cache is not None else db_fn(t, p, season)
        for it in items or []:
            pid = it.get("player_id")
            if pid is not None and pid in seen:
                continue
            if pid is not None:
                seen.add(pid)
            out.append(it)
    return out


# ==============================================================================
# ARCHETYPE / ROLE FIT (context only — does NOT affect the score)
# ==============================================================================
# Read-only explainability: how well a candidate's receiving role matches the
# type of targets being vacated (a slot/possession vacancy "fits" a slot player
# more than an outside burner). Surfaced as a label/insight; the numeric score
# stays usage-based, since archetype data only exists for 2024-25 and can't be
# backtested. Requires PFF-style fields (aDOT, slot/wide rate, YAC) which are
# exported to cache/archetype_{season}.json. Degrades to None when unavailable.

def _archetype_vector(profile: Optional[Dict]) -> Optional[Tuple[float, float, float]]:
    """Normalized role vector (slot tendency, target depth, YAC) or None."""
    if not profile:
        return None
    slot, adot, yac = profile.get("slot_rate"), profile.get("adot"), profile.get("yac_per_rec")
    if slot is None and adot is None and yac is None:
        return None
    return (
        _clamp(_safe_float(slot) / 100.0, 0.0, 1.0),   # 0=outside .. 1=pure slot
        _clamp(_safe_float(adot) / 20.0, 0.0, 1.0),    # target depth (aDOT)
        _clamp(_safe_float(yac) / 10.0, 0.0, 1.0),     # yards after catch
    )


def _vacated_archetype_profile(departed_players: List[Dict]) -> Optional[Tuple[float, float, float]]:
    """Target-weighted average role vector of the departed players, or None."""
    acc, wsum = [0.0, 0.0, 0.0], 0.0
    for dp in departed_players or []:
        vec = _archetype_vector(dp.get("archetype"))
        if vec is None:
            continue
        w = max(_safe_float(dp.get("targets", 0)), 1.0)
        for i in range(3):
            acc[i] += vec[i] * w
        wsum += w
    return (acc[0] / wsum, acc[1] / wsum, acc[2] / wsum) if wsum > 0 else None


def _describe_role(vec: Tuple[float, float, float]) -> str:
    """Short human label for a role vector, e.g. 'slot, short-area'."""
    slot, adot, _ = vec
    align = "slot" if slot >= 0.6 else "outside" if slot <= 0.35 else "slot/outside"
    depth = "deep" if adot >= 0.6 else "short-area" if adot <= 0.4 else "intermediate"
    return f"{align}, {depth}"


def compute_archetype_fit(
    player_archetype: Optional[Dict], team: str, position: str, vacated_cache: Optional[Dict],
) -> Optional[Dict]:
    """How well the candidate's role matches the vacated targets (WR/TE only).

    Returns {fit, label, candidate_role, vacated_role} or None when archetype
    data is missing on either side. Pure context — never feeds the score.
    """
    if position not in PASS_CATCHER_POSITIONS:
        return None
    cand = _archetype_vector(player_archetype)
    if cand is None:
        return None
    departed: List[Dict] = []
    for key in _opportunity_group_keys(team, position):
        v = (vacated_cache or {}).get(key)
        if v:
            departed += v.get("departed_players", []) or []
    vac = _vacated_archetype_profile(departed)
    if vac is None:
        return None
    dist = math.sqrt(sum((cand[i] - vac[i]) ** 2 for i in range(3)))
    sim = max(0.0, 1.0 - dist / math.sqrt(3.0))
    # Realistic role distances compress into ~0.6-1.0, so calibrate labels there:
    # a clean alignment mismatch (slot vs outside) lands ~0.61 -> "low".
    label = "high" if sim >= 0.85 else "medium" if sim >= 0.70 else "low"
    return {
        "fit": round(sim, 2),
        "label": label,
        "candidate_role": _describe_role(cand),
        "vacated_role": _describe_role(vac),
    }


# ==============================================================================
# COMPONENT 1: OPPORTUNITY OPENED SCORE
# ==============================================================================

def calculate_opportunity_opened_score(
        player_id: str,
        team: str,
        position: str,
        season: int,
        vacated_cache: Optional[Dict] = None,
        air_yards_data: Optional[Dict] = None,
        incumbents_cache: Optional[Dict] = None,
        arrivals_cache: Optional[Dict] = None,
        player_prev_usage: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """
    Score (0-100) based on the vacated opportunity actually AVAILABLE to this
    player — the team's vacated work divided among the credible competitors
    contesting it, not the gross team total credited to everyone.

    Args:
        vacated_cache: Optional dict mapping (team, position) to vacated opportunity.
                      If provided, uses O(1) cache lookup instead of DB query.
        air_yards_data: Optional dict with air yards context for WR/TE quality bonus:
                        - 'vacated_air_yards' (int): total air yards from departed WRs/TEs
                        - 'avg_depth_of_target' (float): average aDOT of departed targets
        incumbents_cache, arrivals_cache: Optional rosters used to count the
                        credible competitors who will share the vacated work. When
                        omitted, no dilution is applied (competitor count = 1).
    """
    # OPTIMIZED: Use cache if provided, otherwise fall back to DB query.
    # WR/TE pool receiving opportunity (see _pooled_vacated); RB/QB unchanged.
    vac_opp = _pooled_vacated(vacated_cache, team, position, season)

    if not vac_opp:
        return 0.0, {
            "player_id": player_id,
            "team": team,
            "position": position,
            "season": season,
            "vacated_targets": 0.0,
            "vacated_carries": 0.0,
            "vacated_snap_share": 0.0,
            "raw_score": 0.0,
            "snap_bonus": 0.0,
            "departed_players": [],
        }

    vacated_targets = _safe_float(vac_opp.get("targets", 0))
    vacated_carries = _safe_float(vac_opp.get("carries", 0))
    vacated_snap_share = _safe_float(vac_opp.get("snap_share", 0))
    departed_players = vac_opp.get("departed_players", [])

    # Contested-share dilution: split the vacated work among credible competitors
    # in proportion to each player's claim (usage + draft pedigree), so the score
    # reflects what is realistically AVAILABLE to this player rather than the
    # gross team total. QB starter snaps are not shared, so they stay undivided.
    share_fraction, competitor_count = _opportunity_share(
        player_id, team, position, incumbents_cache, arrivals_cache
    )
    share_targets = vacated_targets * share_fraction
    share_carries = vacated_carries * share_fraction
    share_snap = vacated_snap_share * share_fraction

    if position in ["WR", "TE"]:
        target_score = min((share_targets / PER_COMPETITOR_TARGETS_WR_TE) * 100.0, 100.0)
        raw_score = target_score
    elif position == "RB":
        carry_score = min((share_carries / PER_COMPETITOR_CARRIES_RB) * 70.0, 70.0)
        target_score = min((share_targets / PER_COMPETITOR_TARGETS_RB) * 30.0, 30.0)
        raw_score = carry_score + target_score
    elif position == "QB":
        # Starter snaps are not shared among competitors — use the gross value.
        # But a RETURNING starter's job did not "open": only credit a QB stepping
        # into a vacated starting role, not one who already held it last season
        # (otherwise a backup leaving falsely reads as an opening for the starter).
        #
        # Sleeper doesn't always populate avg_off_snap_pct for QBs, so we use
        # three independent signals — any one is enough to confirm incumbency.
        _pu = player_prev_usage or {}
        qb_prev_snap     = _safe_float(_pu.get("snap_share", 0))
        qb_prev_games    = _safe_float(_pu.get("games", 0))
        qb_prev_attempts = _safe_float(_pu.get("pass_attempts", _pu.get("attempts", 0)))
        _was_starter = (
            qb_prev_snap     >= QB_STARTER_SNAP_THRESHOLD  or
            qb_prev_games    >= QB_STARTER_GAMES_MIN        or
            qb_prev_attempts >= QB_STARTER_ATTEMPTS_MIN
        )
        if _was_starter:
            raw_score = 0.0
        else:
            raw_score = 100.0 if vacated_snap_share >= QB_STARTER_SNAP_THRESHOLD else 0.0
    else:
        raw_score = 0.0

    # QB snap bonus only when a starting job actually opened (raw_score > 0).
    if position == "QB":
        snap_for_bonus = vacated_snap_share if raw_score > 0 else 0.0
    else:
        snap_for_bonus = share_snap
    snap_bonus = min(snap_for_bonus * 50.0, MAX_SNAP_SHARE_BONUS)

    # --- Air yards quality bonus (WR/TE only) ---
    air_yards_bonus = 0.0
    vacated_air_yards = 0
    avg_depth_of_target = 0.0

    if position in ["WR", "TE"] and air_yards_data:
        vacated_air_yards = _safe_int(air_yards_data.get("vacated_air_yards", 0), 0)
        avg_depth_of_target = _safe_float(air_yards_data.get("avg_depth_of_target", 0.0), 0.0)

        # Volume bonus: normalized against elite WR1 air yards season (the
        # player's diluted share, consistent with target/carry dilution).
        volume_bonus = _normalize_to_one(vacated_air_yards * share_fraction, MAX_VACATED_AIR_YARDS) * AIR_YARDS_ELITE_BONUS

        # Depth bonus: routes that go downfield are harder to replace and more valuable
        if avg_depth_of_target >= AIR_YARDS_ELITE_ADOT:
            depth_bonus = AIR_YARDS_ELITE_BONUS
        elif avg_depth_of_target >= AIR_YARDS_GOOD_ADOT:
            depth_bonus = AIR_YARDS_GOOD_BONUS
        elif avg_depth_of_target >= AIR_YARDS_AVERAGE_ADOT:
            depth_bonus = AIR_YARDS_AVERAGE_BONUS
        else:
            depth_bonus = 0.0

        air_yards_bonus = _clamp(volume_bonus * 0.6 + depth_bonus * 0.4, 0.0, AIR_YARDS_ELITE_BONUS)

    final_score = _clamp(raw_score + snap_bonus + air_yards_bonus, 0.0, 100.0)

    details = {
        "player_id": player_id,
        "team": team,
        "position": position,
        "season": season,
        "vacated_targets": round(vacated_targets, 1),
        "vacated_carries": round(vacated_carries, 1),
        "vacated_snap_share": round(vacated_snap_share, 3),
        "competitor_count": competitor_count,
        "share_targets": round(share_targets, 1),
        "share_carries": round(share_carries, 1),
        "raw_score": round(raw_score, 2),
        "snap_bonus": round(snap_bonus, 2),
        "air_yards_bonus": round(air_yards_bonus, 2),
        "vacated_air_yards": vacated_air_yards,
        "avg_depth_of_target": round(avg_depth_of_target, 2),
        "departed_players": departed_players,
    }

    return round(final_score, 2), details


# ==============================================================================
# COMPONENT 2: COMPETITION REMOVED SCORE
# ==============================================================================

def _departure_competition_load(position: str, dep: Dict) -> Tuple[float, Dict]:
    dep_targets = _safe_float(dep.get("last_season_targets", 0))
    dep_carries = _safe_float(dep.get("last_season_carries", 0))
    dep_routes = _safe_float(dep.get("last_season_routes", 0))
    dep_snap = _safe_float(dep.get("last_season_snap_share", 0))
    dep_points = _safe_float(dep.get("last_season_fantasy_points", 0))
    dep_attempts = _safe_float(dep.get("last_season_pass_attempts", 0))
    dep_games = max(_safe_float(dep.get("last_season_games", 0)), 1.0)

    if position in ["WR", "TE"]:
        load = (
                dep_targets * 0.50 +
                dep_routes * 0.20 +
                dep_snap * 100 * 0.20 +
                dep_points * 0.10
        )
        primary_usage = dep_targets
    elif position == "RB":
        weighted_work = dep_carries + (dep_targets * 1.2)
        load = (
                weighted_work * 0.55 +
                dep_snap * 100 * 0.20 +
                dep_points * 0.25
        )
        primary_usage = weighted_work
    elif position == "QB":
        load = (
                dep_attempts * 0.50 +
                dep_carries * 0.10 +
                dep_snap * 100 * 0.25 +
                dep_points * 0.15
        )
        primary_usage = dep_attempts
    else:
        load = 0.0
        primary_usage = 0.0

    details = {
        "targets": round(dep_targets, 1),
        "carries": round(dep_carries, 1),
        "routes": round(dep_routes, 1),
        "snap_share": round(dep_snap, 3),
        "fantasy_points": round(dep_points, 1),
        "pass_attempts": round(dep_attempts, 1),
        "games": round(dep_games, 1),
        "primary_usage": round(primary_usage, 1),
        "competition_load": round(load, 2),
    }
    return load, details


def calculate_competition_removed_score(
        player_id: str,
        team: str,
        position: str,
        season: int,
        player_prev_usage: Dict,
        departures_cache: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Score (0-100) based on meaningful same-position competition leaving the roster.

    Args:
        departures_cache: Optional dict mapping (team, position) to list of departures.
                         If provided, uses O(1) cache lookup instead of DB query.
    """
    # OPTIMIZED: Use cache if provided, otherwise fall back to DB query.
    # WR/TE pool pass-catcher departures so a TE sees a departing WR as removed
    # competition (and vice versa); RB/QB unchanged.
    departures = _pooled_list(
        departures_cache, get_departures_by_team_position, team, position, season
    )

    if not departures:
        return 0.0, {
            "player_id": player_id,
            "team": team,
            "position": position,
            "season": season,
            "player_baseline": {},
            "total_departure_load": 0.0,
            "team_relief_bonus": 0.0,
            "key_departures": [],
        }

    baseline = _build_player_baseline(position, player_prev_usage or {})
    player_role_score = baseline["stabilized_role_score"]

    total_score = 0.0
    total_departure_load = 0.0
    key_departures: List[Dict] = []

    for dep in departures:
        departure_load, dep_details = _departure_competition_load(position, dep)
        if departure_load <= 0:
            continue

        total_departure_load += departure_load
        relative_relief = departure_load / max(player_role_score, 1.0)

        if relative_relief >= 2.2:
            departure_score = 28.0
            threat_level = "high"
        elif relative_relief >= 1.25:
            departure_score = 18.0
            threat_level = "medium"
        elif relative_relief >= 0.65:
            departure_score = 10.0
            threat_level = "low"
        else:
            departure_score = 4.0
            threat_level = "minimal"

        importance_bonus = 0.0
        if position in ["WR", "TE"]:
            if dep_details["targets"] >= 90:
                importance_bonus += 6.0
            elif dep_details["targets"] >= 60:
                importance_bonus += 3.0
        elif position == "RB":
            combined_work = dep_details["carries"] + dep_details["targets"]
            if combined_work >= 220:
                importance_bonus += 6.0
            elif combined_work >= 140:
                importance_bonus += 3.0
        elif position == "QB":
            if dep_details["snap_share"] >= 0.70:
                importance_bonus += 12.0
            elif dep_details["snap_share"] >= 0.35:
                importance_bonus += 5.0

        player_departure_score = departure_score + importance_bonus
        total_score += player_departure_score

        key_departures.append({
            "name": dep.get("player_name"),
            "player_id": dep.get("player_id"),
            "change_type": dep.get("change_type"),
            "threat_level": threat_level,
            "relative_relief": round(relative_relief, 2),
            "departure_score": round(player_departure_score, 2),
            **dep_details,
        })

    if position in ["WR", "TE"]:
        team_relief_bonus = min(total_departure_load / 18.0, 12.0)
    elif position == "RB":
        team_relief_bonus = min(total_departure_load / 22.0, 12.0)
    elif position == "QB":
        team_relief_bonus = min(total_departure_load / 30.0, 15.0)
    else:
        team_relief_bonus = 0.0

    final_score = min(total_score + team_relief_bonus, 100.0)
    key_departures.sort(key=lambda x: x["departure_score"], reverse=True)

    details = {
        "player_id": player_id,
        "team": team,
        "position": position,
        "season": season,
        "player_baseline": baseline,
        "total_departure_load": round(total_departure_load, 2),
        "team_relief_bonus": round(team_relief_bonus, 2),
        "key_departures": key_departures,
    }

    return round(final_score, 2), details


# ==============================================================================
# COMPONENT 3: COMPETITION ADDED PENALTY
# ==============================================================================

def _get_contract_signal(arrival: Dict) -> float:
    contract = arrival.get("contract_metadata") or {}
    guaranteed = _safe_float(contract.get("guaranteed", 0))
    apy = _safe_float(contract.get("apy", 0))
    years = _safe_float(contract.get("years", 0))

    guaranteed_signal = _normalize_to_one(guaranteed, 15_000_000)
    apy_signal = _normalize_to_one(apy, 12_000_000)
    years_signal = _normalize_to_one(years, 4)

    return round(
        guaranteed_signal * 0.45 +
        apy_signal * 0.40 +
        years_signal * 0.15,
        3
    )


def _get_draft_signal(arrival: Dict) -> Tuple[float, Dict]:
    draft_meta = arrival.get("draft_metadata") or {}
    draft_round = draft_meta.get("round")
    draft_pick = draft_meta.get("pick")

    if draft_round == 1:
        signal = 1.00
    elif draft_round == 2:
        signal = 0.82
    elif draft_round == 3:
        signal = 0.62
    elif draft_round == 4:
        signal = 0.42
    elif draft_round == 5:
        signal = 0.28
    elif draft_round in [6, 7]:
        signal = 0.16
    else:
        signal = 0.08

    return signal, {
        "draft_round": draft_round,
        "draft_pick": draft_pick,
    }


def _calculate_arrival_role_threat(position: str, arrival: Dict) -> Tuple[float, Dict]:
    prev_targets = _safe_float(arrival.get("last_season_targets", 0))
    prev_carries = _safe_float(arrival.get("last_season_carries", 0))
    prev_routes = _safe_float(arrival.get("last_season_routes", 0))
    prev_snap = _safe_float(arrival.get("last_season_snap_share", 0))
    prev_points = _safe_float(arrival.get("last_season_fantasy_points", 0))
    prev_attempts = _safe_float(arrival.get("last_season_pass_attempts", 0))

    if position in ["WR", "TE"]:
        usage_signal = (
                _normalize_to_one(prev_targets, 110) * 0.45 +
                _normalize_to_one(prev_routes, 450) * 0.20 +
                _normalize_to_one(prev_snap, 0.80) * 0.20 +
                _normalize_to_one(prev_points, 220) * 0.15
        )
        primary_usage = prev_targets
    elif position == "RB":
        weighted_work = prev_carries + (prev_targets * 1.35)
        usage_signal = (
                _normalize_to_one(weighted_work, 260) * 0.55 +
                _normalize_to_one(prev_snap, 0.70) * 0.20 +
                _normalize_to_one(prev_points, 250) * 0.25
        )
        primary_usage = weighted_work
    elif position == "QB":
        usage_signal = (
                _normalize_to_one(prev_attempts, 550) * 0.50 +
                _normalize_to_one(prev_snap, 0.85) * 0.25 +
                _normalize_to_one(prev_points, 320) * 0.25
        )
        primary_usage = prev_attempts
    else:
        usage_signal = 0.0
        primary_usage = 0.0

    change_type = arrival.get("change_type")

    if change_type == "draft":
        draft_signal, draft_info = _get_draft_signal(arrival)
        contract_signal = 0.0
        threat_score = usage_signal * 0.15 + draft_signal * 0.85
    elif change_type in ["free_agent", "trade"]:
        contract_signal = _get_contract_signal(arrival)
        draft_signal = 0.0
        draft_info = {}
        threat_score = usage_signal * 0.72 + contract_signal * 0.28
    else:
        draft_signal = 0.0
        contract_signal = 0.0
        draft_info = {}
        threat_score = usage_signal

    details = {
        "change_type": change_type,
        "usage_signal": round(usage_signal, 3),
        "draft_signal": round(draft_signal, 3),
        "contract_signal": round(contract_signal, 3),
        "primary_usage": round(primary_usage, 1),
        "prev_targets": round(prev_targets, 1),
        "prev_carries": round(prev_carries, 1),
        "prev_routes": round(prev_routes, 1),
        "prev_snap_share": round(prev_snap, 3),
        "prev_fantasy_points": round(prev_points, 1),
        "prev_pass_attempts": round(prev_attempts, 1),
        **draft_info,
    }

    return _clamp(threat_score, 0.0, 1.0), details


def _threat_score_to_penalty(threat_score: float, position: str) -> Tuple[float, str]:
    if position == "QB":
        max_penalty = 35.0
    elif position == "RB":
        max_penalty = 28.0
    elif position in ["WR", "TE"]:
        max_penalty = 24.0
    else:
        max_penalty = 20.0

    penalty = -1.0 * max_penalty * threat_score

    if threat_score >= 0.78:
        threat_level = "high"
    elif threat_score >= 0.48:
        threat_level = "medium"
    elif threat_score >= 0.22:
        threat_level = "low"
    else:
        threat_level = "minimal"

    return round(penalty, 2), threat_level


def calculate_competition_added_penalty(
        player_id: str,
        team: str,
        position: str,
        season: int,
        arrivals_cache: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Negative score for added same-position competition.

    Args:
        arrivals_cache: Optional dict mapping (team, position) to list of arrivals.
                       If provided, uses O(1) cache lookup instead of DB query.
    """
    # OPTIMIZED: Use cache if provided, otherwise fall back to DB query.
    # WR/TE pool pass-catcher arrivals so an incoming WR counts as added
    # competition for a TE (and vice versa); RB/QB unchanged.
    arrivals = _pooled_list(
        arrivals_cache, get_arrivals_by_team_position, team, position, season
    )

    if not arrivals:
        return 0.0, {
            "player_id": player_id,
            "team": team,
            "position": position,
            "season": season,
            "threats_added": [],
            "raw_penalty_total": 0.0,
            "stacked_penalty_total": 0.0,
            "stacking_adjustment": 0.0,
            "final_penalty": 0.0,
        }

    threats: List[Dict] = []
    raw_penalty_total = 0.0

    for arrival in arrivals:
        if arrival.get("player_id") == player_id:
            continue  # Don't count the player as their own competitor
        threat_score, threat_details = _calculate_arrival_role_threat(position, arrival)
        if threat_score <= 0:
            continue

        penalty_value, threat_level = _threat_score_to_penalty(threat_score, position)
        raw_penalty_total += penalty_value

        threats.append({
            "name": arrival.get("player_name"),
            "player_id": arrival.get("player_id"),
            "threat_score": round(threat_score, 3),
            "threat_level": threat_level,
            "penalty": penalty_value,
            **threat_details,
        })

    if not threats:
        return 0.0, {
            "player_id": player_id,
            "team": team,
            "position": position,
            "season": season,
            "threats_added": [],
            "raw_penalty_total": 0.0,
            "stacked_penalty_total": 0.0,
            "stacking_adjustment": 0.0,
            "final_penalty": 0.0,
        }

    threats.sort(key=lambda x: x["threat_score"], reverse=True)

    stacked_penalty = 0.0
    stacking_weights = [1.00, 0.75, 0.55]

    for idx, threat in enumerate(threats):
        base_penalty = threat["penalty"]
        weight = stacking_weights[idx] if idx < len(stacking_weights) else 0.35
        adjusted_penalty = base_penalty * weight
        stacked_penalty += adjusted_penalty
        threat["stacking_weight"] = weight
        threat["adjusted_penalty"] = round(adjusted_penalty, 2)

    if position == "QB":
        min_penalty_cap = -50.0
    elif position == "RB":
        min_penalty_cap = -42.0
    elif position == "WR":
        min_penalty_cap = -38.0
    elif position == "TE":
        min_penalty_cap = -32.0
    else:
        min_penalty_cap = -35.0

    final_penalty = max(stacked_penalty, min_penalty_cap)

    details = {
        "player_id": player_id,
        "team": team,
        "position": position,
        "season": season,
        "raw_penalty_total": round(raw_penalty_total, 2),
        "stacked_penalty_total": round(stacked_penalty, 2),
        "stacking_adjustment": round(stacked_penalty - raw_penalty_total, 2),
        "final_penalty": round(final_penalty, 2),
        "threats_added": threats,
    }

    return round(final_penalty, 2), details


# ==============================================================================
# COMPONENT 4: TEAM ENVIRONMENT SCORE
# ==============================================================================

def calculate_team_environment_score(
        team: str,
        position: str,
        season: int,
        team_stats_cache: Optional[Dict] = None,
        coaching_changes: Optional[Dict] = None,
        qb_change_data: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Score (0-100) based on how favorable the team offensive environment is
    for a breakout at the given position.

    Args:
        team_stats_cache: Optional dict mapping team to team stats dict.
                         If provided, uses O(1) cache lookup instead of file load.
        coaching_changes: Optional dict with coaching context:
                          - 'new_oc' (bool): New offensive coordinator hired
                          - 'new_hc' (bool): New head coach hired
                          - 'oc_prior_pass_rate' (float): New OC's prior team pass rate
                          - 'oc_prior_team' (str): New OC's previous team (informational)
        qb_change_data: Optional dict with QB situation:
                        - 'qb_changed' (bool): Starting QB changed from last season
                        - 'change_type' ('upgrade'|'downgrade'|'lateral'|'unknown')
                        - 'new_qb_passer_rating' (float): New QB's prior passer rating
                        - 'old_qb_passer_rating' (float): Departed QB's passer rating
    """
    # OPTIMIZED: Use cache if provided, otherwise fall back to file load
    if team_stats_cache is not None:
        team_stats = team_stats_cache.get(team, {})
    else:
        team_stats = get_team_stats(team, season) or {}

    # NFL league averages - used as floor when cached stats are missing/zero.
    # Zero values indicate absent enrichment (not a genuinely zero-offense team)
    # so we substitute league averages rather than produce artificially low scores.
    _NFL_PASS_ATT_PG = 33.5
    _NFL_RUSH_ATT_PG = 25.5
    _NFL_PASS_YDS_PG = 228.0
    _NFL_RUSH_YDS_PG = 110.0
    _NFL_PASS_TD_PG = 1.65
    _NFL_RUSH_TD_PG = 0.85
    _NFL_POINTS_PG = 22.5
    _NFL_RED_ZONE_TRIPS_PG = 3.2
    _NFL_SACKS_ALLOWED_PG = 2.4

    def _stat_or_avg(key: str, avg: float) -> float:
        v = _safe_float(team_stats.get(key, 0.0))
        return v if v > 0.01 else avg

    pass_att_pg = _stat_or_avg("pass_att_pg", _NFL_PASS_ATT_PG)
    rush_att_pg = _stat_or_avg("rush_att_pg", _NFL_RUSH_ATT_PG)
    pass_yds_pg = _stat_or_avg("pass_yds_pg", _NFL_PASS_YDS_PG)
    rush_yds_pg = _stat_or_avg("rush_yds_pg", _NFL_RUSH_YDS_PG)
    pass_td_pg = _stat_or_avg("pass_td_pg", _NFL_PASS_TD_PG)
    rush_td_pg = _stat_or_avg("rush_td_pg", _NFL_RUSH_TD_PG)
    points_pg = _stat_or_avg("points_pg", _NFL_POINTS_PG)
    red_zone_trips_pg = _stat_or_avg("red_zone_trips_pg", _NFL_RED_ZONE_TRIPS_PG)
    sacks_allowed_pg = _stat_or_avg("sacks_allowed_pg", _NFL_SACKS_ALLOWED_PG)

    # Track whether we fell back to league averages for explainability
    _using_derived_stats = _safe_float(team_stats.get("pass_att_pg", 0.0)) <= 0.01

    total_plays_pg = pass_att_pg + rush_att_pg
    total_yds_pg = pass_yds_pg + rush_yds_pg
    total_td_pg = pass_td_pg + rush_td_pg
    pass_rate = pass_att_pg / total_plays_pg if total_plays_pg > 0 else 0.5
    run_rate = 1.0 - pass_rate
    yards_per_play = total_yds_pg / total_plays_pg if total_plays_pg > 0 else 5.0

    volume_score = _normalize_range(total_plays_pg, 55.0, 70.0) * 20.0

    scoring_score = (
            _normalize_range(points_pg, 16.0, 30.0) * 12.0 +
            _normalize_range(total_td_pg, 1.6, 3.8) * 8.0
    )

    efficiency_score = (
            _normalize_range(yards_per_play, 4.7, 6.4) * 10.0 +
            (1.0 - _normalize_range(sacks_allowed_pg, 1.0, 4.0)) * 5.0
    )

    red_zone_score = _normalize_range(red_zone_trips_pg, 2.0, 4.5) * 15.0

    if position in ["WR", "TE"]:
        position_fit_score = (
                _normalize_range(pass_rate, 0.48, 0.67) * 10.0 +
                _normalize_range(pass_yds_pg, 180.0, 310.0) * 6.0 +
                _normalize_range(pass_td_pg, 1.1, 2.4) * 4.0
        )
    elif position == "RB":
        position_fit_score = (
                _normalize_range(run_rate, 0.33, 0.52) * 8.0 +
                _normalize_range(rush_att_pg, 20.0, 33.0) * 7.0 +
                _normalize_range(rush_td_pg, 0.5, 1.8) * 5.0
        )
    elif position == "QB":
        position_fit_score = (
                _normalize_range(pass_rate, 0.48, 0.67) * 8.0 +
                _normalize_range(pass_att_pg, 27.0, 41.0) * 7.0 +
                _normalize_range(pass_td_pg, 1.1, 2.4) * 5.0
        )
    else:
        position_fit_score = 10.0

    context_bonus = 0.0
    if position in ["WR", "TE"]:
        if pass_rate >= 0.58 and pass_td_pg >= 2.0:
            context_bonus += 5.0
        elif pass_rate <= 0.47 and pass_yds_pg < 200:
            context_bonus -= 4.0
    elif position == "RB":
        if rush_att_pg >= 28 and total_td_pg >= 2.5:
            context_bonus += 5.0
        elif rush_att_pg < 21 and points_pg < 20:
            context_bonus -= 4.0
    elif position == "QB":
        if pass_att_pg >= 36 and pass_td_pg >= 2.0:
            context_bonus += 5.0
        elif pass_att_pg < 28 and total_td_pg < 2.0:
            context_bonus -= 4.0

    # --- OC / Coaching change modifier ---
    coaching_bonus = 0.0
    coaching_note = "no_change"

    if coaching_changes:
        new_oc = coaching_changes.get("new_oc", False)
        new_hc = coaching_changes.get("new_hc", False)
        oc_prior_pass_rate = _safe_float(coaching_changes.get("oc_prior_pass_rate", 0.0), 0.0)

        if new_hc:
            coaching_bonus += HC_CHANGE_UNCERTAINTY_PENALTY
            coaching_note = "new_hc_uncertainty"

        if new_oc and oc_prior_pass_rate > 0:
            if position in ["WR", "TE"]:
                if oc_prior_pass_rate >= OC_PASS_HEAVY_THRESHOLD:
                    coaching_bonus += OC_PASS_HEAVY_WR_BONUS
                    coaching_note = "new_oc_pass_heavy"
                elif oc_prior_pass_rate <= OC_RUN_HEAVY_THRESHOLD:
                    coaching_bonus += OC_RUN_HEAVY_WR_PENALTY
                    coaching_note = "new_oc_run_heavy"
                else:
                    coaching_note = "new_oc_balanced"
            elif position == "RB":
                if oc_prior_pass_rate <= OC_RUN_HEAVY_THRESHOLD:
                    coaching_bonus += OC_RUN_HEAVY_RB_BONUS
                    coaching_note = "new_oc_run_heavy"
                elif oc_prior_pass_rate >= OC_PASS_HEAVY_THRESHOLD:
                    coaching_bonus += OC_PASS_HEAVY_RB_PENALTY
                    coaching_note = "new_oc_pass_heavy"
        elif new_oc:
            # OC changed but prior scheme unknown - small uncertainty penalty
            coaching_bonus += HC_CHANGE_UNCERTAINTY_PENALTY
            coaching_note = "new_oc_unknown_scheme"

    # --- QB change modifier (WR/TE primary, minor for RB/QB) ---
    qb_change_bonus = 0.0
    qb_change_note = "no_change"

    if qb_change_data and qb_change_data.get("qb_changed"):
        change_type = qb_change_data.get("change_type", "unknown")
        new_passer_rating = _safe_float(qb_change_data.get("new_qb_passer_rating", 0.0), 0.0)

        if position in ["WR", "TE"]:
            if change_type == "upgrade":
                qb_change_bonus = QB_UPGRADE_WR_BONUS
            elif change_type == "downgrade":
                qb_change_bonus = QB_DOWNGRADE_WR_PENALTY
            elif change_type == "lateral":
                qb_change_bonus = QB_LATERAL_CHANGE
            else:
                # Unknown change type: derive from passer rating if available
                if new_passer_rating >= QB_TIER_ELITE_RATING:
                    qb_change_bonus = QB_TIER_WR_SCORES['elite']
                elif new_passer_rating >= QB_TIER_GOOD_RATING:
                    qb_change_bonus = QB_TIER_WR_SCORES['good']
                elif new_passer_rating >= QB_TIER_AVERAGE_RATING:
                    qb_change_bonus = QB_TIER_WR_SCORES['average']
                elif new_passer_rating > 0:
                    qb_change_bonus = QB_TIER_WR_SCORES['poor']

            qb_change_note = change_type

    total_score = _clamp(
        volume_score +
        scoring_score +
        efficiency_score +
        red_zone_score +
        position_fit_score +
        context_bonus +
        coaching_bonus +
        qb_change_bonus,
        0.0,
        100.0,
    )

    details = {
        "team": team,
        "position": position,
        "season": season,
        "volume_score": round(volume_score, 2),
        "scoring_score": round(scoring_score, 2),
        "efficiency_score": round(efficiency_score, 2),
        "red_zone_score": round(red_zone_score, 2),
        "position_fit_score": round(position_fit_score, 2),
        "context_bonus": round(context_bonus, 2),
        "coaching_bonus": round(coaching_bonus, 2),
        "coaching_note": coaching_note,
        "qb_change_bonus": round(qb_change_bonus, 2),
        "qb_change_note": qb_change_note,
        "total_plays_pg": round(total_plays_pg, 2),
        "pass_rate": round(pass_rate, 3),
        "run_rate": round(run_rate, 3),
        "total_yds_pg": round(total_yds_pg, 2),
        "yards_per_play": round(yards_per_play, 2),
        "points_pg": round(points_pg, 2),
        "red_zone_trips_pg": round(red_zone_trips_pg, 2),
        "pass_att_pg": round(pass_att_pg, 2),
        "rush_att_pg": round(rush_att_pg, 2),
        "pass_yds_pg": round(pass_yds_pg, 2),
        "rush_yds_pg": round(rush_yds_pg, 2),
        "pass_td_pg": round(pass_td_pg, 2),
        "rush_td_pg": round(rush_td_pg, 2),
        "sacks_allowed_pg": round(sacks_allowed_pg, 2),
        "stats_source": "derived_from_yards" if _using_derived_stats else "cached",
    }

    return round(total_score, 2), details


# ==============================================================================
# COMPONENT 5: PLAYER READINESS SCORE
# ==============================================================================

def calculate_player_readiness_score(
        player_id: str,
        position: str,
        season: int,
        player_metadata: Dict,
        prev_usage: Dict,
        is_drafted_rookie: bool = False,
        draft_capital: Optional[Dict] = None,
        injury_status: Optional[str] = None,
        injury_history: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Score (0-100) based on player's ability to capitalize on opportunity.

    Args:
        injury_status: Current injury designation ('healthy', 'questionable',
                       'doubtful', 'out', 'ir', 'pup'). Applies immediate penalty.
        injury_history: Dict with 'games_missed_last_season' (int) and optional
                        'chronic' (bool). Applies history-based discount.
    """
    age = _safe_float(player_metadata.get("age", 25), 25.0)
    years_exp = _safe_int(player_metadata.get("years_exp", 0), 0)
    prev_usage = prev_usage or {}

    if years_exp == 1:
        exp_score = SECOND_YEAR_SCORE
    elif years_exp == 2:
        exp_score = THIRD_YEAR_SCORE
    elif years_exp in [3, 4]:
        exp_score = YEAR_3_4_SCORE
    elif years_exp == 0:
        exp_score = ROOKIE_SCORE
    elif years_exp >= 5 and age < VETERAN_AGE_THRESHOLD:
        exp_score = 15
    else:
        exp_score = VETERAN_SCORE

    efficiency_score = 0.0
    raw_efficiency_score = 0.0
    sample_multiplier = 1.0
    # Position-specific adjustments (set below)
    wr_fp_penalty = 0.0
    skill_lift = 0.0

    if position == "WR":
        yards_per_target = _safe_float(prev_usage.get("yards_per_target", 0))
        catch_rate = _safe_float(prev_usage.get("catch_rate", 0))
        prev_targets = _safe_float(prev_usage.get("targets", 0))

        if yards_per_target >= WR_ELITE_YARDS_PER_TARGET:
            ypt_score = EFFICIENCY_YPT_MAX
        elif yards_per_target >= WR_GOOD_YARDS_PER_TARGET:
            ypt_score = 15
        elif yards_per_target >= WR_AVERAGE_YARDS_PER_TARGET:
            ypt_score = 10
        else:
            ypt_score = 5

        if catch_rate >= WR_ELITE_CATCH_RATE:
            cr_score = EFFICIENCY_CATCH_RATE_MAX
        elif catch_rate >= WR_GOOD_CATCH_RATE:
            cr_score = 10
        elif catch_rate >= WR_AVERAGE_CATCH_RATE:
            cr_score = 5
        else:
            cr_score = 0

        raw_efficiency_score = ypt_score + cr_score
        sample_multiplier = _sample_confidence(prev_targets, full_confidence=50, min_confidence=0.35)
        efficiency_score = raw_efficiency_score * sample_multiplier

        # ── WR false-positive penalty: contested-catch profiles ─────────────
        # High draft capital + poor efficiency proxy = overrated prospect
        if (
            prev_targets >= WR_FP_MIN_TARGETS
            and catch_rate < WR_FP_CATCH_RATE_THRESHOLD
            and yards_per_target < WR_FP_YPT_THRESHOLD
            and not is_drafted_rookie   # only penalize non-rookies with real data
        ):
            draft_round = _safe_int((draft_capital or {}).get("round", 5), 5)
            if draft_round == 1:
                wr_fp_penalty = WR_FP_PENALTY_R1
            elif draft_round == 2:
                wr_fp_penalty = WR_FP_PENALTY_R2
            else:
                wr_fp_penalty = WR_FP_PENALTY_OTHER

        # ── Skill-over-draft lift: Day-2/3 WRs with elite efficiency ────────
        # Captures Kupp/Nacua archetypes that routinely beat their draft slot
        if (
            not is_drafted_rookie
            and prev_targets >= WR_SKILL_LIFT_MIN_TARGETS
            and yards_per_target >= WR_SKILL_LIFT_YPT_THRESHOLD
            and catch_rate >= WR_SKILL_LIFT_CATCH_THRESHOLD
        ):
            draft_round = _safe_int((draft_capital or {}).get("round", 5), 5)
            if draft_round == 1:
                skill_lift = WR_SKILL_LIFT_R1
            elif draft_round == 2:
                skill_lift = WR_SKILL_LIFT_R2
            elif draft_round <= 4:
                skill_lift = WR_SKILL_LIFT_R3_R4
            else:
                skill_lift = WR_SKILL_LIFT_UDFA

    elif position == "TE":
        yards_per_target = _safe_float(prev_usage.get("yards_per_target", 0))
        catch_rate = _safe_float(prev_usage.get("catch_rate", 0))
        prev_targets = _safe_float(prev_usage.get("targets", 0))

        # TE uses slightly higher efficiency thresholds and more shrinkage toward mean
        if yards_per_target >= TE_ELITE_YPT:
            ypt_score = EFFICIENCY_YPT_MAX
        elif yards_per_target >= TE_GOOD_YPT:
            ypt_score = 14
        elif yards_per_target >= WR_AVERAGE_YARDS_PER_TARGET:
            ypt_score = 9
        else:
            ypt_score = 4

        if catch_rate >= TE_ELITE_CATCH_RATE:
            cr_score = EFFICIENCY_CATCH_RATE_MAX
        elif catch_rate >= TE_GOOD_CATCH_RATE:
            cr_score = 9
        elif catch_rate >= WR_AVERAGE_CATCH_RATE:
            cr_score = 4
        else:
            cr_score = 0

        raw_efficiency_score = ypt_score + cr_score
        # Higher min_confidence shrinks TE scores toward mean - stabilizes outliers
        sample_multiplier = _sample_confidence(
            prev_targets, full_confidence=50, min_confidence=TE_SAMPLE_MIN_CONFIDENCE
        )
        efficiency_score = raw_efficiency_score * sample_multiplier

    elif position == "RB":
        yards_per_carry = _safe_float(prev_usage.get("yards_per_carry", 0))
        yards_per_target = _safe_float(prev_usage.get("yards_per_target", 0))
        prev_carries = _safe_float(prev_usage.get("carries", 0))
        prev_targets = _safe_float(prev_usage.get("targets", 0))

        if yards_per_carry >= RB_ELITE_YARDS_PER_CARRY:
            ypc_score = EFFICIENCY_YPC_MAX
        elif yards_per_carry >= RB_GOOD_YARDS_PER_CARRY:
            ypc_score = 15
        elif yards_per_carry >= RB_AVERAGE_YARDS_PER_CARRY:
            ypc_score = 10
        else:
            ypc_score = 5

        if yards_per_target >= RB_ELITE_YARDS_PER_TARGET:
            rec_score = EFFICIENCY_RECEIVING_RB_MAX
        elif yards_per_target >= RB_GOOD_YARDS_PER_TARGET:
            rec_score = 10
        elif yards_per_target > 0:
            rec_score = 5
        else:
            rec_score = 0

        raw_efficiency_score = ypc_score + rec_score
        carry_multiplier = _sample_confidence(prev_carries, full_confidence=80, min_confidence=0.35)
        target_multiplier = _sample_confidence(prev_targets, full_confidence=25, min_confidence=0.35)
        sample_multiplier = _weighted_average([(carry_multiplier, 0.7), (target_multiplier, 0.3)])
        efficiency_score = raw_efficiency_score * sample_multiplier

        # ── Skill-over-draft lift: Day-2/3 RBs with elite efficiency ────────
        if (
            not is_drafted_rookie
            and prev_carries >= RB_SKILL_LIFT_MIN_CARRIES
            and yards_per_carry >= RB_SKILL_LIFT_YPC_THRESHOLD
        ):
            draft_round = _safe_int((draft_capital or {}).get("round", 5), 5)
            if draft_round == 2:
                skill_lift = RB_SKILL_LIFT_R2
            elif draft_round >= 3:
                skill_lift = RB_SKILL_LIFT_R3_R4

    elif position == "QB":
        pass_td_rate = _safe_float(prev_usage.get("pass_td_rate", 0))
        yards_per_attempt = _safe_float(prev_usage.get("yards_per_attempt", 0))
        pass_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)), 0)

        ypa_score = _score_bucket(
            yards_per_attempt,
            [(8.0, 18), (7.2, 14), (6.5, 9), (5.8, 4)],
            default=0,
        )
        td_rate_score = _score_bucket(
            pass_td_rate,
            [(0.055, 17), (0.045, 12), (0.035, 8), (0.025, 4)],
            default=0,
        )

        raw_efficiency_score = ypa_score + td_rate_score
        sample_multiplier = _sample_confidence(pass_attempts, full_confidence=250, min_confidence=0.40)
        efficiency_score = raw_efficiency_score * sample_multiplier

    if is_drafted_rookie and draft_capital:
        round_num = _safe_int(draft_capital.get("round", 7), 7)

        if round_num == 1:
            draft_score = DRAFT_CAPITAL_ROUND_1
        elif round_num == 2:
            draft_score = DRAFT_CAPITAL_ROUND_2
        elif round_num == 3:
            draft_score = DRAFT_CAPITAL_ROUND_3
        elif round_num <= 5:
            draft_score = DRAFT_CAPITAL_ROUND_4_5
        else:
            draft_score = 0

        usage_baseline_score = 0.0
    else:
        prev_targets = _safe_float(prev_usage.get("targets", 0))
        prev_carries = _safe_float(prev_usage.get("carries", 0))
        prev_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)), 0)

        if position in ["WR", "TE"]:
            if prev_targets >= WR_ESTABLISHED_TARGETS:
                usage_baseline_score = ESTABLISHED_USAGE_SCORE
            elif prev_targets >= WR_BACKUP_TARGETS:
                usage_baseline_score = BACKUP_USAGE_SCORE
            elif prev_targets >= WR_ROTATION_TARGETS:
                usage_baseline_score = ROTATION_USAGE_SCORE
            elif prev_targets > 0:
                usage_baseline_score = MINIMAL_USAGE_SCORE
            else:
                usage_baseline_score = 0.0

        elif position == "RB":
            if prev_carries >= RB_ESTABLISHED_CARRIES:
                usage_baseline_score = ESTABLISHED_USAGE_SCORE
            elif prev_carries >= RB_BACKUP_CARRIES:
                usage_baseline_score = BACKUP_USAGE_SCORE
            elif prev_carries >= RB_ROTATION_CARRIES:
                usage_baseline_score = ROTATION_USAGE_SCORE
            elif prev_carries > 0:
                usage_baseline_score = MINIMAL_USAGE_SCORE
            else:
                usage_baseline_score = 0.0

        elif position == "QB":
            if prev_attempts >= 450:
                usage_baseline_score = ESTABLISHED_USAGE_SCORE
            elif prev_attempts >= 250:
                usage_baseline_score = BACKUP_USAGE_SCORE
            elif prev_attempts >= 100:
                usage_baseline_score = ROTATION_USAGE_SCORE
            elif prev_attempts > 0:
                usage_baseline_score = MINIMAL_USAGE_SCORE
            else:
                usage_baseline_score = 0.0
        else:
            usage_baseline_score = 0.0

        draft_score = 0.0

    base_score = (
        exp_score + efficiency_score + draft_score
        if is_drafted_rookie
        else exp_score + efficiency_score + usage_baseline_score
    )

    # Apply position-specific adjustments
    base_score += skill_lift + wr_fp_penalty

    # --- Injury status modifier ---
    injury_status_penalty = 0.0
    injury_history_penalty = 0.0
    injury_status_used = 'healthy'

    if injury_status:
        normalized = injury_status.lower().strip()
        injury_status_used = normalized
        injury_status_penalty = _safe_float(
            INJURY_STATUS_PENALTIES.get(normalized, 0), 0.0
        )

    if injury_history:
        games_missed = _safe_int(injury_history.get('games_missed_last_season', 0), 0)
        if games_missed >= INJURY_HISTORY_GAMES_MISSED_SEVERE:
            injury_history_penalty = INJURY_HISTORY_SEVERE_PENALTY
        elif games_missed >= INJURY_HISTORY_GAMES_MISSED_MODERATE:
            injury_history_penalty = INJURY_HISTORY_MODERATE_PENALTY
        if injury_history.get('chronic'):
            injury_history_penalty = min(injury_history_penalty - 5, -5)

    total_score = _clamp(
        base_score + injury_status_penalty + injury_history_penalty,
        0.0, 100.0
    )

    details = {
        "player_id": player_id,
        "position": position,
        "season": season,
        "age": age,
        "years_exp": years_exp,
        "exp_score": exp_score,
        "raw_efficiency_score": round(raw_efficiency_score, 2),
        "efficiency_score": round(efficiency_score, 2),
        "efficiency_sample_multiplier": round(sample_multiplier, 3),
        "draft_score": round(draft_score, 2) if is_drafted_rookie else None,
        "usage_baseline_score": round(usage_baseline_score, 2) if not is_drafted_rookie else None,
        "is_rookie": is_drafted_rookie,
        "skill_lift": round(skill_lift, 2),
        "contested_catch_penalty": round(wr_fp_penalty, 2),
        "injury_status": injury_status_used,
        "injury_status_penalty": round(injury_status_penalty, 2),
        "injury_history_penalty": round(injury_history_penalty, 2),
    }

    return round(total_score, 2), details


# ==============================================================================
# COMPONENT 6: ROLE TRAJECTORY SCORE
# ==============================================================================

def _offseason_role_trajectory_score(
        player_id: str,
        as_of_date: date,
        prev_usage: Dict,
        current_team: Optional[str],
        position: Optional[str]
) -> Tuple[float, Dict]:
    """
    Offseason trajectory:
    - rewards established prior role
    - discounts tiny priors
    - adds modest team-context bonus
    """
    prev_usage = prev_usage or {}
    position = position or "WR"

    baseline = _build_player_baseline(position, prev_usage)
    stabilized_role = _safe_float(baseline.get("stabilized_role_score", 0), 0)
    prev_snap_share = _safe_float(prev_usage.get("snap_share", 0), 0)
    prev_targets = _safe_float(prev_usage.get("targets", 0), 0)
    prev_carries = _safe_float(prev_usage.get("carries", 0), 0)
    prev_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)), 0)

    if position in ["WR", "TE"]:
        role_size_score = _normalize_range(prev_snap_share, 0.05, 0.80) * 38.0
        usage_strength_score = _normalize_range(prev_targets, 10.0, 110.0) * 32.0
        sample_score = _position_sample_confidence(position, prev_usage) * 10.0
    elif position == "RB":
        total_touches = prev_carries + prev_targets
        role_size_score = _normalize_range(prev_snap_share, 0.05, 0.75) * 36.0
        usage_strength_score = _normalize_range(total_touches, 15.0, 230.0) * 34.0
        sample_score = _position_sample_confidence(position, prev_usage) * 10.0
    elif position == "QB":
        role_size_score = _normalize_range(prev_snap_share, 0.05, 0.95) * 40.0
        usage_strength_score = _normalize_range(prev_attempts, 25.0, 575.0) * 30.0
        sample_score = _position_sample_confidence(position, prev_usage) * 10.0
    else:
        role_size_score = _normalize_range(prev_snap_share, 0.05, 0.80) * 35.0
        usage_strength_score = 20.0
        sample_score = 8.0

    team_bonus = 0.0
    team_tier = "unknown"

    if current_team:
        try:
            team_env_score, team_env_details = calculate_team_environment_score(
                current_team,
                position,
                as_of_date.year
            )
            if team_env_score >= 75:
                team_bonus = 20.0
                team_tier = "elite"
            elif team_env_score >= 60:
                team_bonus = 15.0
                team_tier = "good"
            elif team_env_score >= 45:
                team_bonus = 10.0
                team_tier = "average"
            else:
                team_bonus = 5.0
                team_tier = "poor"
        except Exception:
            team_env_score = None
            team_env_details = {}
            team_bonus = 8.0
            team_tier = "unknown"
    else:
        team_env_score = None
        team_env_details = {}
        team_bonus = 8.0
        team_tier = "unknown"

    total_score = _clamp(
        role_size_score + usage_strength_score + sample_score + team_bonus,
        0.0,
        100.0,
    )

    details = {
        "player_id": player_id,
        "phase_mode": "offseason",
        "position": position,
        "prev_snap_share": round(prev_snap_share, 3),
        "prev_targets": round(prev_targets, 1),
        "prev_carries": round(prev_carries, 1),
        "prev_pass_attempts": round(prev_attempts, 1),
        "stabilized_role_score": round(stabilized_role, 2),
        "role_size_score": round(role_size_score, 2),
        "usage_strength_score": round(usage_strength_score, 2),
        "sample_score": round(sample_score, 2),
        "team_bonus": round(team_bonus, 2),
        "team_tier": team_tier,
        "team_environment_score": round(team_env_score, 2) if team_env_score is not None else None,
        "team_environment_details": team_env_details if current_team else {},
    }

    return round(total_score, 2), details


def _inseason_role_trajectory_score(
        player_id: str,
        as_of_date: date,
        lookback_days: int
) -> Tuple[float, Dict]:
    """
    In-season trajectory:
    compares recent window vs previous window using stabilized deltas.
    """
    current_metrics = get_player_advanced_metrics(player_id, as_of_date, lookback_days)
    previous_date = as_of_date - timedelta(days=lookback_days)
    previous_metrics = get_player_advanced_metrics(player_id, previous_date, lookback_days)

    if not current_metrics or not previous_metrics:
        return OFFSEASON_NEUTRAL_SCORE, {
            "player_id": player_id,
            "phase_mode": "in_season",
            "note": "Insufficient data, neutral score",
            "lookback_days": lookback_days,
        }

    curr_snap = _safe_float(current_metrics.get("snap_share", 0))
    prev_snap = _safe_float(previous_metrics.get("snap_share", 0))

    curr_opp = _safe_float(current_metrics.get("opportunity_share", 0))
    prev_opp = _safe_float(previous_metrics.get("opportunity_share", 0))

    curr_rz = _safe_float(current_metrics.get("red_zone_usage", 0))
    prev_rz = _safe_float(previous_metrics.get("red_zone_usage", 0))

    curr_role = _safe_float(current_metrics.get("role_score", 0))
    prev_role = _safe_float(previous_metrics.get("role_score", 0))

    curr_sample = max(
        _safe_float(current_metrics.get("sample_size", 0)),
        _safe_float(current_metrics.get("snaps", 0)),
        _safe_float(current_metrics.get("opportunities", 0)),
        _safe_float(current_metrics.get("targets", 0)) + _safe_float(current_metrics.get("carries", 0)),
    )
    prev_sample = max(
        _safe_float(previous_metrics.get("sample_size", 0)),
        _safe_float(previous_metrics.get("snaps", 0)),
        _safe_float(previous_metrics.get("opportunities", 0)),
        _safe_float(previous_metrics.get("targets", 0)) + _safe_float(previous_metrics.get("carries", 0)),
    )

    window_confidence = _weighted_average([
        (_sample_confidence(curr_sample, full_confidence=30, min_confidence=0.45), 0.55),
        (_sample_confidence(prev_sample, full_confidence=30, min_confidence=0.45), 0.45),
    ])

    snap_delta = curr_snap - prev_snap
    opp_delta = curr_opp - prev_opp
    rz_delta = curr_rz - prev_rz
    role_delta = curr_role - prev_role

    snap_score = _normalize_range(snap_delta, -0.15, 0.25) * 28.0
    opp_score = _normalize_range(opp_delta, -0.12, 0.22) * 34.0
    rz_score = _normalize_range(rz_delta, -0.08, 0.18) * 18.0
    role_score = _normalize_range(role_delta, -10.0, 25.0) * 15.0

    raw_total = snap_score + opp_score + rz_score + role_score
    stabilized_total = raw_total * window_confidence

    # Small boost if both snap share and opp share are moving up together
    synergy_bonus = 0.0
    if snap_delta > 0.05 and opp_delta > 0.04:
        synergy_bonus += 5.0
    elif snap_delta > 0.02 and opp_delta > 0.02:
        synergy_bonus += 2.5

    total_score = _clamp(stabilized_total + synergy_bonus, 0.0, 100.0)

    details = {
        "player_id": player_id,
        "phase_mode": "in_season",
        "lookback_days": lookback_days,
        "window_confidence": round(window_confidence, 3),
        "curr_sample": round(curr_sample, 1),
        "prev_sample": round(prev_sample, 1),
        "curr_snap_share": round(curr_snap, 3),
        "prev_snap_share": round(prev_snap, 3),
        "curr_opportunity_share": round(curr_opp, 3),
        "prev_opportunity_share": round(prev_opp, 3),
        "curr_red_zone_usage": round(curr_rz, 3),
        "prev_red_zone_usage": round(prev_rz, 3),
        "curr_role_score_metric": round(curr_role, 2),
        "prev_role_score_metric": round(prev_role, 2),
        "snap_delta": round(snap_delta, 3),
        "opp_delta": round(opp_delta, 3),
        "rz_delta": round(rz_delta, 3),
        "role_delta": round(role_delta, 2),
        "snap_score": round(snap_score, 2),
        "opp_score": round(opp_score, 2),
        "rz_score": round(rz_score, 2),
        "role_score_component": round(role_score, 2),
        "raw_total": round(raw_total, 2),
        "stabilized_total": round(stabilized_total, 2),
        "synergy_bonus": round(synergy_bonus, 2),
    }

    return round(total_score, 2), details


def calculate_role_trajectory_score(
        player_id: str,
        as_of_date: date,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        phase: str = "in_season",
        prev_usage: Dict = None,
        current_team: str = None,
        position: str = None
) -> Tuple[float, Dict]:
    """
    Score (0-100) based on role trajectory.

    Offseason:
    - prior role strength
    - prior usage foothold
    - sample stability
    - team environment support

    In-season:
    - recent-vs-previous trend in snap share, opportunity share,
      red-zone usage, and role score
    - discounts noisy small-sample spikes
    """
    if phase in ["offseason", "post_free_agency", "post_draft", "preseason"]:
        if not prev_usage:
            return 28.0, {
                "player_id": player_id,
                "phase_mode": "offseason",
                "note": "No previous season data, low baseline",
            }
        return _offseason_role_trajectory_score(
            player_id=player_id,
            as_of_date=as_of_date,
            prev_usage=prev_usage,
            current_team=current_team,
            position=position,
        )

    return _inseason_role_trajectory_score(
        player_id=player_id,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
    )


# ==============================================================================
# COMPONENT 7: CONFIDENCE SCORE
# ==============================================================================

def calculate_confidence_score(
        player_id: str,
        prev_usage: Dict,
        phase: str,
        data_quality_metrics: Dict
) -> Tuple[float, Dict]:
    """
    Score (0-100) indicating confidence in the projection.
    """
    prev_usage = prev_usage or {}
    data_quality_metrics = data_quality_metrics or {}

    games_played = _safe_float(prev_usage.get("games", 0))
    total_touches = _safe_float(prev_usage.get("targets", 0)) + _safe_float(prev_usage.get("carries", 0))
    pass_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)), 0)
    total_volume = max(total_touches, pass_attempts)

    if games_played >= FULL_SEASON_GAMES and total_volume >= FULL_SEASON_TOUCHES:
        sample_score = SAMPLE_FULL_SCORE
    elif games_played >= HALF_SEASON_GAMES and total_volume >= HALF_SEASON_TOUCHES:
        sample_score = SAMPLE_HALF_SCORE
    elif games_played >= QUARTER_SEASON_GAMES and total_volume >= QUARTER_SEASON_TOUCHES:
        sample_score = SAMPLE_QUARTER_SCORE
    elif games_played > 0:
        sample_score = SAMPLE_MINIMAL_SCORE
    else:
        sample_score = SAMPLE_ROOKIE_SCORE

    completeness = 0.0
    if data_quality_metrics.get("has_efficiency_data"):
        completeness += HAS_EFFICIENCY_DATA_SCORE
    if data_quality_metrics.get("has_advanced_metrics"):
        completeness += HAS_ADVANCED_METRICS_SCORE
    if data_quality_metrics.get("has_usage_history"):
        completeness += HAS_USAGE_HISTORY_SCORE

    usage_variance = _safe_float(data_quality_metrics.get("usage_variance", 1.0), 1.0)
    if usage_variance < VERY_CONSISTENT_VARIANCE:
        consistency_score = CONSISTENCY_HIGH_SCORE
    elif usage_variance < CONSISTENT_VARIANCE:
        consistency_score = CONSISTENCY_GOOD_SCORE
    elif usage_variance < MODERATE_VARIANCE:
        consistency_score = CONSISTENCY_MODERATE_SCORE
    else:
        consistency_score = CONSISTENCY_LOW_SCORE

    phase_score = PHASE_CERTAINTY.get(phase, 5)

    total_score = min(sample_score + completeness + consistency_score + phase_score, 100.0)

    details = {
        "player_id": player_id,
        "sample_score": sample_score,
        "completeness": completeness,
        "consistency_score": consistency_score,
        "phase_score": phase_score,
        "games_played": round(games_played, 1),
        "total_touches": round(total_touches, 1),
        "pass_attempts": round(pass_attempts, 1),
        "total_volume": round(total_volume, 1),
        "usage_variance": round(usage_variance, 3),
    }

    return round(total_score, 2), details
