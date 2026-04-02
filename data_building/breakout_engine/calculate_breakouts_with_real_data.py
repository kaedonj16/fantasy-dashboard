#!/usr/bin/env python3
"""
Calculate breakout scores using real player age data from usage_table.

This version:
- uses real player age data from usage_table
- filters out already-established stars
- filters out players with almost no prior foothold unless they qualify
  for a strong-opportunity exception
- removes verbose print output
"""

import os
from datetime import date
from typing import Any, Dict, List, Tuple, Optional

from dashboard_services.service import age_from_bday
from data_building.breakout_engine import BreakoutEngine
from data_building.external_data.player_history import usage_rows_json_path_for_season
from utils.utils import load_players_index, load_usage_table, read_json

# Ensure DATABASE_URL is set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = f"postgresql://{os.environ.get('USER')}@localhost:5432/brfantasy"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_season_aware_usage_data(season: int, week: int = 0, season_type: str = 'off') -> List[Dict]:
    """
    Load appropriate usage data based on season phase.

    - Offseason: Use last season's stats
    - Early season (weeks 1-4): Use both last season and current season (blended)
    - Mid/late season (week 5+): Use current season stats only

    Args:
        season: Current season year
        week: Current week number
        season_type: Season type ('off', 'pre', 'regular', 'post')

    Returns:
        List of player usage dictionaries
    """
    import json
    import os

    season_type = season_type.lower().strip()
    is_offseason = season_type in ['off', 'pre']
    is_early_season = season_type == 'regular' and week <= 4

    # In offseason/preseason: use last season's data
    if is_offseason:
        last_season_file = f"cache/player_history/usage_rows_{season - 1}.json"

        if os.path.exists(last_season_file):
            print(f"[season-aware] Offseason detected - loading {season - 1} usage data")
            return read_json(usage_rows_json_path_for_season(season-1))
        else:
            print(f"[season-aware] Warning: {last_season_file} not found")
            return []

    # Early season (weeks 1-4): blend last season with current season
    elif is_early_season:
        print(f"[season-aware] Early season (week {week}) - blending {season - 1} and {season} data")

        # Load current season (sparse data)
        current_usage = load_usage_table() or []

        # Load last season (full data)
        last_season_file = f"cache/player_history/usage_rows_{season - 1}.json"
        if os.path.exists(last_season_file):
            with open(last_season_file, 'r') as f:
                last_season_usage = read_json(usage_rows_json_path_for_season(season - 1))
        else:
            last_season_usage = []

        # Merge: prefer current season if player has games, otherwise use last season
        merged_by_id = {}

        # Start with last season as baseline
        for player in last_season_usage:
            player_id = str(player.get('id'))
            if player_id:
                merged_by_id[player_id] = player

        # Overlay current season data (higher priority)
        for player in current_usage:
            player_id = str(player.get('id'))
            games = player.get('usage', {}).get('games', 0)

            if player_id and games > 0:
                merged_by_id[player_id] = player

        print(f"[season-aware] Blended {len(merged_by_id)} players (last season baseline + current season overlay)")
        return list(merged_by_id.values())

    # Mid/late season: use current season only
    else:
        print(f"[season-aware] Mid/late season (week {week}) - using {season} current data only")
        return load_usage_table() or []


def build_usage_maps(usage_table: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    usage_by_id: Dict[str, Dict] = {}
    age_by_id: Dict[str, float] = {}

    for player in usage_table:
        player_id = player.get("id")
        if not player_id:
            continue

        pid = str(player_id)
        usage_by_id[pid] = player

        age = player.get("age")
        if age not in (None, ""):
            try:
                age_by_id[pid] = float(age)
            except (TypeError, ValueError):
                pass

    return usage_by_id, age_by_id


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_position(position: Optional[str], payload: Dict[str, Any]) -> Optional[str]:
    """
    Resolve position from:
    1. explicit function arg
    2. top-level payload["position"]
    3. nested payload["usage"]["position"] if ever present
    """
    if position:
        return str(position).upper()

    top_level_pos = payload.get("position")
    if top_level_pos:
        return str(top_level_pos).upper()

    usage = payload.get("usage") or {}
    usage_pos = usage.get("position")
    if usage_pos:
        return str(usage_pos).upper()

    return None


def _extract_usage_metrics(payload_or_usage: Dict[str, Any]) -> Dict[str, float]:
    """
    Accepts either:
    - full player payload with nested 'usage'
    - a direct usage dict
    - an already-normalized dict

    Returns a normalized usage view with keys:
    - games
    - snap_share
    - targets
    - carries
    - routes
    - target_share
    - pass_attempts
    - fantasy_points
    - ppg
    """
    if not payload_or_usage:
        return {
            "games": 0.0,
            "snap_share": 0.0,
            "targets": 0.0,
            "carries": 0.0,
            "routes": 0.0,
            "target_share": 0.0,
            "pass_attempts": 0.0,
            "fantasy_points": 0.0,
            "ppg": 0.0,
        }

    # If this is a full payload, usage is nested
    usage = payload_or_usage.get("usage") if isinstance(payload_or_usage, dict) else None
    if isinstance(usage, dict):
        source = usage
    else:
        source = payload_or_usage

    games = _safe_float(source.get("games", 0))

    # snap share:
    # prefer direct normalized keys if present, otherwise use avg_off_snap_pct
    snap_share = _safe_float(
        source.get(
            "snap_share",
            source.get("avg_off_snap_pct", 0)
        )
    )

    # targets:
    # prefer totals if populated, otherwise derive from avg * games
    total_targets = _safe_float(source.get("targets", source.get("total_targets", 0)))
    avg_targets = _safe_float(source.get("avg_targets", 0))
    targets = total_targets if total_targets > 0 else (avg_targets * games)

    # carries:
    total_carries = _safe_float(source.get("carries", source.get("total_carries", 0)))
    avg_carries = _safe_float(source.get("avg_carries", 0))
    carries = total_carries if total_carries > 0 else (avg_carries * games)

    # pass attempts:
    total_pass_attempts = _safe_float(
        source.get("pass_attempts", source.get("attempts", source.get("total_pass_att", 0)))
    )
    avg_pass_attempts = _safe_float(source.get("avg_pass_att", 0))
    pass_attempts = total_pass_attempts if total_pass_attempts > 0 else (avg_pass_attempts * games)

    # routes:
    # not present in your sample payload, so default to 0 unless you add it upstream
    routes = _safe_float(source.get("routes", source.get("total_routes", 0)))

    # target share:
    target_share = _safe_float(source.get("target_share", 0))

    # fantasy:
    # your payload has PPR points per game, not season total
    ppg = _safe_float(
        source.get("ppr_ppg", source.get("ppg", source.get("fantasy_ppg", 0)))
    )

    fantasy_points = _safe_float(source.get("fantasy_points", 0))
    if fantasy_points <= 0 and games > 0 and ppg > 0:
        fantasy_points = ppg * games

    return {
        "games": games,
        "snap_share": snap_share,
        "targets": targets,
        "carries": carries,
        "routes": routes,
        "target_share": target_share,
        "pass_attempts": pass_attempts,
        "fantasy_points": fantasy_points,
        "ppg": ppg,
    }


def _is_established_star(position: Optional[str], prev_usage: Dict) -> bool:
    """
    Hard exclude true established players.

    Works with either:
    - normalized usage dict
    - full player payload containing nested usage
    """
    prev_usage = prev_usage or {}
    position = _resolve_position(position, prev_usage)
    metrics = _extract_usage_metrics(prev_usage)

    if not position:
        return False

    snap_share = metrics["snap_share"]
    targets = metrics["targets"]
    carries = metrics["carries"]
    routes = metrics["routes"]
    target_share = metrics["target_share"]
    pass_attempts = metrics["pass_attempts"]
    fantasy_points = metrics["fantasy_points"]
    games = max(metrics["games"], 0.0)
    ppg = fantasy_points / games if games > 0 else metrics["ppg"]

    if position == "RB":
        touches = carries + targets
        return (
            ppg >= 15.0 or
            touches >= 240 or
            (snap_share >= 0.62 and touches >= 180)
        )

    if position == "WR":
        return (
            ppg >= 14.5 or
            targets >= 120 or
            target_share >= 0.25 or
            (snap_share >= 0.80 and routes >= 475 and targets >= 100)
        )

    if position == "TE":
        return (
            ppg >= 11.5 or
            targets >= 100 or
            target_share >= 0.23
        )

    if position == "QB":
        return (
            ppg >= 18.0 or
            pass_attempts >= 525 or
            snap_share >= 0.90
        )

    return False


def _is_true_dust(position: Optional[str], prev_usage: Dict, flag: bool, player_name: str = "Unknown") -> bool:
    """
    Hard exclude only the absolute no-role players.

    Works with either:
    - normalized usage dict
    - full player payload containing nested usage
    """
    prev_usage = prev_usage or {}
    position = _resolve_position(position, prev_usage)
    metrics = _extract_usage_metrics(prev_usage)

    if not position:
        return False

    snap_share = metrics["snap_share"]
    targets = metrics["targets"]
    carries = metrics["carries"]
    routes = metrics["routes"]
    pass_attempts = metrics["pass_attempts"]

    if position in ["WR", "TE"]:
        return targets < 20

    if position == "RB":
        return (carries + targets) < 30

    if position == "QB":
        return pass_attempts < 25

    return False

def _candidate_band_multiplier(position: str, prev_usage: Dict, candidate: Any) -> Tuple[float, str]:
    """
    Soft weighting instead of overusing hard exclusions.

    Returns:
        multiplier, status
    """
    prev_usage = prev_usage or {}

    targets = _safe_float(prev_usage.get("targets", 0))
    carries = _safe_float(prev_usage.get("carries", 0))
    routes = _safe_float(prev_usage.get("routes", 0))
    snap_share = _safe_float(prev_usage.get("snap_share", 0))
    pass_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)))

    opp_opened = _safe_float(getattr(candidate, "opportunity_opened_score", 0))
    comp_removed = _safe_float(getattr(candidate, "competition_removed_score", 0))
    comp_added = _safe_float(getattr(candidate, "competition_added_penalty", 0))
    readiness = _safe_float(getattr(candidate, "player_readiness_score", 0))
    trajectory = _safe_float(getattr(candidate, "role_trajectory_score", 0))

    situation_bonus = 0.0
    if opp_opened >= 70:
        situation_bonus += 0.10
    if comp_removed >= 20:
        situation_bonus += 0.10
    if comp_added > -10:
        situation_bonus += 0.05
    if readiness >= 40:
        situation_bonus += 0.05
    if trajectory >= 40:
        situation_bonus += 0.05

    if position in ["WR", "TE"]:
        # Ideal band is not 10 targets, but also not 120+
        if targets < 15 and routes < 80 and snap_share < 0.12:
            mult, status = 0.20, "longshot"
        elif targets < 30:
            mult, status = 0.65, "viable_small_role"
        elif targets < 70:
            mult, status = 1.00, "ideal_breakout_band"
        elif targets < 100:
            mult, status = 0.82, "near_established"
        else:
            mult, status = 0.45, "too_established"

    elif position == "RB":
        touches = carries + targets
        if touches < 25 and snap_share < 0.12:
            mult, status = 0.20, "longshot"
        elif touches < 60:
            mult, status = 0.65, "viable_small_role"
        elif touches < 160:
            mult, status = 1.00, "ideal_breakout_band"
        elif touches < 220:
            mult, status = 0.82, "near_established"
        else:
            mult, status = 0.45, "too_established"

    elif position == "QB":
        if pass_attempts < 40 and snap_share < 0.10:
            mult, status = 0.20, "longshot"
        elif pass_attempts < 150:
            mult, status = 0.65, "viable_small_role"
        elif pass_attempts < 400:
            mult, status = 1.00, "ideal_breakout_band"
        elif pass_attempts < 500:
            mult, status = 0.82, "near_established"
        else:
            mult, status = 0.45, "too_established"
    else:
        mult, status = 1.0, "eligible"

    mult = min(mult + situation_bonus, 1.10)
    return round(mult, 2), status

def _qualifies_opportunity_exception(candidate: Any) -> bool:
    """
    Keeps rare low-foothold players if the situation is unusually strong.
    """
    opportunity_opened_score = _safe_float(getattr(candidate, "opportunity_opened_score", 0))
    competition_removed_score = _safe_float(getattr(candidate, "competition_removed_score", 0))
    competition_added_penalty = _safe_float(getattr(candidate, "competition_added_penalty", 0))
    role_trajectory_score = _safe_float(getattr(candidate, "role_trajectory_score", 0))
    player_readiness_score = _safe_float(getattr(candidate, "player_readiness_score", 0))

    return (
        opportunity_opened_score >= 75 and
        competition_removed_score >= 20 and
        competition_added_penalty > -8 and
        (role_trajectory_score >= 35 or player_readiness_score >= 35)
    )


def _candidate_multiplier(position: str, prev_usage: Dict) -> float:
    """
    Goldilocks multiplier:
    - too little role -> downweight
    - ideal breakout band -> full weight
    - near-established -> downweight
    - already-established stars are hard-excluded elsewhere
    """
    prev_usage = prev_usage or {}

    targets = _safe_float(prev_usage.get("targets", 0))
    carries = _safe_float(prev_usage.get("carries", 0))
    pass_attempts = _safe_float(prev_usage.get("pass_attempts", prev_usage.get("attempts", 0)))

    if position in ["WR", "TE"]:
        if targets < 20:
            return 0.25
        if targets < 40:
            return 0.75
        if targets < 90:
            return 1.00
        if targets < 110:
            return 0.70
        return 0.0

    if position == "RB":
        touches = carries + targets
        if touches < 35:
            return 0.25
        if touches < 75:
            return 0.75
        if touches < 180:
            return 1.00
        if touches < 220:
            return 0.70
        return 0.0

    if position == "QB":
        if pass_attempts < 60:
            return 0.25
        if pass_attempts < 180:
            return 0.75
        if pass_attempts < 420:
            return 1.00
        if pass_attempts < 500:
            return 0.70
        return 0.0

    return 1.0


def _classify_candidate(candidate: Any, prev_usage: Dict, flag: bool) -> Tuple[bool, str, float]:
    position = candidate.get("position", None)

    if not position:
        return False, "missing_position", 0.0

    if _is_established_star(position, prev_usage):
        return False, "excluded_star", 0.0

    if _is_true_dust(position, prev_usage, flag, candidate.get("name", "Unknown")):
        return False, "excluded_true_dust", 0.0

    multiplier, status = _candidate_band_multiplier(position, prev_usage, candidate)
    return True, status, multiplier

def apply_candidate_filter(candidates: List[Any], usage_by_id: Dict[str, Dict]) -> Tuple[List[Any], Dict[str, int]]:
    kept: List[Any] = []
    summary = {
        "input_candidates": len(candidates),
        "kept_candidates": 0,
        "excluded_star": 0,
        "excluded_true_dust": 0,
        "excluded_age": 0,
        "excluded_other": 0,
        "ideal_breakout_band": 0,
        "viable_small_role": 0,
        "near_established": 0,
        "longshot": 0,
        "too_established": 0,
    }

    for candidate in candidates:
        player_id = str(candidate.get("player_id", ""))
        prev_usage = usage_by_id.get(player_id, {})
        
        # Age filter: exclude players over 26 except QBs
        position = candidate.get("position", "").upper()
        age = candidate.get("age")
        if age is not None:
            age_float = float(age)
            if position != "QB" and age_float > 26:
                summary["excluded_age"] += 1
                continue

        flag = (player_id == "13789" or player_id == "13079")

        is_eligible, status, multiplier = _classify_candidate(candidate, prev_usage, flag)

        try:
            setattr(candidate, "breakout_candidate_status", status)
            setattr(candidate, "breakout_candidate_multiplier", multiplier)
        except Exception:
            pass

        if not is_eligible:
            if status == "excluded_star":
                summary["excluded_star"] += 1
            elif status == "excluded_true_dust":
                summary["excluded_true_dust"] += 1
            else:
                summary["excluded_other"] += 1
            continue

        try:
            candidate.raw_breakout_opportunity_score = _safe_float(candidate.breakout_opportunity_score)
            candidate.breakout_opportunity_score = round(
                candidate.raw_breakout_opportunity_score * multiplier,
                2
            )
        except Exception:
            pass

        if status in summary:
            summary[status] += 1

        kept.append(candidate)

    kept.sort(key=lambda x: _safe_float(getattr(x, "breakout_opportunity_score", 0)), reverse=True)
    summary["kept_candidates"] = len(kept)
    return kept, summary

def main() -> Dict[str, Any]:
    from dashboard_services.api import get_nfl_state

    # Get current NFL state
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season", 2026))
    week = int(nfl_state.get("week", 0))
    season_type = str(nfl_state.get("season_type", "off"))

    engine = BreakoutEngine(season=season, as_of_date=date.today())

    print(f"[main] Season: {season}, Week: {week}, Type: {season_type}")

    players_index = load_players_index() or {}

    # Load season-aware usage data
    usage_table = load_season_aware_usage_data(season, week, season_type)

    usage_by_id, age_by_id = build_usage_maps(usage_table)

    all_players = []
    for player_id, player_data in players_index.items():
        pos = player_data.get("pos")
        team = player_data.get("team")

        if pos in ["QB", "RB", "WR", "TE"] and team:
            age = age_from_bday(player_data.get("bDay"))

            if age is not None and age < 26:
                years_exp = max(0, int(age - 21.5))

                all_players.append({
                    "player_id": player_id,
                    "player_name": player_data.get("name", "Unknown"),
                    "team": team,
                    "position": pos,
                    "age": age,
                    "years_exp": years_exp,
                })

    filtered_candidates, filter_summary = apply_candidate_filter(all_players, usage_by_id)
    print("Filtered candidates:", len(filtered_candidates))
    print(filter_summary)

    candidates = engine.calculate_breakout_scores(filtered_candidates, min_score=0)

    saved_count = engine.save_scores(candidates)

    return {
        "season": engine.season,
        "phase": engine.phase,
        "as_of_date": engine.as_of_date.isoformat(),
        "players_loaded": len(all_players),
        "raw_candidates": len(candidates),
        "filtered_candidates": len(filtered_candidates),
        "saved_count": saved_count,
        "filter_summary": filter_summary,
    }


if __name__ == "__main__":
    main()