from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from dashboard_services.api import get_tank01_player_gamelogs


def _to_float(x: Optional[str]) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _to_int(x: Optional[str]) -> int:
    try:
        if x is None:
            return 0
        return int(x)
    except (TypeError, ValueError):
        return 0


def _aggregate_player_games(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of Tank01 game objects for one player, compute usage/production metrics.

    This is where you decide what goes into your "usage profile" for the trade model.
    """

    if not games:
        return {
            "games": 0,
            "avg_off_snap_pct": 0.0,
            "avg_off_snaps": 0.0,
            "avg_targets": 0.0,
            "avg_receptions": 0.0,
            "avg_rec_yards": 0.0,
            "avg_rec_tds": 0.0,
            "avg_carries": 0.0,
            "avg_rush_yards": 0.0,
            "avg_rush_tds": 0.0,
            "ppr_ppg": 0.0,
            "half_ppr_ppg": 0.0,
            "std_ppg": 0.0,
        }

    # Sort games by gameID (YYYYMMDD_...) so we can easily take "last N" if needed later
    games_sorted = sorted(games, key=lambda g: g.get("gameID", ""))

    total_off_snap_pct = 0.0
    total_off_snaps = 0

    total_targets = 0
    total_receptions = 0
    total_rec_yards = 0
    total_rec_tds = 0

    total_carries = 0
    total_rush_yards = 0
    total_rush_tds = 0

    ppr_points = []
    half_ppr_points = []
    std_points = []

    counted_games = 0

    for g in games_sorted:
        snap = g.get("snapCounts") or {}
        receiving = g.get("Receiving") or {}
        rushing = g.get("Rushing") or {}
        fantasy_default = g.get("fantasyPointsDefault") or {}
        fantasy_points = g.get("fantasyPoints")  # sometimes this is PPR

        off_snap_pct = _to_float(snap.get("offSnapPct"))
        off_snap = _to_int(snap.get("offSnap"))

        targets = _to_int(receiving.get("targets"))
        receptions = _to_int(receiving.get("receptions"))
        rec_yds = _to_int(receiving.get("recYds"))
        rec_td = _to_int(receiving.get("recTD"))

        carries = _to_int(rushing.get("carries"))
        rush_yds = _to_int(rushing.get("rushYds"))
        rush_td = _to_int(rushing.get("rushTD"))

        std_fpts = _to_float(fantasy_default.get("standard"))
        ppr_fpts = _to_float(fantasy_default.get("PPR"))
        half_ppr_fpts = _to_float(fantasy_default.get("halfPPR"))

        # If fantasyPointsDefault is missing, fall back to "fantasyPoints" (likely PPR)
        if ppr_fpts == 0.0 and fantasy_points is not None:
            ppr_fpts = _to_float(fantasy_points)

        total_off_snap_pct += off_snap_pct
        total_off_snaps += off_snap
        total_targets += targets
        total_receptions += receptions
        total_rec_yards += rec_yds
        total_rec_tds += rec_td
        total_carries += carries
        total_rush_yards += rush_yds
        total_rush_tds += rush_td

        std_points.append(std_fpts)
        ppr_points.append(ppr_fpts)
        half_ppr_points.append(half_ppr_fpts)

        counted_games += 1

    if counted_games == 0:
        counted_games = 1  # avoid division by zero

    def _avg(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _std(values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return var ** 0.5

    metrics = {
        "games": counted_games,
        "avg_off_snap_pct": total_off_snap_pct / counted_games,  # 0–1
        "avg_off_snaps": total_off_snaps / counted_games,
        "avg_targets": total_targets / counted_games,
        "avg_receptions": total_receptions / counted_games,
        "avg_rec_yards": total_rec_yards / counted_games,
        "avg_rec_tds": total_rec_tds / counted_games,
        "avg_carries": total_carries / counted_games,
        "avg_rush_yards": total_rush_yards / counted_games,
        "avg_rush_tds": total_rush_tds / counted_games,
        "std_ppg": _std(ppr_points),  # volatility in PPR scoring
        "ppr_ppg": _avg(ppr_points),
        "half_ppr_ppg": _avg(half_ppr_points),
        "std_scoring_ppg": _avg(std_points),
    }

    return metrics


def build_tank01_usage_map_from_players_index(
        players_index: Dict[str, Dict[str, Any]],
        *,
        season: Optional[int] = None,
        per_name: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    High-level service:

    - Takes your players_index:
        {
          "8150": {
              "name": "Kyren Williams",
              "team": "LAR",
              "tankId": "4430737",
              "byeWeek": 8
          },
          ...
        }

    - For each player with a tankId, calls Tank01
    - Aggregates game logs into usage metrics
    - Returns a map:

        if per_name=True:
            { "Kyren Williams": {metrics...}, ... }

        else:
            { "8150": { "name": "Kyren Williams", "tankId": "4430737", "metrics": {...} }, ... }
    """

    result: Dict[str, Dict[str, Any]] = {}

    for sleeper_id, meta in players_index.items():
        tank_id = meta.get("tankId")
        name = meta.get("name") or f"Player {sleeper_id}"

        if not tank_id:
            continue

        try:
            games = get_tank01_player_gamelogs(str(tank_id), season=season)
            metrics = _aggregate_player_games(games)
        except Exception as e:
            # You may want to log this instead of print, depending on your setup
            print(f"[Tank01 usage] Failed for {name} ({tank_id}): {e}")
            metrics = {
                "games": 0,
                "avg_off_snap_pct": 0.0,
                "avg_off_snaps": 0.0,
                "avg_targets": 0.0,
                "avg_receptions": 0.0,
                "avg_rec_yards": 0.0,
                "avg_rec_tds": 0.0,
                "avg_carries": 0.0,
                "avg_rush_yards": 0.0,
                "avg_rush_tds": 0.0,
                "ppr_ppg": 0.0,
                "half_ppr_ppg": 0.0,
                "std_ppg": 0.0,
            }

        if per_name:
            key = name
            result[key] = metrics
        else:
            key = sleeper_id
            result[key] = {
                "name": name,
                "tankId": tank_id,
                "team": meta.get("team"),
                "metrics": metrics,
            }

    return result

def get_player_usage(tank_player_id: str, season: Optional[int] = None):
    gamelogs = get_tank01_player_gamelogs(tank_player_id, season)
    return compute_usage_from_tank_body(gamelogs)


def compute_usage_from_tank_body(body):
    # If body is not a list of dicts, bail out safely
    if not isinstance(body, list):
        # e.g. body might be "No player stats found" for some guys
        return {
            "games": 0,
            "avg_snap_pct": 0.0,
            "avg_targets": 0.0,
            "avg_routes": 0.0,
            "fantasy_ppg": 0.0,
        }

    games = 0
    total_snap_pct = 0.0
    total_targets = 0.0
    total_fantasy = 0.0

    for g in body:
        # extra safety: skip anything that isn't a dict
        if not isinstance(g, dict):
            continue

        snap_counts = g.get("snapCounts", {}) or {}
        receiving = g.get("Receiving", {}) or {}

        off_snap_pct = float(snap_counts.get("offSnapPct", 0) or 0)
        targets = float(receiving.get("targets", 0) or 0)
        fantasy = float(g.get("fantasyPoints", 0) or 0)

        total_snap_pct += off_snap_pct
        total_targets += targets
        total_fantasy += fantasy
        games += 1

    if games == 0:
        return {
            "games": 0,
            "avg_snap_pct": 0.0,
            "avg_targets": 0.0,
            "fantasy_ppg": 0.0,
        }

    return {
        "games": games,
        "avg_snap_pct": total_snap_pct / games,
        "avg_targets": total_targets / games,
        "fantasy_ppg": total_fantasy / games,
    }
