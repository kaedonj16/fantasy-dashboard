# dashboard_services/value_cache.py

from __future__ import annotations

import time
import json
import os
from typing import Dict, Any, List, Optional

from dashboard_services.utils import load_players_index
from dashboard_services.usage_cache import get_usage_map_for_season
from dashboard_services.trade_value_model import PlayerFeatures, build_value_table
from dashboard_services.trade_value_data import load_projections_for_season


VALUE_CACHE_TTL = 7 * 24 * 60 * 60  # also weekly

# In-memory cache: { (league_id, season): { "ts": float, "data": {...} } }
_VALUE_CACHE: Dict[str, Dict[str, Any]] = {}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _value_cache_key(league_id: str, season: int) -> str:
    return f"{league_id}:{season}"


def _value_cache_path(league_id: str, season: int) -> str:
    return os.path.join(CACHE_DIR, f"trade_values_{league_id}_{season}.json")


def _load_value_from_disk(league_id: str, season: int) -> Dict[str, Any] | None:
    path = _value_cache_path(league_id, season)
    print(f"[value_cache] Loading {path}")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        ts = payload.get("ts")
        data = payload.get("data")
        if ts is None or data is None:
            return None
        if time.time() - ts > VALUE_CACHE_TTL:
            return None
        return payload
    except Exception:
        return None


def _save_value_to_disk(league_id: str, season: int, ts: float, data: Dict[str, float]) -> None:
    path = _value_cache_path(league_id, season)
    payload = {"ts": ts, "data": data}
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[value_cache] Failed to write disk cache: {e}")


def _build_player_features_from_sources(
    league_id: str,
    season: int,
) -> List[PlayerFeatures]:
    """
    Bring everything together to build PlayerFeatures for all relevant players:
    - players_index (name, team, position, age if you add it)
    - usage metrics from Tank01 (by player name)
    - projections for your league scoring
    """
    players_index = load_players_index()

    # This may return either:
    #  1) { "Player Name": {metrics...}, ... }
    #  2) { "ts": <float>, "data": { "Player Name": {metrics...}, ... } }
    raw_usage = get_usage_map_for_season(season)

    # Normalize to: { "Player Name": {metrics...}, ... }
    if isinstance(raw_usage, dict) and "data" in raw_usage and isinstance(raw_usage["data"], dict):
        usage_by_name: Dict[str, Dict[str, Any]] = raw_usage["data"]
    else:
        usage_by_name = raw_usage  # already name -> metrics

    projections = load_projections_for_season(season)  # {player_id (sleeper?), fpts}

    features: List[PlayerFeatures] = []

    for player_id, meta in list(players_index.items())[:800]:
        name = meta.get("name", f"Player {player_id}")
        pos = meta.get("pos")
        team = meta.get("team")
        age = meta.get("age")  # if you have DOB, you can precompute age elsewhere

        # Only skill positions
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        # usage_by_name is keyed by player name, like "Jaxon Smith-Njigba"
        usage = usage_by_name.get(name, {}) or {}

        projected_points = float(projections.get(player_id, 0.0))

        feat = PlayerFeatures(
            player_id=str(player_id),
            name=name,
            position=pos,
            age=age,
            team=team,
            projected_points=projected_points,
            snap_share=usage.get("avg_off_snap_pct"),
            trade_value=usage.get("trade_value"),
            target_share=None,           # you can wire these later
            carry_share=None,
            route_participation=None,
            redzone_share=None,
            team_offense_tier=None,
            team_pass_rate=None,
            team_play_volume=None,
            injury_risk_flag=None,
            suspension_risk_flag=None,
        )
        features.append(feat)

    return features

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Adjust to your actual import
# from .models import PlayerFeatures


@dataclass
class PosDistribution:
    proj_mean: float
    proj_std: float
    ppr_mean: float
    ppr_std: float
    snap_mean: float
    snap_std: float


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def build_pos_distributions(players: List["PlayerFeatures"]) -> Dict[str, PosDistribution]:
    """
    Build mean/std for each position so we can z-score players
    relative to other players at the same position.
    """
    buckets = defaultdict(lambda: {"proj": [], "ppr": [], "snap": []})

    for f in players:
        pos = f.position
        if not pos:
            continue

        if f.projected_points is not None:
            buckets[pos]["proj"].append(float(f.projected_points))

        ppr_ppg = getattr(f, "ppr_ppg", None)
        if ppr_ppg is not None:
            buckets[pos]["ppr"].append(float(ppr_ppg))

        if f.snap_share is not None:
            buckets[pos]["snap"].append(float(f.snap_share))

    out: Dict[str, PosDistribution] = {}

    for pos, vals in buckets.items():
        proj_mean, proj_std = _mean_std(vals["proj"])
        ppr_mean, ppr_std = _mean_std(vals["ppr"])
        snap_mean, snap_std = _mean_std(vals["snap"])

        out[pos] = PosDistribution(
            proj_mean=proj_mean,
            proj_std=proj_std,
            ppr_mean=ppr_mean,
            ppr_std=ppr_std,
            snap_mean=snap_mean,
            snap_std=snap_std,
        )

    return out


def get_value_table_for_league(league_id: str, season: int) -> Dict[str, float]:
    """
    Weekly cached trade-value table for a given league/season:

    Returns:
      { player_id (str): value_score (float), ... }
    """
    now = time.time()
    key = _value_cache_key(league_id, season)

    # 1) In-memory
    entry = _VALUE_CACHE.get(key)
    if entry and now - entry["ts"] <= VALUE_CACHE_TTL:
        return entry["data"]

    # 2) Disk
    disk_entry = _load_value_from_disk(league_id, season)
    if disk_entry:
        _VALUE_CACHE[key] = disk_entry
        return disk_entry["data"]
    # 3) Rebuild
    print(f"[value_cache] Rebuilding trade value table for league {league_id}, season {season}...")
    features = _build_player_features_from_sources(league_id, season)
    pos_dists = build_pos_distributions(features)
    table = build_value_table(features, pos_dists)  # {player_id: value}

    new_entry = {"ts": now, "data": table}
    _VALUE_CACHE[key] = new_entry
    _save_value_to_disk(league_id, season, now, table)

    return table
