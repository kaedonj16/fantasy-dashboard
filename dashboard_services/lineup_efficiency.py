"""
Historical lineup efficiency (manager skill) for the playoff simulator.

Efficiency = actual points started / optimal possible points, measured across a
manager's COMPLETED prior seasons. A manager who routinely leaves points on the
bench scores below 1.0; a sharp one approaches it. The playoff simulator scales
each team's projected weekly mean by this factor so good managers' odds get a
small bump and sloppy ones a small drag.

The computation walks prior-season matchups (one API call per past week), so it
must never run inside an interactive request. `get_efficiency` only ever reads a
disk cache and, when the cache is missing or stale, kicks off a background
thread to (re)compute it. Until that lands, callers get {} and the simulator
falls back to a neutral 1.0 — managers are never penalized for missing data.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR     = os.path.join(os.path.dirname(__file__), "..", "cache", "efficiency")
_CACHE_TTL     = 7 * 24 * 3600     # recompute at most weekly
_MAX_SEASONS   = 3                  # how many prior seasons to average
_EFF_FLOOR     = 0.80              # clamp so one disaster week can't tank a team
_EFF_CEIL      = 1.00
_DEFAULT_AVG   = 0.95             # league-average efficiency when some data exists

# Guard against launching duplicate background computes for the same league.
_INFLIGHT: set[str] = set()
_INFLIGHT_LOCK = threading.Lock()


def _cache_path(platform: str, league_id: str) -> str:
    safe = f"{platform}_{league_id}".replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _read_cache(platform: str, league_id: str) -> Optional[dict]:
    try:
        with open(_cache_path(platform, league_id)) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(platform: str, league_id: str, payload: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = _cache_path(platform, league_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, _cache_path(platform, league_id))
    except Exception as exc:
        logger.warning("[efficiency] cache write failed: %s", exc)


def get_efficiency(platform: str, league_id: str, season: int) -> Dict[int, float]:
    """Return {roster_id: efficiency} for the current season (cache-only).

    Triggers a background recompute when the cache is missing or stale, but
    never blocks: returns the cached map (or {} when there's nothing yet).
    """
    cache = _read_cache(platform, league_id)
    fresh = bool(cache) and (time.time() - float(cache.get("ts", 0)) < _CACHE_TTL)
    if not fresh:
        _kick_background_compute(platform, league_id, season)
    if not cache:
        return {}
    eff = cache.get("efficiency") or {}
    return {int(rid): float(v) for rid, v in eff.items()}


def _kick_background_compute(platform: str, league_id: str, season: int) -> None:
    key = f"{platform}:{league_id}:{season}"
    with _INFLIGHT_LOCK:
        if key in _INFLIGHT:
            return
        _INFLIGHT.add(key)

    def _worker():
        try:
            compute_and_cache_efficiency(platform, league_id, season)
        except Exception as exc:
            logger.warning("[efficiency] background compute failed: %s", exc)
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(key)

    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Optimal-lineup helper (local copy so this module has no app.py dependency)
# ---------------------------------------------------------------------------

def _optimal_total(players, players_points, positions, roster_positions) -> float:
    slot_counts: dict = defaultdict(int)
    for s in roster_positions:
        slot_counts[str(s).upper()] += 1

    by_pos: dict = defaultdict(list)
    for pid in players:
        pos = str(positions.get(str(pid)) or "").upper()
        by_pos[pos].append(float(players_points.get(str(pid)) or 0))
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    used: dict = defaultdict(int)
    total = 0.0
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB"):
        pool = by_pos.get(pos, [])
        for _ in range(slot_counts.get(pos, 0)):
            i = used[pos]
            if i < len(pool):
                total += pool[i]
                used[pos] += 1

    flex_n = slot_counts.get("FLEX", 0)
    flex_pool = sorted(
        [p for pos in ("RB", "WR", "TE") for p in by_pos.get(pos, [])[used[pos]:]],
        reverse=True,
    )
    total += sum(flex_pool[:flex_n])

    sf_n = slot_counts.get("SUPER_FLEX", 0) + slot_counts.get("SFLEX", 0)
    remaining_flex = flex_pool[flex_n:]
    sf_pool = sorted(by_pos.get("QB", [])[used["QB"]:] + remaining_flex, reverse=True)
    total += sum(sf_pool[:sf_n])
    return total


# ---------------------------------------------------------------------------
# Heavy computation (background only)
# ---------------------------------------------------------------------------

def compute_and_cache_efficiency(platform: str, league_id: str, season: int) -> Dict[int, float]:
    """Compute per-manager efficiency from prior seasons and cache it.

    Maps efficiency by owner_id so it follows a manager across seasons (dynasty
    leagues get a new league_id each year), then projects onto the current
    season's roster_ids. Teams with no usable history get the league average.
    """
    from dashboard_services.api import build_league_history_map
    from dashboard_services.platform_api import get_league, get_matchups, get_rosters

    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
        positions = {str(pid): (info or {}).get("pos", "") for pid, info in idx.items()}
    except Exception:
        positions = {}

    history_map = build_league_history_map(platform, league_id, season) or {}
    prior_seasons = sorted((s for s in history_map if int(s) < int(season)), reverse=True)[:_MAX_SEASONS]

    actual_by_owner: dict = defaultdict(float)
    optimal_by_owner: dict = defaultdict(float)

    for s in prior_seasons:
        lid = str(history_map[s])
        try:
            league = get_league(platform, lid, int(s)) or {}
            settings = league.get("settings") or league.get("league_settings") or {}
            playoff_start = int(settings.get("playoff_week_start") or 15)
            roster_positions = league.get("roster_positions") or []
            rosters = get_rosters(platform, lid, int(s)) or []
        except Exception:
            continue

        rid_to_owner = {
            int(r.get("roster_id")): str(r.get("owner_id") or r.get("roster_id"))
            for r in rosters if r.get("roster_id") is not None
        }

        for week in range(1, playoff_start):
            try:
                matchups = get_matchups(platform, lid, week, int(s)) or []
            except Exception:
                continue
            for entry in matchups:
                rid = entry.get("roster_id")
                if rid is None:
                    continue
                owner = rid_to_owner.get(int(rid))
                if not owner:
                    continue
                players_points = entry.get("players_points") or {}
                players = entry.get("players") or list(players_points.keys())
                starters = entry.get("starters") or []
                actual = entry.get("points")
                if actual is None:
                    actual = sum(float(players_points.get(str(p)) or 0) for p in starters)
                optimal = _optimal_total(players, players_points, positions, roster_positions)
                if optimal and optimal > 0:
                    actual_by_owner[owner]  += float(actual or 0)
                    optimal_by_owner[owner] += float(optimal)

    eff_by_owner: dict = {}
    for owner, opt in optimal_by_owner.items():
        if opt > 0:
            eff_by_owner[owner] = min(_EFF_CEIL, max(_EFF_FLOOR, actual_by_owner[owner] / opt))

    league_avg = (sum(eff_by_owner.values()) / len(eff_by_owner)) if eff_by_owner else _DEFAULT_AVG

    # Project onto current-season roster ids.
    efficiency: Dict[int, float] = {}
    try:
        cur_rosters = get_rosters(platform, league_id, int(season)) or []
        for r in cur_rosters:
            rid = r.get("roster_id")
            if rid is None:
                continue
            owner = str(r.get("owner_id") or rid)
            efficiency[int(rid)] = round(eff_by_owner.get(owner, league_avg), 4)
    except Exception:
        pass

    payload = {
        "ts": time.time(),
        "season": int(season),
        "league_avg": round(league_avg, 4),
        # Empty when no prior data so the simulator stays neutral (1.0).
        "efficiency": {str(k): v for k, v in efficiency.items()} if eff_by_owner else {},
    }
    _write_cache(platform, league_id, payload)
    logger.info(
        "[efficiency] cached %d teams for %s/%s (avg %.3f, %d prior seasons)",
        len(payload["efficiency"]), platform, league_id, league_avg, len(prior_seasons),
    )
    return {int(k): float(v) for k, v in payload["efficiency"].items()}
