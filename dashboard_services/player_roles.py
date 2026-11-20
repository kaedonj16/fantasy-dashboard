"""
Compare Roles engine: per-player role classification, target maps and
rushing gap maps for the Compare page.

Two data sources, both fail-soft:
  * Weekly usage series (data_building/weekly_metrics) drive the role
    engine: Job tier, axes, tags, security, trend, timeline, comparables.
  * nflverse play-by-play (direct parquet, mirroring
    data_building/oline_ratings) drives the target map (pass zones),
    the gap map (rushing lanes) and QB rushing splits.

Play-action comes from FTN charting, not raw PBP: the weekly
advanced-metrics store (player_weekly_advanced_metrics, read via
data_building.advanced_metrics.get_player_weekly_adv_series) carries
per-week play_action_rate / play_action_epa built from the FTN
is_play_action flag. Those feed the QB summary strip and the
Play-Action Merchant tag; when the store has no charted weeks for a
window, both are omitted rather than faked.

Pure helpers (percentiles, tiers, tags, zone/lane aggregation over plain
row dicts) are kept pandas-free so the engine tests run on the stdlib.
The pandas aggregation entry point is build_pbp_tables().
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZONE_ROWS = ["Deep", "Inter", "Short", "Behind"]
ZONE_COLS = ["left", "middle", "right"]
LANES = ["LE", "LT", "LG", "M", "RG", "RT", "RE"]

_LANE_BY_LOC_GAP = {
    ("left", "end"): "LE",
    ("left", "tackle"): "LT",
    ("left", "guard"): "LG",
    ("middle", None): "M",
    ("right", "guard"): "RG",
    ("right", "tackle"): "RT",
    ("right", "end"): "RE",
}

_TEAM_ALIAS = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS"}

_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
_PBP_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "pbp"
)

# The only PBP columns the zone/lane aggregation reads. The raw nflverse
# frame carries ~372 columns; materialising all of them (plus a
# to_dict("records") over the full frame) peaked at ~1.65 GB RSS and
# OOM-killed the web worker in production (PR #3193 follow-up). Every
# read of the parquet is pruned to this list.
_PBP_COLUMNS = [
    "week", "season_type", "play_type", "posteam", "yardline_100",
    "down", "ydstogo", "yards_gained", "qb_spike", "qb_kneel",
    "qb_scramble", "two_point_attempt", "passer_player_id",
    "receiver_player_id", "rusher_player_id", "pass_location",
    "air_yards", "complete_pass", "pass_touchdown", "interception",
    "first_down_pass", "yards_after_catch", "run_location", "run_gap",
    "rush_touchdown", "first_down_rush",
]

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _f(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        out = float(v)
        return 0.0 if math.isnan(out) else out
    except (TypeError, ValueError):
        return 0.0


def _pctile(value: float, values: List[float]) -> float:
    """Percentile (0-100) of value within values, rank-based."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 50.0
    if n == 1:
        return 50.0
    below = sum(1 for v in vals if v < value)
    equal = sum(1 for v in vals if v == value)
    rank = below + (equal - 1) / 2.0
    return round(rank / (n - 1) * 100.0, 1)


def passer_rating(att: float, comp: float, yards: float, td: float, inter: float) -> float:
    """Standard NFL passer rating, clamped 0-158.3."""
    if att <= 0:
        return 0.0
    a = min(max((comp / att - 0.3) * 5, 0), 2.375)
    b = min(max((yards / att - 3) * 0.25, 0), 2.375)
    c = min(max((td / att) * 20, 0), 2.375)
    d = min(max(2.375 - (inter / att) * 25, 0), 2.375)
    return round((a + b + c + d) / 6 * 100, 1)


def zone_row_for_air_yards(air_yards: float) -> str:
    if air_yards < 0:
        return "Behind"
    if air_yards <= 9:
        return "Short"
    if air_yards <= 19:
        return "Inter"
    return "Deep"


def lane_for(run_location: Any, run_gap: Any) -> Optional[str]:
    loc = str(run_location or "").strip().lower()
    gap = str(run_gap or "").strip().lower()
    if loc not in ("left", "middle", "right"):
        return None
    if loc == "middle":
        return "M"
    if gap in ("", "nan", "none"):
        return None
    return _LANE_BY_LOC_GAP.get((loc, gap))


def is_success(down: Any, ydstogo: Any, yards: Any) -> bool:
    d, need, gained = _f(down), _f(ydstogo), _f(yards)
    if d <= 0 or need <= 0:
        return False
    if d == 1:
        return gained >= 0.4 * need
    if d == 2:
        return gained >= 0.6 * need
    return gained >= need


# ---------------------------------------------------------------------------
# Weekly-series features
# ---------------------------------------------------------------------------


def _weekly_opp_share(row: Dict[str, Any], pos: str) -> Optional[float]:
    """One week's opportunity share (%) for the player's position."""
    if pos in ("WR", "TE"):
        v = row.get("target_share")
        return _f(v) if v is not None else None
    if pos == "RB":
        tgt, car = _f(row.get("targets")), _f(row.get("carries"))
        ts, cs = row.get("target_share"), row.get("carry_share")
        team_tgt = tgt / (_f(ts) / 100.0) if ts and _f(ts) > 0 else 0.0
        team_car = car / (_f(cs) / 100.0) if cs and _f(cs) > 0 else 0.0
        denom = team_tgt + team_car
        if denom > 0 and (tgt + car) > 0:
            return round((tgt + car) / denom * 100.0, 1)
        if ts is not None and cs is not None:
            return round((_f(ts) + _f(cs)) / 2.0, 1)
        return _f(ts) if ts is not None else (_f(cs) if cs is not None else None)
    if pos == "QB":
        v = row.get("snap_pct")
        return _f(v) if v is not None else None
    return None


def player_features(series: List[Dict[str, Any]], pos: str,
                    pbp_feats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Aggregate one player's window series into role features."""
    rows = sorted(series or [], key=lambda r: int(r.get("week") or 0))
    games = len(rows)
    snaps = sum(_f(r.get("snaps")) for r in rows)
    targets = sum(_f(r.get("targets")) for r in rows)
    carries = sum(_f(r.get("carries")) for r in rows)
    receptions = sum(_f(r.get("receptions")) for r in rows)
    pass_att = sum(_f(r.get("pass_att")) for r in rows)
    rz_opp = sum(_f(r.get("rz_targets")) + _f(r.get("rz_carries")) for r in rows)
    opp = pass_att if pos == "QB" else targets + carries
    shares = [s for s in (_weekly_opp_share(r, pos) for r in rows) if s is not None]
    snap_pcts = [_f(r.get("snap_pct")) for r in rows if r.get("snap_pct") is not None]
    feats: Dict[str, Any] = {
        "games": games,
        "snaps": int(snaps),
        "targets": targets,
        "carries": carries,
        "receptions": receptions,
        "pass_att": pass_att,
        "opp": opp,
        "opp_share": (sum(shares) / len(shares)) if shares else 0.0,
        "snap_pct": (sum(snap_pcts) / len(snap_pcts)) if snap_pcts else 0.0,
        "rz_opp": rz_opp,
        "rz_share": (rz_opp / opp * 100.0) if opp > 0 else 0.0,
        "rz_per_game": (rz_opp / games) if games else 0.0,
        "rec_share_of_opp": (targets / opp) if opp > 0 and pos == "RB" else 0.0,
        "catch_rate": (receptions / targets) if targets > 0 else 0.0,
        "yards_per_target": (sum(_f(r.get("rec_yards")) for r in rows) / targets) if targets > 0 else 0.0,
        "yards_per_catch": (sum(_f(r.get("rec_yards")) for r in rows) / receptions) if receptions > 0 else 0.0,
        "yards_per_touch": (
            (sum(_f(r.get("rec_yards")) + _f(r.get("rush_yards")) for r in rows)
             / (receptions + carries)) if (receptions + carries) > 0 else 0.0),
        "rush_ypc": (sum(_f(r.get("rush_yards")) for r in rows) / carries) if carries > 0 else 0.0,
        "rush_yd_per_game": (sum(_f(r.get("rush_yards")) for r in rows) / games) if games else 0.0,
        "dropbacks_per_game": (pass_att / games) if games else 0.0,
        "start_share": (
            100.0 * sum(1 for p in snap_pcts if p >= 50) / len(snap_pcts)
        ) if snap_pcts else 0.0,
        "pass_td_per_game": (sum(_f(r.get("pass_tds")) for r in rows) / games) if games else 0.0,
        "ppr_per_dropback": (sum(_f(r.get("ppr_pts")) for r in rows) / pass_att) if pass_att > 0 else 0.0,
        "weekly_opp_shares": shares,
        "weeks": [int(r.get("week") or 0) for r in rows],
    }
    if pbp_feats:
        feats.update({f"pbp_{k}": v for k, v in pbp_feats.items()})
    return feats


# Axis feature keys per position: axis name -> feature key on the feats dict.
AXIS_FEATURES: Dict[str, Dict[str, str]] = {
    "WR": {"Volume": "opp_share", "Deep Work": "pbp_deep_rate",
           "Red Zone": "rz_share", "Explosiveness": "pbp_explosive_rate"},
    "TE": {"Volume": "opp_share", "Deep Work": "pbp_deep_rate",
           "Red Zone": "rz_share", "Explosiveness": "pbp_explosive_rate"},
    "RB": {"Volume": "opp_share", "Receiving": "rec_share_of_opp",
           "Red Zone": "rz_share", "Explosiveness": "pbp_explosive_run_rate"},
    "QB": {"Volume": "dropbacks_per_game", "Deep Work": "pbp_deep_att_rate",
           "Mobility": "rush_yd_per_game", "Efficiency": "pbp_rating",
           "Red Zone": "pbp_rz_att_rate"},
}
# Fallbacks when PBP features are unavailable for a player.
AXIS_FALLBACKS = {
    "pbp_deep_rate": "yards_per_target",
    "pbp_explosive_rate": "yards_per_catch",
    "pbp_explosive_run_rate": "rush_ypc",
    "pbp_deep_att_rate": "pass_td_per_game",
    "pbp_rating": "ppr_per_dropback",
    "pbp_rz_att_rate": "pass_td_per_game",
}


def _axis_raw(feats: Dict[str, Any], key: str) -> float:
    if key in feats and feats.get(key) is not None:
        return _f(feats[key])
    fb = AXIS_FALLBACKS.get(key)
    return _f(feats.get(fb)) if fb else 0.0


def compute_pool(pool: Dict[str, Dict[str, Any]],
                 pbp_feats_by_player: Optional[Dict[str, Dict[str, float]]] = None
                 ) -> Dict[str, Dict[str, Any]]:
    """Features + axes + job score for every player in the pool.

    pool: {player_id: {"pos": str, "team": str, "name": str,
                       "series": [weekly rows]}}
    Returns {player_id: {"feats":..., "axes": {...}, "score": float,
                         "team_rank_score": float}}
    """
    pbp_feats_by_player = pbp_feats_by_player or {}
    feats_by_pid: Dict[str, Dict[str, Any]] = {}
    for pid, entry in pool.items():
        pos = (entry.get("pos") or "").upper()
        if pos not in AXIS_FEATURES:
            continue
        feats_by_pid[pid] = player_features(
            entry.get("series") or [], pos, pbp_feats_by_player.get(pid))

    # Team rank at position (by opportunity share).
    by_team_pos: Dict[tuple, List[str]] = {}
    for pid, entry in pool.items():
        if pid in feats_by_pid:
            by_team_pos.setdefault((entry.get("team") or "", entry.get("pos")), []).append(pid)
    team_rank_score: Dict[str, float] = {}
    rank_pts = {1: 100.0, 2: 70.0, 3: 45.0}
    for pids in by_team_pos.values():
        ordered = sorted(pids, key=lambda p: feats_by_pid[p]["opp_share"], reverse=True)
        for i, pid in enumerate(ordered, start=1):
            team_rank_score[pid] = rank_pts.get(i, 25.0)

    # Percentile pools per position per raw feature key.
    by_pos: Dict[str, List[str]] = {}
    for pid, entry in pool.items():
        if pid in feats_by_pid:
            by_pos.setdefault(entry.get("pos"), []).append(pid)

    out: Dict[str, Dict[str, Any]] = {}
    for pid, feats in feats_by_pid.items():
        pos = pool[pid].get("pos")
        peers = by_pos.get(pos, [])
        axes: Dict[str, int] = {}
        feat_pct: Dict[str, float] = {}
        for axis, key in AXIS_FEATURES[pos].items():
            raw = _axis_raw(feats, key)
            pool_vals = [_axis_raw(feats_by_pid[p], key) for p in peers]
            axes[axis] = int(round(_pctile(raw, pool_vals)))
        for key in ("opp_share", "snap_pct", "catch_rate", "rz_per_game",
                    "dropbacks_per_game", "start_share", "rec_share_of_opp",
                    "pbp_adot", "pbp_yac_per_rec", "pbp_deep_rate",
                    "pbp_explosive_rate", "pbp_rating",
                    "pbp_play_action_rate", "pbp_play_action_epa"):
            pool_vals = [_f(feats_by_pid[p].get(key)) for p in peers]
            feat_pct[key] = _pctile(_f(feats.get(key)), pool_vals)
        if pos == "QB":
            score = 0.6 * _pctile(feats["dropbacks_per_game"],
                                  [feats_by_pid[p]["dropbacks_per_game"] for p in peers]) \
                + 0.4 * feats["start_share"]
        else:
            score = (0.5 * feat_pct["opp_share"] + 0.3 * feat_pct["snap_pct"]
                     + 0.2 * team_rank_score.get(pid, 25.0))
        out[pid] = {"feats": feats, "axes": axes, "score": round(score, 1),
                    "feat_pct": feat_pct,
                    "team_rank_score": team_rank_score.get(pid, 25.0)}
    return out


# ---------------------------------------------------------------------------
# Job tiers, tags, security, trend
# ---------------------------------------------------------------------------


def job_for(pos: str, score: float, feats: Dict[str, Any]) -> str:
    if pos == "WR":
        if score >= 85:
            return "Alpha WR1"
        if score >= 70:
            return "WR1"
        if score >= 55:
            return "WR2"
        if score >= 40:
            return "WR3"
        return "Rotational"
    if pos == "RB":
        if feats.get("rec_share_of_opp", 0) >= 0.40 and score < 65 \
                and feats.get("targets", 0) >= feats.get("carries", 0) * 0.8:
            return "Passing-Down Specialist"
        if score >= 85:
            return "Workhorse"
        if score >= 65:
            return "Lead 1A"
        if score >= 45:
            return "Committee 1B"
        return "Depth"
    if pos == "TE":
        if feats.get("snap_pct", 0) >= 55 and feats.get("opp_share", 0) < 8 and score < 60:
            return "Blocking-First"
        if score >= 80:
            return "Alpha TE1"
        if score >= 60:
            return "TE1"
        if score >= 40:
            return "Committee TE"
        return "Blocking-First" if feats.get("snap_pct", 0) >= 55 else "Committee TE"
    if pos == "QB":
        if score >= 80:
            return "Franchise QB1"
        if score >= 60:
            return "QB1"
        if score >= 40:
            return "QB2-Streamer"
        return "Timeshare-Backup"
    return "Rotational"


# (axis, tag, confirming feature-percentile key)
_TAG_RULES: Dict[str, List[tuple]] = {
    "WR": [("Volume", "Target Hog", "opp_share"),
           ("Deep Work", "Deep Threat", "pbp_deep_rate"),
           ("Red Zone", "Red Zone Threat", "rz_per_game"),
           ("Explosiveness", "YAC Creator", "pbp_yac_per_rec")],
    "TE": [("Volume", "Target Hog", "opp_share"),
           ("Deep Work", "Deep Threat", "pbp_deep_rate"),
           ("Red Zone", "Red Zone Threat", "rz_per_game"),
           ("Explosiveness", "YAC Creator", "pbp_yac_per_rec")],
    "RB": [("Receiving", "Receiving Back", "rec_share_of_opp"),
           ("Red Zone", "Goal-Line Back", "rz_per_game"),
           ("Explosiveness", "Home-Run Hitter", "pbp_explosive_rate")],
    "QB": [("Deep Work", "Gunslinger", "pbp_deep_rate"),
           ("Mobility", "Dual Threat", "opp_share"),
           ("Efficiency", "Surgical", "pbp_rating")],
}
_VERSATILE_PAIRS = {
    "WR": ("Volume", "Deep Work"),
    "TE": ("Volume", "Deep Work"),
    "RB": ("Receiving", "Explosiveness"),
    "QB": ("Efficiency", "Deep Work"),
}


def tags_for(pos: str, axes: Dict[str, int], feat_pct: Dict[str, float],
             feats: Dict[str, Any]) -> List[str]:
    pair = _VERSATILE_PAIRS.get(pos)
    if pair and axes.get(pair[0], 0) >= 80 and axes.get(pair[1], 0) >= 80:
        return ["Versatile"]
    tags: List[str] = []
    # Possession (WR/TE): high volume, low deep work, sure hands.
    if pos in ("WR", "TE") and axes.get("Volume", 0) >= 70 \
            and axes.get("Deep Work", 100) < 55 and feat_pct.get("catch_rate", 0) >= 50:
        tags.append("Possession")
    if pos == "QB" and axes.get("Efficiency", 0) >= 70 \
            and axes.get("Deep Work", 100) < 55 and feat_pct.get("pbp_rating", 0) >= 50:
        tags.append("Quick Game")
    # Play-Action Merchant (QB): FTN play-action usage in the top tier
    # of QBs (rate percentile >= 70, the usage bar the other tags use)
    # with play-action EPA as the confirming efficiency signal.
    if pos == "QB" and feats.get("pbp_play_action_rate") is not None \
            and feat_pct.get("pbp_play_action_rate", 0) >= 70 \
            and feat_pct.get("pbp_play_action_epa", 0) >= 50:
        tags.append("Play-Action Merchant")
    for axis, tag, confirm_key in _TAG_RULES.get(pos, []):
        if len(tags) >= 2:
            break
        if tag in tags:
            continue
        if axes.get(axis, 0) >= 70 and feat_pct.get(confirm_key, 0) >= 50:
            tags.append(tag)
    return tags[:2]


def _cv(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.5
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return 1.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / mean


def security_for(pid: str, pool: Dict[str, Dict[str, Any]],
                 computed: Dict[str, Dict[str, Any]]):
    """(security, security_note, lead_pid). Contingent first."""
    entry = pool[pid]
    pos, team = entry.get("pos"), entry.get("team")
    my = computed[pid]
    teammates = [p for p, e in pool.items()
                 if p != pid and e.get("pos") == pos and e.get("team") == team
                 and p in computed]
    lead_pid = None
    if teammates:
        lead_pid = max(teammates, key=lambda p: computed[p]["feats"]["opp_share"])
        # Only a lead if he actually out-shares the player.
        if computed[lead_pid]["feats"]["opp_share"] <= my["feats"]["opp_share"]:
            lead_pid = None
    if lead_pid:
        lead_series = {int(r.get("week") or 0): r
                       for r in pool[lead_pid].get("series") or []}
        total_prod = 0.0
        down_prod = 0.0
        for r in entry.get("series") or []:
            prod = _f(r.get("ppr_pts")) or (
                _f(r.get("targets")) + _f(r.get("carries")) + _f(r.get("pass_att")))
            total_prod += prod
            lead_row = lead_series.get(int(r.get("week") or 0))
            if lead_row is None or _f(lead_row.get("snap_pct")) < 25:
                down_prod += prod
        if total_prod > 0 and down_prod / total_prod >= 0.60:
            lead_name = pool[lead_pid].get("name") or "the room lead"
            return "Contingent", f"Production came with {lead_name} sidelined or limited", lead_pid
    pct = my["feat_pct"].get("opp_share", 0)
    cv = _cv(my["feats"].get("weekly_opp_shares") or [])
    if pct >= 75 and cv <= 0.35:
        return "Locked", None, lead_pid
    if pct >= 45 and cv <= 0.65:
        return "Stable", None, lead_pid
    return "Volatile", None, lead_pid


def trend_for(feats: Dict[str, Any]) -> str:
    shares = feats.get("weekly_opp_shares") or []
    if len(shares) < 2:
        return "Steady"
    last4 = shares[-4:]
    season = sum(shares) / len(shares)
    recent = sum(last4) / len(last4)
    if season <= 0:
        return "Steady"
    rel = (recent - season) / season
    if rel >= 0.15:
        return "Rising"
    if rel <= -0.15:
        return "Fading"
    return "Steady"


def build_role(pid: str, pool: Dict[str, Dict[str, Any]],
               computed: Dict[str, Dict[str, Any]],
               prior_score: Optional[float] = None) -> Dict[str, Any]:
    entry = pool[pid]
    pos = entry.get("pos")
    mine = computed[pid]
    feats = mine["feats"]
    score = mine["score"]
    force_provisional = False
    if prior_score is not None:
        score = round(0.7 * score + 0.3 * prior_score, 1)
        force_provisional = True
    job = job_for(pos, score, feats)
    security, note, _lead = security_for(pid, pool, computed)
    confidence = "Provisional" if (force_provisional or feats["games"] < 5
                                   or feats["snaps"] < 100) else "High"
    return {
        "job": job,
        "tags": tags_for(pos, mine["axes"], mine["feat_pct"], feats),
        "security": security,
        "security_note": note,
        "confidence": confidence,
        "trend": trend_for(feats),
        "axes": mine["axes"],
        "games": int(feats["games"]),
        "snaps": int(feats["snaps"]),
        "_score": score,
    }


def build_timeline(pid: str, pool: Dict[str, Dict[str, Any]],
                   pbp_feats_by_player: Optional[Dict[str, Dict[str, float]]] = None
                   ) -> List[Dict[str, Any]]:
    weeks = sorted({int(r.get("week") or 0)
                    for r in pool[pid].get("series") or [] if r.get("week")})
    if not weeks:
        return []
    segments: List[Dict[str, Any]] = []
    prev_job = None
    prev_share = None
    pos = pool[pid].get("pos")
    for w in weeks:
        # Restrict the per-week pool to the player's own position: a
        # player's score/job inside compute_pool depends only on
        # same-position peers (percentiles are per position, team rank
        # is per team+position), so this is output-identical for the
        # target player and ~4x cheaper than recomputing every position
        # for every week (the old loop was ~10 full-pool computes per
        # payload build).
        sub_pool = {}
        for p, e in pool.items():
            if e.get("pos") != pos:
                continue
            sub = [r for r in (e.get("series") or []) if int(r.get("week") or 0) <= w]
            if sub:
                sub_pool[p] = {**e, "series": sub}
        if pid not in sub_pool:
            continue
        comp = compute_pool(sub_pool, pbp_feats_by_player)
        if pid not in comp:
            continue
        job = job_for(pool[pid].get("pos"), comp[pid]["score"], comp[pid]["feats"])
        share = comp[pid]["feats"]["opp_share"]
        if job != prev_job:
            reason = None
            if prev_job is not None:
                if prev_share is not None and share > prev_share + 2:
                    reason = f"opportunity share jumped after Week {w - 1}"
                elif prev_share is not None and share < prev_share - 2:
                    reason = f"opportunity share fell after Week {w - 1}"
                else:
                    reason = f"role changed after Week {w - 1}"
            segments.append({"from_week": w, "to_week": w, "job": job,
                             "reason": reason})
            prev_job = job
        else:
            segments[-1]["to_week"] = w
        prev_share = share
    return segments


def build_comparables(pid: str, pool: Dict[str, Dict[str, Any]],
                      computed: Dict[str, Dict[str, Any]],
                      jobs: Dict[str, str]):
    pos = pool[pid].get("pos")
    axis_names = list(AXIS_FEATURES.get(pos, {}).keys())
    mine = computed[pid]["axes"]

    def _vec(axes):
        return [axes.get(a, 0) / 100.0 for a in axis_names]

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    scored = []
    for other, comp in computed.items():
        if other == pid or pool[other].get("pos") != pos:
            continue
        if comp["feats"]["games"] < 1:
            continue
        sim = _cos(_vec(mine), _vec(comp["axes"]))
        scored.append((other, sim))
    if not scored:
        return [], None
    my_job = jobs.get(pid)
    ranked = sorted(scored, key=lambda t: (jobs.get(t[0]) == my_job, t[1]),
                    reverse=True)
    comps = []
    for other, sim in ranked[:3]:
        o_axes = computed[other]["axes"]
        diffs = sorted(axis_names, key=lambda a: abs(mine.get(a, 0) - o_axes.get(a, 0)))
        closest, farthest = diffs[0], diffs[-1]
        comps.append({
            "player_id": other,
            "name": pool[other].get("name") or "",
            "pos": pos,
            "team": pool[other].get("team") or "",
            "similarity": int(round(sim * 100)),
            "match_line": f"Closest on {closest} ({mine.get(closest, 0)} vs {o_axes.get(closest, 0)})",
            "differ_line": f"Biggest gap on {farthest} ({mine.get(farthest, 0)} vs {o_axes.get(farthest, 0)})",
        })
    twin_pid, twin_sim = max(scored, key=lambda t: t[1])
    twin = {"player_id": twin_pid, "name": pool[twin_pid].get("name") or "",
            "pos": pos, "team": pool[twin_pid].get("team") or "",
            "similarity": int(round(twin_sim * 100))}
    return comps, twin


# ---------------------------------------------------------------------------
# PBP aggregation (row-dict core + pandas wrapper)
# ---------------------------------------------------------------------------


def _blank_zone():
    return {"primary": 0, "secondary": 0, "yards": 0.0, "td": 0, "int": 0,
            "first_downs": 0, "air": 0.0, "big": 0,
            "primary_last4": 0, "primary_base": 0}


def _accumulate_pass(rows, key_field, last4_weeks, base_weeks, window_weeks=None):
    """Aggregate pass rows into per-player zone tables."""
    per_player: Dict[str, Dict[tuple, dict]] = {}
    league: Dict[tuple, dict] = {}
    team_air: Dict[str, float] = {}
    player_air: Dict[str, float] = {}
    totals: Dict[str, dict] = {}
    for r in rows:
        if str(r.get("play_type") or "") != "pass":
            continue
        if _f(r.get("qb_spike")) == 1:
            continue
        recv = r.get("receiver_player_id")
        if recv is None or str(recv).strip() in ("", "nan", "None"):
            continue
        loc = str(r.get("pass_location") or "").strip().lower()
        if loc not in ZONE_COLS:
            continue
        air = r.get("air_yards")
        air = 0.0 if air is None else _f(air)
        row_name = zone_row_for_air_yards(air)
        key = (row_name, loc)
        pid = str(r.get(key_field) or "")
        if not pid:
            continue
        wk = int(_f(r.get("week")))
        in_window = window_weeks is None or wk in window_weeks
        for store in (per_player.setdefault(pid, {}), league):
            cell = store.setdefault(key, _blank_zone())
            if not in_window:
                if wk in last4_weeks:
                    cell["primary_last4"] += 1
                if wk in base_weeks:
                    cell["primary_base"] += 1
                continue
            cell["primary"] += 1
            cell["secondary"] += 1 if _f(r.get("complete_pass")) == 1 else 0
            cell["yards"] += _f(r.get("yards_gained"))
            cell["td"] += 1 if _f(r.get("pass_touchdown")) == 1 else 0
            cell["int"] += 1 if _f(r.get("interception")) == 1 else 0
            cell["first_downs"] += 1 if _f(r.get("first_down_pass")) == 1 else 0
            cell["air"] += air
            cell["big"] += 1 if _f(r.get("yards_gained")) >= 20 else 0
            if wk in last4_weeks:
                cell["primary_last4"] += 1
            if wk in base_weeks:
                cell["primary_base"] += 1
        if not in_window:
            continue
        team = str(r.get("posteam") or "")
        team_air[team] = team_air.get(team, 0.0) + air
        recv_pid = str(recv)
        player_air[recv_pid] = player_air.get(recv_pid, 0.0) + air
        t = totals.setdefault(pid, {"att": 0, "comp": 0, "yards": 0.0, "td": 0,
                                    "int": 0, "air": 0.0, "deep": 0, "rz": 0,
                                    "yac": 0.0, "rec": 0, "big_rec": 0})
        t["att"] += 1
        t["comp"] += 1 if _f(r.get("complete_pass")) == 1 else 0
        t["yards"] += _f(r.get("yards_gained"))
        t["td"] += 1 if _f(r.get("pass_touchdown")) == 1 else 0
        t["int"] += 1 if _f(r.get("interception")) == 1 else 0
        t["air"] += air
        t["deep"] += 1 if row_name == "Deep" else 0
        yl = r.get("yardline_100")
        t["rz"] += 1 if (yl is not None and _f(yl) <= 20) else 0
        if _f(r.get("complete_pass")) == 1:
            t["rec"] += 1
            t["yac"] += _f(r.get("yards_after_catch"))
            t["big_rec"] += 1 if _f(r.get("yards_gained")) >= 20 else 0
    return {"zones": per_player, "league": league, "totals": totals,
            "team_air": team_air, "player_air": player_air}


def _blank_lane():
    return {"carries": 0, "yards": 0.0, "td": 0, "first_downs": 0,
            "stuffed": 0, "explosive": 0, "success": 0, "longest": 0,
            "carries_last4": 0, "carries_base": 0}


def _accumulate_rush(rows, last4_weeks, base_weeks, window_weeks=None):
    per_player: Dict[str, Dict[str, dict]] = {}
    league: Dict[str, dict] = {}
    qb: Dict[str, dict] = {}
    direction: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if str(r.get("play_type") or "") != "run":
            continue
        if _f(r.get("qb_kneel")) == 1 or _f(r.get("qb_spike")) == 1:
            continue
        if _f(r.get("two_point_attempt")) == 1:
            continue
        rusher = r.get("rusher_player_id")
        if rusher is None or str(rusher).strip() in ("", "nan", "None"):
            continue
        pid = str(rusher)
        yards = _f(r.get("yards_gained"))
        wk = int(_f(r.get("week")))
        in_window = window_weeks is None or wk in window_weeks
        lane = lane_for(r.get("run_location"), r.get("run_gap"))
        if in_window:
            q = qb.setdefault(pid, {"att": 0, "yards": 0.0, "td": 0,
                                    "scramble": 0})
            q["att"] += 1
            q["yards"] += yards
            q["td"] += 1 if _f(r.get("rush_touchdown")) == 1 else 0
            q["scramble"] += 1 if _f(r.get("qb_scramble")) == 1 else 0
            loc = str(r.get("run_location") or "").strip().lower()
            if loc in ("left", "middle", "right"):
                d = direction.setdefault(pid, {"left": 0, "middle": 0, "right": 0})
                d[loc] += 1
        if lane is None:
            continue
        for store in (per_player.setdefault(pid, {}), league):
            cell = store.setdefault(lane, _blank_lane())
            if not in_window:
                if wk in last4_weeks:
                    cell["carries_last4"] += 1
                if wk in base_weeks:
                    cell["carries_base"] += 1
                continue
            cell["carries"] += 1
            cell["yards"] += yards
            cell["td"] += 1 if _f(r.get("rush_touchdown")) == 1 else 0
            cell["first_downs"] += 1 if _f(r.get("first_down_rush")) == 1 else 0
            cell["stuffed"] += 1 if yards <= 0 else 0
            cell["explosive"] += 1 if yards >= 10 else 0
            cell["success"] += 1 if is_success(r.get("down"), r.get("ydstogo"), yards) else 0
            cell["longest"] = max(cell["longest"], int(yards))
            if wk in last4_weeks:
                cell["carries_last4"] += 1
            if wk in base_weeks:
                cell["carries_base"] += 1
    return {"lanes": per_player, "league": league, "qb": qb,
            "direction": direction}


def build_pbp_tables_from_rows(rows: List[Dict[str, Any]],
                               xwalk: Optional[Dict[str, str]] = None,
                               last4_weeks: Optional[set] = None,
                               base_weeks: Optional[set] = None,
                               window_weeks: Optional[set] = None) -> Dict[str, Any]:
    """Build zone/lane tables from plain PBP row dicts (GSIS ids).

    xwalk maps GSIS -> Sleeper; ids missing from the crosswalk are kept
    under their GSIS id so partial crosswalks still aggregate.
    """
    xwalk = xwalk or {}
    last4_weeks = last4_weeks or set()
    base_weeks = base_weeks if base_weeks is not None else set()

    def _map(pid):
        return xwalk.get(pid, pid)

    # Single pass, copying only the rows each accumulator will read
    # (the accumulators filter on play_type anyway). The original code
    # copied every row three times; on full-width PBP records that
    # duplication was a large share of the build's peak memory.
    recv_rows, pass_rows, rush_rows = [], [], []
    for r in rows:
        play_type = str(r.get("play_type") or "")
        if play_type == "pass":
            r2 = dict(r)
            if r2.get("receiver_player_id"):
                r2["receiver_player_id"] = _map(str(r2["receiver_player_id"]))
            recv_rows.append(r2)
            r3 = dict(r)
            if r3.get("passer_player_id"):
                r3["passer_player_id"] = _map(str(r3["passer_player_id"]))
            pass_rows.append(r3)
        elif play_type == "run":
            r2 = dict(r)
            if r2.get("rusher_player_id"):
                r2["rusher_player_id"] = _map(str(r2["rusher_player_id"]))
            rush_rows.append(r2)

    receiving = _accumulate_pass(recv_rows, "receiver_player_id", last4_weeks, base_weeks, window_weeks)
    passing = _accumulate_pass(pass_rows, "passer_player_id", last4_weeks, base_weeks, window_weeks)
    rushing = _accumulate_rush(rush_rows, last4_weeks, base_weeks, window_weeks)

    # Per-player PBP features for the axes engine.
    feats: Dict[str, Dict[str, float]] = {}
    for pid, t in receiving["totals"].items():
        if t["att"] > 0:
            feats.setdefault(pid, {}).update({
                "deep_rate": t["deep"] / t["att"] * 100.0,
                "adot": t["air"] / t["att"],
                "explosive_rate": (t["big_rec"] / t["rec"] * 100.0) if t["rec"] else 0.0,
                "yac_per_rec": (t["yac"] / t["rec"]) if t["rec"] else 0.0,
            })
    for pid, t in passing["totals"].items():
        if t["att"] > 0:
            feats.setdefault(pid, {}).update({
                "deep_att_rate": t["deep"] / t["att"] * 100.0,
                "rating": passer_rating(t["att"], t["comp"], t["yards"], t["td"], t["int"]),
                "rz_att_rate": t["rz"] / t["att"] * 100.0,
            })
    for pid, lanes in rushing["lanes"].items():
        car = sum(c["carries"] for c in lanes.values())
        big = sum(1 for c in lanes.values() if c["longest"] >= 10)
        exp = sum(c["explosive"] for c in lanes.values())
        if car > 0:
            feats.setdefault(pid, {})["explosive_run_rate"] = exp / car * 100.0
            feats[pid]["big_lanes"] = float(big)
    return {"receiving": receiving, "passing": passing, "rushing": rushing,
            "feats": feats}


def build_pbp_tables(df, xwalk: Optional[Dict[str, str]] = None,
                     last4_weeks: Optional[set] = None,
                     base_weeks: Optional[set] = None,
                     window_weeks: Optional[set] = None) -> Dict[str, Any]:
    """Pandas entry point: DataFrame -> build_pbp_tables_from_rows."""
    if df is None or len(df) == 0:
        return build_pbp_tables_from_rows([], xwalk, last4_weeks, base_weeks, window_weeks)
    # Prune to the columns the aggregation reads before materialising
    # row dicts: to_dict("records") over the full ~372-column frame was
    # the production OOM (see _PBP_COLUMNS).
    cols = [c for c in _PBP_COLUMNS if c in df.columns]
    rows = df[cols].to_dict("records") if cols else df.to_dict("records")
    return build_pbp_tables_from_rows(rows, xwalk, last4_weeks, base_weeks, window_weeks)


# ---------------------------------------------------------------------------
# PBP loading (mirrors oline_ratings direct-parquet pattern) + bounded caches
# ---------------------------------------------------------------------------

_PBP_TABLES_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_TABLES_MAX = 6
# One tables build at a time per worker. A payload build aggregates a
# full season of PBP; without this, concurrent first-hits for different
# players each launched their own build and the combined peak memory
# OOM-killed the web worker in production (PR #3193 follow-up). The
# lock is only held for the (now column-pruned) build itself; cache
# hits never touch it.
_TABLES_BUILD_LOCK = threading.Lock()


def load_pbp(season: int):
    """Load one season of nflverse PBP (tests monkeypatch this).

    Prefers the on-disk cache under cache/pbp/, else reads the nflverse
    GitHub release parquet directly and best-effort caches it. The read
    is pruned to _PBP_COLUMNS and the frame is NOT retained: callers
    build the (small) aggregate tables and drop the frame.
    """
    season = int(season)
    try:
        import pandas as pd
    except Exception:
        return None

    def _read_pruned(source):
        try:
            return pd.read_parquet(source, columns=list(_PBP_COLUMNS))
        except Exception:
            # A parquet missing one of the pruned columns (an upstream
            # rename) must not fail the whole load: fall back to a full
            # read and select what exists.
            full = pd.read_parquet(source)
            cols = [c for c in _PBP_COLUMNS if c in full.columns]
            return full[cols] if cols else full

    df = None
    path = os.path.join(_PBP_CACHE_DIR, f"play_by_play_{season}.parquet")
    try:
        if os.path.exists(path):
            df = _read_pruned(path)
    except Exception:
        df = None
    if df is None:
        try:
            df = _read_pruned(_PBP_URL.format(year=season))
            try:
                os.makedirs(_PBP_CACHE_DIR, exist_ok=True)
                df.to_parquet(path)
            except Exception:
                pass
        except Exception:
            return None
    if df is not None and "season_type" in df.columns:
        try:
            df = df[df["season_type"].astype(str).str.upper() == "REG"]
        except Exception:
            pass
    return df


def get_gsis_crosswalk() -> Dict[str, str]:
    try:
        from data_building.external_data.nflverse_metrics import _gsis_to_sleeper
        return _gsis_to_sleeper() or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Play-action (FTN charting via the weekly advanced-metrics store)
# ---------------------------------------------------------------------------

_PA_SERIES_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_PA_SERIES_MAX = 128


def _pa_series(pid: str, season: int) -> List[Dict[str, Any]]:
    """Weekly advanced-metric rows for one player/season (patchable).

    Same store the Advanced Metrics page reads:
    data_building.advanced_metrics.get_player_weekly_adv_series over
    player_weekly_advanced_metrics, keyed by Sleeper id. Rows carry
    play_action_rate / play_action_epa (FTN is_play_action, dropbacks
    only, seasons >= 2022) plus the w_dropbacks volume weight.
    Fail-soft: [] on any error (e.g. no database configured).
    """
    key = (str(pid), int(season))
    with _CACHE_LOCK:
        hit = _PA_SERIES_CACHE.get(key)
        if hit is not None:
            _PA_SERIES_CACHE.move_to_end(key)
            return hit
    rows: List[Dict[str, Any]] = []
    try:
        from data_building.advanced_metrics import get_player_weekly_adv_series
        rows = get_player_weekly_adv_series(str(pid), int(season)) or []
    except Exception:
        rows = []
    with _CACHE_LOCK:
        _PA_SERIES_CACHE[key] = rows
        while len(_PA_SERIES_CACHE) > _PA_SERIES_MAX:
            _PA_SERIES_CACHE.popitem(last=False)
    return rows


def aggregate_pa_rows(rows: List[Dict[str, Any]], weeks) -> Optional[Dict[str, float]]:
    """Dropback-weighted play-action rate/EPA over a set of weeks.

    Mirrors advanced_metrics.get_player_weekly_adv_range: both
    play_action_rate and play_action_epa weight by w_dropbacks.
    Returns None when no charted dropbacks fall in the window, so the
    caller omits play-action for that window instead of faking it.
    """
    week_set = {int(w) for w in (weeks or [])}
    num_r = den_r = num_e = den_e = 0.0
    for r in rows or []:
        if int(_f(r.get("week"))) not in week_set:
            continue
        w = _f(r.get("w_dropbacks"))
        if w <= 0:
            continue
        if r.get("play_action_rate") is not None:
            num_r += _f(r["play_action_rate"]) * w
            den_r += w
        if r.get("play_action_epa") is not None:
            num_e += _f(r["play_action_epa"]) * w
            den_e += w
    if den_r <= 0:
        return None
    out: Dict[str, float] = {"play_action_rate": round(num_r / den_r, 1)}
    if den_e > 0:
        out["play_action_epa"] = round(num_e / den_e, 3)
    return out


def load_pa_feats(win_pool: Dict[str, Dict[str, Any]], season: int
                  ) -> Dict[str, Dict[str, float]]:
    """{pid: play-action feats} for QBs in an already-windowed pool.

    The window is each player's own windowed series weeks, so season /
    last-4 / last-season payloads each aggregate exactly the weeks the
    rest of the payload covers for that player.
    """
    out: Dict[str, Dict[str, float]] = {}
    for pid, entry in (win_pool or {}).items():
        if (entry.get("pos") or "") != "QB":
            continue
        weeks = [int(r.get("week") or 0) for r in entry.get("series") or []]
        agg = aggregate_pa_rows(_pa_series(pid, season), weeks)
        if agg:
            out[pid] = agg
    return out


def _tables_cache_fresh(key):
    """Cached tables for key when present and inside the TTL."""
    with _CACHE_LOCK:
        hit = _PBP_TABLES_CACHE.get(key)
        if hit is not None and time.time() - hit[0] < 3600:
            _PBP_TABLES_CACHE.move_to_end(key)
            return hit[1]
    return None


def get_pbp_tables(season: int, window: str) -> Optional[Dict[str, Any]]:
    key = (int(season), window)
    fresh = _tables_cache_fresh(key)
    if fresh is not None:
        return fresh
    # Build at most one tables set at a time per worker, then re-check:
    # concurrent first-hits for other players queue here and pick up
    # the finished tables instead of each building their own copy.
    # (The previous code also returned None forever once an entry
    # expired; an expired entry now rebuilds.)
    with _TABLES_BUILD_LOCK:
        fresh = _tables_cache_fresh(key)
        if fresh is not None:
            return fresh
        df = load_pbp(season)
        if df is None or len(df) == 0:
            return None
        try:
            weeks = sorted({int(w) for w in df["week"].dropna().unique()})
        except Exception:
            weeks = []
        last4 = set(weeks[-4:])
        if window == "last4":
            base = set(weeks[-8:-4])
            use_weeks = last4
        else:
            base = set(weeks)
            use_weeks = set(weeks)
        try:
            sub = df[df["week"].isin(sorted(use_weeks | base))] if window == "last4" else df
        except Exception:
            sub = df
        tables = build_pbp_tables(sub, get_gsis_crosswalk(), last4, base,
                                  window_weeks=set(weeks) if window != "last4" else last4)
        del df, sub
        with _CACHE_LOCK:
            _PBP_TABLES_CACHE[key] = (time.time(), tables)
            while len(_PBP_TABLES_CACHE) > _TABLES_MAX:
                _PBP_TABLES_CACHE.popitem(last=False)
        return tables


# ---------------------------------------------------------------------------
# Pool loading (weekly metrics + players index)
# ---------------------------------------------------------------------------


def load_player_index() -> Dict[str, Any]:
    try:
        from utils.utils import load_players_index
        return load_players_index() or {}
    except Exception:
        return {}


def load_pool(season: int) -> Dict[str, Dict[str, Any]]:
    """{player_id: {pos, team, name, series}} for a season (tests patch this)."""
    from data_building.weekly_metrics import get_weekly_series_by_player
    idx = load_player_index()
    by_player = get_weekly_series_by_player(int(season), 18) or {}
    pool: Dict[str, Dict[str, Any]] = {}
    for pid, series in by_player.items():
        if not series:
            continue
        meta = idx.get(str(pid)) or {}
        pos = (series[-1].get("position") or meta.get("pos") or "").upper()
        if pos not in AXIS_FEATURES:
            continue
        pool[str(pid)] = {
            "pos": pos,
            "team": meta.get("team") or "",
            "name": meta.get("name") or meta.get("full_name") or str(pid),
            "series": series,
        }
    return pool


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _window_series(series, window: str):
    rows = sorted(series or [], key=lambda r: int(r.get("week") or 0))
    if window == "last4":
        return rows[-4:]
    return rows


def _share(part: float, total: float) -> float:
    return round(part / total * 100.0, 1) if total > 0 else 0.0


def _trend_label(last4: float, base: float) -> str:
    diff = last4 - base
    if diff >= 3:
        return "Rising"
    if diff <= -3:
        return "Falling"
    return "Steady"


def _zone_payload(player_zones, league_zones, is_qb: bool,
                  totals, team_air_total: float, player_air: float):
    total_primary = sum(c["primary"] for c in player_zones.values())
    total_secondary = sum(c["secondary"] for c in player_zones.values())
    total_last4 = sum(c["primary_last4"] for c in player_zones.values())
    total_base = sum(c["primary_base"] for c in player_zones.values())
    league_primary = sum(c["primary"] for c in league_zones.values())
    zones = []
    td_cells, big_cells = [], []
    for row_name in ZONE_ROWS:
        for col in ZONE_COLS:
            cell = player_zones.get((row_name, col)) or _blank_zone()
            lg = league_zones.get((row_name, col)) or _blank_zone()
            lg_catch = (lg["secondary"] / lg["primary"]) if lg["primary"] else None
            lg_rating = (passer_rating(lg["primary"], lg["secondary"], lg["yards"],
                                       lg["td"], lg["int"]) if is_qb and lg["primary"] else None)
            share_p = _share(cell["primary"], total_primary)
            zones.append({
                "row": row_name, "col": col,
                "primary": int(cell["primary"]),
                "secondary": int(cell["secondary"]),
                "share_primary": share_p,
                "share_secondary": _share(cell["secondary"], total_secondary),
                "yards": int(cell["yards"]),
                "td": int(cell["td"]),
                "int": int(cell["int"]) if is_qb else 0,
                "first_downs": int(cell["first_downs"]),
                "avg_depth": round(cell["air"] / cell["primary"], 1) if cell["primary"] else 0.0,
                "catch_rate": (round(cell["secondary"] / cell["primary"], 3)
                               if cell["primary"] else None),
                "rating": (passer_rating(cell["primary"], cell["secondary"],
                                         cell["yards"], cell["td"], cell["int"])
                           if is_qb and cell["primary"] else None),
                "league_catch_rate": (round(lg_catch, 3) if lg_catch is not None else None),
                "league_rating": lg_rating,
                "trend_last4": _share(cell["primary_last4"], total_last4),
                "trend_base": _share(cell["primary_base"], total_base),
                "trend_label": _trend_label(_share(cell["primary_last4"], total_last4),
                                            _share(cell["primary_base"], total_base)),
            })
            if cell["td"] > 0:
                td_cells.append({"row": row_name, "col": col, "count": int(cell["td"])})
            if cell["big"] > 0:
                big_cells.append({"row": row_name, "col": col})
    att = totals.get("att", 0) if totals else 0
    comp = totals.get("comp", 0) if totals else 0
    summary = {
        "adot": round(totals["air"] / att, 1) if att and totals else 0.0,
        "air_yards_share": (round(player_air / team_air_total * 100.0, 1)
                            if not is_qb and team_air_total > 0 else None),
        "catch_rate": round(comp / att, 3) if att else 0.0,
        "ypt_or_ypa": round(totals["yards"] / att, 1) if att and totals else 0.0,
        "deep_rate": round(totals["deep"] / att * 100.0, 1) if att and totals else 0.0,
        "comp_pct": round(comp / att * 100.0, 1) if att else 0.0,
    }
    if league_primary == 0:
        pass  # baselines simply stay as computed (None rates when empty)
    return zones, summary, td_cells, big_cells


def _gap_payload(player_lanes, league_lanes, qb_row, directions):
    total_car = sum(c["carries"] for c in player_lanes.values())
    total_last4 = sum(c["carries_last4"] for c in player_lanes.values())
    total_base = sum(c["carries_base"] for c in player_lanes.values())
    league_car = sum(c["carries"] for c in league_lanes.values())
    league_yards = sum(c["yards"] for c in league_lanes.values())
    lanes = []
    td_lanes, big_lanes = [], []
    tot_yards = sum(c["yards"] for c in player_lanes.values())
    tot_td = sum(c["td"] for c in player_lanes.values())
    tot_fd = sum(c["first_downs"] for c in player_lanes.values())
    tot_stuffed = sum(c["stuffed"] for c in player_lanes.values())
    tot_exp = sum(c["explosive"] for c in player_lanes.values())
    tot_success = sum(c["success"] for c in player_lanes.values())
    for lane in LANES:
        cell = player_lanes.get(lane) or _blank_lane()
        lg = league_lanes.get(lane) or _blank_lane()
        lanes.append({
            "lane": lane,
            "share": _share(cell["carries"], total_car),
            "carries": int(cell["carries"]),
            "yards": int(cell["yards"]),
            "ypc": round(cell["yards"] / cell["carries"], 1) if cell["carries"] else 0.0,
            "td": int(cell["td"]),
            "first_downs": int(cell["first_downs"]),
            "stuffed_pct": _share(cell["stuffed"], cell["carries"]),
            "explosive_pct": _share(cell["explosive"], cell["carries"]),
            "longest": int(cell["longest"]),
            "league_share": _share(lg["carries"], league_car),
            "league_ypc": round(lg["yards"] / lg["carries"], 1) if lg["carries"] else 0.0,
            "trend_last4": _share(cell["carries_last4"], total_last4),
            "trend_base": _share(cell["carries_base"], total_base),
        })
        if cell["td"] > 0:
            td_lanes.append(lane)
        if cell["longest"] >= 20:
            big_lanes.append(lane)
    tendency = {"left": 0.0, "middle": 0.0, "right": 0.0}
    if directions:
        tot_dir = sum(directions.values())
        for k in tendency:
            tendency[k] = _share(directions.get(k, 0), tot_dir)
    summary = {
        "carries": int(total_car),
        "ypc": round(tot_yards / total_car, 1) if total_car else 0.0,
        "success_rate": _share(tot_success, total_car),
        "stuffed_pct": _share(tot_stuffed, total_car),
        "explosive_pct": _share(tot_exp, total_car),
        "rush_td": int(tot_td),
    }
    _ = (tot_fd, league_yards)
    return {"lanes": lanes, "summary": summary, "tendency": tendency,
            "td_lanes": td_lanes, "big_lanes": big_lanes}


def build_player_payload(sleeper_id: str, season: int, window: str,
                         pool: Optional[Dict[str, Dict[str, Any]]] = None,
                         tables: Optional[Dict[str, Any]] = None,
                         prior_pool: Optional[Dict[str, Dict[str, Any]]] = None,
                         pa_feats: Optional[Dict[str, Dict[str, float]]] = None
                         ) -> Optional[Dict[str, Any]]:
    """Assemble the full /api/player-role payload. None if player unknown."""
    season = int(season)
    window = window if window in ("season", "last4", "lastseason") else "season"
    eff_season = season - 1 if window == "lastseason" else season
    if pool is None:
        try:
            pool = load_pool(eff_season)
        except Exception:
            pool = {}
    pid = str(sleeper_id)
    if pid not in pool:
        return None
    entry = pool[pid]
    pos = entry.get("pos")

    # Window the pool series.
    win_pool: Dict[str, Dict[str, Any]] = {}
    for p, e in pool.items():
        s = _window_series(e.get("series") or [], window)
        if s:
            win_pool[p] = {**e, "series": s}
    if pid not in win_pool:
        return None

    if tables is None:
        try:
            tables = get_pbp_tables(eff_season, window)
        except Exception:
            tables = None
    pbp_feats = (tables or {}).get("feats") or {}

    # Play-action feats (FTN) merge into the PBP feature dict so they
    # flow through player_features as pbp_play_action_rate / _epa.
    if pa_feats is None:
        try:
            pa_feats = load_pa_feats(win_pool, eff_season)
        except Exception:
            pa_feats = {}
    if pa_feats:
        merged_feats = {p: dict(f) for p, f in pbp_feats.items()}
        for p, pa in pa_feats.items():
            merged_feats.setdefault(p, {}).update(pa)
        pbp_feats = merged_feats

    computed = compute_pool(win_pool, pbp_feats)

    # Early-season prior: blend last season's job score at low weight.
    prior_score = None
    if window == "season":
        all_weeks = [int(r.get("week") or 0) for r in win_pool[pid]["series"]]
        if all_weeks and max(all_weeks) <= 3:
            try:
                pp = prior_pool if prior_pool is not None else load_pool(eff_season - 1)
            except Exception:
                pp = {}
            if pid in pp:
                prior_comp = compute_pool(
                    {p: {**e, "series": e.get("series") or []} for p, e in pp.items()},
                    {})
                if pid in prior_comp:
                    prior_score = prior_comp[pid]["score"]

    role = build_role(pid, win_pool, computed, prior_score=prior_score)
    role_out = {k: v for k, v in role.items() if not k.startswith("_")}
    jobs = {p: job_for(e.get("pos"), computed[p]["score"], computed[p]["feats"])
            for p, e in win_pool.items() if p in computed}
    timeline = build_timeline(pid, win_pool, pbp_feats)
    comparables, twin = build_comparables(pid, win_pool, computed, jobs)

    # Target map / gap map from PBP tables.
    target_map = None
    gap_map = None
    qb_rush = None
    if tables:
        is_qb = pos == "QB"
        side = tables["passing"] if is_qb else tables["receiving"]
        player_zones = side["zones"].get(pid) or {}
        totals = side["totals"].get(pid) or {}
        if totals.get("att", 0) > 0:
            team = _TEAM_ALIAS.get(entry.get("team") or "", entry.get("team") or "")
            team_air_total = 0.0
            for tname, val in side["team_air"].items():
                if _TEAM_ALIAS.get(tname, tname) == team:
                    team_air_total += val
            zones, summary, td_cells, big_cells = _zone_payload(
                player_zones, side["league"], is_qb, totals,
                team_air_total, side["player_air"].get(pid, 0.0))
            if is_qb and pa_feats.get(pid):
                summary["play_action_rate"] = pa_feats[pid].get("play_action_rate")
                if pa_feats[pid].get("play_action_epa") is not None:
                    summary["play_action_epa"] = pa_feats[pid]["play_action_epa"]
            target_map = {"kind": "passing" if is_qb else "receiving",
                          "zones": zones, "summary": summary,
                          "td_cells": td_cells, "big_cells": big_cells}
        lanes = tables["rushing"]["lanes"].get(pid) or {}
        if sum(c["carries"] for c in lanes.values()) > 0:
            gap_map = _gap_payload(lanes, tables["rushing"]["league"],
                                   tables["rushing"]["qb"].get(pid),
                                   tables["rushing"]["direction"].get(pid))
        if is_qb:
            q = tables["rushing"]["qb"].get(pid)
            if q and q["att"] > 0:
                dirs = tables["rushing"]["direction"].get(pid) or {}
                top_dir = max(dirs, key=lambda k: dirs[k]) if dirs else "middle"
                qb_rush = {
                    "att": int(q["att"]), "yards": int(q["yards"]),
                    "ypc": round(q["yards"] / q["att"], 1) if q["att"] else 0.0,
                    "td": int(q["td"]),
                    "scramble_att": int(q["scramble"]),
                    "designed_att": int(q["att"] - q["scramble"]),
                    "top_direction": top_dir,
                }

    weeks = sorted({int(r.get("week") or 0) for r in win_pool[pid]["series"]})
    return {
        "player_id": pid,
        "name": entry.get("name") or "",
        "pos": pos,
        "team": entry.get("team") or "",
        "season": eff_season,
        "window": window,
        "weeks": weeks,
        "role": role_out,
        "timeline": timeline,
        "target_map": target_map,
        "gap_map": gap_map,
        "qb_rush": qb_rush,
        "comparables": comparables,
        "twin": twin,
    }
