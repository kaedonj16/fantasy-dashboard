"""Shared per-player weekly scoring distribution model.

One engine, two consumers: the Lineup Lab (Start/Sit tab) and the playoff odds
sim. Every player gets a weekly scoring *profile*::

    {"player_id", "pos", "mean", "std", "skew_alpha", "dud_risk",
     "n_games", "spike", "factors"}

The contract, and the thing that keeps this honest:

* **The mean is never touched here.** It is passed in from the caller's
  projection (start/sit / Sleeper). Sleeper's feed already bakes in role
  changes, teammate injuries, and matchup context. Re-adjusting the mean here
  would double-count.
* Everything in this module tunes the **shape** around that mean: how wide the
  spread is, how skewed it is, and whether there's a dud-week left tail.

Shape pipeline per player
--------------------------
1. Gather weekly fantasy scores (this season + two priors) from Sleeper weekly
   stat files, scored with a fixed PPR-ish map (shape only; rescaled to the
   caller's mean so league scoring stays correct).
2. Blend variances: this season vs history vs position baseline, weighted by
   sample size. Early season leans on history; ~week 8+ is nearly all current.
3. Apply factor adjustments (each recorded in ``factors`` for transparency):
   role change, teammate-out volume lock, QB-out / new-QB regime uncertainty,
   TD dependency (skew), boom/bust profile, team target concentration,
   questionable-but-active dud mixture, Vegas game script, short week / travel,
   catchable-ball %, unrealized air yards (ceiling).

Every factor degrades gracefully: missing data -> neutral, never an exception.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

# Bump when the math changes so downstream cache keys invalidate.
MODEL_VERSION = 2

# ---------------------------------------------------------------------------
# Fixed scoring map for SHAPE computation only. Never used for means; the
# resulting std is rescaled to the caller's projection (see _rescale_std).
# Full-PPR-ish so WR/TE shapes aren't understated.
# ---------------------------------------------------------------------------
_STD_SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "rec": 1.0,          # receptions
    "fum_lost": -2.0,
    "pass_2pt": 2.0,
    "rush_2pt": 2.0,
    "rec_2pt": 2.0,
}

# Opportunity keys: a week with none of these is a DNP/bye, not a dud.
_OPP_KEYS = ("pass_att", "rush_att", "rec_tgt", "targets", "rush_carries")

# Position baseline: std = m * ppg + b. Mirrors the playoff sim's _POS_STD so
# the fallback behaves identically where profiles ~= position averages.
_POS_STD: Dict[str, Tuple[float, float]] = {
    "QB": (0.28, 3.0),
    "RB": (0.45, 2.0),
    "WR": (0.50, 2.0),
    "TE": (0.42, 1.5),
    "K": (0.00, 4.0),
    "DEF": (0.00, 5.5),
}
_POS_STD_DEFAULT = (0.42, 2.0)

# Base skew of weekly fantasy scores (right-skewed: boom weeks, thin left
# tail). Matches the playoff sim's skew-normal alpha.
_BASE_SKEW_ALPHA = 2.0

# Stack correlations (Gaussian copula inputs; see correlation_pairs).
_CORR_QB_PASS_CATCHER = 0.30
_CORR_RB_DST = -0.15
_CORR_K_QB = 0.15

# Questionable-but-active dud mixture probability.
_Q_DUD_RISK = 0.12

# In-memory caches.
_WEEK_FILES_CACHE: Dict[int, Tuple[float, Dict[str, List[dict]]]] = {}
_PLAYERS_INDEX_CACHE: Dict[str, Any] = {"mtime": 0.0, "data": {}}
_PROFILE_CACHE: Dict[Tuple, Tuple[float, dict]] = {}
_PROFILE_TTL = 6 * 3600


# ---------------------------------------------------------------------------
# Data access (all defensive; empty on any failure)
# ---------------------------------------------------------------------------

def _week_files(season: int) -> Dict[str, List[dict]]:
    """pid -> list of raw weekly stat rows for a season, cached 6h."""
    now = time.time()
    hit = _WEEK_FILES_CACHE.get(season)
    if hit and now - hit[0] < _PROFILE_TTL:
        return hit[1]
    out: Dict[str, List[dict]] = {}
    try:
        pattern = os.path.join(
            str(CACHE_DIR), "sleeper_stats", f"sleeper_stats_s{int(season)}_w*.json"
        )
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as fh:
                    weekly = json.load(fh) or {}
            except Exception:
                continue
            for pid, row in weekly.items():
                if isinstance(row, dict):
                    out.setdefault(str(pid), []).append(row)
    except Exception:
        logger.debug("player_distributions: week files unreadable", exc_info=True)
    _WEEK_FILES_CACHE[season] = (now, out)
    return out


def _players_index() -> Dict[str, dict]:
    """Sleeper players index (injury_status, team, position, depth chart)."""
    try:
        from data_building.updates.update_players import path_players_index
        path = str(path_players_index())
        mtime = os.path.getmtime(path)
        if mtime != _PLAYERS_INDEX_CACHE["mtime"]:
            with open(path) as fh:
                data = json.load(fh) or {}
            # Index may be {pid: {...}} or {"players": {...}}; normalize.
            if isinstance(data, dict) and "players" in data and isinstance(data["players"], dict):
                data = data["players"]
            _PLAYERS_INDEX_CACHE.update(mtime=mtime, data=data if isinstance(data, dict) else {})
    except Exception:
        logger.debug("player_distributions: players index unreadable", exc_info=True)
    return _PLAYERS_INDEX_CACHE["data"]


def _fnum(row: dict, key: str) -> float:
    try:
        v = row.get(key)
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fixed_fp(row: dict) -> float:
    return sum(_fnum(row, k) * w for k, w in _STD_SCORING.items())


def _played(row: dict) -> bool:
    return any(_fnum(row, k) > 0 for k in _OPP_KEYS)


def _weekly_scores(pid: str, season: int) -> Tuple[List[float], List[float], List[float]]:
    """(fantasy pts, opportunities, td pts) per played week, oldest first."""
    pts, opps, td_pts = [], [], []
    for row in _week_files(season).get(str(pid), []):
        if not _played(row):
            continue
        pts.append(_fixed_fp(row))
        opps.append(_fnum(row, "rec_tgt") + _fnum(row, "targets")
                    + _fnum(row, "rush_att") + _fnum(row, "rush_carries"))
        td_pts.append((_fnum(row, "pass_td") * 4.0 + (_fnum(row, "rush_td") + _fnum(row, "rec_td")) * 6.0))
    return pts, opps, td_pts


def _wvar(xs: List[float], weights: Optional[List[float]] = None) -> Tuple[float, float]:
    """Weighted variance + effective sample size. Empty -> (0, 0)."""
    if not xs:
        return 0.0, 0.0
    w = weights or [1.0] * len(xs)
    sw = sum(w)
    if sw <= 0:
        return 0.0, 0.0
    mean = sum(x * wi for x, wi in zip(xs, w)) / sw
    var = sum(wi * (x - mean) ** 2 for x, wi in zip(xs, w)) / sw
    n_eff = sw * sw / sum(wi * wi for wi in w) if any(wi for wi in w) else 0.0
    return var, n_eff


def _pos_baseline_std(pos: str, ppg: float) -> float:
    m, b = _POS_STD.get(pos, _POS_STD_DEFAULT)
    return m * ppg + b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_inputs_signature(season: int, week: int) -> str:
    """Hash of everything profiles depend on; downstream cache keys should
    include this so mid-week injury/usage changes invalidate."""
    try:
        from data_building.updates.update_players import path_players_index
        path = str(path_players_index())
        idx_sig = f"{os.path.getmtime(path):.0f}:{os.path.getsize(path)}"
    except Exception:
        idx_sig = "no-index"
    raw = f"v{MODEL_VERSION}:{season}:{week}:{idx_sig}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_profiles(
    requests: List[dict],
    season: int,
    week: int,
    ctx: Optional[dict] = None,
) -> Dict[str, dict]:
    """Build profiles for many players at once (shared data fetched once).

    requests: [{"player_id", "pos", "mean"}, ...]
    ctx: optional pre-fetched context (see module docstring / _CTX_DEFAULTS).
         Missing keys are resolved internally where possible, else neutral.
    Returns {player_id: profile}.
    """
    ctx = ctx or {}
    players = _players_index()
    season_files_cur = _week_files(season)
    # Team target concentration for the current season (Herfindahl of
    # targets). Computed lazily on the first cache miss below: when every
    # requested profile is served from _PROFILE_CACHE, the league-wide
    # pass is skipped entirely.
    concentration: Optional[Dict[str, float]] = None

    # Per-player profile cache: same key shape and TTL as the
    # get_player_profile wrapper. Callers that pass ctx overrides always
    # build fresh — their inputs are not part of the key, exactly as
    # before this cache was consulted here.
    use_cache = not ctx
    sig = profile_inputs_signature(season, week) if use_cache else ""
    now = time.time()

    out: Dict[str, dict] = {}
    for req in requests:
        pid = str(req.get("player_id"))
        pos = (req.get("pos") or "").upper()
        mean = _safe_float(req.get("mean"), 0.0)
        key = (pid, pos, round(float(mean), 2), season, week, sig)
        if use_cache:
            hit = _PROFILE_CACHE.get(key)
            if hit and now - hit[0] < _PROFILE_TTL:
                out[pid] = hit[1]
                continue
        try:
            if concentration is None:
                concentration = _team_concentration(season, players)
            prof = _build_one(pid, pos, mean, season, week, ctx, players, concentration)
        except Exception:
            logger.debug("player_distributions: profile failed for %s", pid, exc_info=True)
            # Build-error fallbacks are never cached: a transient failure
            # must not stick for the whole TTL.
            out[pid] = _fallback_profile(pid, pos, mean, "build_error")
            continue
        out[pid] = prof
        if use_cache:
            _PROFILE_CACHE[key] = (now, prof)
    return out


def get_player_profile(
    player_id: str,
    pos: str,
    mean: float,
    season: int,
    week: int,
    ctx: Optional[dict] = None,
) -> dict:
    """Single-player convenience wrapper (6h in-memory cache)."""
    key = (str(player_id), pos, round(float(mean or 0), 2), season, week,
           profile_inputs_signature(season, week))
    now = time.time()
    hit = _PROFILE_CACHE.get(key)
    if hit and now - hit[0] < _PROFILE_TTL:
        return hit[1]
    prof = build_profiles(
        [{"player_id": player_id, "pos": pos, "mean": mean}], season, week, ctx
    )[str(player_id)]
    _PROFILE_CACHE[key] = (now, prof)
    return prof


def correlation_pairs(
    pids: List[str],
    season: int,
    ctx: Optional[dict] = None,
) -> Dict[Tuple[str, str], float]:
    """Pairwise correlations for the Gaussian copula.

    Same-team QB<->pass catcher 0.30, RB<->DST -0.15, K<->QB 0.15.
    Returns {(pid_a, pid_b): rho} with pid_a < pid_b.
    """
    players = _players_index()
    info: Dict[str, Tuple[str, str]] = {}
    for pid in pids:
        p = players.get(str(pid)) or {}
        info[str(pid)] = (str(p.get("team") or ""), str(p.get("position") or "").upper())
    pairs: Dict[Tuple[str, str], float] = {}
    ids = [str(p) for p in pids]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            team_a, pos_a = info[a]
            team_b, pos_b = info[b]
            if not team_a or team_a != team_b:
                continue
            rho = 0.0
            pair = {pos_a, pos_b}
            if "QB" in pair and pair & {"WR", "TE", "RB"}:
                rho = _CORR_QB_PASS_CATCHER
            elif pair == {"RB", "DEF"}:
                rho = _CORR_RB_DST
            elif pair == {"K", "QB"}:
                rho = _CORR_K_QB
            if rho:
                pairs[(a, b)] = rho
    return pairs


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _fallback_profile(pid: str, pos: str, mean: float, reason: str) -> dict:
    return {
        "player_id": pid,
        "pos": pos,
        "mean": mean,
        "std": _pos_baseline_std(pos, mean),
        "skew_alpha": _BASE_SKEW_ALPHA,
        "dud_risk": 0.0,
        "n_games": 0.0,
        "spike": None,
        "factors": {"fallback": reason},
    }


def _team_concentration(season: int, players: Dict[str, dict]) -> Dict[str, float]:
    """Herfindahl index of team target shares (top-4), 0..1. High = funnel."""
    team_targets: Dict[str, Dict[str, float]] = {}
    for pid, rows in _week_files(season).items():
        p = players.get(str(pid)) or {}
        team = str(p.get("team") or "")
        if not team:
            continue
        tot = sum(_fnum(r, "rec_tgt") + _fnum(r, "targets") for r in rows)
        if tot > 0:
            team_targets.setdefault(team, {})[str(pid)] = tot
    out: Dict[str, float] = {}
    for team, tgts in team_targets.items():
        total = sum(tgts.values())
        if total <= 0:
            continue
        shares = sorted((v / total for v in tgts.values()), reverse=True)[:4]
        out[team] = sum(s * s for s in shares)
    return out


def _build_one(
    pid: str,
    pos: str,
    mean: float,
    season: int,
    week: int,
    ctx: dict,
    players: Dict[str, dict],
    concentration: Dict[str, float],
) -> dict:
    factors: Dict[str, Any] = {}
    pinfo = players.get(pid) or {}
    team = str(pinfo.get("team") or "")
    status = str(ctx.get("injury_status", {}).get(pid)
                  or pinfo.get("injury_status") or "").strip().upper()

    # ---- Step 1+2: gather + blend -------------------------------------
    cur_pts, cur_opps, cur_td = _weekly_scores(pid, season)
    hist_pts: List[float] = []
    hist_w: List[float] = []
    hist_opps: List[float] = []
    for back, s in (1, season - 1), (2, season - 2):
        pts, opps, _ = _weekly_scores(pid, s)
        hist_pts.extend(pts)
        hist_opps.extend(opps)
        hist_w.extend([1.0] * len(pts))

    # Role change: caller override wins; else auto-detect from opportunity swing.
    role_change = ctx.get("role_change", {}).get(pid)
    if role_change is None:
        role_change = _detect_role_change(cur_opps, hist_opps)
    if role_change:
        hist_w = [w * 0.25 for w in hist_w]  # old regime heavily discounted
        factors["role_change"] = True

    var_cur, n_cur = _wvar(cur_pts)
    var_hist, n_hist = _wvar(hist_pts, hist_w)

    if pos in ("K", "DEF") or (n_cur + n_hist) < 1:
        # Kickers/defenses: not enough signal in stat-file scoring; baseline.
        base_std = _pos_baseline_std(pos, mean)
        n_eff = 0.0
        factors["baseline_only"] = True
    else:
        base_var_pos = _pos_baseline_std(pos, mean) ** 2
        n_prior = max(0.0, 4.0 - n_cur - n_hist)  # pseudo-games at baseline
        tot = n_cur + n_hist + n_prior
        if tot <= 0:
            base_std, n_eff = _pos_baseline_std(pos, mean), 0.0
        else:
            var = (n_cur * var_cur + n_hist * var_hist + n_prior * base_var_pos) / tot
            # Rescale: shape came from the fixed map; size follows the mean.
            raw_mean = ((sum(cur_pts) + sum(hist_pts)) / max(1, len(cur_pts) + len(hist_pts)))
            scale = (mean / raw_mean) if raw_mean > 1e-6 and mean > 0 else 1.0
            base_std = math.sqrt(max(var, 0.0)) * scale
            n_eff = n_cur + n_hist
            factors.update({
                "n_cur": round(n_cur, 1), "n_hist": round(n_hist, 1),
                "rescale": round(scale, 2),
            })

    std = max(base_std, 1.0)
    skew = _BASE_SKEW_ALPHA
    dud_risk = 0.0

    # ---- Step: TD dependency -> skew + spread --------------------------
    td_pts_total = sum(cur_td)
    fp_total = sum(cur_pts)
    td_share = (td_pts_total / fp_total) if fp_total > 1e-6 else 0.0
    td_share = max(0.0, min(1.0, td_share))
    if td_share > 0.05 and pos in ("RB", "WR", "TE", "QB"):
        skew = _BASE_SKEW_ALPHA + 3.0 * td_share
        std *= (1.0 + 0.5 * td_share)
        factors["td_share"] = round(td_share, 2)

    # ---- Step: boom/bust profile ---------------------------------------
    # (skipped for baseline-only players: the label would be derived from the
    # same baseline it then multiplies — circular.)
    if not factors.get("baseline_only"):
        label = str(ctx.get("consistency", {}).get(pid) or _profile_label(std, mean)).lower()
        prof_mult = {"steady": 0.9, "volatile": 1.1, "boom-or-bust": 1.25,
                     "boom": 1.25, "bust": 1.25}.get(label, 1.0)
        # Don't stack absurdly with TD dependency: cap combined lift at 1.4x.
        pre = std
        std *= prof_mult
        if td_share > 0.05 and std / max(pre / prof_mult, 1e-6) > 1.4:
            std = (pre / prof_mult) * 1.4
        if prof_mult != 1.0:
            factors["profile"] = label

    # ---- Step: target concentration ------------------------------------
    if team and pos in ("WR", "TE", "RB") and concentration.get(team, 0) > 0.30:
        # Is this player a top-2 target on his team?
        top2 = _is_top2_target(pid, team, season, players)
        if top2 is True:
            std *= 0.95
            factors["funnel_top2"] = True
        elif top2 is False:
            std *= 1.15
            factors["funnel_scraps"] = True

    # ---- Step: teammate / QB regime ------------------------------------
    if ctx.get("teammate_out", {}).get(pid):
        std *= 0.9  # volume concentrates; floor gets safer
        factors["teammate_out"] = True
    if pos in ("WR", "TE", "RB") and team:
        if ctx.get("qb_out", {}).get(team):
            std *= 1.15
            factors["qb_out"] = True
        elif ctx.get("new_qb", {}).get(team):
            std *= 1.2
            factors["new_qb"] = True

    # ---- Step: questionable-but-active dud mixture ---------------------
    if status == "QUESTIONABLE":
        dud_risk = _Q_DUD_RISK
        std *= 1.1  # approximation for consumers without mixture support
        factors["questionable"] = True

    # ---- Step: Vegas game script (shape only) ---------------------------
    spread = ctx.get("vegas_spread", {}).get(team)
    if spread is not None:
        try:
            sp = float(spread)
            if sp >= 7 and pos == "RB":
                std *= 0.9; factors["fav_rb"] = True
            elif sp >= 7 and pos == "WR":
                std *= 1.1; factors["fav_wr"] = True
            elif sp <= -7 and pos == "RB":
                std *= 1.15; factors["dog_rb"] = True
            elif sp <= -7 and pos == "WR":
                std *= 0.95; factors["dog_wr"] = True
        except (TypeError, ValueError):
            pass

    # ---- Step: short week / travel --------------------------------------
    tags = ctx.get("game_tags", {}).get(pid) or {}
    if tags.get("short_week") or tags.get("weird_travel"):
        std *= 1.1
        factors["weird_game"] = True

    # ---- Step: catchable ball % (feature-gated) --------------------------
    cz = ctx.get("catchable_z", {}).get(pid)
    if cz is not None and pos in ("WR", "TE"):
        try:
            cz = float(cz)
            if cz >= 1.0:
                std *= 0.92; factors["catchable_high"] = True
            elif cz <= -1.0:
                std *= 1.08; factors["catchable_low"] = True
        except (TypeError, ValueError):
            pass

    # ---- Step: unrealized air yards -> ceiling ---------------------------
    ayz = ctx.get("unrealized_ay_z", {}).get(pid)
    if ayz is not None and pos in ("WR", "TE"):
        try:
            skew += min(1.5, max(0.0, float(ayz)) * 0.75)
            if float(ayz) > 1.0:
                factors["unrealized_ay"] = True
        except (TypeError, ValueError):
            pass

    std = min(max(std, 1.0), 30.0)

    # Spike: the player's best single game this season. Demonstrated
    # explosions set a floor on claimed upside elsewhere (Chase Upside):
    # a 90th-percentile fit can never see a 40-point tail. Needs >= 2
    # played games so one fluke doesn't define a player.
    spike = max(cur_pts) if len(cur_pts) >= 2 else None

    return {
        "player_id": pid,
        "pos": pos,
        "mean": mean,          # passed through; never modified here
        "std": round(std, 2),
        "skew_alpha": round(skew, 2),
        "dud_risk": round(dud_risk, 3),
        "n_games": round(n_eff, 1),
        "spike": round(spike, 1) if spike is not None else None,
        "factors": factors,
    }


def _detect_role_change(cur_opps: List[float], hist_opps: List[float]) -> bool:
    """Opportunity-per-game swing > 40% with >= 2 current games."""
    if len(cur_opps) < 2 or not hist_opps:
        return False
    cur_avg = sum(cur_opps) / len(cur_opps)
    hist_avg = sum(hist_opps) / len(hist_opps)
    if hist_avg < 1e-6:
        return cur_avg > 3.0
    return abs(cur_avg - hist_avg) / hist_avg > 0.40


def _profile_label(std: float, mean: float) -> str:
    if mean <= 0:
        return "volatile"
    cv = std / mean
    if cv < 0.45:
        return "steady"
    if cv < 0.65:
        return "volatile"
    return "boom-or-bust"


def _is_top2_target(pid: str, team: str, season: int, players: Dict[str, dict]) -> Optional[bool]:
    """Is pid a top-2 target-getter on his team this season? None if unknown."""
    try:
        tgts: List[Tuple[str, float]] = []
        for other_pid, rows in _week_files(season).items():
            p = players.get(str(other_pid)) or {}
            if str(p.get("team") or "") != team:
                continue
            tot = sum(_fnum(r, "rec_tgt") + _fnum(r, "targets") for r in rows)
            if tot > 0:
                tgts.append((str(other_pid), tot))
        tgts.sort(key=lambda t: -t[1])
        top2 = {t[0] for t in tgts[:2]}
        if not tgts:
            return None
        return pid in top2
    except Exception:
        return None
