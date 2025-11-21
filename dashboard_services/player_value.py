# dashboard_services/player_value.py

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, Iterable, Optional

from dashboard_services.sleeper_usage import build_usage_map_for_season
from dashboard_services.utils import load_players_index

# Rough dynasty trade-value scales per position
BASE_SCALE_BY_POSITION: Dict[str, float] = {
    "QB": 130.0,
    "RB": 150.0,
    "WR": 150.0,
    "TE": 120.0,
}


def _usage_score(pos: str, u: Dict[str, float]) -> float:
    """
    Turn usage + production into a 0–1 score.
    PPR is the main driver, usage fills in around it.
    """
    g = u.get("games", 0) or 0
    if g <= 0:
        return 0.0

    ppr_ppg = float(u.get("ppr_ppg", 0.0) or 0.0)
    snap_pct = float(u.get("avg_off_snap_pct", 0.0) or 0.0)
    targets = float(u.get("avg_targets", 0.0) or 0.0)
    carries = float(u.get("avg_carries", 0.0) or 0.0)
    yards = float(u.get("avg_rec_yards", 0.0) or 0.0) + float(u.get("avg_rush_yards", 0.0) or 0.0)
    tds = float(u.get("avg_rec_tds", 0.0) or 0.0) + float(u.get("avg_rush_tds", 0.0) or 0.0)

    # Normalize to 0–1 ranges
    # 25 PPR is elite, 12+ is strong
    ppr_score = min(1.0, ppr_ppg / 25.0)
    snap_score = min(1.0, snap_pct / 0.85)  # 85%+ snaps ~ 1
    vol_score = min(1.0, (targets + carries) / 18.0)  # 18+ opps ~ 1
    yards_score = min(1.0, yards / 110.0)  # 110+ yards ~ 1
    td_score = min(1.0, tds / 1.0)  # 1 TD per game is insane

    # Production dominates, usage refines
    score = (
            0.60 * ppr_score +
            0.10 * snap_score +
            0.10 * vol_score +
            0.10 * yards_score +
            0.10 * td_score
    )

    return max(0.0, min(1.0, score))


def _percentile(sorted_vals, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pct = max(0.0, min(100.0, pct))
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def build_positional_context(
        usage_by_pid: Dict[str, Dict[str, float]],
        players_index: Dict[str, Dict],
        min_games: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Build per-position baselines:
    {
      "WR": {
        "replacement_ppg": float,
        "mean_ppg": float,
        "std_ppg": float,
      },
      ...
    }
    """
    per_pos: Dict[str, list[float]] = defaultdict(list)

    for pid, u in usage_by_pid.items():
        meta = players_index.get(str(pid), {})
        pos = meta.get("pos") or meta.get("position")
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        g = u.get("games", 0) or 0
        if g < min_games:
            continue

        ppr_ppg = float(u.get("ppr_ppg", 0.0) or 0.0)
        if ppr_ppg <= 0:
            continue

        per_pos[pos].append(ppr_ppg)

    ctx: Dict[str, Dict[str, float]] = {}
    for pos, vals in per_pos.items():
        vals = sorted(vals)
        mean_ppg = float(statistics.mean(vals))
        std_ppg = float(statistics.pstdev(vals)) if len(vals) > 1 else 1.0

        # "Replacement" ~ 60th percentile of all players at that position in your data
        replacement_ppg = _percentile(vals, 60.0)

        ctx[pos] = {
            "replacement_ppg": replacement_ppg,
            "mean_ppg": mean_ppg,
            "std_ppg": max(std_ppg, 0.5),  # avoid zero
        }

    return ctx


def _vor_components(
        pos: str,
        u: Dict[str, float],
        pos_ctx: Dict[str, Dict[str, float]],
) -> tuple[float, float]:
    """
    Return (vor_ppg, vor_score_0_1).
    vor_ppg is raw extra PPG vs replacement.
    vor_score_0_1 maps that into ~0–1.
    """
    ppr_ppg = float(u.get("ppr_ppg", 0.0) or 0.0)
    ctx = pos_ctx.get(pos)
    if ctx is None:
        return 0.0, 0.0

    replacement = ctx["replacement_ppg"]
    vor_ppg = max(0.0, ppr_ppg - replacement)

    # Translate extra PPG into diminishing-returns 0–1 (5+ PPG over repl is crazy)
    # vor_score ~ 0.63 at +3 PPG, ~0.86 at +5
    vor_score = 1.0 - math.exp(-vor_ppg / 3.0)

    return vor_ppg, max(0.0, min(1.0, vor_score))


def _age_factor(pos: str, age: Optional[float]) -> float:
    """
    0–1 age score by position. Higher = better 3-year outlook.
    If age is unknown, assume a neutral-ish 0.85.
    """
    if age is None:
        return 0.85

    if pos == "RB":
        # RBs fall off faster
        if age <= 22: return 1.0
        if age <= 24: return 0.95
        if age <= 25: return 0.90
        if age <= 26: return 0.80
        if age <= 27: return 0.70
        if age <= 28: return 0.55
        if age <= 30: return 0.45
        return 0.35

    if pos == "WR":
        # WRs stay strong longer
        if age <= 22: return 1.0
        if age <= 24: return 0.98
        if age <= 26: return 0.95
        if age <= 27: return 0.92
        if age <= 28: return 0.85
        if age <= 29: return 0.78
        if age <= 31: return 0.65
        if age <= 33: return 0.50
        return 0.40

    if pos == "QB":
        if age <= 24: return 0.95
        if age <= 28: return 1.0
        if age <= 31: return 0.95
        if age <= 34: return 0.90
        if age <= 37: return 0.80
        if age <= 40: return 0.65
        return 0.50

    if pos == "TE":
        if age <= 24: return 0.95
        if age <= 26: return 1.0
        if age <= 28: return 0.95
        if age <= 30: return 0.85
        if age <= 32: return 0.70
        if age <= 34: return 0.55
        return 0.45

    # Fallback
    return 0.8


def _age_from_bday(bday: Optional[str], season: int) -> Optional[float]:
    """
    Compute decimal age in years as of Sept 1 of `season`.

    Returns values like:
      22.4, 26.7, 30.1, etc.
    """
    if not bday:
        return None

    try:
        # Handle formats like "1999-05-14" or "1999-05-14T00:00:00Z"
        parts = bday.split("T")[0].split("/")
        month, day, year = map(int, parts[:3])
        dob = date(year, month, day)

        # Dynasty age baseline: as of Sept 1 of the current season
        as_of = date(season, 9, 1)

        # Convert to decimal years
        days = (as_of - dob).days
        age = days / 365.25

        # Round to one decimal place (26.7 style)
        return round(age, 1)

    except Exception:
        return None


def _age_longevity_score(pos: str, age: Optional[float]) -> float:
    """
    0–1 score for expected *three-year* window based on position + age.

    - RB: early peak, fast decline
    - WR / TE: longer prime
    - QB: very long prime

    We keep it simple & piecewise, no heavy math.
    """
    if age is None:
        # Unknown age -> neutral-ish but not great
        return 0.6

    # Clamp age to reasonable bounds to avoid weird inputs
    a = max(20.0, min(38.0, float(age or 0.0)))
    pos = (pos or "").upper()

    # ---- Running backs: sharp drop after ~25–26 ----
    if pos == "RB":
        if a <= 22:   return 1.00  # rookie phenom
        if a <= 23:   return 0.95
        if a <= 24:   return 0.90
        if a <= 25:   return 0.80
        if a <= 26:   return 0.70  # JT range
        if a <= 27:   return 0.55
        if a <= 28:   return 0.40
        if a <= 30:   return 0.30
        return 0.20

    # ---- Wide receivers: longer window, especially 22–26 ----
    if pos == "WR":
        if a <= 22:   return 1.00  # elite prospect (JSN range)
        if a <= 23:   return 0.98
        if a <= 24:   return 0.96
        if a <= 25:   return 0.93
        if a <= 26:   return 0.90
        if a <= 27:   return 0.85
        if a <= 28:   return 0.78
        if a <= 29:   return 0.72
        if a <= 30:   return 0.65
        if a <= 32:   return 0.55
        return 0.40

    # ---- Tight ends: late bloomers, long-ish prime ----
    if pos == "TE":
        if a <= 23:   return 0.90
        if a <= 24:   return 0.95
        if a <= 27:   return 1.00
        if a <= 29:   return 0.90
        if a <= 31:   return 0.80
        if a <= 33:   return 0.65
        return 0.50

    # ---- QBs: very long window ----
    if pos == "QB":
        if a <= 24:   return 0.95
        if a <= 28:   return 1.00
        if a <= 30:   return 0.95
        if a <= 32:   return 0.90
        if a <= 34:   return 0.80
        if a <= 36:   return 0.70
        return 0.55

    # Fallback for weird positions (FB, etc.)
    if a <= 23:       return 0.90
    if a <= 26:       return 0.80
    if a <= 29:       return 0.65
    return 0.50


def compute_dynasty_trade_value(
        pos: str,
        age: Optional[float],
        usage: Dict[str, float],
        pos_ctx: Dict[str, Dict[str, float]],
        *,
        market_ranks: Optional[Dict[str, int]] = None,
) -> float:
    """
    Full dynasty-style trade value for one player.

    Components:
      - usage_score: 0–1 based on production & usage
      - VOR: PPG above replacement at the same position
      - age_factor: multiplier based on age & position
      - market_score (optional): blend in public market/ECR
      - log-based scaling: make elite players separate more at the top
    """
    pos = pos or "WR"  # default if missing
    usage_score = _usage_score(pos, usage)
    vor_ppg, vor_score = _vor_components(pos, usage, pos_ctx)
    age_mult = _age_factor(pos, age)
    # market = _market_score(player_id, market_ranks)

    # Core 0–1 score (model-based)
    # Heavier weight on production/usage, but VOR matters too
    core_score = (
            0.55 * usage_score +
            0.25 * vor_score +
            0.20 * max(0.0, min(1.1, age_mult))  # age_mult is ~0.7–1.1; clamp-ish
    )

    # Blend with market if available (acts like a "correction" term)
    # if market is not None:
    #     core_score = 0.70 * core_score + 0.30 * market

    # Clamp 0–1 again
    core_score = max(0.0, min(1.0, core_score))

    # Log-based scaling on VOR so 1–2 PPG bumps matter,
    # but 6+ PPG over replacement gets insane value
    # log_factor ~ 1.0 at vor=0, ~1.5 at +3, ~1.9 at +6
    log_factor = 1.0 + math.log1p(vor_ppg) / math.log(1 + 6.0)

    base_scale = BASE_SCALE_BY_POSITION.get(pos, 140.0)

    value = base_scale * core_score * log_factor

    # Round for sanity
    return round(value, 1)


def _build_multi_year_usage(
        season: int,
        weeks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Combine usage / production across (season, season-1, season-2)
    with recency weighting.

    Returns:
      {
        sleeper_id: {
          "games_equiv": float,
          "ppg_weighted": float,
          "opp_weighted": float,   # targets + carries
          "snaps_weighted": float,
        },
        ...
      }
    """
    year_weights = [
        (season, 0.75),
        (season - 1, 0.15),
        (season - 2, 0.10),
    ]

    agg: Dict[str, Dict[str, float]] = {}

    for yr, wt in year_weights:
        try:
            usage_map = build_usage_map_for_season(yr, weeks)
        except Exception as e:
            print(f"[value] warning: failed usage for {yr}: {e}")
            continue

        for pid, u in usage_map.items():
            pid_str = str(pid)
            g = float(u.get("games", 0) or 0.0)
            ppg = float(u.get("ppr_ppg", 0.0) or 0.0)
            tgts = float(u.get("avg_targets", 0.0) or 0.0)
            car = float(u.get("avg_carries", 0.0) or 0.0)
            snaps = float(u.get("avg_off_snaps", 0.0) or 0.0)

            a = agg.setdefault(pid_str, {
                "games_equiv": 0.0,
                "ppg_weighted": 0.0,
                "opp_weighted": 0.0,
                "snaps_weighted": 0.0,
            })

            a["games_equiv"] += wt * g
            a["ppg_weighted"] += wt * ppg
            a["opp_weighted"] += wt * (tgts + car)
            a["snaps_weighted"] += wt * snaps

    return agg


def _production_component(
        u: Dict[str, float],
        pos: str,
        age: float
) -> float:
    """
    Full dynasty production component:
    - weighted 3-year production (ppg, opp, snaps)
    - sample size reliability
    - age-adjusted longevity for 3-year projection
    Returns a 0–1 score.
    """

    # ----- Extract weighted production inputs -----
    g_eq = u.get("games_equiv", 0.0) or 0.0
    ppg = u.get("ppg_weighted", 0.0) or 0.0
    opp = u.get("opp_weighted", 0.0) or 0.0  # touches+targets / game, weighted
    snaps = u.get("snaps_weighted", 0.0) or 0.0

    # ----- Normalize production to 0–1 -----
    ppg_score = min(1.0, ppg / 24.0)  # 24+ PPG = elite
    opp_score = min(1.0, opp / 18.0)  # 18 opp/g = elite usage
    snaps_score = min(1.0, snaps / 60.0)  # 60 snaps/g = full-time role

    # ----- Sample size reliability -----
    reliability = min(1.0, g_eq / 10.0)  # 10 weighted games => full trust

    # ----- Pure performance component (no age) -----
    perf_component = (
                             0.55 * ppg_score +
                             0.30 * opp_score +
                             0.15 * snaps_score
                     ) * reliability

    # ----- Add age longevity -----
    # (0–1 score based on position + age curve)
    age_component = _age_longevity_score(pos, age)  # from earlier

    # ----- Final dynasty-weighted blend -----
    dynasty_score = (
            0.70 * perf_component +  # production drives 70%
            0.30 * age_component  # longevity drives 30%
    )

    return max(0.0, min(1.0, dynasty_score))


def horizon_age_factor(pos: str, age):
    """
    Weighted average of age_factor(pos, age + t) for t = 0..2
    Handles missing age gracefully.
    """
    # fallback when age is None
    if age is None:
        # default peak ages per position
        default_age = {
            "RB": 24,
            "WR": 25,
            "QB": 27,
            "TE": 26,
        }.get(pos, 25)
        age = default_age

    YEAR_WEIGHTS = [0.5, 0.3, 0.2]

    num = 0.0
    den = 0.0
    for t, w in enumerate(YEAR_WEIGHTS):
        num += w * _age_factor(pos, age + t)
        den += w

    return num / den if den > 0 else _age_factor(pos, age)


def _production_component_fixed(u: dict, pos: str) -> float:
    """
    Unified production model with:
      - QB passing integration
      - TE softened
    """

    # default fallback = fantasy points
    ppg = float(u.get("half_ppr_ppg") or 0.0)

    # --- QUARTERBACKS --------------------------
    if pos == "QB":
        att = u.get("avg_pass_att", 0)
        yds = u.get("avg_pass_yds", 0)
        tds = u.get("avg_pass_tds", 0)
        ints = u.get("avg_pass_int", 0)

        # scaled formula (empirically tuned)
        score = (
                (yds / 300.0) * 0.50 +
                (tds / 3.5) * 0.60 +
                (att / 40.0) * 0.20 -
                (ints / 2.5) * 0.20 +
                (ppg / 30.0) * 0.50
        )

        return max(0.0, min(1.0, score))

    # --- RUNNING BACKS -------------------------
    if pos == "RB":
        carries = u.get("avg_carries", 0)
        yds = u.get("avg_rush_yards", 0)
        recs = u.get("avg_receptions", 0)

        score = (
                (carries / 18.0) * 0.40 +
                (yds / 90.0) * 0.40 +
                (recs / 4.0) * 0.20 +
                (ppg / 25.0) * 0.50
        )
        return min(1.0, score)

    # --- WIDE RECEIVERS ------------------------
    if pos == "WR":
        tgt = u.get("avg_targets", 0)
        rec = u.get("avg_receptions", 0)
        yds = u.get("avg_rec_yards", 0)

        score = (
                (tgt / 11.0) * 0.45 +
                (rec / 7.0) * 0.30 +
                (yds / 90.0) * 0.40 +
                (ppg / 22.0) * 0.50
        )
        return min(1.0, score)

    # --- TIGHT ENDS (tuned down) --------------
    if pos == "TE":
        tgt = u.get("avg_targets", 0)
        yds = u.get("avg_rec_yards", 0)

        score = (
                (tgt / 9.0) * 0.30 +
                (yds / 75.0) * 0.25 +
                (ppg / 18.0) * 0.35  # TE scoring deflated
        )
        return min(1.0, score)

    return 0.0


def build_value_table_for_usage(
        season: int,
        weeks: Iterable[int],
) -> Dict[str, float]:
    """
    Build a dynasty-style value table on a 0–999.9 scale using:
      - multi-year weighted production
      - QB passing integration
      - TE weighting tuned down
      - 3-year age curve
      - positional scarcity (VORP)
    """
    players_index = load_players_index()
    multi_usage = _build_multi_year_usage(season, weeks)

    # ----------------------------------------
    # 1) Collect production and metadata
    # ----------------------------------------
    per_pid: Dict[str, Dict[str, object]] = {}

    for pid, u in multi_usage.items():
        meta = players_index.get(pid, {})
        pos = meta.get("pos") or meta.get("position")

        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        bday = meta.get("bDay") or meta.get("dob")
        age = _age_from_bday(bday, season)

        # === NEW === QB now uses full passing integration
        prod_raw = _production_component_fixed(u, pos)

        ppg = float(u.get("half_ppr_ppg") or 0.0)

        per_pid[pid] = {
            "pos": pos,
            "age": age,
            "prod_raw": prod_raw,
            "games_equiv": float(u.get("games_equiv", 0.0) or 0.0),
            "ppg": ppg,
        }

    if not per_pid:
        return {}

    # ----------------------------------------
    # 2) Normalize production WITHIN positions
    # ----------------------------------------
    pos_groups: Dict[str, list[float]] = {}
    for info in per_pid.values():
        pos = info["pos"]
        pos_groups.setdefault(pos, []).append(info["prod_raw"])

    pos_minmax = {
        pos: (
            min(vals),
            max(vals)
        ) if max(vals) > min(vals) else (0.0, 1.0)
        for pos, vals in pos_groups.items()
    }

    # ----------------------------------------
    # 3) Position scores + 3-year age horizon
    # ----------------------------------------

    # UPDATED — lower TE & QB weight
    POS_WEIGHTS = {
        "QB": (0.70, 0.10),  # was (0.80,0.20)
        "RB": (0.65, 0.35),
        "WR": (0.70, 0.30),
        "TE": (0.55, 0.20),  # TE production reduced
    }

    pos_scores = {}

    for pid, info in per_pid.items():
        pos = info["pos"]
        prod_raw = info["prod_raw"]
        age = info["age"]
        games_eq = info["games_equiv"]

        vmin, vmax = pos_minmax[pos]
        prod_norm = (prod_raw - vmin) / (vmax - vmin) if vmax > vmin else prod_raw

        age_curve = horizon_age_factor(pos, age)
        w_prod, w_age = POS_WEIGHTS[pos]

        base = w_prod * prod_norm + w_age * age_curve

        reliability = min(1.0, games_eq / 10.0)

        pos_scores[pid] = base * reliability

    # ----------------------------------------
    # 4) Scarcity (VORP)
    # ----------------------------------------
    # -----------------------------------------------------
    # 4) VORP (Value Over Replacement) with 3-year horizon
    # -----------------------------------------------------
    # starter assumptions (1QB league)
    STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    NUM_TEAMS = 10

    # how “important” each position’s scarcity is in a 1QB league
    POS_SCARCITY_WEIGHT = {
        "QB": 0.25,  # QBs matter least in 1QB
        "RB": 1.00,
        "WR": 1.00,
        "TE": 0.60,  # TE scarcity helps, but not insane
    }

    # 4a. Compute dynasty-adjusted PPG per player (3-year horizon)
    ppg_by_pos: Dict[str, list[tuple[str, float]]] = {}
    for pid, info in per_pid.items():
        pos = info["pos"]
        age = info["age"]
        ppg = info["ppg"]

        current_af = _age_factor(pos, age)
        future_af = horizon_age_factor(pos, age)

        if current_af > 0:
            horizon_scale = future_af / current_af
        else:
            horizon_scale = future_af

        dynasty_ppg = ppg * horizon_scale

        # store for reuse
        info["dynasty_ppg"] = dynasty_ppg
        ppg_by_pos.setdefault(pos, []).append((pid, dynasty_ppg))

    # 4b. Determine replacement PPG per position
    replacement_ppg: Dict[str, float] = {}
    for pos, lst in ppg_by_pos.items():
        if not lst:
            replacement_ppg[pos] = 0.0
            continue

        # sort descending by dynasty_ppg
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)

        # number of starting spots for this position in the league
        starter_slots = STARTERS.get(pos, 1) * NUM_TEAMS

        # index for replacement: just outside typical starters
        # e.g. for WR: 2*10 starters = 20 -> use ~ WR24-ish
        rep_index = int(starter_slots * 1.2)
        rep_index = max(0, min(rep_index, len(lst_sorted) - 1))

        replacement_ppg[pos] = lst_sorted[rep_index][1]

    # 4c. Compute VOR (dynasty_ppg - replacement_ppg)
    vor_map: Dict[str, float] = {}
    for pid, info in per_pid.items():
        pos = info["pos"]
        dyn_ppg = info["dynasty_ppg"]
        rep_line = replacement_ppg.get(pos, 0.0)
        vor = max(dyn_ppg - rep_line, 0.0)
        vor_map[pid] = vor

    # 4d. Normalize VOR across all players and blend with pos_scores
    if vor_map:
        max_vor = max(vor_map.values())
    else:
        max_vor = 0.0

    # how much weight to give scarcity vs. the base production/age score
    SCARCITY_ALPHA = 0.6  # 0 = ignore VOR, 1 = only VOR

    final_scores: Dict[str, float] = {}
    for pid, base_score in pos_scores.items():
        pos = per_pid[pid]["pos"]
        vor = vor_map.get(pid, 0.0)

        # normalize global VOR to [0,1]
        vor_norm = (vor / max_vor) if max_vor > 0 else 0.0

        pos_weight = POS_SCARCITY_WEIGHT.get(pos, 1.0)

        scarcity_component = vor_norm * pos_weight

        # blend: mostly scarcity, but base production/age still matters
        blended = (1.0 - SCARCITY_ALPHA) * base_score + SCARCITY_ALPHA * scarcity_component

        # clamp to [0,1] for safety
        final_scores[pid] = max(0.0, min(1.0, blended))

    # -----------------------------------------------------
    # 5) Global normalization to 0–999.9
    # -----------------------------------------------------
    all_vals = list(final_scores.values())
    gmin = min(all_vals)
    gmax = max(all_vals)

    value_table: Dict[str, float] = {}
    for pid, v in final_scores.items():
        if gmax <= gmin:
            s01 = 0.0
        else:
            s01 = (v - gmin) / (gmax - gmin)
        value_table[pid] = round(s01 * 999.9, 1)

    return value_table
