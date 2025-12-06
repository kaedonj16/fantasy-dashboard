# dashboard_services/player_value.py

from __future__ import annotations

from typing import Dict, Optional

from dashboard_services.utils import load_usage_table


# Rough dynasty trade-value scales per position

def _age_factor(pos: str, age: Optional[float]) -> float:
    """
    0–1 age score by position. Higher = better 3-year outlook.
    If age is unknown, assume a neutral-ish 0.85.
    """
    if age is None:
        return 0.85

    if pos == "RB":
        # RBs fall off faster
        if age <= 22: return 0.95
        if age <= 24: return 0.90
        if age <= 25: return 0.85
        if age <= 26: return 0.45
        if age <= 27: return 0.30
        return 0.10

    if pos == "WR":
        # WRs stay strong longer
        if age <= 22: return 1.0
        if age <= 24: return 1.0
        if age <= 26: return 0.95
        if age <= 27: return 0.92
        if age <= 28: return 0.85
        if age <= 29: return 0.70
        if age <= 31: return 0.625
        return 0.50

    if pos == "QB":
        if age <= 24: return 0.95
        if age <= 28: return 1.0
        if age <= 31: return 0.80
        if age <= 34: return 0.60
        if age <= 37: return 0.40
        if age <= 40: return 0.35
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


def horizon_age_factor(pos: str, age: float) -> float:
    # Later years matter just as much as now → older players get hit harder
    weights = [0.4, 0.35, 0.25]

    num = 0.0
    den = 0.0
    for t, w in enumerate(weights):
        num += w * _age_factor(pos, age + t)
        den += w

    base = num / den if den else 0.0

    # Make the curve more “spiky”: young studs stand out more,
    # aging players drop off faster.
    return base ** 1.2


def _production_component_fixed(u: dict, pos: str) -> float:
    """
    Unified production model with:
      - QB passing integration
      - TE softened
    """

    # default fallback = fantasy points
    ppg = float(u.get("ppr_ppg") or 0.0)

    # --- QUARTERBACKS --------------------------
    if pos == "QB":
        yds = float(u.get("avg_pass_yds", 0) or 0)
        tds = float(u.get("avg_pass_tds", 0) or 0)
        ints = float(u.get("avg_pass_int", 0) or 0)

        # scaled formula (empirically tuned)
        # NOTE: INTs are a penalty (negative contribution)
        score = (
            (yds / 300.0) * 0.50 +
            (tds / 3.5) * 0.60 -
            (ints / 2.5) * 0.20 +
            (ppg / 30.0) * 0.50
        )

        return max(0.0, min(1.0, score))

    # --- RUNNING BACKS -------------------------
    if pos == "RB":
        carries = float(u.get("avg_carries", 0) or 0)
        yds = float(u.get("avg_rush_yards", 0) or 0)
        recs = float(u.get("avg_receptions", 0) or 0)

        score = (
            (carries / 18.0) * 0.40 +
            (yds / 90.0) * 0.40 +
            (recs / 4.0) * 0.20 +
            (ppg / 25.0) * 0.50
        )
        return max(0.0, min(1.0, score))

    # --- WIDE RECEIVERS ------------------------
    if pos == "WR":
        tgt = float(u.get("avg_targets", 0) or 0)
        rec = float(u.get("avg_receptions", 0) or 0)
        yds = float(u.get("avg_rec_yards", 0) or 0)

        score = (
            (tgt / 11.0) * 0.45 +
            (rec / 7.0) * 0.30 +
            (yds / 90.0) * 0.40 +
            (ppg / 22.0) * 0.50
        )
        return max(0.0, min(1.0, score))

    # --- TIGHT ENDS (tuned down) --------------
    if pos == "TE":
        tgt = float(u.get("avg_targets", 0) or 0)
        yds = float(u.get("avg_rec_yards", 0) or 0)

        score = (
            (tgt / 9.0) * 0.30 +
            (yds / 75.0) * 0.25 +
            (ppg / 19.5) * 0.35  # TE scoring deflated
        )
        return max(0.0, min(1.0, score))

    return 0.0


def build_value_table_for_usage() -> Dict[str, float]:
    """
    Build dynasty-style value table on 0–999.9 scale using:
      - PPR PPG (within-position)
      - Dynasty age window (3-year horizon)
      - VORP (value over replacement) by position
      - Explicit positional weighting (e.g. QBs downweighted in 1QB)

    Expects load_usage_table() -> list of:
      {
        "id": str,
        "name": str,
        "team": str,
        "position": "QB"/"RB"/"WR"/"TE",
        "age": float or int,
        "usage": {
          "ppr_ppg": float,
          "rec_rz_tgt_pg": float (optional),
          "rush_rz_att_pg": float (optional),
          ...
        }
      }
    """
    # -----------------------------
    # 0) Load usage objects
    # -----------------------------
    lst = load_usage_table()
    if not isinstance(lst, list):
        raise ValueError("usage table must be a list of player objects")

    players_index: Dict[str, dict] = {}
    usage_table: Dict[str, dict] = {}

    for obj in lst:
        pid = str(obj.get("id") or "").strip()
        if not pid:
            continue
        players_index[pid] = {
            "name": obj.get("name"),
            "team": obj.get("team"),
            "pos": obj.get("position"),
            "age": obj.get("age"),
        }
        usage_table[pid] = obj.get("usage") or {}

    # ---------------------------------------------------
    # 0.5) Filter to "relevant" fantasy players only
    # ---------------------------------------------------
    POS_WHITELIST = {"QB", "RB", "WR", "TE"}

    def is_relevant(meta: dict, usage: dict) -> bool:
        pos = meta.get("pos")
        if pos not in POS_WHITELIST:
            return False

        if not usage:
            return False

        games = float(usage.get("games") or 0)
        ppg = float(usage.get("ppr_ppg") or 0)
        snaps = float(usage.get("avg_off_snaps") or 0)
        opps = float(usage.get("avg_targets") or 0) + float(usage.get("avg_carries") or 0)

        # You can tweak these thresholds, but this is a good "fantasy relevant" cut:
        # - played at least 3 games, OR
        # - scoring ~flex-ish points, OR
        # - meaningful snaps/opportunities
        if games >= 3:
            return True
        if ppg >= 6:  # ~flex floor
            return True
        if snaps >= 20:  # on the field a decent amount
            return True
        if opps >= 3:  # 3+ touches/targets per game
            return True

        return False

    filtered_players_index: Dict[str, dict] = {}
    filtered_usage_table: Dict[str, dict] = {}

    for pid, meta in players_index.items():
        u = usage_table.get(pid, {})
        if is_relevant(meta, u):
            filtered_players_index[pid] = meta
            filtered_usage_table[pid] = u

    players_index = filtered_players_index
    usage_table = filtered_usage_table

    # If we somehow filtered *everything* out, bail gracefully
    if not players_index:
        return {}

    # ----------------------------------------
    # 1) Collect metadata + base production
    # ----------------------------------------
    per_pid: Dict[str, dict] = {}

    for pid, u in usage_table.items():
        meta = players_index.get(pid, {})
        pos = meta.get("pos")

        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        # Treat missing age as "unknown" so _age_factor uses neutral 0.85
        raw_age = meta.get("age")
        age: Optional[float]
        if raw_age is None or raw_age == "":
            age = None
        else:
            try:
                age = float(raw_age)
            except (TypeError, ValueError):
                age = None

        ppg = float(u.get("ppr_ppg") or 0.0)

        # QB / non-QB production component (your custom logic)
        prod_raw = _production_component_fixed(u, pos)

        # Red-zone usage
        rz_targets = float(u.get("rec_rz_tgt_pg") or 0.0)
        rz_carries = float(u.get("rush_rz_att_pg") or 0.0)

        per_pid[pid] = {
            "pos": pos,
            "age": age if age is not None else 0.0,
            "age_opt": age,
            "ppg": ppg,
            "prod_raw": prod_raw,
            "rz_targets": rz_targets,
            "rz_carries": rz_carries,
        }

    if not per_pid:
        return {}

    # ----------------------------------------
    # 2) Normalize PPG and RZ usage by position
    # ----------------------------------------
    by_pos_ppg: Dict[str, list[float]] = {}
    by_pos_rz: Dict[str, list[float]] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]

        by_pos_ppg.setdefault(pos, []).append(p["ppg"])

        rz_metric = p["rz_targets"] + p["rz_carries"]
        by_pos_rz.setdefault(pos, []).append(rz_metric)
        p["rz_metric"] = rz_metric

    pos_ppg_minmax: Dict[str, tuple[float, float]] = {}
    for pos, vals in by_pos_ppg.items():
        vmin, vmax = min(vals), max(vals)
        pos_ppg_minmax[pos] = (vmin, vmax) if vmax > vmin else (0.0, 1.0)

    pos_rz_minmax: Dict[str, tuple[float, float]] = {}
    for pos, vals in by_pos_rz.items():
        vmin, vmax = min(vals), max(vals)
        pos_rz_minmax[pos] = (vmin, vmax) if vmax > vmin else (0.0, 1.0)

    # ----------------------------------------
    # 3) Age + production + RZ score
    # ----------------------------------------
    POS_WEIGHTS = {
        "QB": (0.625, 0.25, 0.00),
        "RB": (0.425, 0.40, 0.30),
        "WR": (0.625, 0.30, 0.25),
        "TE": (0.325, 0.30, 0.35),
    }

    pos_scores: Dict[str, float] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]

        vmin_ppg, vmax_ppg = pos_ppg_minmax[pos]
        ppg_norm = (p["ppg"] - vmin_ppg) / (vmax_ppg - vmin_ppg) if vmax_ppg > vmin_ppg else 0.0

        vmin_rz, vmax_rz = pos_rz_minmax[pos]
        rz_norm = (p["rz_metric"] - vmin_rz) / (vmax_rz - vmin_rz) if vmax_rz > vmin_rz else 0.0

        # Use the optional age for the horizon curve; if None, horizon_age_factor
        # will effectively treat them as neutral via _age_factor’s logic.
        age_for_horizon = p["age_opt"] if p["age_opt"] is not None else 26.0
        age_curve = horizon_age_factor(pos, age_for_horizon)

        w_ppg, w_age, w_rz = POS_WEIGHTS[pos]

        pos_scores[pid] = (
            w_ppg * ppg_norm +
            w_age * age_curve +
            w_rz * rz_norm
        )

    # ----------------------------------------
    # 4) VORP (dynasty-PPG horizon)
    # ----------------------------------------
    STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    NUM_TEAMS = 10

    dynasty_ppg_by_pos: Dict[str, list[tuple[str, float]]] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        age_for_horizon = p["age_opt"] if p["age_opt"] is not None else 26.0

        current_af = _age_factor(pos, age_for_horizon)
        future_af = horizon_age_factor(pos, age_for_horizon)
        horizon_scale = (future_af / current_af) if current_af else future_af

        dynasty_ppg = p["ppg"] * horizon_scale
        p["dynasty_ppg"] = dynasty_ppg

        dynasty_ppg_by_pos.setdefault(pos, []).append((pid, dynasty_ppg))

    replacement_ppg: Dict[str, float] = {}
    for pos, lst in dynasty_ppg_by_pos.items():
        if not lst:
            replacement_ppg[pos] = 0.0
            continue

        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        starter_slots = STARTERS[pos] * NUM_TEAMS
        idx = int(starter_slots * 1.2)
        idx = max(0, min(idx, len(lst_sorted) - 1))
        replacement_ppg[pos] = lst_sorted[idx][1]

    vor_map: Dict[str, float] = {}
    for pid, p in per_pid.items():
        rep = replacement_ppg[p["pos"]]
        vor = p["dynasty_ppg"] - rep
        vor_map[pid] = max(vor, 0.0)

    max_vor = max(vor_map.values()) if vor_map else 1.0

    # ----------------------------------------
    # 5) Blend base score + scarcity → global 0–999.9
    # ----------------------------------------
    final_scores: Dict[str, float] = {}
    SCARCITY_ALPHA = 0.4  # how much VORP shapes the final value

    for pid, base_score in pos_scores.items():
        vor_norm = vor_map[pid] / max_vor if max_vor > 0 else 0.0
        blended = (1 - SCARCITY_ALPHA) * base_score + SCARCITY_ALPHA * vor_norm
        final_scores[pid] = max(0.0, min(1.0, blended))

    vals = list(final_scores.values())
    gmin, gmax = min(vals), max(vals)

    # hyperparams for shape
    GAMMA = 0.6  # < 1 → lifts mid/lower values
    FLOOR = 0.05  # 5% baseline so “real” players aren’t near 0

    value_table: Dict[str, float] = {}

    for pid, v in final_scores.items():
        if gmax <= gmin:
            s01 = 0.0
        else:
            # base linear 0–1
            s01 = (v - gmin) / (gmax - gmin)

        # concave transform: makes mid-tier relatively “fatter”
        s_curve = s01 ** GAMMA

        # soft floor so everything with non-zero score gets some value
        s_mix = FLOOR + (1.0 - FLOOR) * s_curve  # in [FLOOR, 1]

        value_table[pid] = round(s_mix * 999.9, 1)

    return value_table
