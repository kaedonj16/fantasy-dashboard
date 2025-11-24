# dashboard_services/player_value.py

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, Optional

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
        if age <= 22: return 1.0
        if age <= 24: return 0.95
        if age <= 25: return 0.90
        if age <= 26: return 0.75
        if age <= 27: return 0.65
        if age <= 28: return 0.50
        if age <= 30: return 0.40
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
    ppg = float(u.get("ppr_ppg") or 0.0)

    # --- QUARTERBACKS --------------------------
    if pos == "QB":
        yds = u.get("avg_pass_yds", 0)
        tds = u.get("avg_pass_tds", 0)
        ints = u.get("avg_pass_int", 0)

        # scaled formula (empirically tuned)
        score = (
                (yds / 300.0) * 0.50 +
                (tds / 3.5) * 0.60 +
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
                (ppg / 19.5) * 0.35  # TE scoring deflated
        )
        return min(1.0, score)

    return 0.0



def build_value_table_for_usage() -> Dict[str, float]:
    """
    Build value table using PPR PPG + RZ usage + dynasty horizon + VORP.
    """

    # --- LOAD LIST-BASED VALUE TABLE ---
    lst = load_usage_table()
    if not isinstance(lst, list):
        raise ValueError("value_table must be a list of player objects")

    players_index = {}
    usage_table = {}

    for obj in lst:
        pid = str(obj.get("id"))
        if not pid:
            continue
        players_index[pid] = {
            "name": obj.get("name"),
            "team": obj.get("team"),
            "pos": obj.get("position"),
            "age": obj.get("age"),
        }
        usage_table[pid] = obj.get("usage") or {}

    # ----------------------------------------
    # 1) Collect metadata + base production
    # ----------------------------------------
    per_pid = {}

    for pid, u in usage_table.items():
        meta = players_index.get(pid, {})
        pos = meta.get("pos")

        if pos not in {"QB", "RB", "WR", "TE"}:
            continue

        age = float(meta.get("age") or 0.0)
        ppg = float(u.get("ppr_ppg") or 0.0)

        # QB passing integration (kept from your old model)
        prod_raw = _production_component_fixed(u, pos)

        # NEW — red-zone usage
        rz_targets = float(u.get("rec_rz_tgt_pg") or 0.0)
        rz_carries = float(u.get("rush_rz_att_pg") or 0.0)

        per_pid[pid] = {
            "pos": pos,
            "age": age,
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
    by_pos_ppg = {}
    by_pos_rz = {}

    for pid, p in per_pid.items():
        pos = p["pos"]

        by_pos_ppg.setdefault(pos, []).append(p["ppg"])

        # combine rz inputs into one metric
        rz_metric = p["rz_targets"] + p["rz_carries"]
        by_pos_rz.setdefault(pos, []).append(rz_metric)

        p["rz_metric"] = rz_metric

    pos_ppg_minmax = {
        pos: (min(v), max(v)) if max(v) > min(v) else (0, 1)
        for pos, v in by_pos_ppg.items()
    }

    pos_rz_minmax = {
        pos: (min(v), max(v)) if max(v) > min(v) else (0, 1)
        for pos, v in by_pos_rz.items()
    }

    # ----------------------------------------
    # 3) Age + production + RZ score
    # ----------------------------------------
    POS_WEIGHTS = {
        "QB": (0.80, 0.25, 0.00),  # production dominates, light age, tiny RZ
        "RB": (0.45, 0.35, 0.20),  # VERY age-weighted, medium RZ, low PPG
        "WR": (0.625, 0.35, 0.25),  # PPG-driven with meaningful age & RZ boost
        "TE": (0.45, 0.25, 0.30),  # redzone matters most for TD volatility
    }

    pos_scores = {}

    for pid, p in per_pid.items():
        pos = p["pos"]

        # normalize ppg
        vmin, vmax = pos_ppg_minmax[pos]
        ppg_norm = (p["ppg"] - vmin) / (vmax - vmin) if vmax > vmin else 0

        # normalize RZ
        rmin, rmax = pos_rz_minmax[pos]
        rz_norm = (p["rz_metric"] - rmin) / (rmax - rmin) if rmax > rmin else 0

        age_curve = horizon_age_factor(pos, p["age"])

        w_ppg, w_age, w_rz = POS_WEIGHTS[pos]

        pos_scores[pid] = (
            w_ppg * ppg_norm +
            w_age * age_curve +
            w_rz * rz_norm
        )

    # ----------------------------------------
    # 4) VORP (still using dynasty-PPG)
    # ----------------------------------------
    STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    NUM_TEAMS = 10

    dynasty_ppg_by_pos = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        age = p["age"]

        current_af = _age_factor(pos, age)
        future_af = horizon_age_factor(pos, age)
        horizon_scale = (future_af / current_af) if current_af else future_af

        dynasty_ppg = p["ppg"] * horizon_scale
        p["dynasty_ppg"] = dynasty_ppg

        dynasty_ppg_by_pos.setdefault(pos, []).append((pid, dynasty_ppg))

    replacement_ppg = {}

    for pos, lst in dynasty_ppg_by_pos.items():
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        starter_slots = STARTERS[pos] * NUM_TEAMS
        idx = int(starter_slots * 1.2)
        idx = max(0, min(idx, len(lst_sorted) - 1))
        replacement_ppg[pos] = lst_sorted[idx][1]

    vor_map = {}
    for pid, p in per_pid.items():
        vor = p["dynasty_ppg"] - replacement_ppg[p["pos"]]
        vor_map[pid] = max(vor, 0)

    max_vor = max(vor_map.values()) if vor_map else 1

    final_scores = {}
    SCARCITY_ALPHA = 0.4

    for pid, base_score in pos_scores.items():
        vor_norm = vor_map[pid] / max_vor
        scarcity_weight = 1.0  # optional mix by position
        scarcity = vor_norm * scarcity_weight

        blended = (1 - SCARCITY_ALPHA) * base_score + SCARCITY_ALPHA * scarcity
        final_scores[pid] = max(0.0, min(1.0, blended))

    # ----------------------------------------
    # 5) Global normalization → 0–999
    # ----------------------------------------
    vals = list(final_scores.values())
    gmin, gmax = min(vals), max(vals)

    value_table = {}
    for pid, v in final_scores.items():
        if gmax <= gmin:
            s01 = 0
        else:
            s01 = (v - gmin) / (gmax - gmin)
        value_table[pid] = round(s01 * 999.9, 1)


    return value_table

