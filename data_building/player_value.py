from __future__ import annotations

from typing import Dict, Optional

from dashboard_services.api import get_nfl_state
from dashboard_services.utils import load_usage_table
from data_building.player_history import load_player_history_df, build_player_history_features


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _age_factor(pos: str, age: Optional[float]) -> float:
    """
    0–1 age score by position. Higher = better 3-year outlook.
    If age is unknown, assume a neutral-ish 0.85.
    """
    if age is None:
        return 0.85

    if pos == "RB":
        if age <= 22: return 0.95
        if age <= 24: return 0.90
        if age <= 25: return 0.85
        if age <= 26: return 0.45
        if age <= 27: return 0.30
        return 0.10

    if pos == "WR":
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

    return 0.8


def horizon_age_factor(pos: str, age: Optional[float]) -> float:
    if age is None:
        age = 26.0

    weights = [0.4, 0.35, 0.25]

    num = 0.0
    den = 0.0
    for t, w in enumerate(weights):
        num += w * _age_factor(pos, age + t)
        den += w

    base = num / den if den else 0.0
    return base ** 1.2


def _production_component_fixed(u: dict, pos: str) -> float:
    """
    Unified current-season production model.
    """
    ppg = _safe_float(u.get("ppr_ppg"))

    if pos == "QB":
        yds = _safe_float(u.get("avg_pass_yds"))
        tds = _safe_float(u.get("avg_pass_tds"))
        ints = _safe_float(u.get("avg_pass_int"))

        score = (
                (yds / 300.0) * 0.50 +
                (tds / 3.5) * 0.60 -
                (ints / 2.5) * 0.20 +
                (ppg / 30.0) * 0.50
        )
        return max(0.0, min(1.0, score))

    if pos == "RB":
        carries = _safe_float(u.get("avg_carries"))
        yds = _safe_float(u.get("avg_rush_yards"))
        recs = _safe_float(u.get("avg_receptions"))

        score = (
                (carries / 18.0) * 0.40 +
                (yds / 90.0) * 0.40 +
                (recs / 4.0) * 0.20 +
                (ppg / 25.0) * 0.50
        )
        return max(0.0, min(1.0, score))

    if pos == "WR":
        tgt = _safe_float(u.get("avg_targets"))
        rec = _safe_float(u.get("avg_receptions"))
        yds = _safe_float(u.get("avg_rec_yards"))

        score = (
                (tgt / 11.0) * 0.45 +
                (rec / 7.0) * 0.30 +
                (yds / 90.0) * 0.40 +
                (ppg / 22.0) * 0.50
        )
        return max(0.0, min(1.0, score))

    if pos == "TE":
        tgt = _safe_float(u.get("avg_targets"))
        yds = _safe_float(u.get("avg_rec_yards"))

        score = (
                (tgt / 9.0) * 0.30 +
                (yds / 75.0) * 0.25 +
                (ppg / 19.5) * 0.35
        )
        return max(0.0, min(1.0, score))

    return 0.0


def availability_score(u: dict, pos: str, games_possible: float = 17.0) -> float:
    g_played = _safe_float(u.get("games"))
    g_possible = _safe_float(u.get("games_possible"), games_possible)

    if g_possible <= 0:
        g_possible = games_possible

    raw_rate = (g_played / g_possible) if g_possible > 0 else 0.0

    k = 6.0
    prior = 0.88
    rate = (raw_rate * g_possible + prior * k) / (g_possible + k)

    floor_by_pos = {"QB": 0.80, "RB": 0.75, "WR": 0.78, "TE": 0.76}
    floor = floor_by_pos.get(pos, 0.75)

    return max(floor, min(1.0, rate))


def _normalize_by_pos(values_by_pid: Dict[str, float], pos_by_pid: Dict[str, str]) -> Dict[str, float]:
    by_pos: Dict[str, list[float]] = {}
    for pid, val in values_by_pid.items():
        pos = pos_by_pid.get(pid)
        if pos is None:
            continue
        by_pos.setdefault(pos, []).append(val)

    bounds: Dict[str, tuple[float, float]] = {}
    for pos, vals in by_pos.items():
        if not vals:
            bounds[pos] = (0.0, 1.0)
            continue
        vmin = min(vals)
        vmax = max(vals)
        bounds[pos] = (vmin, vmax if vmax > vmin else vmin + 1.0)

    out: Dict[str, float] = {}
    for pid, val in values_by_pid.items():
        pos = pos_by_pid.get(pid)
        if pos is None:
            out[pid] = 0.0
            continue
        vmin, vmax = bounds[pos]
        out[pid] = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    return out


def build_value_table_for_usage() -> Dict[str, float]:
    """
    Build dynasty-style value table on 0–999.9 scale using:
      - current production
      - 3-year weighted production
      - career ceiling
      - 1-year / 2-year trend
      - age curve
      - red-zone usage
      - availability
      - VORP scarcity
    """

    lst = load_usage_table()

    if not isinstance(lst, list):
        raise ValueError("usage table must be a list of player objects")

    history_df = load_player_history_df()

    history_features_df = build_player_history_features(history_df) if not history_df.empty else None

    history_by_pid: Dict[str, dict] = {}
    if history_features_df is not None and not history_features_df.empty:
        for _, row in history_features_df.iterrows():
            sleeper_id = str(row.get("sleeper_id") or "").strip()
            if sleeper_id:
                history_by_pid[sleeper_id] = row.to_dict()

    players_index: Dict[str, dict] = {}
    usage_table: Dict[str, dict] = {}

    missing_pid = 0
    missing_position = 0
    missing_usage = 0
    flattened_usage_fallback = 0

    for obj in lst:
        pid = str(obj.get("id") or "").strip()
        if not pid:
            missing_pid += 1
            continue

        pos = obj.get("position") or obj.get("pos")
        if not pos:
            missing_position += 1

        usage = obj.get("usage")
        if not isinstance(usage, dict):
            # fallback for flattened rows
            usage = {
                "games": obj.get("games"),
                "ppr_ppg": obj.get("ppr_ppg"),
                "avg_off_snaps": obj.get("avg_off_snaps"),
                "avg_targets": obj.get("avg_targets"),
                "avg_carries": obj.get("avg_carries"),
                "avg_off_snap_pct": obj.get("avg_off_snap_pct"),
                "avg_receptions": obj.get("avg_receptions"),
                "avg_rec_yards": obj.get("avg_rec_yards"),
                "avg_rec_tds": obj.get("avg_rec_tds"),
                "avg_rush_yards": obj.get("avg_rush_yards"),
                "avg_rush_tds": obj.get("avg_rush_tds"),
                "avg_pass_att": obj.get("avg_pass_att"),
                "avg_pass_cmp": obj.get("avg_pass_cmp"),
                "avg_pass_yds": obj.get("avg_pass_yds"),
                "avg_pass_tds": obj.get("avg_pass_tds"),
                "avg_pass_int": obj.get("avg_pass_int"),
                "rec_rz_tgt_pg": obj.get("rec_rz_tgt_pg"),
                "rush_rz_att_pg": obj.get("rush_rz_att_pg"),
                "target_share": obj.get("target_share"),
            }
            flattened_usage_fallback += 1

        if not usage:
            missing_usage += 1

        players_index[pid] = {
            "name": obj.get("name"),
            "team": obj.get("team"),
            "pos": pos,
            "age": obj.get("age"),
        }
        usage_table[pid] = usage

    POS_WHITELIST = {"QB", "RB", "WR", "TE"}

    nfl_state = get_nfl_state() or {}
    season_type = str(nfl_state.get("season_type") or "").lower()
    offseason_mode = season_type == "off"

    filter_reasons = {
        "bad_pos": 0,
        "empty_usage_no_history": 0,
        "games": 0,
        "ppg": 0,
        "snaps": 0,
        "opps": 0,
        "history_ppg": 0,
        "history_ceiling": 0,
        "history_seasons": 0,
        "passed": 0,
    }

    def _safe_float_local(x, default=0.0):
        try:
            if x is None:
                return float(default)
            if isinstance(x, str) and not x.strip():
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def is_relevant(pid: str, meta: dict, usage: dict) -> bool:
        pos = str(meta.get("pos") or "").upper()
        if pos not in POS_WHITELIST:
            filter_reasons["bad_pos"] += 1
            return False

        hist = history_by_pid.get(pid, {}) or {}

        # ---------- current-season signals ----------
        games = _safe_float_local((usage or {}).get("games"))
        ppg = _safe_float_local((usage or {}).get("ppr_ppg"))
        snaps = _safe_float_local((usage or {}).get("avg_off_snaps"))
        opps = (
                _safe_float_local((usage or {}).get("avg_targets")) +
                _safe_float_local((usage or {}).get("avg_carries"))
        )

        # ---------- historical signals ----------
        hist_last_year_ppg = _safe_float_local(hist.get("last_year_ppg"))
        hist_weighted_ppg = _safe_float_local(hist.get("three_year_weighted_ppg"))
        hist_career_best = _safe_float_local(hist.get("career_best_ppg"))
        hist_seasons = _safe_float_local(hist.get("seasons_played"))

        # =========================================================
        # IN SEASON:
        # Prefer current-year relevance, but allow strong history fallback
        # =========================================================
        if not offseason_mode:
            if games >= 3:
                filter_reasons["games"] += 1
                filter_reasons["passed"] += 1
                return True
            if ppg >= 6:
                filter_reasons["ppg"] += 1
                filter_reasons["passed"] += 1
                return True
            if snaps >= 20:
                filter_reasons["snaps"] += 1
                filter_reasons["passed"] += 1
                return True
            if opps >= 3:
                filter_reasons["opps"] += 1
                filter_reasons["passed"] += 1
                return True

            # in-season fallback for suspended / injured / early-week players
            if hist_weighted_ppg >= 8:
                filter_reasons["history_ppg"] += 1
                filter_reasons["passed"] += 1
                return True
            if hist_career_best >= 12:
                filter_reasons["history_ceiling"] += 1
                filter_reasons["passed"] += 1
                return True
            if hist_seasons >= 2 and hist_last_year_ppg >= 6:
                filter_reasons["history_seasons"] += 1
                filter_reasons["passed"] += 1
                return True

            filter_reasons["empty_usage_no_history"] += 1
            return False

        # =========================================================
        # OFFSEASON:
        # Current usage is often empty, so history becomes the base
        # =========================================================
        if hist_weighted_ppg >= 6:
            filter_reasons["history_ppg"] += 1
            filter_reasons["passed"] += 1
            return True
        if hist_last_year_ppg >= 6:
            filter_reasons["history_ppg"] += 1
            filter_reasons["passed"] += 1
            return True
        if hist_career_best >= 10:
            filter_reasons["history_ceiling"] += 1
            filter_reasons["passed"] += 1
            return True
        if hist_seasons >= 2 and hist_career_best >= 7:
            filter_reasons["history_seasons"] += 1
            filter_reasons["passed"] += 1
            return True

        # still allow current-year relevance if somehow usage exists
        if games >= 2:
            filter_reasons["games"] += 1
            filter_reasons["passed"] += 1
            return True
        if ppg >= 5:
            filter_reasons["ppg"] += 1
            filter_reasons["passed"] += 1
            return True
        if snaps >= 15:
            filter_reasons["snaps"] += 1
            filter_reasons["passed"] += 1
            return True
        if opps >= 2:
            filter_reasons["opps"] += 1
            filter_reasons["passed"] += 1
            return True

        filter_reasons["empty_usage_no_history"] += 1
        return False

    filtered_players_index: Dict[str, dict] = {}
    filtered_usage_table: Dict[str, dict] = {}

    for pid, meta in players_index.items():
        u = usage_table.get(pid, {}) or {}
        if is_relevant(pid, meta, u):
            filtered_players_index[pid] = meta
            filtered_usage_table[pid] = u

    players_index = filtered_players_index
    usage_table = filtered_usage_table

    if not players_index:
        return {}

    per_pid: Dict[str, dict] = {}
    per_pid_bad_pos = 0

    for pid, u in usage_table.items():
        meta = players_index.get(pid, {})
        pos = str(meta.get("pos") or "").upper()

        if pos not in POS_WHITELIST:
            per_pid_bad_pos += 1
            continue

        raw_age = meta.get("age")
        if raw_age is None or raw_age == "":
            age = None
        else:
            try:
                age = float(raw_age)
            except (TypeError, ValueError):
                age = None

        hist = history_by_pid.get(pid, {})

        avail = availability_score(u, pos)
        ppg = _safe_float_local(u.get("ppr_ppg"))
        prod_raw = _production_component_fixed(u, pos)

        rz_targets = _safe_float_local(u.get("rec_rz_tgt_pg"))
        rz_carries = _safe_float_local(u.get("rush_rz_att_pg"))
        rz_metric = rz_targets + rz_carries

        last_year_ppg = _safe_float_local(hist.get("last_year_ppg"), ppg)
        prev_year_ppg = _safe_float_local(hist.get("prev_year_ppg"), ppg)
        weighted_ppg_3yr = _safe_float_local(hist.get("three_year_weighted_ppg"), ppg)
        career_best_ppg = _safe_float_local(hist.get("career_best_ppg"), ppg)
        career_avg_ppg = _safe_float_local(hist.get("career_avg_ppg"), ppg)

        last_year_snap_pct = _safe_float_local(hist.get("last_year_snap_pct"))
        weighted_snap_3yr = _safe_float_local(hist.get("three_year_weighted_snap_pct"))
        last_year_target_share = _safe_float_local(hist.get("last_year_target_share"),
                                                   _safe_float_local(u.get("target_share")))
        weighted_target_share_3yr = _safe_float_local(
            hist.get("three_year_weighted_target_share"),
            _safe_float_local(u.get("target_share"))
        )

        ppg_trend_1yr = _safe_float_local(hist.get("ppg_trend_1yr"))
        ppg_trend_2yr = _safe_float_local(hist.get("ppg_trend_2yr"))
        target_share_trend_1yr = _safe_float_local(hist.get("target_share_trend_1yr"))
        seasons_played = _safe_float_local(hist.get("seasons_played"), 1.0)

        per_pid[pid] = {
            "pos": pos,
            "age_opt": age,
            "avail": avail,
            "ppg": ppg,
            "prod_raw": prod_raw,
            "rz_metric": rz_metric,
            "last_year_ppg": last_year_ppg,
            "prev_year_ppg": prev_year_ppg,
            "weighted_ppg_3yr": weighted_ppg_3yr,
            "career_best_ppg": career_best_ppg,
            "career_avg_ppg": career_avg_ppg,
            "last_year_snap_pct": last_year_snap_pct,
            "weighted_snap_3yr": weighted_snap_3yr,
            "last_year_target_share": last_year_target_share,
            "weighted_target_share_3yr": weighted_target_share_3yr,
            "ppg_trend_1yr": ppg_trend_1yr,
            "ppg_trend_2yr": ppg_trend_2yr,
            "target_share_trend_1yr": target_share_trend_1yr,
            "seasons_played": seasons_played,
        }

    if not per_pid:
        return {}

    pos_by_pid = {pid: p["pos"] for pid, p in per_pid.items()}

    current_ppg_norm = _normalize_by_pos(
        {pid: p["ppg"] for pid, p in per_pid.items()},
        pos_by_pid,
    )
    weighted_ppg_3yr_norm = _normalize_by_pos(
        {pid: p["weighted_ppg_3yr"] for pid, p in per_pid.items()},
        pos_by_pid,
    )
    career_best_ppg_norm = _normalize_by_pos(
        {pid: p["career_best_ppg"] for pid, p in per_pid.items()},
        pos_by_pid,
    )
    rz_norm = _normalize_by_pos(
        {pid: p["rz_metric"] for pid, p in per_pid.items()},
        pos_by_pid,
    )
    target_share_norm = _normalize_by_pos(
        {pid: p["weighted_target_share_3yr"] for pid, p in per_pid.items()},
        pos_by_pid,
    )
    snap_norm = _normalize_by_pos(
        {pid: p["weighted_snap_3yr"] for pid, p in per_pid.items()},
        pos_by_pid,
    )

    def trend_to_unit(x: float, scale: float = 6.0) -> float:
        clipped = max(-scale, min(scale, x))
        return (clipped + scale) / (2.0 * scale)

    POS_WEIGHTS = {
        "QB": {
            "current_ppg": 0.24,
            "history_ppg": 0.28,
            "career_ceiling": 0.10,
            "current_prod": 0.12,
            "age": 0.16,
            "rz": 0.00,
            "target_share": 0.00,
            "snap": 0.00,
            "trend": 0.10,
        },
        "RB": {
            "current_ppg": 0.20,
            "history_ppg": 0.20,
            "career_ceiling": 0.08,
            "current_prod": 0.12,
            "age": 0.16,
            "rz": 0.11,
            "target_share": 0.00,
            "snap": 0.05,
            "trend": 0.08,
        },
        "WR": {
            "current_ppg": 0.18,
            "history_ppg": 0.24,
            "career_ceiling": 0.10,
            "current_prod": 0.10,
            "age": 0.15,
            "rz": 0.08,
            "target_share": 0.09,
            "snap": 0.03,
            "trend": 0.03,
        },
        "TE": {
            "current_ppg": 0.16,
            "history_ppg": 0.24,
            "career_ceiling": 0.08,
            "current_prod": 0.10,
            "age": 0.14,
            "rz": 0.12,
            "target_share": 0.10,
            "snap": 0.03,
            "trend": 0.03,
        },
    }

    pos_scores: Dict[str, float] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        age_for_horizon = p["age_opt"] if p["age_opt"] is not None else 26.0
        age_curve = horizon_age_factor(pos, age_for_horizon)

        if pos == "QB":
            trend_score = (
                    0.7 * trend_to_unit(p["ppg_trend_1yr"], scale=5.0) +
                    0.3 * trend_to_unit(p["ppg_trend_2yr"], scale=8.0)
            )
        else:
            trend_score = (
                    0.55 * trend_to_unit(p["ppg_trend_1yr"], scale=4.0) +
                    0.20 * trend_to_unit(p["ppg_trend_2yr"], scale=6.0) +
                    0.25 * trend_to_unit(p["target_share_trend_1yr"], scale=0.08)
            )

        experience_bonus = min(1.0, 0.70 + 0.10 * p["seasons_played"])
        w = POS_WEIGHTS[pos]

        base = (
                w["current_ppg"] * current_ppg_norm.get(pid, 0.0) +
                w["history_ppg"] * weighted_ppg_3yr_norm.get(pid, 0.0) * experience_bonus +
                w["career_ceiling"] * career_best_ppg_norm.get(pid, 0.0) * experience_bonus +
                w["current_prod"] * p["prod_raw"] +
                w["age"] * age_curve +
                w["rz"] * rz_norm.get(pid, 0.0) +
                w["target_share"] * target_share_norm.get(pid, 0.0) +
                w["snap"] * snap_norm.get(pid, 0.0) +
                w["trend"] * trend_score
        )

        base *= (0.82 + 0.18 * p["avail"])
        pos_scores[pid] = max(0.0, min(1.0, base))

    STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    NUM_TEAMS = 10

    dynasty_ppg_by_pos: Dict[str, list[tuple[str, float]]] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        age_for_horizon = p["age_opt"] if p["age_opt"] is not None else 26.0

        current_af = _age_factor(pos, age_for_horizon)
        future_af = horizon_age_factor(pos, age_for_horizon)
        horizon_scale = (future_af / current_af) if current_af else future_af

        dynasty_ppg = p["weighted_ppg_3yr"] * horizon_scale * p["avail"]
        p["dynasty_ppg"] = dynasty_ppg

        dynasty_ppg_by_pos.setdefault(pos, []).append((pid, dynasty_ppg))

    replacement_ppg: Dict[str, float] = {}
    for pos, lst_pos in dynasty_ppg_by_pos.items():
        if not lst_pos:
            replacement_ppg[pos] = 0.0
            continue

        lst_sorted = sorted(lst_pos, key=lambda x: x[1], reverse=True)
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

    final_scores: Dict[str, float] = {}
    SCARCITY_ALPHA = 0.28

    for pid, base_score in pos_scores.items():
        vor_norm = vor_map[pid] / max_vor if max_vor > 0 else 0.0
        blended = (1 - SCARCITY_ALPHA) * base_score + SCARCITY_ALPHA * vor_norm
        final_scores[pid] = max(0.0, min(1.0, blended))

    vals = list(final_scores.values())
    gmin, gmax = min(vals), max(vals)

    GAMMA = 0.60
    FLOOR = 0.05

    value_table: Dict[str, float] = {}

    for pid, v in final_scores.items():
        if gmax <= gmin:
            s01 = 0.0
        else:
            s01 = (v - gmin) / (gmax - gmin)

        s_curve = s01 ** GAMMA
        s_mix = FLOOR + (1.0 - FLOOR) * s_curve
        value_table[pid] = round(s_mix * 999.9, 1)

    return value_table
