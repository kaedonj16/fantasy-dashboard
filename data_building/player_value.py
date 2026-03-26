from __future__ import annotations

import math
from typing import Dict, Optional

from dashboard_services.api import get_nfl_state
from utils.utils import load_usage_table
from data_building.external_data.player_history import load_player_history_df, build_player_history_features
from data_building.external_data.player_investment import load_player_investment_context

CORE_POSITIONS = {"QB", "RB", "WR", "TE"}

# 10-team, 1QB defaults. If you later make this league-aware,
# these are the first values to parameterize.
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
NUM_TEAMS = 10

# 1QB replacement / scarcity assumptions
REPLACEMENT_MULT = {"QB": 1.50, "RB": 1.20, "WR": 1.20, "TE": 1.25}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip()


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _smoothstep01(x: float) -> float:
    x = _clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


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
        if vmax <= vmin:
            vmax = vmin + 1.0
        bounds[pos] = (vmin, vmax)

    out: Dict[str, float] = {}
    for pid, val in values_by_pid.items():
        pos = pos_by_pid.get(pid)
        if pos is None:
            out[pid] = 0.0
            continue
        vmin, vmax = bounds[pos]
        out[pid] = _clip((val - vmin) / (vmax - vmin))
    return out


def _peak_age_score(age: Optional[float], peak: float, left_width: float, right_width: float) -> float:
    """Smooth peak curve in [0,1]."""
    if age is None:
        return 0.85

    age = float(age)

    if age <= peak:
        d = (peak - age) / max(left_width, 0.001)
        return _clip(1.0 - 0.35 * (d ** 1.4), 0.35, 1.0)

    d = (age - peak) / max(right_width, 0.001)
    return _clip(1.0 - 0.70 * (d ** 1.15), 0.08, 1.0)


def _age_factor(pos: str, age: Optional[float]) -> float:
    """Smooth dynasty age curve. Higher = better 3-year outlook."""
    if age is None:
        return 0.85

    pos = (pos or "").upper()

    if pos == "RB":
        return _peak_age_score(age, peak=23.5, left_width=3.5, right_width=4.5)
    if pos == "WR":
        return _peak_age_score(age, peak=25.0, left_width=4.5, right_width=7.5)
    if pos == "QB":
        return _peak_age_score(age, peak=28.5, left_width=5.5, right_width=10.5)
    if pos == "TE":
        return _peak_age_score(age, peak=26.5, left_width=4.5, right_width=8.5)

    return 0.80


def horizon_age_factor(pos: str, age: Optional[float]) -> float:
    """3-year dynasty horizon age factor."""
    if age is None:
        age = 26.0

    weights = [0.45, 0.35, 0.20]
    vals = [_age_factor(pos, float(age) + i) for i in range(len(weights))]
    base = sum(w * v for w, v in zip(weights, vals)) / sum(weights)
    return base ** 1.10


def _production_component_fixed(u: dict, pos: str) -> float:
    """
    Current descriptive production / opportunity score in [0,1].

    Important:
    - QB intentionally does NOT use ppg directly here, because current PPG is
      already incorporated elsewhere in the model and double-counting was
      inflating efficient pocket passers.
    """
    ppg = _safe_float(u.get("ppr_ppg"))

    if pos == "QB":
        pass_yds = _safe_float(u.get("avg_pass_yds"))
        pass_tds = _safe_float(u.get("avg_pass_tds"))
        pass_int = _safe_float(u.get("avg_pass_int"))
        pass_att = _safe_float(u.get("avg_pass_att"))

        rush_yds = _safe_float(u.get("avg_rush_yards"))
        rush_tds = _safe_float(u.get("avg_rush_tds"))

        volume = _clip(pass_att / 38.0)
        rush_floor = _clip(rush_yds / 35.0)
        rush_td_rate = _clip(rush_tds / 0.5)

        score = (
                0.22 * _clip(pass_yds / 300.0) +
                0.24 * _clip(pass_tds / 2.5) +
                0.24 * _clip(rush_yds / 40.0) +
                0.18 * rush_td_rate +
                0.06 * volume +
                0.16 * rush_floor -
                0.08 * _clip(pass_int / 1.5)
        )
        return _clip(score)

    if pos == "RB":
        carries = _safe_float(u.get("avg_carries"))
        rush_yds = _safe_float(u.get("avg_rush_yards"))
        recs = _safe_float(u.get("avg_receptions"))
        tgts = _safe_float(u.get("avg_targets"))
        rush_tds = _safe_float(u.get("avg_rush_tds"))
        rec_tds = _safe_float(u.get("avg_rec_tds"))

        score = (
                0.28 * _clip(ppg / 21.5) +
                0.18 * _clip(carries / 17.5) +
                0.16 * _clip(rush_yds / 85.0) +
                0.18 * _clip(recs / 4.2) +
                0.10 * _clip(tgts / 5.2) +
                0.10 * _clip((rush_tds + rec_tds) / 0.9)
        )
        return _clip(score)

    if pos == "WR":
        tgts = _safe_float(u.get("avg_targets"))
        recs = _safe_float(u.get("avg_receptions"))
        rec_yds = _safe_float(u.get("avg_rec_yards"))
        rec_tds = _safe_float(u.get("avg_rec_tds"))
        target_share = _safe_float(u.get("target_share"))

        score = (
                0.28 * _clip(ppg / 20.0) +
                0.22 * _clip(tgts / 10.5) +
                0.15 * _clip(recs / 6.8) +
                0.18 * _clip(rec_yds / 88.0) +
                0.07 * _clip(rec_tds / 0.65) +
                0.10 * _clip(target_share / 0.28)
        )
        return _clip(score)

    if pos == "TE":
        tgts = _safe_float(u.get("avg_targets"))
        recs = _safe_float(u.get("avg_receptions"))
        rec_yds = _safe_float(u.get("avg_rec_yards"))
        rec_tds = _safe_float(u.get("avg_rec_tds"))
        target_share = _safe_float(u.get("target_share"))

        score = (
                0.30 * _clip(ppg / 16.5) +
                0.20 * _clip(tgts / 8.0) +
                0.12 * _clip(recs / 5.5) +
                0.16 * _clip(rec_yds / 65.0) +
                0.08 * _clip(rec_tds / 0.55) +
                0.14 * _clip(target_share / 0.24)
        )
        return _clip(score)

    return 0.0


def availability_score(u: dict, pos: str, games_possible: float = 17.0) -> float:
    g_played = _safe_float(u.get("games"))
    g_possible = _safe_float(u.get("games_possible"), games_possible)

    if g_possible <= 0:
        g_possible = games_possible

    raw_rate = (g_played / g_possible) if g_possible > 0 else 0.0

    k = 6.0
    prior = {"QB": 0.91, "RB": 0.83, "WR": 0.87, "TE": 0.85}.get(pos, 0.85)
    rate = (raw_rate * g_possible + prior * k) / (g_possible + k)

    floor = {"QB": 0.80, "RB": 0.74, "WR": 0.78, "TE": 0.76}.get(pos, 0.74)
    return _clip(max(floor, rate))


def _current_confidence(u: dict, pos: str) -> float:
    games = _safe_float(u.get("games"))
    snaps = _safe_float(u.get("avg_off_snap_pct"))
    if snaps <= 0:
        snaps = _safe_float(u.get("avg_off_snaps")) / {"QB": 65.0, "RB": 45.0, "WR": 50.0, "TE": 45.0}.get(pos, 50.0)

    opps = _safe_float(u.get("avg_targets")) + _safe_float(u.get("avg_carries"))

    games_conf = _smoothstep01(games / {"QB": 5.0, "RB": 6.0, "WR": 6.0, "TE": 6.0}.get(pos, 6.0))
    snaps_conf = _clip(snaps if snaps <= 1.25 else snaps / 70.0)
    opp_conf = _smoothstep01(opps / {"QB": 28.0, "RB": 12.0, "WR": 9.0, "TE": 7.0}.get(pos, 8.0))

    if pos == "QB":
        return _clip(0.60 * games_conf + 0.25 * snaps_conf + 0.15 * opp_conf)
    return _clip(0.50 * games_conf + 0.30 * snaps_conf + 0.20 * opp_conf)


def _history_confidence(hist: dict) -> float:
    seasons_played = _safe_float(hist.get("seasons_played"), 0.0)
    games_last_3yr = _safe_float(hist.get("games_last_3yr"), 0.0)
    c1 = _smoothstep01(seasons_played / 3.0)
    c2 = _smoothstep01(games_last_3yr / 34.0)
    return _clip(0.55 * c1 + 0.45 * c2)


def _usage_role_security(u: dict, hist: dict, pos: str) -> float:
    snap_now = _safe_float(u.get("avg_off_snap_pct"))
    if snap_now <= 0:
        snap_now = _safe_float(u.get("avg_off_snaps")) / {"QB": 65.0, "RB": 45.0, "WR": 50.0, "TE": 45.0}.get(pos, 50.0)

    snap_hist = _safe_float(hist.get("three_year_weighted_snap_pct"), snap_now)
    target_share_now = _safe_float(u.get("target_share"))
    target_share_hist = _safe_float(hist.get("three_year_weighted_target_share"), target_share_now)
    trend_ppg = _safe_float(hist.get("ppg_trend_1yr"))
    trend_tgt = _safe_float(hist.get("target_share_trend_1yr"))

    if pos == "QB":
        starter_signal = _clip(_safe_float(u.get("avg_pass_att")) / 32.0)
        security = (
                0.22 * _clip(snap_now) +
                0.18 * _clip(snap_hist) +
                0.32 * starter_signal +
                0.10 * _clip((trend_ppg + 4.0) / 8.0) +
                0.18 * _clip(_safe_float(hist.get("career_best_ppg")) / 24.0)
        )
        return _clip(security)

    opp_signal = _safe_float(u.get("avg_targets")) + _safe_float(u.get("avg_carries"))
    opp_norm = {
        "RB": _clip(opp_signal / 16.0),
        "WR": _clip(opp_signal / 10.5),
        "TE": _clip(opp_signal / 8.0),
    }.get(pos, _clip(opp_signal / 10.0))

    share_norm = _clip(target_share_hist / {"RB": 0.16, "WR": 0.28, "TE": 0.24}.get(pos, 0.25))
    trend_mix = 0.70 * _clip((trend_ppg + 4.0) / 8.0) + 0.30 * _clip((trend_tgt + 0.06) / 0.12)

    security = (
            0.25 * _clip(snap_now) +
            0.20 * _clip(snap_hist) +
            0.25 * opp_norm +
            0.18 * share_norm +
            0.12 * trend_mix
    )
    return _clip(security)


def _investment_score(invest: dict, pos: str, age: Optional[float]) -> float:
    draft_capital = _safe_float(invest.get("draft_capital_score"))
    draft_capital_pct = _safe_float(invest.get("draft_capital_pos_pct"))
    contract_score = _safe_float(invest.get("contract_score"))
    team_investment = _safe_float(invest.get("team_investment_score"))
    years_to_fa = _safe_float(invest.get("years_to_fa"))
    apy_pct = _safe_float(invest.get("contract_apy_pos_pct"))
    guaranteed_pct = _safe_float(invest.get("guaranteed_pct_pos_pct"))

    if draft_capital_pct > 1.5:
        draft_capital_pct /= 100.0
    if apy_pct > 1.5:
        apy_pct /= 100.0
    if guaranteed_pct > 1.5:
        guaranteed_pct /= 100.0

    years_score = _clip(years_to_fa / {"QB": 4.0, "RB": 3.0, "WR": 4.0, "TE": 4.0}.get(pos, 4.0))
    raw = (
            0.22 * _clip(draft_capital / 1000.0) +
            0.18 * _clip(draft_capital_pct) +
            0.18 * _clip(contract_score / 1000.0) +
            0.18 * _clip(team_investment / 1000.0) +
            0.12 * years_score +
            0.07 * _clip(apy_pct) +
            0.05 * _clip(guaranteed_pct)
    )

    if age is not None and age <= {"QB": 27, "RB": 24, "WR": 26, "TE": 26}.get(pos, 26):
        raw *= 1.06

    return _clip(raw)


def _trend_score(hist: dict, pos: str) -> float:
    ppg_trend_1yr = _safe_float(hist.get("ppg_trend_1yr"))
    ppg_trend_2yr = _safe_float(hist.get("ppg_trend_2yr"))
    target_share_trend_1yr = _safe_float(hist.get("target_share_trend_1yr"))

    def trend_to_unit(x: float, scale: float) -> float:
        x = max(-scale, min(scale, x))
        return (x + scale) / (2.0 * scale)

    if pos == "QB":
        return _clip(
            0.72 * trend_to_unit(ppg_trend_1yr, 5.0) +
            0.28 * trend_to_unit(ppg_trend_2yr, 8.0)
        )

    return _clip(
        0.55 * trend_to_unit(ppg_trend_1yr, 4.0) +
        0.20 * trend_to_unit(ppg_trend_2yr, 6.0) +
        0.25 * trend_to_unit(target_share_trend_1yr, 0.08)
    )


def _risk_penalty(
        pos: str,
        age: Optional[float],
        avail: float,
        current_conf: float,
        hist_conf: float,
        role_security: float,
        seasons_played: float,
) -> float:
    age_risk = 1.0 - _age_factor(pos, age)
    sample_risk = 1.0 - _clip(0.60 * current_conf + 0.40 * hist_conf)
    role_risk = 1.0 - role_security
    injury_risk = 1.0 - avail
    exp_risk = 1.0 - _clip(seasons_played / 3.0)

    if pos == "RB":
        penalty = (
                0.28 * age_risk +
                0.24 * injury_risk +
                0.20 * role_risk +
                0.18 * sample_risk +
                0.10 * exp_risk
        )
    elif pos == "QB":
        penalty = (
                0.18 * age_risk +
                0.14 * injury_risk +
                0.30 * role_risk +
                0.22 * sample_risk +
                0.16 * exp_risk
        )
    else:
        penalty = (
                0.22 * age_risk +
                0.20 * injury_risk +
                0.22 * role_risk +
                0.22 * sample_risk +
                0.14 * exp_risk
        )

    return _clip(penalty)


def _is_relevant(pid: str, meta: dict, usage: dict, history_by_pid: Dict[str, dict], offseason_mode: bool) -> bool:
    pos = _safe_str(meta.get("pos")).upper()
    if pos not in CORE_POSITIONS:
        return False

    hist = history_by_pid.get(pid, {}) or {}

    games = _safe_float(usage.get("games"))
    ppg = _safe_float(usage.get("ppr_ppg"))
    snaps = _safe_float(usage.get("avg_off_snaps"))
    opps = _safe_float(usage.get("avg_targets")) + _safe_float(usage.get("avg_carries"))

    hist_last_year_ppg = _safe_float(hist.get("last_year_ppg"))
    hist_weighted_ppg = _safe_float(hist.get("three_year_weighted_ppg"))
    hist_career_best = _safe_float(hist.get("career_best_ppg"))
    hist_seasons = _safe_float(hist.get("seasons_played"))

    if not offseason_mode:
        if games >= 3:
            return True
        if ppg >= 6:
            return True
        if snaps >= 20:
            return True
        if opps >= 3:
            return True
        if hist_weighted_ppg >= 8:
            return True
        if hist_career_best >= 12:
            return True
        if hist_seasons >= 2 and hist_last_year_ppg >= 6:
            return True
        return False

    if hist_weighted_ppg >= 6:
        return True
    if hist_last_year_ppg >= 6:
        return True
    if hist_career_best >= 10:
        return True
    if hist_seasons >= 2 and hist_career_best >= 7:
        return True
    if games >= 2:
        return True
    if ppg >= 5:
        return True
    if snaps >= 15:
        return True
    if opps >= 2:
        return True

    return False


def _proven_elite_bonus(pos: str, career_best_ppg: float, weighted_ppg_3yr: float, seasons_played: float) -> float:
    if pos == "WR":
        if career_best_ppg >= 17.5 and weighted_ppg_3yr >= 15.0 and seasons_played >= 2:
            return 1.0
        if career_best_ppg >= 15.5 and weighted_ppg_3yr >= 13.5 and seasons_played >= 2:
            return 0.7
        return 0.0

    if pos == "RB":
        if career_best_ppg >= 19.0 and weighted_ppg_3yr >= 15.0 and seasons_played >= 2:
            return 1.0
        if career_best_ppg >= 16.0 and weighted_ppg_3yr >= 13.0 and seasons_played >= 2:
            return 0.6
        return 0.0

    if pos == "TE":
        if career_best_ppg >= 14.0 and weighted_ppg_3yr >= 11.5 and seasons_played >= 2:
            return 1.0
        return 0.0

    if pos == "QB":
        if career_best_ppg >= 23.0 and weighted_ppg_3yr >= 20.0 and seasons_played >= 2:
            return 1.0
        return 0.0

    return 0.0


def _apply_qb_market_compression(
        final_scores: Dict[str, float],
        pos_by_pid: Dict[str, str],
        elite_norm: Dict[str, float],
        ceiling_norm: Dict[str, float],
        per_pid: Dict[str, dict],
) -> Dict[str, float]:
    for pid, score in list(final_scores.items()):
        if pos_by_pid.get(pid) != "QB":
            continue

        elite = elite_norm.get(pid, 0.0)
        ceiling = ceiling_norm.get(pid, 0.0)
        p = per_pid.get(pid, {}) or {}

        rush_yds = _safe_float(p.get("rush_yds_pg"))
        rush_tds = _safe_float(p.get("rush_tds_pg"))
        rush_norm = _clip((rush_yds / 35.0) + (rush_tds / 0.5), 0.0, 1.0)

        elite_soft = elite ** 0.80
        ceiling_soft = ceiling ** 0.90

        base = 0.42
        elite_boost = 0.22 * elite_soft
        ceiling_boost = 0.05 * ceiling_soft
        rushing_boost = 0.13 * rush_norm

        qb_keep = base + elite_boost + ceiling_boost + rushing_boost
        qb_keep = min(qb_keep, 0.74)

        final_scores[pid] = _clip(score * qb_keep)

    return final_scores


def _apply_te_market_compression(
        final_scores: Dict[str, float],
        pos_by_pid: Dict[str, str],
        elite_norm: Dict[str, float],
        ceiling_norm: Dict[str, float],
) -> Dict[str, float]:
    for pid, score in list(final_scores.items()):
        if pos_by_pid.get(pid) != "TE":
            continue

        elite = elite_norm.get(pid, 0.0)
        ceiling = ceiling_norm.get(pid, 0.0)

        keep = (
                0.79
                + 0.09 * (elite ** 0.90)
                + 0.03 * (ceiling ** 0.95)
        )
        keep = min(keep, 0.89)
        final_scores[pid] = _clip(score * keep)

    return final_scores


def build_value_table_for_usage() -> Dict[str, float]:
    """
    Dynasty value formula using:
      - smooth age curve
      - current + historical production blend
      - role security
      - draft capital / contract / team investment
      - availability
      - trend
      - risk penalty
      - scarcity based on replacement / starter / elite edge
      - 1QB QB compression

    Returns:
      Dict[player_id, value_0_to_999_9]
    """
    lst = load_usage_table()
    if not isinstance(lst, list):
        raise ValueError("usage table must be a list of player objects")

    nfl_state = get_nfl_state() or {}
    season_type = _safe_str(nfl_state.get("season_type")).lower()
    offseason_mode = season_type == "off"

    history_df = load_player_history_df()
    history_features_df = build_player_history_features(
        history_df) if history_df is not None and not history_df.empty else None

    history_by_pid: Dict[str, dict] = {}
    if history_features_df is not None and not history_features_df.empty:
        for _, row in history_features_df.iterrows():
            sleeper_id = _safe_str(row.get("sleeper_id"))
            if sleeper_id:
                history_by_pid[sleeper_id] = row.to_dict()

    investment_df = load_player_investment_context()
    investment_by_pid: Dict[str, dict] = {}
    if investment_df is not None and not investment_df.empty and "sleeper_id" in investment_df.columns:
        for _, row in investment_df.iterrows():
            sleeper_id = _safe_str(row.get("sleeper_id"))
            if sleeper_id:
                investment_by_pid[sleeper_id] = row.to_dict()

    players_index: Dict[str, dict] = {}
    usage_table: Dict[str, dict] = {}

    for obj in lst:
        pid = _safe_str(obj.get("id"))
        if not pid:
            continue

        pos = obj.get("position") or obj.get("pos")
        usage = obj.get("usage")
        if not isinstance(usage, dict):
            usage = {
                "games": obj.get("games"),
                "ppr_ppg": obj.get("ppr_ppg"),
                "avg_off_snaps": obj.get("avg_off_snaps"),
                "avg_off_snap_pct": obj.get("avg_off_snap_pct"),
                "avg_targets": obj.get("avg_targets"),
                "avg_carries": obj.get("avg_carries"),
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

        players_index[pid] = {
            "name": obj.get("name"),
            "team": obj.get("team"),
            "pos": pos,
            "age": obj.get("age"),
        }
        usage_table[pid] = usage or {}

    filtered_players_index: Dict[str, dict] = {}
    filtered_usage_table: Dict[str, dict] = {}

    for pid, meta in players_index.items():
        usage = usage_table.get(pid, {}) or {}
        if _is_relevant(pid, meta, usage, history_by_pid, offseason_mode):
            filtered_players_index[pid] = meta
            filtered_usage_table[pid] = usage

    players_index = filtered_players_index
    usage_table = filtered_usage_table

    if not players_index:
        return {}

    per_pid: Dict[str, dict] = {}

    for pid, usage in usage_table.items():
        meta = players_index.get(pid, {})
        pos = _safe_str(meta.get("pos")).upper()
        if pos not in CORE_POSITIONS:
            continue

        raw_age = meta.get("age")
        try:
            age: Optional[float] = float(raw_age) if raw_age not in (None, "") else None
        except (TypeError, ValueError):
            age = None

        hist = history_by_pid.get(pid, {}) or {}
        invest = investment_by_pid.get(pid, {}) or {}

        avail = availability_score(usage, pos)
        prod_now = _production_component_fixed(usage, pos)
        current_ppg = _safe_float(usage.get("ppr_ppg"))

        last_year_ppg = _safe_float(hist.get("last_year_ppg"), current_ppg)
        prev_year_ppg = _safe_float(hist.get("prev_year_ppg"), current_ppg)
        weighted_ppg_3yr = _safe_float(hist.get("three_year_weighted_ppg"), max(current_ppg, last_year_ppg))
        career_best_ppg = _safe_float(hist.get("career_best_ppg"), max(current_ppg, last_year_ppg))
        career_avg_ppg = _safe_float(hist.get("career_avg_ppg"), weighted_ppg_3yr)

        snap_hist = _safe_float(hist.get("three_year_weighted_snap_pct"))
        target_share_hist = _safe_float(hist.get("three_year_weighted_target_share"),
                                        _safe_float(usage.get("target_share")))
        seasons_played = _safe_float(hist.get("seasons_played"), 1.0)

        current_conf = _current_confidence(usage, pos)
        hist_conf = _history_confidence(hist)

        if offseason_mode:
            current_weight = 0.28 * current_conf
            hist_weight = 0.72 + 0.20 * hist_conf
        else:
            current_weight = 0.52 * current_conf + 0.18
            hist_weight = 0.48 + 0.20 * hist_conf

        denom = max(current_weight + hist_weight, 1e-9)
        blended_prod = (
                               current_weight * current_ppg +
                               hist_weight * weighted_ppg_3yr
                       ) / denom

        ceiling_proxy = 0.65 * career_best_ppg + 0.35 * max(current_ppg, last_year_ppg)
        floor_proxy = 0.70 * career_avg_ppg + 0.30 * last_year_ppg

        rz_metric = _safe_float(usage.get("rec_rz_tgt_pg")) + _safe_float(usage.get("rush_rz_att_pg"))
        role_security = _usage_role_security(usage, hist, pos)
        trend_score = _trend_score(hist, pos)
        age_curve = horizon_age_factor(pos, age)
        invest_score = _investment_score(invest, pos, age)

        rush_yds_pg = _safe_float(usage.get("avg_rush_yards"))
        rush_tds_pg = _safe_float(usage.get("avg_rush_tds"))

        risk_penalty = _risk_penalty(
            pos=pos,
            age=age,
            avail=avail,
            current_conf=current_conf,
            hist_conf=hist_conf,
            role_security=role_security,
            seasons_played=seasons_played,
        )
        proven_elite = _proven_elite_bonus(
            pos,
            career_best_ppg,
            weighted_ppg_3yr,
            seasons_played,
        )

        per_pid[pid] = {
            "pos": pos,
            "age": age,
            "avail": avail,
            "current_conf": current_conf,
            "hist_conf": hist_conf,
            "current_ppg": current_ppg,
            "blended_prod": blended_prod,
            "prod_now": prod_now,
            "ceiling_proxy": ceiling_proxy,
            "floor_proxy": floor_proxy,
            "rz_metric": rz_metric,
            "target_share_hist": target_share_hist,
            "snap_hist": snap_hist,
            "role_security": role_security,
            "trend_score": trend_score,
            "age_curve": age_curve,
            "invest_score": invest_score,
            "risk_penalty": risk_penalty,
            "weighted_ppg_3yr": weighted_ppg_3yr,
            "career_best_ppg": career_best_ppg,
            "career_avg_ppg": career_avg_ppg,
            "last_year_ppg": last_year_ppg,
            "prev_year_ppg": prev_year_ppg,
            "seasons_played": seasons_played,
            "proven_elite": proven_elite,
            "rush_yds_pg": rush_yds_pg,
            "rush_tds_pg": rush_tds_pg,
        }

    if not per_pid:
        return {}

    pos_by_pid = {pid: p["pos"] for pid, p in per_pid.items()}

    blended_prod_norm = _normalize_by_pos({pid: p["blended_prod"] for pid, p in per_pid.items()}, pos_by_pid)
    current_ppg_norm = _normalize_by_pos({pid: p["current_ppg"] for pid, p in per_pid.items()}, pos_by_pid)
    prod_now_norm = _normalize_by_pos({pid: p["prod_now"] for pid, p in per_pid.items()}, pos_by_pid)
    ceiling_norm = _normalize_by_pos({pid: p["ceiling_proxy"] for pid, p in per_pid.items()}, pos_by_pid)
    floor_norm = _normalize_by_pos({pid: p["floor_proxy"] for pid, p in per_pid.items()}, pos_by_pid)
    rz_norm = _normalize_by_pos({pid: p["rz_metric"] for pid, p in per_pid.items()}, pos_by_pid)
    target_share_norm = _normalize_by_pos({pid: p["target_share_hist"] for pid, p in per_pid.items()}, pos_by_pid)
    snap_norm = _normalize_by_pos({pid: p["snap_hist"] for pid, p in per_pid.items()}, pos_by_pid)

    POS_WEIGHTS = {
        "QB": {
            "blended_prod": 0.16,
            "current_prod": 0.20,
            "ceiling": 0.16,
            "floor": 0.03,
            "age": 0.09,
            "role": 0.11,
            "trend": 0.07,
            "invest": 0.04,
            "rz": 0.00,
            "share": 0.00,
            "snap": 0.04,
        },
        "RB": {
            "blended_prod": 0.22,
            "current_prod": 0.10,
            "ceiling": 0.08,
            "floor": 0.06,
            "age": 0.13,
            "role": 0.12,
            "trend": 0.05,
            "invest": 0.10,
            "rz": 0.08,
            "share": 0.00,
            "snap": 0.06,
        },
        "WR": {
            "blended_prod": 0.27,
            "current_prod": 0.07,
            "ceiling": 0.12,
            "floor": 0.11,
            "age": 0.09,
            "role": 0.12,
            "trend": 0.03,
            "invest": 0.06,
            "rz": 0.04,
            "share": 0.06,
            "snap": 0.03,
        },
        "TE": {
            "blended_prod": 0.24,
            "current_prod": 0.07,
            "ceiling": 0.08,
            "floor": 0.10,
            "age": 0.09,
            "role": 0.10,
            "trend": 0.03,
            "invest": 0.05,
            "rz": 0.07,
            "share": 0.10,
            "snap": 0.07,
        },
    }

    pos_scores: Dict[str, float] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        w = POS_WEIGHTS[pos]

        base = (
                w["blended_prod"] * blended_prod_norm.get(pid, 0.0) +
                w["current_prod"] * prod_now_norm.get(pid, 0.0) +
                w["ceiling"] * ceiling_norm.get(pid, 0.0) +
                w["floor"] * floor_norm.get(pid, 0.0) +
                w["age"] * p["age_curve"] +
                w["role"] * p["role_security"] +
                w["trend"] * p["trend_score"] +
                w["invest"] * p["invest_score"] +
                w["rz"] * rz_norm.get(pid, 0.0) +
                w["share"] * target_share_norm.get(pid, 0.0) +
                w["snap"] * snap_norm.get(pid, 0.0)
        )

        if pos == "QB":
            ceiling_gap = max(
                ceiling_norm.get(pid, 0.0) - floor_norm.get(pid, 0.0),
                0.0
            )
            base *= (0.92 + 0.14 * ceiling_gap)

            rush_yds_pg = p["rush_yds_pg"]
            rush_tds_pg = p["rush_tds_pg"]
            rush_profile = _clip((rush_yds_pg / 35.0) + (rush_tds_pg / 0.5), 0.0, 1.0)

            if rush_yds_pg < 10 and rush_tds_pg < 0.20:
                base *= 0.84
            elif rush_yds_pg < 20 and rush_tds_pg < 0.30:
                base *= 0.90
            elif rush_profile < 0.55:
                base *= 0.95

        confidence_multiplier = 0.82 + 0.10 * p["current_conf"] + 0.08 * p["hist_conf"]
        availability_multiplier = 0.84 + 0.16 * p["avail"]

        adjusted = base * confidence_multiplier * availability_multiplier
        adjusted *= (0.90 - 0.22 * p["risk_penalty"])

        pos_scores[pid] = _clip(adjusted)

    dynasty_strength_by_pos: Dict[str, list[tuple[str, float]]] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]

        if pos == "QB":
            rush_bonus = 1.0 + 0.12 * _clip((p["rush_yds_pg"] / 35.0) + (p["rush_tds_pg"] / 0.5), 0.0, 1.0)
        else:
            rush_bonus = 1.0

        dynasty_strength = (
                                   0.46 * p["blended_prod"] +
                                   0.22 * p["ceiling_proxy"] +
                                   0.14 * p["floor_proxy"] +
                                   0.10 * p["role_security"] * max(p["blended_prod"], 1.0) +
                                   0.08 * p["invest_score"] * max(p["blended_prod"], 1.0)
                           ) * p["age_curve"] * p["avail"] * rush_bonus

        dynasty_strength_by_pos.setdefault(pos, []).append((pid, dynasty_strength))

    replacement_map: Dict[str, float] = {}
    starter_map: Dict[str, float] = {}
    elite_map: Dict[str, float] = {}

    elite_cutoffs = {
        "QB": 4,
        "RB": 12,
        "WR": 18,
        "TE": 5,
    }

    for pos, lst_pos in dynasty_strength_by_pos.items():
        if not lst_pos:
            replacement_map[pos] = 0.0
            starter_map[pos] = 0.0
            elite_map[pos] = 0.0
            continue

        lst_sorted = sorted(lst_pos, key=lambda x: x[1], reverse=True)

        starter_slots = STARTERS[pos] * NUM_TEAMS
        replacement_idx = int(starter_slots * REPLACEMENT_MULT[pos])
        replacement_idx = max(0, min(replacement_idx, len(lst_sorted) - 1))
        starter_idx = max(0, min(starter_slots - 1, len(lst_sorted) - 1))
        elite_idx = max(0, min(elite_cutoffs[pos] - 1, len(lst_sorted) - 1))

        replacement_map[pos] = lst_sorted[replacement_idx][1]
        starter_map[pos] = lst_sorted[starter_idx][1]
        elite_map[pos] = lst_sorted[elite_idx][1]

    replacement_edge: Dict[str, float] = {}
    starter_edge: Dict[str, float] = {}
    elite_edge: Dict[str, float] = {}

    for pid, p in per_pid.items():
        pos = p["pos"]
        own = next(val for player_id, val in dynasty_strength_by_pos[pos] if player_id == pid)

        replacement_edge[pid] = max(own - replacement_map[pos], 0.0)
        starter_edge[pid] = max(own - starter_map[pos], 0.0)
        elite_edge[pid] = max(own - elite_map[pos], 0.0)

    replacement_norm = _normalize_by_pos(replacement_edge, pos_by_pid)
    starter_norm = _normalize_by_pos(starter_edge, pos_by_pid)
    elite_norm = _normalize_by_pos(elite_edge, pos_by_pid)

    final_scores: Dict[str, float] = {}

    SCARCITY_ALPHA = {
        "QB": 0.05,
        "RB": 0.32,
        "WR": 0.25,
        "TE": 0.30,
    }

    for pid, base_score in pos_scores.items():
        pos = pos_by_pid[pid]
        scarcity = (
                0.42 * replacement_norm.get(pid, 0.0) +
                0.33 * starter_norm.get(pid, 0.0) +
                0.25 * elite_norm.get(pid, 0.0)
        )
        alpha = SCARCITY_ALPHA[pos]
        final_scores[pid] = _clip((1.0 - alpha) * base_score + alpha * scarcity)

    final_scores = _apply_qb_market_compression(
        final_scores,
        pos_by_pid,
        elite_norm,
        ceiling_norm,
        per_pid,
    )

    final_scores = _apply_te_market_compression(
        final_scores,
        pos_by_pid,
        elite_norm,
        ceiling_norm,
    )

    vals = list(final_scores.values())
    gmin = min(vals) if vals else 0.0
    gmax = max(vals) if vals else 1.0

    GAMMA = 0.72
    FLOOR = 0.03
    ELITE_BOOST_SCALE = 0.035

    value_table: Dict[str, float] = {}

    for pid, v in final_scores.items():
        if gmax <= gmin:
            s01 = 0.0
        else:
            s01 = (v - gmin) / (gmax - gmin)

        s_curve = s01 ** GAMMA
        elite_bonus = ELITE_BOOST_SCALE * (elite_norm.get(pid, 0.0) ** 1.8)
        s_mix = FLOOR + (1.0 - FLOOR) * _clip(s_curve + elite_bonus)
        value_table[pid] = round(s_mix * 999.9, 1)

    return value_table
