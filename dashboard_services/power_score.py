"""Unified PowerScore engine.

Two scoring modes share the same z-score math:

1. ``performance_power_scores`` — results-only (standings, historical week
   views, career/tour graphs). Reconstructible from weekly scores alone.
2. ``blended_team_scores`` — canonical live power rankings. Adds luck-adjusted
   record, slot-legal starter value, momentum, consistency, past SoS, rest-of-
   season schedule ease, and playoff odds. Weights shift by season phase.

Improvements vs the prior dual formulas:
  - One shared z-score implementation
  - Scoring volume uses AVG (PPG) instead of total PF (less double-count with
    the luck-adjusted record term)
  - Season-phase weights (preseason → early → mid → late)
  - Starter value fills real lineup slots when available
  - Past SoS + ROS ease + playoff odds enter the live blend
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

_CORE_POS = {"QB", "RB", "WR", "TE"}

# Results-only blend (standings / historical / graphs). Weights sum to 1.0.
# Past SoS is optional — when missing its weight redistributes.
PERFORMANCE_WEIGHTS = {
    "win": 0.18,
    "avg": 0.26,
    "last3": 0.22,
    "consistency": 0.12,
    "ceiling": 0.12,
    "sos": 0.10,
}

# Live blended rankings by season phase. Component key "pf" is the scoring
# volume z-score (AVG/PPG). Weights sum to 1.0; missing optional components
# (playoff / ros / momentum) redistribute onto the remaining terms.
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "preseason": {
        "pf": 0.05,
        "record": 0.05,
        "value": 0.70,
        "momentum": 0.00,
        "consistency": 0.00,
        "sos": 0.05,
        "ros": 0.00,
        "playoff": 0.15,
    },
    "early": {
        "pf": 0.16,
        "record": 0.18,
        "value": 0.28,
        "momentum": 0.08,
        "consistency": 0.06,
        "sos": 0.08,
        "ros": 0.04,
        "playoff": 0.12,
    },
    "mid": {
        "pf": 0.16,
        "record": 0.24,
        "value": 0.18,
        "momentum": 0.12,
        "consistency": 0.08,
        "sos": 0.08,
        "ros": 0.04,
        "playoff": 0.10,
    },
    "late": {
        "pf": 0.14,
        "record": 0.26,
        "value": 0.12,
        "momentum": 0.14,
        "consistency": 0.08,
        "sos": 0.08,
        "ros": 0.06,
        "playoff": 0.12,
    },
}


def z_scores(values: Sequence[float]) -> list[float]:
    """Population z-scores; zeros when variance is zero or n < 2."""
    n = len(values)
    if n < 2:
        return [0.0] * n
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def season_phase_from_progress(
    *,
    games_played: int = 0,
    current_week: Optional[int] = None,
) -> str:
    """Map season progress onto preseason / early / mid / late."""
    games = max(0, int(games_played or 0))
    if games <= 0:
        return "preseason"
    if games <= 4:
        return "early"
    if games <= 9:
        return "mid"
    return "late"


def _normalize_weights(
    weights: Mapping[str, float],
    available: Iterable[str],
) -> dict[str, float]:
    """Drop unavailable keys and renormalize so remaining weights sum to 1."""
    avail = set(available)
    kept = {k: float(v) for k, v in weights.items() if k in avail and float(v) > 0}
    total = sum(kept.values())
    if total <= 0:
        keys = [k for k in weights if k in avail]
        if not keys:
            return {}
        share = 1.0 / len(keys)
        return {k: share for k in keys}
    return {k: v / total for k, v in kept.items()}


def _weighted_sum(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(float(components.get(k, 0.0)) * float(w) for k, w in weights.items())


# ── Starter value ─────────────────────────────────────────────────────────────


def starter_lineup_value(
    player_ids: Sequence[Any],
    model_value_lookup: Mapping[str, Mapping[str, Any]],
    *,
    redraft_key: str,
    roster_positions: Optional[Sequence[Any]] = None,
) -> float:
    """Win-now starter strength: slot-legal lineup when positions are known.

    Falls back to the top-8 core (QB/RB/WR/TE) redraft sum when lineup slots
    are missing or produce an empty fill — same safety net as the old formula.
    """
    pairs: list[tuple[str, str, float]] = []  # pid, pos, redraft
    for raw in player_ids or []:
        pid = str(raw or "").strip()
        if not pid:
            continue
        row = model_value_lookup.get(pid) or {}
        pos = str(row.get("position") or "").upper()
        try:
            rval = float(row.get(redraft_key) or 0.0)
        except (TypeError, ValueError):
            rval = 0.0
        pairs.append((pid, pos, rval))

    if not pairs:
        return 0.0

    score_by_pid = {pid: rval for pid, _pos, rval in pairs}
    pos_by_pid = {pid: pos for pid, pos, _rval in pairs}

    if roster_positions:
        try:
            from utils.starter_lineup import derive_starters_from_slots

            starters = derive_starters_from_slots(
                [pid for pid, _pos, _rval in pairs],
                roster_positions,
                pos_by_pid=pos_by_pid,
                score_by_pid=score_by_pid,
            )
            if starters:
                return round(sum(score_by_pid.get(pid, 0.0) for pid in starters), 1)
        except Exception:
            pass

    core = sorted(
        (rval for _pid, pos, rval in pairs if pos in _CORE_POS),
        reverse=True,
    )
    return round(sum(core[:8]), 1)


def performance_component_z(
    *,
    win_pct: Sequence[float],
    avg: Sequence[float],
    last3: Sequence[float],
    consistency: Sequence[float],
    ceiling: Sequence[float],
    past_sos: Optional[Sequence[float]] = None,
) -> dict[str, list[float]]:
    """Z-score each performance component (same-length sequences)."""
    out = {
        "win": z_scores(list(win_pct)),
        "avg": z_scores(list(avg)),
        "last3": z_scores(list(last3)),
        "consistency": z_scores(list(consistency)),
        "ceiling": z_scores(list(ceiling)),
    }
    if past_sos is not None:
        out["sos"] = z_scores(list(past_sos))
    return out


def performance_power_scores(
    *,
    win_pct: Sequence[float],
    avg: Sequence[float],
    last3: Sequence[float],
    consistency: Sequence[float],
    ceiling: Sequence[float],
    past_sos: Optional[Sequence[float]] = None,
) -> list[float]:
    """Results-only PowerScore list aligned to the input sequences."""
    comps = performance_component_z(
        win_pct=win_pct,
        avg=avg,
        last3=last3,
        consistency=consistency,
        ceiling=ceiling,
        past_sos=past_sos,
    )
    weights = _normalize_weights(PERFORMANCE_WEIGHTS, comps.keys())
    n = len(win_pct)
    scores: list[float] = []
    for i in range(n):
        row = {k: comps[k][i] for k in comps}
        scores.append(round(_weighted_sum(row, weights), 6))
    return scores


def approximate_power_score_frame(team_stats: Any) -> Any:
    """Attach a performance PowerScore column to a pandas team_stats frame.

    Used by career/tour graphs when the full finalize_team_stats pipeline
    (incl. SOS) is not available. Requires Win%, AVG, and ideally MAX/STD;
    synthesizes Last3 from AVG when missing.
    """
    import pandas as pd

    if team_stats is None or getattr(team_stats, "empty", True):
        return team_stats
    df = team_stats
    n = len(df)
    if n == 0:
        return df

    win = df["Win%"].fillna(0.0).astype(float) if "Win%" in df.columns else pd.Series([0.0] * n)
    avg = df["AVG"].fillna(0.0).astype(float) if "AVG" in df.columns else pd.Series([0.0] * n)
    if "Last3" in df.columns:
        last3 = df["Last3"].fillna(0.0).astype(float)
    else:
        last3 = avg
    if "STD" in df.columns:
        cons = (-df["STD"].fillna(0.0).astype(float)).tolist()
    else:
        cons = [0.0] * n
    if "MAX" in df.columns:
        ceiling = df["MAX"].fillna(0.0).astype(float).tolist()
    else:
        ceiling = avg.tolist()
    past = None
    if "past_sos" in df.columns:
        past = df["past_sos"].fillna(100.0).astype(float).tolist()

    scores = performance_power_scores(
        win_pct=win.tolist(),
        avg=avg.tolist(),
        last3=last3.tolist(),
        consistency=cons,
        ceiling=ceiling,
        past_sos=past,
    )
    df = df.copy()
    df["PowerScore"] = scores
    comps = performance_component_z(
        win_pct=win.tolist(),
        avg=avg.tolist(),
        last3=last3.tolist(),
        consistency=cons,
        ceiling=ceiling,
        past_sos=past,
    )
    df["Z_WinPercentage"] = comps["win"]
    df["Z_Avg"] = comps["avg"]
    df["Z_Last3"] = comps["last3"]
    df["Z_Consistency"] = comps["consistency"]
    df["Z_Ceiling"] = comps["ceiling"]
    return df


# ── Blended (canonical live) PowerScore ───────────────────────────────────────


def blended_team_scores(
    teams: list[dict],
    *,
    phase: str = "mid",
) -> list[dict]:
    """Score and sort team dicts; returns the same list sorted best-first.

    Each team may carry raw fields:
      avg (preferred) or pf, luck_adj_win, starter_value, momentum, consistency,
      sos, ros_ease (higher = easier remaining schedule), playoff_pct

    Writes ``power_score``, ``power_components``, and ``rank``.
    Component key ``pf`` is the scoring-volume z-score (AVG/PPG based).
    """
    if not teams:
        return teams

    def _avg_of(t: dict) -> float:
        if t.get("avg") is not None:
            try:
                return float(t["avg"])
            except (TypeError, ValueError):
                pass
        try:
            return float(t.get("pf") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    scoring_z = z_scores([_avg_of(t) for t in teams])
    record_z = z_scores([float(t.get("luck_adj_win") or 0.0) for t in teams])
    value_z = z_scores([float(t.get("starter_value") or 0.0) for t in teams])
    mom_z = z_scores([float(t.get("momentum") or 0.0) for t in teams])
    con_z = z_scores([float(t.get("consistency") or 0.0) for t in teams])
    sos_z = z_scores([
        float(t["sos"]) if t.get("sos") is not None else 0.5 for t in teams
    ])

    has_ros = any(t.get("ros_ease") is not None for t in teams)
    ros_z = (
        z_scores([
            float(t["ros_ease"]) if t.get("ros_ease") is not None else 0.5
            for t in teams
        ])
        if has_ros
        else [0.0] * len(teams)
    )

    has_playoff = any(t.get("playoff_pct") is not None for t in teams)
    playoff_z = (
        z_scores([
            float(t["playoff_pct"]) if t.get("playoff_pct") is not None else 0.0
            for t in teams
        ])
        if has_playoff
        else [0.0] * len(teams)
    )

    base_weights = PHASE_WEIGHTS.get(phase) or PHASE_WEIGHTS["mid"]
    available = {"pf", "record", "value", "momentum", "consistency", "sos"}
    if has_ros:
        available.add("ros")
    if has_playoff:
        available.add("playoff")
    available = {
        k for k in available
        if float(base_weights.get(k, 0.0)) > 0 or k in ("pf", "record", "value")
    }
    weights = _normalize_weights(base_weights, available)

    for i, team in enumerate(teams):
        comps = {
            "pf": round(scoring_z[i], 2),
            "record": round(record_z[i], 2),
            "value": round(value_z[i], 2),
            "momentum": round(mom_z[i], 2),
            "consistency": round(con_z[i], 2),
            "sos": round(sos_z[i], 2),
        }
        if has_ros:
            comps["ros"] = round(ros_z[i], 2)
        if has_playoff:
            comps["playoff"] = round(playoff_z[i], 2)
        team["power_components"] = comps
        team["power_score"] = round(
            _weighted_sum({k: float(comps[k]) for k in comps}, weights),
            3,
        )

    teams.sort(key=lambda t: t.get("power_score") or 0.0, reverse=True)
    for rank, team in enumerate(teams, start=1):
        team["rank"] = rank
    return teams


def value_scale_score(
    totals: Sequence[float],
    *,
    floor: float = 100.0,
    span: float = 60.0,
) -> list[float]:
    """Offseason display scale: ``floor + total/max * span``."""
    vals = [float(v or 0.0) for v in totals]
    raw_max = max(vals) if vals else 0.0
    if raw_max <= 0:
        return [float(floor) for _ in vals]
    return [round(floor + v / raw_max * span, 2) for v in vals]
