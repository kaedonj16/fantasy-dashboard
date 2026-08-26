"""Stable, position-aware Market vs ADP calculations."""
from __future__ import annotations

import bisect
import math
from datetime import datetime, timezone
from statistics import median

from .config import MIN_SIGNAL_CONFIDENCE, SEASON_MAX_AGE

SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
MIN_CURVE_SAMPLES = 4
TARGET_BINS = 8
DEFAULT_SEASON_GAMES = 17
BASIS_CAPS = {
    "team_environment": 12.0,
    "rolling_market": 25.0,
    "prediction_market": 25.0,
    "season_props": 40.0,
    "blended": 30.0,
}


def _normalize_scoring_type(scoring_type: str | None) -> str:
    s = str(scoring_type or "redraft").strip().lower()
    if s in ("startup", "dynasty"):
        return "dynasty"
    if s == "rookie":
        return "rookie"
    return "redraft"


def _adp_keys(is_superflex: bool = False, scoring_type: str = "redraft") -> tuple[str, ...]:
    scoring = _normalize_scoring_type(scoring_type)
    if scoring == "dynasty":
        if is_superflex:
            return ("sf_avg_pick", "avg_pick", "sf_redraft_avg_pick", "redraft_avg_pick")
        return ("avg_pick", "redraft_avg_pick", "redraft_adp", "adp")
    if scoring == "rookie":
        if is_superflex:
            return ("sf_rookie_avg_pick", "rookie_avg_pick", "sf_avg_pick", "avg_pick")
        return ("rookie_avg_pick", "avg_pick", "redraft_avg_pick")
    if is_superflex:
        return ("sf_redraft_avg_pick", "redraft_avg_pick", "redraft_adp", "adp")
    return ("redraft_avg_pick", "redraft_adp", "adp", "sf_redraft_avg_pick")


def _resolve_adp(player: dict, is_superflex: bool = False,
                 scoring_type: str = "redraft") -> float | None:
    for key in _adp_keys(is_superflex, scoring_type):
        try:
            adp = float(player.get(key))
        except (TypeError, ValueError):
            continue
        if adp > 0:
            return adp
    return None


def _position(player: dict) -> str:
    return str(player.get("position") or player.get("pos") or "").upper()


def _projected_ppg(player: dict) -> float:
    try:
        return float(player.get("proj_ppg") or player.get("projected_ppg") or 0)
    except (TypeError, ValueError):
        return 0.0


def _pava_decreasing(values: list[float], weights: list[int]) -> list[float]:
    """Pool-adjacent-violators fit constrained to non-increasing values."""
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append({"start": index, "end": index, "sum": value * weight, "weight": weight})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["sum"] / left["weight"] >= right["sum"] / right["weight"]:
                break
            blocks[-2:] = [{"start": left["start"], "end": right["end"],
                            "sum": left["sum"] + right["sum"],
                            "weight": left["weight"] + right["weight"]}]
    fitted = [0.0] * len(values)
    for block in blocks:
        value = block["sum"] / block["weight"]
        for index in range(block["start"], block["end"] + 1):
            fitted[index] = value
    return fitted


def build_adp_curve(player_pool: list[dict], position: str,
                    is_superflex: bool = False,
                    scoring_type: str = "redraft") -> tuple[list[float], list[float]]:
    """Build a deterministic binned-median, monotonic curve for one position."""
    pos = str(position or "").upper()
    if pos not in SUPPORTED_POSITIONS:
        return ([], [])
    samples = []
    for player in player_pool:
        if _position(player) != pos:
            continue
        ppg = _projected_ppg(player)
        adp = _resolve_adp(player, is_superflex, scoring_type)
        if ppg > 0 and adp is not None:
            samples.append((ppg, adp))
    samples.sort()
    if len(samples) < MIN_CURVE_SAMPLES:
        return ([], [])

    # Quantile-like contiguous bins remove adjacent-player noise while retaining
    # more resolution for deep positions with larger samples.
    bin_count = min(TARGET_BINS, max(2, len(samples) // 3))
    bin_size = math.ceil(len(samples) / bin_count)
    bins = [samples[index:index + bin_size] for index in range(0, len(samples), bin_size)]
    points = [(samples[0][0], samples[0][1], 1)]
    points.extend((float(median(row[0] for row in group)),
                   float(median(row[1] for row in group)), len(group)) for group in bins if group)
    points.append((samples[-1][0], samples[-1][1], 1))
    xs = [point[0] for point in points]
    ys = _pava_decreasing([point[1] for point in points], [point[2] for point in points])
    return xs, ys


def build_position_curves(player_pool: list[dict], is_superflex: bool = False,
                          scoring_type: str = "redraft"):
    return {position: build_adp_curve(player_pool, position, is_superflex, scoring_type)
            for position in SUPPORTED_POSITIONS}


def interp_adp(curve: tuple[list[float], list[float]], projected_ppg: float) -> float | None:
    """Interpolate within a monotonic curve, clamping outside its observed range."""
    xs, ys = curve
    if len(xs) < 2:
        return None
    target = float(projected_ppg)
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]
    index = bisect.bisect_left(xs, target)
    x0, x1, y0, y1 = xs[index - 1], xs[index], ys[index - 1], ys[index]
    if x1 == x0:
        return (y0 + y1) / 2.0
    return y0 + (target - x0) / (x1 - x0) * (y1 - y0)


def expected_adp(projected_ppg: float, player_pool: list[dict], position: str,
                 is_superflex: bool = False, scoring_type: str = "redraft") -> float | None:
    return interp_adp(build_adp_curve(player_pool, position, is_superflex, scoring_type),
                      projected_ppg)


def _confidence_weight(confidence: float) -> float:
    # Projection points are already confidence-shrunk. This lighter second-stage
    # shrink distinguishes threshold-level context from strong direct evidence
    # without applying confidence twice at full strength.
    return 0.5 + 0.5 * max(0.0, min(1.0, confidence))


def attach_market_vs_adp(players: list[dict], projections: dict[str, dict],
                         is_superflex: bool = False,
                         scoring_type: str = "redraft") -> dict:
    """Attach incremental market-driven ADP movement, never absolute mispricing."""
    curves = build_position_curves(players, is_superflex, scoring_type)
    diagnostics = {key: 0 for key in (
        "qualified", "capped", "projection_only", "low_confidence", "missing_adp",
        "missing_fantasy_points", "missing_projection", "invalid_curve", "stale",
        "unsupported_position", "missing_baseline",
    )}
    diagnostics["curves"] = {position: {
        "samples": sum(1 for player in players if _position(player) == position and
                       _resolve_adp(player, is_superflex, scoring_type) is not None and
                       _projected_ppg(player) > 0),
        "bins": len(curves[position][0]),
    } for position in SUPPORTED_POSITIONS}
    diagnostics["max_positive_edge"] = 0.0
    diagnostics["max_negative_edge"] = 0.0
    diagnostics["examples"] = []

    for player in players:
        market = projections.get(str(player.get("id")))
        if not market:
            diagnostics["missing_projection"] += 1
            continue
        components = market.get("components") or {}
        basis = components.get("basis") or "season_props"
        try:
            confidence = float(market.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        player["market_vs_adp"] = None
        player["market_expected_adp"] = None
        player["market_confidence"] = round(confidence, 2)
        player["market_confidence_label"] = ("High" if confidence >= 0.7 else
                                             "Moderate" if confidence >= 0.5 else
                                             "Low" if confidence > 0 else "Unavailable")
        player["market_basis"] = basis
        if basis == "projection_only":
            diagnostics["projection_only"] += 1
            continue
        if confidence < MIN_SIGNAL_CONFIDENCE:
            diagnostics["low_confidence"] += 1
            continue
        calculated_at = market.get("calculated_at")
        if calculated_at:
            if not isinstance(calculated_at, datetime):
                try:
                    calculated_at = datetime.fromisoformat(str(calculated_at).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    calculated_at = None
            if calculated_at:
                calculated_at = calculated_at if calculated_at.tzinfo else calculated_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - calculated_at > SEASON_MAX_AGE:
                    diagnostics["stale"] += 1
                    continue
        position = _position(player)
        curve = curves.get(position)
        if position not in SUPPORTED_POSITIONS:
            diagnostics["unsupported_position"] += 1
            continue
        if not curve or len(curve[0]) < 2:
            diagnostics["invalid_curve"] += 1
            continue
        actual = _resolve_adp(player, is_superflex, scoring_type)
        if actual is None:
            diagnostics["missing_adp"] += 1
            continue
        try:
            season_points = float(market["fantasy_points"])
        except (KeyError, TypeError, ValueError):
            diagnostics["missing_fantasy_points"] += 1
            continue
        try:
            games = int(components.get("season_games") or DEFAULT_SEASON_GAMES)
        except (TypeError, ValueError):
            diagnostics["missing_fantasy_points"] += 1
            continue
        if games <= 0:
            diagnostics["missing_fantasy_points"] += 1
            continue
        try:
            baseline_points = float(components["baseline_points"])
        except (KeyError, TypeError, ValueError):
            baseline_ppg = _projected_ppg(player)
            if baseline_ppg <= 0:
                diagnostics["missing_baseline"] += 1
                continue
        else:
            baseline_ppg = baseline_points / games
        market_ppg = season_points / games
        baseline_expected = interp_adp(curve, baseline_ppg)
        market_expected = interp_adp(curve, market_ppg)
        if baseline_expected is None or market_expected is None:
            diagnostics["invalid_curve"] += 1
            continue

        raw_delta = baseline_expected - market_expected
        shrunk_delta = raw_delta * _confidence_weight(confidence)
        cap = BASIS_CAPS.get(basis, BASIS_CAPS["blended"])
        final_delta = max(-cap, min(cap, shrunk_delta))
        if final_delta != shrunk_delta:
            diagnostics["capped"] += 1
        player["market_baseline_expected_adp"] = round(baseline_expected, 1)
        player["market_curve_expected_adp"] = round(market_expected, 1)
        player["market_expected_adp"] = round(actual - final_delta, 1)
        player["market_vs_adp"] = round(final_delta, 1)
        player["market_signal"] = ("bullish" if final_delta > 1 else
                                   "bearish" if final_delta < -1 else "aligned")
        diagnostics["qualified"] += 1
        diagnostics["max_positive_edge"] = max(diagnostics["max_positive_edge"], final_delta)
        diagnostics["max_negative_edge"] = min(diagnostics["max_negative_edge"], final_delta)
        if len(diagnostics["examples"]) < 5:
            diagnostics["examples"].append({
                "player": player.get("name") or player.get("id"), "position": position,
                "actual_adp": round(actual, 1), "baseline_ppg": round(baseline_ppg, 2),
                "market_ppg": round(market_ppg, 2),
                "baseline_expected_adp": round(baseline_expected, 1),
                "market_expected_adp": round(market_expected, 1),
                "raw_market_delta": round(raw_delta, 1), "final_market_delta": round(final_delta, 1),
                "confidence": round(confidence, 2), "basis": basis,
            })
    diagnostics["max_positive_edge"] = round(diagnostics["max_positive_edge"], 1)
    diagnostics["max_negative_edge"] = round(diagnostics["max_negative_edge"], 1)
    return diagnostics
