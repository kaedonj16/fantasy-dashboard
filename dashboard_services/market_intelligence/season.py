"""Provider-independent, confidence-weighted season projection adjustments."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import median, pstdev

from .models import MarketProjectionInput
from .projection import STAT_KEYS, build_season_market_projection
from utils.fantasy_scoring import score_stats

MIN_ROLLING_WEEKS = 3
MAX_ROLLING_ADJUSTMENT = 0.10
MAX_TEAM_ADJUSTMENT = 0.03
MIN_INDEPENDENT_CONFIDENCE = 0.35
_RELEVANT_STATS = {"QB": 5, "RB": 4, "WR": 4, "TE": 4}
_TEAM_SENSITIVITY = {"QB": 1.0, "RB": 0.75, "WR": 0.85, "TE": 0.75}


def team_environment_input(player_id: str, position: str, environment,
                           observed_at: datetime | None = None) -> MarketProjectionInput | None:
    """Conservative context from a team's current implied scoring environment.

    The environment has already been centered against covered NFL teams. Even an
    extreme relative score is capped at a three-percent projection adjustment;
    this context can never dominate a player prop or the baseline projection.
    """
    if not isinstance(environment, dict):
        return None
    try:
        score = max(-1.0, min(1.0, float(environment["score"])))
        confidence = max(0.0, min(1.0, float(environment.get("confidence") or 0)))
        coverage = max(0.0, min(1.0, float(environment.get("coverage") or 0)))
    except (KeyError, TypeError, ValueError):
        return None
    pos = str(position or "").upper()
    sensitivity = _TEAM_SENSITIVITY.get(pos)
    if sensitivity is None:
        return None
    adjustment = max(-MAX_TEAM_ADJUSTMENT, min(MAX_TEAM_ADJUSTMENT,
                                               score * MAX_TEAM_ADJUSTMENT * sensitivity))
    if abs(adjustment) < 0.001:
        return None
    return MarketProjectionInput(str(player_id), "season", "team_environment", adjustment,
                                 str(environment.get("source") or "sportsgameodds"),
                                 "team_market", round(confidence, 3),
                                 observed_at or datetime.now(timezone.utc),
                                 {"score": round(score, 3),
                                  "adjustment_pct": round(adjustment, 4),
                                  "implied_team_points": environment.get("implied_points"),
                                  "league_average": environment.get("league_average"),
                                  "coverage": round(coverage, 3),
                                  "sources": ["team_implied_points"]})


def rolling_weekly_inputs(rows: list[dict], now: datetime | None = None,
                          min_weeks: int = MIN_ROLLING_WEEKS,
                          regular_season: bool = True) -> list[MarketProjectionInput]:
    """Turn multiple historical weekly consensuses into per-game rate evidence.

    This never annualizes a line. It emits a weighted *rate* only after distinct
    regular-season weeks meet the sample floor. Bye/inactive/partial/live records
    are excluded by callers or the optional flags accepted here.
    """
    if not regular_season:
        return []
    now = now or datetime.now(timezone.utc)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        try:
            week, line = int(row.get("week")), float(row.get("line"))
        except (TypeError, ValueError):
            continue
        period = str(row.get("period") or "game").lower()
        if (week < 1 or line <= 0 or period not in ("game", "reg", "full", "fullgame") or
                row.get("preseason") or row.get("inactive") or row.get("injury_limited") or
                row.get("partial_game") or row.get("live")):
            continue
        pid, stat = str(row.get("canonical_player_id") or ""), str(row.get("stat_type") or "")
        if pid and stat in STAT_KEYS:
            grouped[(pid, stat)].append({**row, "week": week, "line": line})

    out = []
    for (pid, stat), samples in grouped.items():
        # One consensus per week, newest value wins if a caller supplied duplicates.
        by_week = {sample["week"]: sample for sample in samples}
        ordered = [by_week[w] for w in sorted(by_week)]
        if len(ordered) < min_weeks:
            continue
        weights = [0.72 ** (len(ordered) - 1 - i) for i in range(len(ordered))]
        weighted = sum(r["line"] * w for r, w in zip(ordered, weights)) / sum(weights)
        center = median(r["line"] for r in ordered)
        variance = pstdev(r["line"] for r in ordered) / max(abs(center), 1.0)
        sample_score = min(1.0, len(ordered) / 5.0)
        agreement = max(0.0, 1.0 - variance * 2.0)
        source_conf = sum(float(r.get("confidence") or 0) for r in ordered) / len(ordered)
        confidence = min(0.72, (0.35 * sample_score + 0.35 * agreement + 0.30 * source_conf) * 0.85)
        out.append(MarketProjectionInput(
            pid, "season", stat, round(weighted, 3), "sportsgameodds",
            "rolling_weekly_market", round(confidence, 3), now,
            {"weeks": [r["week"] for r in ordered], "sample_size": len(ordered),
             "weighted_rate": round(weighted, 3), "relative_stddev": round(variance, 3)},
        ))
    return out


def build_adjusted_season_projection(baseline_points: float, position: str,
                                     scoring_settings: dict,
                                     inputs: list[MarketProjectionInput],
                                     games_played: int = 0, season_games: int = 17) -> dict:
    """Anchor at the site projection and blend independent evidence by quality.

    Direct season props own their covered stats. Rolling lines for those same stats
    are retained as provenance but receive no second adjustment. Team context is
    scaled to uncovered components. This prevents correlated evidence from being
    counted once per observation.
    """
    baseline = float(baseline_points or 0)
    pos = str(position or "").upper()
    direct = {i.stat_type: {"line": i.value, "confidence": i.confidence}
              for i in inputs if i.source_type == "season_prop" and i.stat_type in STAT_KEYS}
    rolling = [i for i in inputs if i.source_type == "rolling_weekly_market"]
    team = [i for i in inputs if i.source_type == "team_market"]
    prediction = [i for i in inputs if i.source_type == "prediction_market"]

    points = baseline
    adjustments = {"season_prop_points": 0.0, "rolling_market_points": 0.0,
                   "team_environment_points": 0.0, "prediction_market_points": 0.0}
    sources: dict = {}
    direct_coverage = 0.0
    confidences = []
    if direct and baseline > 0:
        projected = build_season_market_projection(direct, baseline, scoring_settings, pos)
        if projected:
            adjustments["season_prop_points"] = round(projected["points"] - baseline, 2)
            points = projected["points"]
            direct_coverage = projected["coverage"]
            confidence = projected["confidence"]
            confidences.append(confidence)
            sources["season_props"] = {"stats": sorted(direct), "coverage": direct_coverage,
                                       "confidence": confidence}

    # Rolling rates adjust the baseline rate conservatively; they are evidence of a
    # changed role, never presented as a season line or multiplied by games.
    usable_rolling = [i for i in rolling if i.stat_type not in direct]
    if usable_rolling and baseline > 0:
        partial_stats = {STAT_KEYS[i.stat_type]: i.value for i in usable_rolling}
        partial_ppg = score_stats(partial_stats, scoring_settings or {}, pos)
        coverage = min(1.0, len(partial_stats) / _RELEVANT_STATS.get(pos, 4))
        remaining_games = max(1, int(season_games) - max(0, int(games_played)))
        baseline_rate = baseline / float(season_games)
        inferred_ppg = partial_ppg / coverage if coverage else 0
        raw_pct = (inferred_ppg - baseline_rate) / max(baseline_rate, 1.0)
        confidence = sum(i.confidence for i in usable_rolling) / len(usable_rolling)
        pct = max(-MAX_ROLLING_ADJUSTMENT, min(MAX_ROLLING_ADJUSTMENT, raw_pct))
        adjusted_rate = baseline_rate * (1.0 + pct * confidence)
        # Only the remaining schedule is adjusted. This is role-rate evidence—not
        # a weekly line multiplied into a fabricated full-season sportsbook total.
        delta = (adjusted_rate - baseline_rate) * remaining_games * (1.0 - direct_coverage)
        adjustments["rolling_market_points"] = round(delta, 2)
        points += delta
        confidences.append(confidence)
        sources["rolling_weekly_market"] = {
            "stats": {i.stat_type: i.metadata for i in usable_rolling},
            "baseline_rate": round(baseline_rate, 3), "market_rate": round(inferred_ppg, 3),
            "adjusted_rate": round(adjusted_rate, 3), "games_played": int(games_played),
            "remaining_games": remaining_games, "confidence": round(confidence, 3)}

    if team and baseline > 0:
        strongest = max(team, key=lambda i: i.confidence)
        delta = baseline * max(-MAX_TEAM_ADJUSTMENT, min(MAX_TEAM_ADJUSTMENT, strongest.value))
        delta *= strongest.confidence * (1.0 - direct_coverage)
        adjustments["team_environment_points"] = round(delta, 2)
        points += delta
        confidences.append(strongest.confidence * 0.8)
        sources["team_environment"] = {**strongest.metadata, "confidence": strongest.confidence}

    # Structured threshold markets may supply a precomputed, explicitly contextual
    # adjustment_pct. Award/MVP prices alone never turn directly into fantasy points.
    valid_prediction = [i for i in prediction if "adjustment_pct" in i.metadata]
    if valid_prediction and baseline > 0:
        strongest = max(valid_prediction, key=lambda i: i.confidence)
        pct = max(-0.02, min(0.02, float(strongest.metadata["adjustment_pct"])))
        delta = baseline * pct * strongest.confidence * (1.0 - direct_coverage)
        adjustments["prediction_market_points"] = round(delta, 2)
        points += delta
        confidences.append(strongest.confidence * 0.75)
        sources["prediction_markets"] = {"confidence": strongest.confidence,
                                         "stat_type": strongest.stat_type}

    confidence = min(1.0, max(confidences, default=0.0) +
                     0.12 * max(0, len(confidences) - 1))
    if "season_props" in sources and len(sources) == 1:
        basis = "season_props"
    elif "rolling_weekly_market" in sources and len(sources) == 1:
        basis = "rolling_market"
    elif "team_environment" in sources and len(sources) == 1:
        basis = "team_environment"
    elif "prediction_markets" in sources and len(sources) == 1:
        basis = "prediction_market"
    elif sources:
        basis = "blended"
    else:
        basis = "projection_only"
    return {"points": round(points, 2), "coverage": round(direct_coverage, 3),
            "confidence": round(confidence, 3), "basis": basis,
            "meaningful": confidence >= MIN_INDEPENDENT_CONFIDENCE and basis != "projection_only",
            "components": {"baseline_points": baseline,
                           "market_adjusted_points": round(points, 2),
                           "sources": sources, "adjustments": adjustments,
                           "basis": basis, "confidence": round(confidence, 3)}}
