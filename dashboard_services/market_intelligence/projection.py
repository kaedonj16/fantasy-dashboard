from __future__ import annotations

from utils.fantasy_scoring import score_stats

STAT_KEYS = {
    "passing_yards": "pass_yd", "passing_touchdowns": "pass_td",
    "interceptions": "pass_int", "rushing_yards": "rush_yd",
    "rushing_touchdowns": "rush_td", "receptions": "rec",
    "receiving_yards": "rec_yd", "receiving_touchdowns": "rec_td",
    "touchdowns": "rush_td",
}


def build_market_projection(consensus: dict, baseline_stats: dict,
                            scoring_settings: dict, position: str) -> dict | None:
    """Replace only covered baseline components, never convert missing props to zero."""
    if not consensus:
        return None
    stats = dict(baseline_stats or {})
    sources = {key: "baseline" for key in stats}
    confidences = []
    for stat_type, market in consensus.items():
        key = STAT_KEYS.get(stat_type)
        if not key or market.get("line") is None:
            continue
        stats[key] = float(market["line"])
        sources[key] = "sportsgameodds"
        confidences.append(float(market.get("confidence") or 0))
    if not confidences:
        return None
    market_keys = sum(v == "sportsgameodds" for v in sources.values())
    relevant = 5 if position.upper() == "QB" else 4
    coverage = min(1.0, market_keys / relevant)
    confidence = (sum(confidences) / len(confidences)) * (0.5 + 0.5 * coverage)
    baseline_pts = score_stats(baseline_stats or {}, scoring_settings or {}, position)
    raw_pts = score_stats(stats, scoring_settings or {}, position)
    # Shrink sparse or uncertain market differences toward the existing model.
    points = baseline_pts + (raw_pts - baseline_pts) * confidence
    return {"points": round(points, 2), "coverage": round(coverage, 3),
            "confidence": round(confidence, 3), "components": sources,
            "stats": stats, "baseline_stats": dict(baseline_stats or {})}


def build_season_market_projection(consensus: dict, baseline_points: float,
                                   scoring_settings: dict, position: str) -> dict | None:
    """Blend season-long components with the site's season projection.

    Sleeper provides the season point baseline. Covered market components are
    scored directly; the baseline supplies
    the uncovered share. Confidence then shrinks the result toward the baseline.
    Weekly props must never be passed to this function.
    """
    market_stats = {}
    sources = {}
    confidences = []
    for stat_type, market in (consensus or {}).items():
        key = STAT_KEYS.get(stat_type)
        if not key or market.get("line") is None:
            continue
        market_stats[key] = float(market["line"])
        sources[key] = "sportsgameodds"
        confidences.append(float(market.get("confidence") or 0))
    if not confidences or baseline_points <= 0:
        return None
    relevant = 5 if position.upper() == "QB" else 4
    coverage = min(1.0, len(market_stats) / relevant)
    confidence = (sum(confidences) / len(confidences)) * (0.5 + 0.5 * coverage)
    partial_points = score_stats(market_stats, scoring_settings or {}, position)
    hybrid_points = partial_points + float(baseline_points) * (1.0 - coverage)
    points = float(baseline_points) + (hybrid_points - float(baseline_points)) * confidence
    return {"points": round(points, 2), "coverage": round(coverage, 3),
            "confidence": round(confidence, 3), "components": sources,
            "stats": market_stats, "baseline_points": float(baseline_points)}
