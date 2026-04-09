from __future__ import annotations

from typing import Any, Dict, List


def _latest_metric_by_season(metrics_by_season: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    ordered_seasons = sorted(metrics_by_season.keys())
    for season in ordered_seasons:
        for metric, payload in (metrics_by_season.get(season) or {}).items():
            latest[metric] = payload
    return latest


def _calc_completeness(metrics: Dict[str, Dict[str, Any]], expected_count: int) -> float:
    if expected_count <= 0:
        return 0.0
    return round(min(1.0, len(metrics) / expected_count), 4)


def build_rookie_profile(
    player: Dict[str, Any],
    metrics_by_season: Dict[int, Dict[str, Dict[str, Any]]],
    missing: Dict[str, Dict[str, Any]],
    draft_market: Dict[str, Any],
    expected_metric_count: int,
) -> Dict[str, Any]:
    season_span: List[int] = sorted(metrics_by_season.keys())
    latest_metrics = _latest_metric_by_season(metrics_by_season)

    profile = {
        "player_id": player.get("player_id"),
        "name": player.get("name"),
        "school": player.get("school"),
        "position": player.get("position"),
        "class_year": player.get("class_year") or player.get("experience"),
        "age": player.get("age"),
        "rookie_profile": {
            "season_span": season_span,
            "metrics": latest_metrics,
            "metrics_by_season": metrics_by_season,
            "draft_market": draft_market,
            "missing": missing,
            "completeness": _calc_completeness(latest_metrics, expected_metric_count),
            "confidence_summary": {
                "average_metric_confidence": round(
                    sum(float(v.get("confidence") or 0.0) for v in latest_metrics.values()) / max(len(latest_metrics), 1),
                    4,
                ),
                "available_metric_count": len(latest_metrics),
                "missing_metric_count": len(missing),
            },
        },
    }
    return profile
