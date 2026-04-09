from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from data_building.rookie_pipeline.draft_market_sources import (
    build_draft_market_for_player,
    fetch_draft_market_entries,
)
from data_building.rookie_pipeline.ingestion import load_prospects_for_year
from data_building.rookie_pipeline.rookie_identity import build_identity_index, reconcile_player_identity
from data_building.rookie_pipeline.rookie_profile_builder import build_rookie_profile
from data_building.rookie_pipeline.rookie_source_registry import build_rookie_source_registry
from data_building.rookie_pipeline.rookie_sources import RookieMetricSpec, rookie_metric_specs
from data_building.rookie_pipeline.rookie_storage import RookieDiskCache, utc_now_iso, write_rookie_snapshot


def _iter_player_seasons(player: Dict[str, Any], fallback_season: int):
    seasons = player.get("seasons") or []
    if not seasons:
        yield fallback_season, {"season": fallback_season}
        return
    for season_record in seasons:
        season = int(season_record.get("season") or fallback_season)
        yield season, season_record


def _missing_payload(metric: RookieMetricSpec, reason: str) -> Dict[str, Any]:
    return {
        "value": None,
        "missing_reason": reason,
        "best_source_candidate": metric.best_source_candidate,
        "updated_at": utc_now_iso(),
    }


def _source_for_metric(
    metric: RookieMetricSpec,
    player: Dict[str, Any],
    season: int,
    season_record: Dict[str, Any],
    cache: RookieDiskCache,
    source,
) -> Tuple[Optional[Dict[str, Any]], str]:
    player_key = player.get("player_id") or player.get("name") or "unknown"
    cache_hit = cache.read(source.source_name, season, f"{player_key}_{metric.name}", source.source_type)

    if cache_hit.payload and not cache_hit.is_stale:
        payload = cache_hit.payload.get("metric_payload")
        if payload and payload.get("value") is not None:
            return payload, "cache_fresh"

    try:
        fetched = source.fetch_player_season_metrics(player, season_record, [metric])
        payload = fetched.get(metric.name)
        if payload and payload.get("value") is not None:
            cache.write(
                source.source_name,
                season,
                f"{player_key}_{metric.name}",
                {
                    "metric": metric.name,
                    "metric_payload": payload,
                    "player_id": player.get("player_id"),
                    "season": season,
                },
            )
            return payload, "fetched_live"
    except Exception as exc:
        print(f"[rookie_eval] source_error metric={metric.name} source={source.source_name} player={player_key}: {exc}")

    if cache_hit.payload:
        payload = cache_hit.payload.get("metric_payload")
        if payload and payload.get("value") is not None:
            return payload, "cache_stale_fallback"

    return None, "missing"


def run_rookie_evaluation_pipeline(
    draft_year: Optional[int] = None,
    as_of_date: Optional[str] = None,
    player_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build rookie evaluation metrics + consolidated rookie profiles.

    Writes:
      - data/rookie_advanced_metrics_{date}.json
      - data/rookie_profiles_{date}.json
    """
    year = int(draft_year or date.today().year)
    as_of = as_of_date or date.today().isoformat()
    prospects = load_prospects_for_year(year) or []
    if player_limit:
        prospects = prospects[: max(0, int(player_limit))]

    metrics_specs = rookie_metric_specs()
    sources = build_rookie_source_registry()
    cache = RookieDiskCache()

    identity_index = build_identity_index(prospects)
    draft_entries = fetch_draft_market_entries(year)

    by_player_metrics: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    rookie_profiles: List[Dict[str, Any]] = []
    logs = Counter()
    missing_metric_reasons: Dict[str, Counter] = defaultdict(Counter)

    for player in prospects:
        reconciled = reconcile_player_identity(player, identity_index)
        if not reconciled:
            logs["identity_not_found"] += 1
            continue
        if reconciled.ambiguous:
            logs["identity_ambiguous"] += 1
            print(
                f"[rookie_eval] identity_ambiguous player={player.get('name')} candidates={reconciled.candidates} - skipping merge"
            )
            continue

        pid = reconciled.player_id
        metrics_by_season: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        missing: Dict[str, Dict[str, Any]] = {}

        for season, season_record in _iter_player_seasons(player, year):
            for metric in metrics_specs:
                resolved_payload = None
                resolution_mode = "missing"
                for source in sources:
                    payload, mode = _source_for_metric(metric, player, season, season_record, cache, source)
                    if payload:
                        resolved_payload = payload
                        resolution_mode = mode
                        break

                if resolved_payload is not None:
                    metrics_by_season[season][metric.name] = resolved_payload
                    src_type = resolved_payload.get("source_type")
                    if src_type == "derived":
                        logs["metrics_derived"] += 1
                    else:
                        logs["metrics_direct"] += 1
                    if resolution_mode.startswith("cache"):
                        logs[resolution_mode] += 1
                    continue

                reason = "no_reliable_free_source"
                if metric.name == "injury_flags":
                    reason = "public_structured_college_injury_feed_not_connected"
                missing[metric.name] = _missing_payload(metric, reason)
                missing_metric_reasons[metric.name][reason] += 1
                logs["metrics_unavailable"] += 1

        draft_market = build_draft_market_for_player(player.get("name", ""), year, draft_entries)
        profile = build_rookie_profile(
            player,
            dict(metrics_by_season),
            missing,
            draft_market,
            expected_metric_count=len(metrics_specs),
        )
        rookie_profiles.append(profile)
        by_player_metrics[pid] = dict(metrics_by_season)

    metrics_snapshot = {
        "as_of_date": as_of,
        "draft_class_year": year,
        "generated_at": utc_now_iso(),
        "metrics": by_player_metrics,
        "log_summary": dict(logs),
        "missing_breakdown": {k: dict(v) for k, v in missing_metric_reasons.items()},
    }
    profiles_snapshot = {
        "as_of_date": as_of,
        "draft_class_year": year,
        "generated_at": utc_now_iso(),
        "profiles": rookie_profiles,
        "count": len(rookie_profiles),
    }

    metrics_file, _ = write_rookie_snapshot("rookie_advanced_metrics", as_of, metrics_snapshot)
    profiles_file, _ = write_rookie_snapshot("rookie_profiles", as_of, profiles_snapshot)

    print(
        "[rookie_eval] complete "
        f"class={year} prospects={len(rookie_profiles)} "
        f"direct={logs.get('metrics_direct', 0)} derived={logs.get('metrics_derived', 0)} "
        f"unavailable={logs.get('metrics_unavailable', 0)}"
    )

    return {
        "draft_class_year": year,
        "metrics_file": str(metrics_file),
        "profiles_file": str(profiles_file),
        "log_summary": dict(logs),
        "profile_count": len(rookie_profiles),
    }
