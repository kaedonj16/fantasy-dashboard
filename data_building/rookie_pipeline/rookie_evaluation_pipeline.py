from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from data_building.rookie_pipeline.draft_market_sources import (
    build_draft_market_for_player,
    fetch_draft_market_entries,
)
from data_building.rookie_pipeline.rookie_db_storage import save_rookie_evaluation_to_db
from data_building.rookie_pipeline.ingestion import load_prospects_for_year
from data_building.rookie_pipeline.pipeline import load_prospects_from_db
from data_building.rookie_pipeline.rookie_identity import build_identity_index, reconcile_player_identity
from data_building.rookie_pipeline.rookie_profile_builder import build_rookie_profile
from data_building.rookie_pipeline.rookie_source_registry import build_rookie_source_registry
from data_building.rookie_pipeline.sportradar_ncaa import build_sportradar_ncaa_index
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


def _load_eval_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """
    Load prospects for rookie eval, preferring DB-backed staged prospect data.

    Why:
      - `load_prospects_for_year` depends on external APIs/seed files.
      - evaluation should still produce values when the prospects are already
        populated in Postgres via `scripts.populate_rookie_data` / pipeline jobs.
    """
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            db_prospects = load_prospects_from_db(draft_year, conn)
            if db_prospects:
                return db_prospects
    except Exception as exc:
        print(f"[rookie_eval] db_prospect_load_failed class={draft_year}: {exc}")

    return load_prospects_for_year(draft_year) or []


def _missing_payload(metric: RookieMetricSpec, reason: str) -> Dict[str, Any]:
    return {
        "value": None,
        "missing_reason": reason,
        "best_source_candidate": metric.best_source_candidate,
        "updated_at": utc_now_iso(),
    }


# Proxy metrics where a value of exactly 0 is meaningless (no data, not truly zero).
# The derivation functions now return None for these; cached 0s are stale artifacts.
_REJECT_ZERO_METRICS = frozenset({"routes_run", "yprr", "tprr"})


def _cache_value_valid(metric_name: str, value) -> bool:
    """Return False if the cached value should be treated as missing."""
    if value is None:
        return False
    if metric_name in _REJECT_ZERO_METRICS and value == 0:
        return False
    return True


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
        if payload and _cache_value_valid(metric.name, payload.get("value")):
            return payload, "cache_fresh"

    try:
        fetched = source.fetch_player_season_metrics(player, season_record, [metric])
        payload = fetched.get(metric.name)
        if payload and _cache_value_valid(metric.name, payload.get("value")):
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
        else:
            print(f"[rookie_eval] no_value     player={player_key} season={season} metric={metric.name} source={source.source_name} fetched={list(fetched.keys()) if fetched else '[]'}")
    except Exception as exc:
        print(f"[rookie_eval] source_error  player={player_key} season={season} metric={metric.name} source={source.source_name}: {type(exc).__name__}: {exc}")

    if cache_hit.payload:
        payload = cache_hit.payload.get("metric_payload")
        if payload and _cache_value_valid(metric.name, payload.get("value")):
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
    prospects = _load_eval_prospects(year)
    if player_limit:
        prospects = prospects[: max(0, int(player_limit))]

    metrics_specs = rookie_metric_specs()
    cache = RookieDiskCache()

    identity_index = build_identity_index(prospects)
    draft_entries = fetch_draft_market_entries(year)

    # Build Sportradar NCAAFB index for real target data (no-op if key absent)
    prospect_names = [p.get("name", "") for p in prospects if p.get("name")]
    sportradar_index = build_sportradar_ncaa_index(prospect_names)

    sources = build_rookie_source_registry(sportradar_index=sportradar_index)

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
            raw_fields = {k: v for k, v in season_record.items() if v is not None}
            print(f"[source_raw] player={pid} season={season} pos={player.get('position')} raw_fields={raw_fields}")
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
    db_result = {"db_metrics_rows": 0, "db_profiles_rows": 0, "db_runs_rows": 0}
    try:
        db_result = save_rookie_evaluation_to_db(
            as_of_date=as_of,
            draft_class_year=year,
            by_player_metrics=by_player_metrics,
            rookie_profiles=rookie_profiles,
            run_metadata={
                "log_summary": dict(logs),
                "missing_breakdown": {k: dict(v) for k, v in missing_metric_reasons.items()},
                "generated_at": utc_now_iso(),
            },
        )
    except Exception as exc:
        print(f"[rookie_eval] database_save_failed class={year}: {exc}")

    print(
        "[rookie_eval] complete "
        f"class={year} prospects={len(rookie_profiles)} "
        f"direct={logs.get('metrics_direct', 0)} derived={logs.get('metrics_derived', 0)} "
        f"unavailable={logs.get('metrics_unavailable', 0)} "
        f"db_profiles={db_result.get('db_profiles_rows', 0)}"
    )

    return {
        "draft_class_year": year,
        "metrics_file": str(metrics_file),
        "profiles_file": str(profiles_file),
        "log_summary": dict(logs),
        "profile_count": len(rookie_profiles),
        "profiles": rookie_profiles,
        **db_result,
    }
