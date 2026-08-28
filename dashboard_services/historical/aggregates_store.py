"""Request-path loader for historical_profile_aggregates.json.

json + pathlib only. Flask / pandas / parquet stay out. The in-memory cache
keys on file mtime so a cron rebuild is visible on the next request.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from utils.paths import PLAYER_HISTORY_DIR

PROFILE_PATH = PLAYER_HISTORY_DIR / "historical_profile_aggregates.json"
CAREER_PATH_OVERLAY_PATH = PLAYER_HISTORY_DIR / "career_path_overlay.json"
USAGE_VOLUME_OVERLAY_PATH = PLAYER_HISTORY_DIR / "usage_volume_overlay.json"
COHORT_INDEX_PATH = PLAYER_HISTORY_DIR / "cohort_index.json"

from dashboard_services.historical.usage import VOLUME_USAGE_IDS


_CACHE: dict[str, Any] = {"mtime": None, "data": None}


def aggregates_version() -> Any:
    """mtime token for the loaded JSON (+ overlays). None if nothing is cached."""
    return _CACHE.get("mtime")


def _file_mtime(path: Any) -> Any:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _nflverse_mtime_token() -> Any:
    latest = None
    for path in PLAYER_HISTORY_DIR.glob("nflverse_metrics_*.json"):
        mt = _file_mtime(path)
        if mt is not None and (latest is None or mt > latest):
            latest = mt
    return latest


def _load_nflverse_metrics(season: int) -> dict:
    path = PLAYER_HISTORY_DIR / f"nflverse_metrics_{season}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _merge_nflverse_preseason_usage(data: dict) -> dict:
    """Stamp last-year aDOT / RYOE from committed NGS JSON onto live profiles.

    Warehouse preseason rows omit those fields, so Trends filters for them
    would match nobody even though the rate tables exist.
    """
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    years: set[int] = set()
    for rec in by_player.values():
        if not isinstance(rec, dict):
            continue
        try:
            years.add(int(rec.get("previous_season_year")))
        except (TypeError, ValueError):
            continue
    by_year = {year: _load_nflverse_metrics(year) for year in years}
    for pid, rec in by_player.items():
        if not isinstance(rec, dict):
            continue
        try:
            year = int(rec.get("previous_season_year"))
        except (TypeError, ValueError):
            continue
        ngs = by_year.get(year, {}).get(str(pid))
        if not isinstance(ngs, dict):
            continue
        if rec.get("previous_season_adot") is None:
            adot = ngs.get("adot")
            if adot is None:
                adot = ngs.get("avg_depth_of_target")
            if adot is not None:
                rec["previous_season_adot"] = adot
        if rec.get("previous_season_ngs_rush_yards_over_expected_per_att") is None:
            ryoe = ngs.get("ngs_rush_yards_over_expected_per_att")
            if ryoe is not None:
                rec["previous_season_ngs_rush_yards_over_expected_per_att"] = ryoe
    return data


def _merge_career_path_overlay(data: dict, overlay: Mapping[str, Any]) -> dict:
    """Stamp prior top-12 counts and bounce-back rates onto loaded aggregates."""
    if not overlay:
        return data
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    counts = overlay.get("prior_top12_count") if isinstance(overlay.get("prior_top12_count"), dict) else {}
    for pid, count in counts.items():
        rec = by_player.get(str(pid))
        if isinstance(rec, dict) and count is not None:
            rec["prior_top12_count"] = count
    repeat = data.get("repeat_and_breakout") if isinstance(data.get("repeat_and_breakout"), dict) else {}
    bounce = overlay.get("bounce_back") if isinstance(overlay.get("bounce_back"), dict) else {}
    for pos, block in bounce.items():
        if not isinstance(block, dict):
            continue
        dest = repeat.get(pos)
        if isinstance(dest, dict):
            dest.update(block)
    return data


def _merge_usage_volume_overlay(data: dict, overlay: Mapping[str, Any]) -> dict:
    """Stamp last-year volume rates and live totals. Does not replace share/snap/aDOT."""
    if not overlay:
        return data
    usage = data.get("prior_usage") if isinstance(data.get("prior_usage"), dict) else {}
    extra_usage = overlay.get("prior_usage") if isinstance(overlay.get("prior_usage"), dict) else {}
    data["prior_usage"] = usage
    for key, block in extra_usage.items():
        if key in VOLUME_USAGE_IDS and isinstance(block, dict):
            usage[key] = block
    dest_tiers = data.get("prior_usage_by_tier")
    if not isinstance(dest_tiers, dict):
        dest_tiers = {}
        data["prior_usage_by_tier"] = dest_tiers
    extra_tiers = overlay.get("prior_usage_by_tier")
    if isinstance(extra_tiers, dict):
        for tier, metrics in extra_tiers.items():
            if not isinstance(metrics, dict):
                continue
            dest = dest_tiers.setdefault(str(tier), {})
            if not isinstance(dest, dict):
                dest = {}
                dest_tiers[str(tier)] = dest
            for key, block in metrics.items():
                if key in VOLUME_USAGE_IDS and isinstance(block, dict):
                    dest[key] = block
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    extras = overlay.get("preseason_volume") if isinstance(overlay.get("preseason_volume"), dict) else {}
    for pid, extra in extras.items():
        rec = by_player.get(str(pid))
        if isinstance(rec, dict) and isinstance(extra, dict):
            rec.update(extra)
    return data


def _merge_cohort_index(data: dict, overlay: Mapping[str, Any]) -> dict:
    """Attach the compact observation index when the main JSON lacks one.

    Cron rebuilds stamp ``cohort_index`` onto ``historical_profile_aggregates.json``.
    Until that rebuild, a sibling ``cohort_index.json`` overlay is enough for
    ``POST /api/historical-cohort`` without a request-time parquet scan.

    ``preseason_trajectory`` is stamped onto live preseason profiles so Scout
    uses the same YoY buckets as historical matching.
    """
    if not overlay:
        return data
    existing = data.get("cohort_index") if isinstance(data.get("cohort_index"), dict) else {}
    if overlay.get("observations") and not existing.get("observations"):
        data["cohort_index"] = {
            key: value
            for key, value in overlay.items()
            if key != "preseason_trajectory"
        }
    extras = overlay.get("preseason_trajectory")
    if isinstance(extras, dict) and extras:
        pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
        by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
        for pid, extra in extras.items():
            rec = by_player.get(str(pid))
            if not isinstance(rec, dict) or not isinstance(extra, dict):
                continue
            for key, value in extra.items():
                if value and rec.get(key) is None:
                    rec[key] = value
    return data


def load_profile_aggregates(*, path: Optional[Any] = None) -> dict:
    """Return the precomputed JSON, or ``{}`` when the file is missing."""
    target = path if path is not None else PROFILE_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {}
    overlay_mtime = None
    volume_mtime = None
    nflverse_mtime = None
    cohort_mtime = None
    if path is None:
        overlay_mtime = _file_mtime(CAREER_PATH_OVERLAY_PATH)
        volume_mtime = _file_mtime(USAGE_VOLUME_OVERLAY_PATH)
        nflverse_mtime = _nflverse_mtime_token()
        cohort_mtime = _file_mtime(COHORT_INDEX_PATH)
    cache_key = (mtime, overlay_mtime, volume_mtime, nflverse_mtime, cohort_mtime)
    cached = _CACHE.get("data")
    if cached is not None and _CACHE.get("mtime") == cache_key and path is None:
        return cached
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if path is None:
        try:
            overlay = json.loads(CAREER_PATH_OVERLAY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            overlay = {}
        if isinstance(overlay, dict) and overlay:
            _merge_career_path_overlay(data, overlay)
        try:
            volume = json.loads(USAGE_VOLUME_OVERLAY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            volume = {}
        if isinstance(volume, dict) and volume:
            _merge_usage_volume_overlay(data, volume)
        try:
            cohort = json.loads(COHORT_INDEX_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cohort = {}
        if isinstance(cohort, dict) and cohort:
            _merge_cohort_index(data, cohort)
        _merge_nflverse_preseason_usage(data)
        _CACHE["mtime"] = cache_key
        _CACHE["data"] = data
    return data


def stamp_historical_on_payload(payload: Mapping[str, Any]) -> dict:
    """Attach compact historical fields. Never raises into the request path."""
    out = dict(payload or {})
    try:
        from dashboard_services.historical.board import attach_historical_signals

        aggs = load_profile_aggregates()
        if not aggs:
            out["historical_available"] = False
            return out
        attach_historical_signals(out.get("players") or [], aggs)
        out["historical_available"] = True
        out["historical_phase"] = aggs.get("phase")
        out["historical_descriptive_only"] = True
    except Exception:
        out["historical_available"] = False
    return out
