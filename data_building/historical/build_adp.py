"""I/O: historical ADP snapshot backfill and join onto warehouse rows.

Reuses ``adp_service.write_adp_snapshot`` / ``fetch_sleeper_adp`` and
``providers.global_adp`` fetchers. Frozen copies are committed under
``cache/player_history/adp/`` so request-path profile rebuilds never need
live NFL APIs or the gitignored ``data/adp_snapshots/`` disk.

Yahoo has no season axis on the public global endpoint — it is not backfilled.
ESPN boards that fail the preseason quality gate (the "170 wall") are stored
but not used. Superflex / TEP historical ADP is not claimed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from utils.paths import PLAYER_HISTORY_DIR

from dashboard_services.historical.adp import attach_adp_features
from dashboard_services.historical.definitions import (
    ADP_SOURCE_PREFERENCE,
    RELIABLE_SEASON_FLOOR,
    normalize_adp,
    source_map_is_usable,
)

HISTORICAL_ADP_DIR = PLAYER_HISTORY_DIR / "adp"
COVERAGE_PATH = PLAYER_HISTORY_DIR / "historical_adp_coverage.json"
DEFAULT_SEASONS = tuple(range(2018, 2026))
AXIS = "redraft"


def _snapshot_path(source: str, season: int) -> Path:
    return HISTORICAL_ADP_DIR / f"{source}_{AXIS}_{int(season)}.json"


def load_committed_snapshot(source: str, season: int) -> dict:
    path = _snapshot_path(source, season)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def snapshot_adp_map(source: str, season: int) -> dict[str, float]:
    raw = (load_committed_snapshot(source, season).get("adp") or {})
    out: dict[str, float] = {}
    for pid, val in raw.items():
        adp = normalize_adp(val)
        if adp is not None:
            out[str(pid)] = adp
    return out


def persist_committed_snapshot(
    source: str,
    season: int,
    payload: dict,
    *,
    usable: Optional[bool] = None,
) -> Path:
    HISTORICAL_ADP_DIR.mkdir(parents=True, exist_ok=True)
    adp = dict((payload or {}).get("adp") or {})
    if usable is None:
        usable = source_map_is_usable(adp)
    record = {
        "source": source,
        "axis": AXIS,
        "season": int(season),
        "frozen": True,
        "usable": bool(usable),
        "collected_at": (payload or {}).get("collected_at") or time.time(),
        "adp": adp,
        "extra": (payload or {}).get("extra") or {},
        "meta": (payload or {}).get("meta") or {},
        "raw_count": (payload or {}).get("raw_count"),
        "mapped_count": (payload or {}).get("mapped_count") or len(adp),
        "ppr_rank": (payload or {}).get("ppr_rank") or {},
        "fields": (payload or {}).get("fields") or {},
    }
    path = _snapshot_path(source, season)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def _sleeper_ppr_map(rows: dict) -> tuple[dict[str, float], dict[str, dict]]:
    adp: dict[str, float] = {}
    fields: dict[str, dict] = {}
    for pid, row in (rows or {}).items():
        if not isinstance(row, dict):
            continue
        keep = {}
        for key, val in row.items():
            if not str(key).startswith("adp_"):
                continue
            parsed = normalize_adp(val)
            if parsed is not None:
                keep[str(key)] = parsed
        if not keep:
            continue
        fields[str(pid)] = keep
        ppr = normalize_adp(keep.get("adp_ppr"))
        if ppr is not None:
            adp[str(pid)] = ppr
    return adp, fields


def _skip_source(source: str, season: int, skip_existing: bool) -> Optional[dict]:
    if not skip_existing:
        return None
    snap = load_committed_snapshot(source, season)
    if not snap:
        return None
    return {
        "ok": True,
        "skipped": True,
        "n": len(snap.get("adp") or {}),
        "usable": bool(snap.get("usable")),
    }


def fetch_and_commit_season(season: int, *, skip_existing: bool = True) -> dict:
    """Live-fetch one season into committed frozen snapshots. Isolated per source."""
    summary: dict[str, Any] = {"season": int(season), "sources": {}}

    skipped = _skip_source("sleeper", season, skip_existing)
    if skipped:
        summary["sources"]["sleeper"] = skipped
    else:
        try:
            from dashboard_services.adp_service import fetch_sleeper_adp
            raw = fetch_sleeper_adp(int(season)) or {}
            adp, fields = _sleeper_ppr_map(raw)
            persist_committed_snapshot(
                "sleeper",
                season,
                {"adp": adp, "fields": fields, "mapped_count": len(adp), "raw_count": len(raw),
                 "meta": {"ppr": 1.0, "qb_format": "1qb", "draft_type": "redraft"}},
            )
            summary["sources"]["sleeper"] = {"ok": True, "n": len(adp), "usable": source_map_is_usable(adp)}
        except Exception as exc:
            summary["sources"]["sleeper"] = {"ok": False, "error": str(exc)[:200]}

    skipped = _skip_source("mfl", season, skip_existing)
    if skipped:
        summary["sources"]["mfl"] = skipped
    else:
        try:
            from dashboard_services.providers.global_adp import fetch_mfl_adp
            from dashboard_services.adp_service import write_adp_snapshot
            payload = fetch_mfl_adp(int(season), is_ppr=1, fcount=12, is_mock=0) or {}
            persist_committed_snapshot("mfl", season, payload)
            write_adp_snapshot("mfl", AXIS, int(season), payload, frozen=True)
            adp = payload.get("adp") or {}
            summary["sources"]["mfl"] = {"ok": True, "n": len(adp), "usable": source_map_is_usable(adp)}
        except Exception as exc:
            summary["sources"]["mfl"] = {"ok": False, "error": str(exc)[:200]}

    skipped = _skip_source("espn", season, skip_existing)
    if skipped:
        summary["sources"]["espn"] = skipped
    else:
        try:
            from dashboard_services.providers.global_adp import fetch_espn_global_adp
            from dashboard_services.adp_service import write_adp_snapshot
            payload = fetch_espn_global_adp(int(season)) or {}
            persist_committed_snapshot("espn", season, payload)
            if source_map_is_usable(payload.get("adp") or {}):
                write_adp_snapshot("espn", AXIS, int(season), payload, frozen=True)
            adp = payload.get("adp") or {}
            summary["sources"]["espn"] = {"ok": True, "n": len(adp), "usable": source_map_is_usable(adp)}
        except Exception as exc:
            summary["sources"]["espn"] = {"ok": False, "error": str(exc)[:200]}

    summary["sources"]["yahoo"] = {
        "ok": True,
        "n": 0,
        "usable": False,
        "skipped": True,
        "reason": "public Yahoo ADP endpoint has no season axis",
    }
    return summary


def backfill_historical_adp(
    seasons: Sequence[int] = DEFAULT_SEASONS,
    *,
    skip_existing: bool = True,
) -> dict:
    """Fetch + commit frozen snapshots for warehouse seasons."""
    HISTORICAL_ADP_DIR.mkdir(parents=True, exist_ok=True)
    by_season = {}
    for season in seasons:
        if int(season) < RELIABLE_SEASON_FLOOR:
            continue
        print(f"[historical] ADP backfill {season} ...")
        by_season[str(season)] = fetch_and_commit_season(int(season), skip_existing=skip_existing)
    coverage = coverage_from_committed(seasons)
    coverage["fetch"] = by_season
    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    print(f"[historical] ADP coverage → {COVERAGE_PATH}")
    return coverage


def coverage_from_committed(seasons: Sequence[int] = DEFAULT_SEASONS) -> dict:
    by_season = {}
    for season in seasons:
        sources = {}
        for source in ADP_SOURCE_PREFERENCE:
            snap = load_committed_snapshot(source, int(season))
            adp = snap.get("adp") or {}
            sources[source] = {
                "n": len(adp),
                "usable": bool(snap.get("usable")) if snap else False,
            }
        by_season[str(season)] = sources
    return {
        "axis": AXIS,
        "scoring": "ppr",
        "qb_format": "1qb",
        "sf_tep_historical": False,
        "yahoo_historical": False,
        "seasons": list(seasons),
        "by_season": by_season,
    }


def load_adp_by_season(
    seasons: Optional[Sequence[int]] = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """``{season: {source: {sleeper_id: adp}}}`` using only usable committed maps."""
    if seasons is None:
        seasons = DEFAULT_SEASONS
    out: dict[int, dict[str, dict[str, float]]] = {}
    for season in seasons:
        year = int(season)
        srcs: dict[str, dict[str, float]] = {}
        for source in ADP_SOURCE_PREFERENCE:
            snap = load_committed_snapshot(source, year)
            if not snap.get("usable"):
                continue
            mapping = snapshot_adp_map(source, year)
            if mapping:
                srcs[source] = mapping
        if srcs:
            out[year] = srcs
    return out


def attach_historical_adp(rows: Sequence[dict]) -> list[dict]:
    """Join committed snapshots onto warehouse dicts. No-op when cache is empty."""
    seasons = sorted({
        int(r["season"]) for r in rows
        if r.get("season") is not None
    }) if rows else []
    maps = load_adp_by_season(seasons)
    if not maps:
        return [dict(r) for r in rows]
    return attach_adp_features(rows, maps)


if __name__ == "__main__":
    backfill_historical_adp()
