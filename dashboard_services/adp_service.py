"""Shared ADP fetching logic used by both the draft-grades endpoint and the prospects page."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

# Pure-logic format model + capability metadata. Safe to import at module load
# (no I/O, no heavy deps); re-exported below so callers may import either module.
from dashboard_services.adp_formats import (  # noqa: F401
    AdpFormat, SOURCE_CAPABILITIES, source_capability,
    classify_match, rank_sources_by_match, tep_bucket,
    axis_to_draft_type, draft_type_to_axis,
    EXACT, COMPATIBLE, GENERIC, EXCLUDED, MATCH_QUALITY_ORDER,
)

logger = logging.getLogger(__name__)

# Crawler ADP is an expensive GROUP BY over the whole draft_adp table and the
# underlying data only changes once a day (nightly crawl). A player modal can
# ask for it up to ~8x per open (BR Fantasy + Consensus x dynasty/redraft x
# 1QB/SF), so cache each (season, is_sf, scoring_type, min_samples) result in
# process for a short window to keep modals snappy.
#
# The lock makes the query single-flight: without it, several modals opening on
# a cold cache each run the full GROUP BY at once (a stampede) — that both
# starves the DB pool (unrelated queries, e.g. the modal's own player-details
# lookups, start failing) and makes the crawler calls themselves time out. With
# it, one caller computes while the rest wait and then read the fresh cache.
_CRAWLER_ADP_CACHE: Dict[tuple, tuple] = {}
_CRAWLER_ADP_TTL = 600  # seconds
_CRAWLER_ADP_LOCK = threading.Lock()


def _atomic_json_write(path, data) -> None:
    """Write JSON to a temp file then rename - prevents partial reads on crash."""
    import json as _json
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(data, f)
    os.replace(tmp, str(path))


# ── Global-source snapshot store ──────────────────────────────────────────────
# The tokenless global feeds (Yahoo/ESPN/MFL) are refreshed centrally a few times
# a day and their normalized results persisted here, so the request path only ever
# reads a snapshot from disk — never touches the network. This gives three
# properties the plan requires for free: per-provider failure isolation (a fetch
# that fails leaves the last good snapshot untouched), stale-data retention
# (``write_adp_snapshot`` refuses to overwrite good data with an empty result),
# and a durable record for later historical ADP-movement analysis.

def _snapshot_dir():
    from utils.paths import DATA_DIR
    d = DATA_DIR / "adp_snapshots"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("adp_service: could not create snapshot dir", exc_info=True)
    return d


def _snapshot_path(source: str, axis: str, season: int):
    return _snapshot_dir() / f"{source}_{axis}_{int(season)}.json"


def load_adp_snapshot(source: str, axis: str, season: int) -> dict:
    """Full persisted snapshot payload for a source/axis/season, or {} if absent.

    Never raises: a missing or corrupt snapshot degrades to {} so the resolver
    simply treats the source as having no data."""
    import json as _json
    path = _snapshot_path(source, axis, season)
    try:
        if path.exists():
            with open(path) as f:
                data = _json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("adp_service: corrupt ADP snapshot at %s", path, exc_info=True)
    return {}


def snapshot_adp_map(source: str, axis: str, season: int) -> Dict[str, float]:
    """Just the {canonical_id: overall_adp} map from a persisted snapshot."""
    adp = (load_adp_snapshot(source, axis, season) or {}).get("adp") or {}
    out: Dict[str, float] = {}
    for pid, v in adp.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            out[str(pid)] = fv
    return out


def snapshot_freshness(source: str, axis: str, season: int) -> Optional[float]:
    """Unix timestamp the snapshot was collected, or None."""
    ca = (load_adp_snapshot(source, axis, season) or {}).get("collected_at")
    try:
        return float(ca) if ca is not None else None
    except (TypeError, ValueError):
        return None


def write_adp_snapshot(source: str, axis: str, season: int, payload: dict) -> bool:
    """Persist a normalized snapshot, retaining the last good data on empty input.

    Returns True if a new snapshot was written. If ``payload`` carries no ADP rows
    but a non-empty snapshot already exists on disk, the write is *skipped* so an
    upstream outage or empty response never clobbers valid cached data."""
    adp = (payload or {}).get("adp") or {}
    if not adp:
        existing = load_adp_snapshot(source, axis, season)
        if (existing.get("adp") or {}):
            logger.info("adp_service: keeping last-good %s/%s snapshot (empty fetch)",
                        source, axis)
            return False
    record = {
        "source": source,
        "axis": axis,
        "season": int(season),
        "collected_at": time.time(),
        "adp": adp,
        "extra": (payload or {}).get("extra") or {},
        "meta": (payload or {}).get("meta") or {},
        "raw_count": (payload or {}).get("raw_count"),
        "mapped_count": (payload or {}).get("mapped_count"),
    }
    if (payload or {}).get("ppr_rank"):
        record["ppr_rank"] = payload["ppr_rank"]
    try:
        _atomic_json_write(_snapshot_path(source, axis, season), record)
    except Exception:
        logger.warning("adp_service: failed to write %s/%s snapshot", source, axis, exc_info=True)
        return False
    _persist_snapshot_db(record)  # best-effort, never raises
    return True


def _persist_snapshot_db(record: dict) -> None:
    """Best-effort mirror of a snapshot into the adp_snapshots table.

    Disk is the source of truth for the resolver; the table exists for future
    historical ADP-movement queries. Any DB problem is swallowed — a missing
    table, no DSN in a pure test env, or a transient error must never break the
    refresh or the request path."""
    meta = record.get("meta") or {}
    adp = record.get("adp") or {}
    if not adp:
        return
    try:
        from dashboard_services.db import get_conn
        ppr = meta.get("ppr")
        ppr_num = None
        if isinstance(ppr, (int, float)):
            ppr_num = float(ppr)
        draft_type = meta.get("draft_type") or record.get("axis")
        qb_format = meta.get("qb_format")
        num_teams = meta.get("num_teams")
        scope = meta.get("scope")
        extra = record.get("extra") or {}
        rows = []
        for pid, val in adp.items():
            try:
                a = float(val)
            except (TypeError, ValueError):
                continue
            ex = extra.get(pid) or {}
            rows.append((record["source"], int(record["season"]), str(pid), a,
                         draft_type, qb_format, ppr_num, meta.get("te_premium"),
                         num_teams, scope, ex.get("min_pick"), ex.get("max_pick"),
                         ex.get("draft_pct")))
        if not rows:
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO adp_snapshots
                        (source, season, player_id, adp, draft_type, qb_format,
                         ppr, te_premium, num_teams, source_scope,
                         min_pick, max_pick, draft_pct, collected_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
                    ON CONFLICT (source, season, player_id, draft_type, qb_format)
                    DO UPDATE SET adp = EXCLUDED.adp, ppr = EXCLUDED.ppr,
                        te_premium = EXCLUDED.te_premium, num_teams = EXCLUDED.num_teams,
                        source_scope = EXCLUDED.source_scope, min_pick = EXCLUDED.min_pick,
                        max_pick = EXCLUDED.max_pick, draft_pct = EXCLUDED.draft_pct,
                        collected_at = NOW()
                    """,
                    rows,
                )
    except Exception:
        logger.debug("adp_service: adp_snapshots DB mirror skipped", exc_info=True)


# ── Central refresh of the global feeds ───────────────────────────────────────
def refresh_global_adp_sources(season: int) -> dict:
    """Fetch every tokenless global feed and persist its snapshot. For the daily
    cron. Each provider is isolated: one failing never affects the others, and an
    empty fetch keeps the last good snapshot. Returns a per-source summary."""
    summary: Dict[str, dict] = {}

    def _run(source: str, axis: str, fetch):
        try:
            payload = fetch()
            wrote = write_adp_snapshot(source, axis, int(season), payload)
            summary[source] = {"ok": True, "written": wrote,
                               "mapped": (payload or {}).get("mapped_count"),
                               "raw": (payload or {}).get("raw_count")}
        except Exception as exc:  # noqa: BLE001 - isolate each provider
            logger.warning("adp_service: refresh %s failed: %s", source, exc, exc_info=True)
            summary[source] = {"ok": False, "error": type(exc).__name__}

    from dashboard_services.providers import global_adp as _g
    _run("yahoo", "redraft", lambda: _g.fetch_yahoo_global_adp(int(season)))
    _run("espn", "redraft", lambda: _g.fetch_espn_global_adp(int(season)))
    _run("mfl", "redraft", lambda: _g.fetch_mfl_adp(int(season)))
    return summary


def fetch_league_adp_from_db(
    is_sf: bool,
    season: int,
    draft_type: str,
    min_samples: int = 40,
) -> dict:
    """
    Pull ADP from real league draft data aggregated across standard league sizes
    (8–16 teams).  Returns sleeper_id -> {adp_rank, avg_pick, std_pick,
    sample_size, position} or empty dict when data is sparse.
    """
    try:
        from utils.paths import DATA_DIR
        import json as _json

        prefix = f"league_adp_{draft_type}_{'sf' if is_sf else '1qb'}_{season}"
        cache_path = DATA_DIR / f"{prefix}.json"
        # Also search for dated variants (e.g. league_adp_rookie_sf_2026_2026-05-14.json)
        # and prefer the most recent one if the undated file doesn't exist
        if not cache_path.exists():
            import glob as _glob
            dated = sorted(_glob.glob(str(DATA_DIR / f"{prefix}_*.json")))
            if dated:
                cache_path = type(DATA_DIR)(dated[-1])
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 86400:
            try:
                with open(cache_path) as _cf:
                    return _json.load(_cf)
            except Exception:
                logger.warning("adp_service: corrupt cache at %s, rebuilding", cache_path)

        from dashboard_services.db import get_conn
        with get_conn() as conn:
            # Size-normalize before combining across league sizes: a raw pick
            # number is not comparable between a 10- and a 14-team draft (pick 24
            # is round 3 in one, round 2 in the other). Convert each size's row to
            # a round position (avg_pick / num_teams), sample-weight across sizes,
            # then rescale to a reference 12-team pick so avg_pick stays an
            # overall-pick number for callers. Ordering by the round position is
            # equivalent to ordering by the rescaled pick (monotonic).
            rows = conn.execute(
                """
                SELECT
                    da.player_id,
                    SUM((da.avg_pick / da.num_teams::numeric) * da.sample_size)
                        / NULLIF(SUM(da.sample_size), 0) AS norm_round,
                    SUM(da.sample_size) AS sample_size
                FROM draft_adp da
                WHERE da.draft_type   = %s
                  AND da.season       = %s
                  AND da.is_superflex = %s
                  AND da.num_teams BETWEEN 8 AND 16
                GROUP BY da.player_id
                HAVING SUM(da.sample_size) >= %s
                ORDER BY norm_round ASC
                """,
                (draft_type, season, is_sf, min_samples),
            ).fetchall()

        if not rows:
            return {}

        player_ids = [r["player_id"] for r in rows]
        pos_map: Dict[str, str] = {}
        try:
            with get_conn() as conn:
                pv_rows = conn.execute(
                    "SELECT player_id, position FROM player_values WHERE player_id = ANY(%s)",
                    (player_ids,),
                ).fetchall()
                pos_map = {r["player_id"]: (r["position"] or "").upper() for r in pv_rows}
        except Exception:
            logger.warning("adp_service: failed to load position map", exc_info=True)

        result: dict = {}
        pos_counters: Dict[str, int] = {}
        for rank, row in enumerate(rows, start=1):
            pid = str(row["player_id"])
            pos = pos_map.get(pid, "")
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            try:
                _overall = float(row["norm_round"]) * _CRAWLER_REF_SIZE
            except (TypeError, ValueError):
                _overall = float(rank)
            result[pid] = {
                "adp_rank":    rank,
                "avg_pick":    _overall or float(rank),
                "std_pick":    0,
                "pos_rank":    pos_counters[pos],
                "position":    pos,
                "sample_size": int(row["sample_size"] or 0),
            }

        try:
            _atomic_json_write(cache_path, result)
        except Exception:
            logger.warning("adp_service: failed to write cache to %s", cache_path, exc_info=True)
        return result
    except Exception:
        logger.exception("adp_service: fetch_league_adp_from_db failed (sf=%s, season=%s, type=%s)", is_sf, season, draft_type)
        return {}


def fetch_sleeper_adp(season: int) -> dict:
    """Per-player ADP from Sleeper's own projections API (api.sleeper.com).

    Sleeper's season projection objects carry ADP fields (adp_ppr, adp_2qb,
    adp_dynasty_ppr, adp_dynasty_2qb, adp_dynasty_half_ppr, adp_rookie, ...).
    Returns sleeper_id -> {field: value}. Cached daily; {} on any failure so the
    caller can fall back. Server-reachable (unlike FantasyCalc/DraftSharks).
    """
    import json as _json
    from datetime import date
    from utils.paths import DATA_DIR

    cache_path = DATA_DIR / f"sleeper_adp_{season}_{date.today().isoformat()}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as _f:
                return _json.load(_f)
        except Exception:
            logger.warning("adp_service: corrupt Sleeper ADP cache at %s, rebuilding", cache_path)

    url = (
        f"https://api.sleeper.com/projections/nfl/{season}"
        "?season_type=regular&order_by=adp_ppr"
        "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=K&position[]=DEF"
    )
    try:
        import requests as _req
        resp = _req.get(url, timeout=20, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as _exc:
        logger.warning("adp_service: Sleeper ADP fetch failed (season=%s): %s", season, _exc)
        return {}

    _KEYS = (
        "adp_std", "adp_ppr", "adp_half_ppr", "adp_2qb",
        "adp_dynasty_std", "adp_dynasty_ppr", "adp_dynasty_half_ppr", "adp_dynasty_2qb",
        "adp_rookie", "adp_dynasty_rookie",
    )
    out: dict = {}
    for _item in (data or []):
        if not isinstance(_item, dict):
            continue
        _pid = str(_item.get("player_id") or "")
        _st = _item.get("stats") or {}
        if not _pid or not isinstance(_st, dict):
            continue
        _row = {}
        for _k in _KEYS:
            _v = _st.get(_k)
            if _v is not None:
                try:
                    _row[_k] = float(_v)
                except (TypeError, ValueError):
                    logger.debug("suppressed exception", exc_info=True)
        if _row:
            out[_pid] = _row
    try:
        _atomic_json_write(cache_path, out)
    except Exception:
        logger.warning("adp_service: failed to write Sleeper ADP cache to %s", cache_path, exc_info=True)
    return out


def build_model_adp_fallback(is_sf: bool, season: int, filter_undrafted: bool = False) -> dict:
    """
    Build a value-based board from our own model when external ADP is unavailable.
    Returns sleeper_id -> {adp_rank, pos_rank, position}.
    """
    try:
        from dashboard_services.db import get_conn
        value_col = "COALESCE(calibrated_value_sf, value_sf)" if is_sf \
                    else "COALESCE(calibrated_value_1qb, value_1qb)"
        undrafted_clause = "AND rp.draft_confirmed = TRUE" if filter_undrafted else ""
        with get_conn() as _conn:
            rows = _conn.execute(
                f"""
                SELECT rp.sleeper_id, rp.position,
                       {value_col} AS val
                FROM rookie_prospects rp
                LEFT JOIN player_values pv ON pv.player_id = rp.sleeper_id
                WHERE rp.draft_class_year = %s
                  AND rp.sleeper_id IS NOT NULL
                  {undrafted_clause}
                ORDER BY {value_col} DESC NULLS LAST
                """,
                (season,)
            ).fetchall()

        result: dict = {}
        pos_counters: dict = {}
        for rank, row in enumerate(rows, start=1):
            sid = str(row["sleeper_id"])
            pos = str(row["position"] or "").upper()
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            result[sid] = {
                "adp_rank": rank,
                "pos_rank":  pos_counters[pos],
                "position":  pos,
            }
        return result
    except Exception:
        logger.exception("adp_service: build_model_adp_fallback failed (sf=%s, season=%s)", is_sf, season)
        return {}


# ── Unified market-ADP resolver ──────────────────────────────────────────────
# One entry point every surface (keeper tool, rankings, draft room) uses to ask
# for ADP, so the source and scoring axis are chosen in one place instead of ad
# hoc per consumer. New sources (e.g. a replacement dynasty/rookie feed) slot in
# as another adapter below and get registered in ADP_SOURCES.

# Sleeper projection ADP fields per (scoring_type, is_superflex), most- to
# least-preferred.
_SLEEPER_ADP_FIELDS = {
    ("redraft", False): ("adp_ppr", "adp_half_ppr", "adp_std"),
    ("redraft", True):  ("adp_2qb", "adp_ppr", "adp_half_ppr", "adp_std"),
    ("dynasty", False): ("adp_dynasty_ppr", "adp_dynasty_half_ppr", "adp_dynasty_std"),
    ("dynasty", True):  ("adp_dynasty_2qb", "adp_dynasty_ppr", "adp_dynasty_half_ppr", "adp_dynasty_std"),
    # Most classes have no rookie-specific Sleeper ADP, but Sleeper does price the
    # rookies in its overall dynasty startup ADP. Fall back to that so the rookie
    # board isn't empty; the caller ranks the rookies among themselves (restrict_ids
    # + as_rank) to turn those overall picks into a clean 1..N rookie board.
    ("rookie", False):  ("adp_dynasty_rookie", "adp_rookie",
                         "adp_dynasty_ppr", "adp_dynasty_half_ppr", "adp_dynasty_std"),
    ("rookie", True):   ("adp_dynasty_rookie", "adp_rookie",
                         "adp_dynasty_2qb", "adp_dynasty_ppr", "adp_dynasty_half_ppr", "adp_dynasty_std"),
}

# Which market sources are valid per scoring axis, and which the selector UIs
# offer. Only sources with verified capability on an axis appear:
#   redraft — Sleeper, ESPN (global), Yahoo (global), MFL (global PPR), BR Fantasy.
#   dynasty — Sleeper, BR Fantasy. ESPN/Yahoo/MFL global feeds are redraft-only
#             and are deliberately excluded from dynasty (never mix redraft ADP
#             into a dynasty market). MFL exposes no verified dynasty ADP filter.
#   rookie  — Sleeper, BR Fantasy. MFL has no verified rookie ADP filter.
# The globals (espn/yahoo/mfl) are read from centrally-refreshed snapshots, never
# fetched on the request path.
ADP_SOURCES = {
    "redraft": ("sleeper", "espn", "yahoo", "mfl", "brfantasy"),
    "dynasty": ("sleeper", "brfantasy"),
    "rookie":  ("sleeper", "brfantasy"),
}

# Human labels for the ADP sources, for source-selector UIs.
ADP_SOURCE_LABELS = {
    "sleeper":   "Sleeper",
    "espn":      "ESPN",
    "yahoo":     "Yahoo",
    "mfl":       "MFL",
    "brfantasy": "BR Fantasy",
    "consensus": "Consensus",
}

# resolver scoring axis -> draft_adp.draft_type produced by the BR Fantasy crawler.
_CRAWLER_DRAFT_TYPE = {"dynasty": "startup", "redraft": "redraft", "rookie": "rookie"}

# Reference league size the crawler's size-normalized ADP is rescaled onto, so
# the output reads as an overall pick in a standard 12-team draft.
_CRAWLER_REF_SIZE = 12


# Global feeds that only make sense in a selector once they actually have data.
_GLOBAL_SNAPSHOT_SOURCES = frozenset({"espn", "yahoo", "mfl"})


def adp_source_options(scoring_type: str, season: Optional[int] = None):
    """[(value, label)] of the sources valid for a scoring axis, plus Consensus.

    Drives the source-selector dropdowns so each surface offers exactly the
    sources that make sense for what is being drafted (Yahoo/ESPN/MFL redraft
    only, BR Fantasy on every axis).

    When ``season`` is given, a global snapshot-backed source (ESPN/Yahoo/MFL) is
    hidden unless it actually has a non-empty snapshot for that season — so a
    selector never offers a source that would return nothing (Priority 4). Always-
    on sources (Sleeper, BR Fantasy) and Consensus are never gated. With
    ``season=None`` every configured source is listed (legacy behavior)."""
    st = scoring_type if scoring_type in ADP_SOURCES else "redraft"
    values = ["consensus", *ADP_SOURCES[st]]
    season_int: Optional[int] = None
    if season is not None:
        try:
            season_int = int(season)
        except (TypeError, ValueError):
            season_int = None
    if season_int is not None:
        values = [v for v in values
                  if v not in _GLOBAL_SNAPSHOT_SOURCES
                  or snapshot_adp_map(v, st, season_int)]
    return [(v, ADP_SOURCE_LABELS.get(v, v.title())) for v in values]


# Sleeper's projection ADP fields report 999 for players it has no draft data
# for (undrafted / no ADP). It is a sentinel, not a real pick, so treat anything
# at or above it as missing - otherwise every rookie without a Sleeper ADP shows
# a literal "ADP 999.0", and consensus rank-blends those identical 999s against
# real sources and produces nonsense.
_SLEEPER_UNDRAFTED_ADP = 999.0


def _adp_overall_from_row(row: dict, fields) -> Optional[float]:
    for f in fields:
        v = (row or {}).get(f)
        try:
            if v is not None and 0 < float(v) < _SLEEPER_UNDRAFTED_ADP:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _sleeper_adp_source(season: int, is_sf: bool, scoring_type: str) -> Dict[str, float]:
    fields = _SLEEPER_ADP_FIELDS.get((scoring_type, is_sf)) or _SLEEPER_ADP_FIELDS[("redraft", is_sf)]
    out: Dict[str, float] = {}
    for pid, row in (fetch_sleeper_adp(int(season)) or {}).items():
        ov = _adp_overall_from_row(row, fields)
        if ov:
            out[str(pid)] = ov
    return out


def fetch_crawler_adp(season: int, is_sf: bool, scoring_type: str,
                      min_samples: int = 20) -> Dict[str, float]:
    """canonical id -> average draft pick from the draft-crawler aggregate table.

    The crawler stores one ``draft_adp`` row per
    (player, draft_type, season, is_superflex, num_teams). We report the raw
    ``avg_pick`` (overall pick number), sample-size-weighted across the drafts a
    player appears in, so it reads on the same scale as other ADP feeds. Formats
    stay separate: dynasty and rookie are distinct markets (and distinct axes),
    never blended together.

    Dynasty startup, rookie, and keeper/redraft drafts are crawled. Falls back to
    the latest season with data when the requested season is empty (early in a
    season the startups are sparse). Empty on any failure."""
    draft_type = _CRAWLER_DRAFT_TYPE.get(scoring_type)
    if not draft_type:
        return {}

    _ck = (int(season), bool(is_sf), scoring_type, int(min_samples))
    _cached = _CRAWLER_ADP_CACHE.get(_ck)
    if _cached is not None and (time.time() - _cached[0]) < _CRAWLER_ADP_TTL:
        return _cached[1]

    # Single-flight the compute. If another thread is already running it and it
    # takes a while, don't pile up (which would exhaust the worker's threads and
    # make unrelated requests fail) — serve a stale entry if we have one, else
    # empty so the caller falls back to Sleeper-only.
    if not _CRAWLER_ADP_LOCK.acquire(timeout=3.0):
        return _cached[1] if _cached is not None else {}
    try:
        # Re-check: another thread may have filled the cache while we waited.
        _fresh = _CRAWLER_ADP_CACHE.get(_ck)
        if _fresh is not None and (time.time() - _fresh[0]) < _CRAWLER_ADP_TTL:
            return _fresh[1]

        try:
            from dashboard_services.db import get_conn

            def _query(season_val: int):
                with get_conn() as conn:
                    return conn.execute(
                        """
                        SELECT
                            player_id,
                            SUM(avg_pick * sample_size)
                                / NULLIF(SUM(sample_size), 0) AS avg_pick,
                            SUM(sample_size) AS n
                        FROM draft_adp
                        WHERE draft_type   = %s
                          AND season       = %s
                          AND is_superflex = %s
                          AND num_teams BETWEEN 8 AND 18
                        GROUP BY player_id
                        HAVING SUM(sample_size) >= %s
                        ORDER BY avg_pick ASC
                        """,
                        (draft_type, season_val, is_sf, min_samples),
                    ).fetchall()

            rows = _query(int(season))
            if not rows:
                with get_conn() as conn:
                    latest = conn.execute(
                        "SELECT MAX(season) AS s FROM draft_adp "
                        "WHERE draft_type = %s AND is_superflex = %s",
                        (draft_type, is_sf),
                    ).fetchone()
                latest_season = latest and latest["s"]
                if latest_season and int(latest_season) != int(season):
                    rows = _query(int(latest_season))

            out: Dict[str, float] = {}
            for r in rows or []:
                _ap = r["avg_pick"]
                if _ap is None:
                    continue
                try:
                    out[str(r["player_id"])] = float(_ap)
                except (TypeError, ValueError):
                    continue
            _CRAWLER_ADP_CACHE[_ck] = (time.time(), out)
            return out
        except Exception:
            logger.debug("adp_service: crawler ADP source failed", exc_info=True)
            return {}
    finally:
        _CRAWLER_ADP_LOCK.release()


def _crawler_adp_source(season: int, is_sf: bool, scoring_type: str) -> Dict[str, float]:
    return fetch_crawler_adp(int(season), is_sf, scoring_type)


def fetch_yahoo_adp(league_id, token, season: int, is_sf: bool) -> Dict[str, float]:
    """canonical id -> Yahoo average draft pick for a league.

    Thin wrapper over the Yahoo provider so the provider stays the only place
    that talks to Yahoo. Yahoo's draft_analysis is already scored for the
    league's own format, so ``is_sf`` is informational here. Empty on any
    failure (no token, network, mapping) so the resolver falls back."""
    if not (league_id and token):
        return {}
    try:
        from dashboard_services.providers.yahoo_api import get_draft_analysis_adp
        return get_draft_analysis_adp(int(season), str(league_id), str(token)) or {}
    except Exception:
        logger.debug("adp_service: Yahoo ADP fetch failed", exc_info=True)
        return {}


def _yahoo_adp_source(season: int, is_sf: bool, scoring_type: str,
                      league_id, token) -> Dict[str, float]:
    """Yahoo ADP, redraft-only.

    With a league token, use Yahoo's league-format-aware draft_analysis (kept for
    connected Yahoo leagues). Without one, fall back to the public *global* Yahoo
    ADP snapshot (no OAuth) so the "Yahoo" source works for everyone."""
    if scoring_type != "redraft":
        return {}
    if league_id and token:
        try:
            got = fetch_yahoo_adp(league_id, token, int(season), is_sf) or {}
            if got:
                return got
        except Exception:
            logger.debug("adp_service: Yahoo league ADP failed, trying global", exc_info=True)
    return snapshot_adp_map("yahoo", "redraft", int(season))


def _espn_adp_source(season: int, is_sf: bool, scoring_type: str) -> Dict[str, float]:
    """ESPN global redraft ADP from the persisted snapshot (redraft-only).

    Reads only ``averageDraftPosition``; ESPN's separate PPR draft-room rank is
    never surfaced here, so it can never leak into ADP consensus."""
    if scoring_type != "redraft":
        return {}
    return snapshot_adp_map("espn", "redraft", int(season))


def _mfl_adp_source(season: int, is_sf: bool, scoring_type: str) -> Dict[str, float]:
    """MFL global redraft ADP from the persisted snapshot (redraft-only)."""
    if scoring_type != "redraft":
        return {}
    return snapshot_adp_map("mfl", "redraft", int(season))


def espn_ppr_rank(season: int) -> Dict[str, float]:
    """ESPN's PPR draft-room rank (separate from ADP) for platform-room analysis.

    Exposed for future platform-room value features; deliberately NOT part of any
    ADP source map or consensus."""
    return {str(k): float(v) for k, v in
            ((load_adp_snapshot("espn", "redraft", int(season)) or {}).get("ppr_rank") or {}).items()
            if _is_pos_num(v)}


def _is_pos_num(v) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def consensus_adp(source_maps) -> Dict[str, float]:
    """Blend several ``{id: overall_adp}`` maps into one consensus map.

    A single source is returned as-is (raw ADP kept). With two or more, each
    source is ranked independently (1 = earliest) and a player's consensus ADP is
    the average of their ranks across the sources that list them, so one source
    on a different numeric scale can't skew the blend."""
    present = [m for m in source_maps if m]
    if not present:
        return {}
    if len(present) == 1:
        return dict(present[0])
    ranks_per_source = []
    for m in present:
        order = sorted(m.items(), key=lambda kv: kv[1])
        ranks_per_source.append({pid: i + 1 for i, (pid, _v) in enumerate(order)})
    agg: Dict[str, list] = {}
    for ranks in ranks_per_source:
        for pid, r in ranks.items():
            agg.setdefault(pid, []).append(r)
    return {pid: sum(rs) / len(rs) for pid, rs in agg.items()}


def ordinal_rank_adp(adp_map) -> Dict[str, float]:
    """Replace ADP values with their 1-based draft order (1, 2, 3, ...).

    A mean-of-picks ADP never bottoms out at a clean 1.0 (the consensus No. 1 is
    still taken third in some drafts, so the average floats above the floor).
    For a single-source display that should read like a board, ranking by ADP
    turns those raw averages into contiguous slots. Ties break by id for a
    stable order. This is a presentation transform; keep the raw values for any
    math that needs the gaps between picks."""
    order = sorted(adp_map.items(), key=lambda kv: (kv[1], str(kv[0])))
    return {str(pid): float(i) for i, (pid, _v) in enumerate(order, start=1)}


def resolve_market_adp(season: int, is_sf: bool, scoring_type: str = "redraft",
                       source: str = "consensus", league_id=None, token=None,
                       as_rank: bool = False, fallback: bool = True,
                       restrict_ids=None) -> Dict[str, float]:
    """canonical player_id -> overall market ADP for a scoring axis and source.

    scoring_type: ``redraft`` | ``dynasty`` | ``rookie``.
    source:       ``sleeper`` | ``yahoo`` | ``brfantasy`` | ``consensus``.
                  ``yahoo`` is redraft-only and ``brfantasy`` is dynasty/rookie
                  only; requesting a source off its axis yields nothing and the
                  resolver falls back. Empty result means no data (the caller can
                  apply its own fallback, e.g. value rank).
    as_rank:      when True the result is re-ranked to contiguous 1..N draft
                  order (see ``ordinal_rank_adp``) for a clean board display.
    restrict_ids: when given, each source is filtered to just these ids BEFORE
                  ranking/blending. For a rookie board pass the rookie pool so a
                  source's overall ADP (e.g. Sleeper's dynasty fallback) is ranked
                  among the rookies alone, and consensus blends rookie-only ranks
                  rather than mixing a rookie's overall rank with its rookie rank."""
    scoring_type = scoring_type if scoring_type in ("redraft", "dynasty", "rookie") else "redraft"
    valid = ADP_SOURCES.get(scoring_type, ("sleeper",))
    _restrict = set(restrict_ids) if restrict_ids is not None else None

    def _clip(m: Dict[str, float]) -> Dict[str, float]:
        return {k: v for k, v in m.items() if k in _restrict} if _restrict is not None else m

    def _src(name: str) -> Dict[str, float]:
        return _clip(_raw_source_map(name, season, is_sf, scoring_type, league_id, token))

    def _finish(m: Dict[str, float]) -> Dict[str, float]:
        return ordinal_rank_adp(m) if (as_rank and m) else m

    if source == "consensus":
        blended = consensus_adp([_src(n) for n in valid])
        if blended:
            return _finish(blended)
    elif source in valid:
        got = _src(source)
        if got:
            return _finish(got)
    # No cross-source fallback when the caller wants THIS source's own data only
    # (e.g. the per-source ADP columns, where "BR Fantasy" must not silently show
    # Sleeper's numbers on an axis BR Fantasy doesn't cover).
    if not fallback:
        return {}
    # Fallback: sleeper, then the BR Fantasy crawler (dynasty/rookie).
    fallback_m = (_clip(_sleeper_adp_source(season, is_sf, scoring_type))
                  or _clip(_crawler_adp_source(season, is_sf, scoring_type)))
    return _finish(fallback_m)


# ── Shared source dispatch (used by both the simple and detailed resolvers) ────
def _raw_source_map(name: str, season: int, is_sf: bool, scoring_type: str,
                    league_id=None, token=None) -> Dict[str, float]:
    """{canonical_id: overall ADP} for one source, unclipped. Empty off-axis or
    on any failure (each source function isolates its own errors)."""
    if name == "sleeper":
        return _sleeper_adp_source(season, is_sf, scoring_type)
    if name == "brfantasy":
        return _crawler_adp_source(season, is_sf, scoring_type)
    if name == "yahoo":
        return _yahoo_adp_source(season, is_sf, scoring_type, league_id, token)
    if name == "espn":
        return _espn_adp_source(season, is_sf, scoring_type)
    if name == "mfl":
        return _mfl_adp_source(season, is_sf, scoring_type)
    return {}


# ── Capability-aware detailed resolver ────────────────────────────────────────
# The richer path new features use. Unlike resolve_market_adp (which preserves the
# simple {id: adp} contract and equal rank-blend), this classifies each source
# against the requested format, prefers exact over compatible over generic, keeps
# ESPN/Yahoo redraft data out of dynasty, and returns per-player provenance.

_TIER_WEIGHT = {EXACT: 3.0, COMPATIBLE: 2.0, GENERIC: 1.0}

# source_count -> confidence label (Priority 2, item 14).
def _confidence(n: int) -> str:
    if n <= 1:
        return "single-source"
    if n == 2:
        return "low"
    return "normal"


def resolve_market_adp_detailed(
    season: int,
    fmt: Optional[AdpFormat] = None,
    *,
    is_sf: Optional[bool] = None,
    scoring_type: str = "redraft",
    ppr=1.0,
    te_premium: float = 0.0,
    num_teams: Optional[int] = None,
    league_id=None,
    token=None,
    restrict_ids=None,
    min_quality: str = GENERIC,
) -> Dict[str, dict]:
    """canonical_id -> rich consensus record, capability-aware.

    Each record::

        {
          "consensus_adp": 24.4,      # tier-weighted mean of the raw source ADPs
          "source_count": 5,
          "exact_source_count": 2,
          "min_adp": 19.6, "max_adp": 31.2, "spread": 11.6,
          "sources": {"sleeper": 22.3, "espn": 26.1, ...},
          "match_quality": "exact",   # best tier among contributing sources
          "confidence": "normal"      # single-source | low | normal
        }

    Sources whose capability is ``excluded`` for the requested format (e.g. ESPN
    redraft ADP against a dynasty request) never contribute. ``min_quality`` drops
    any source below the given tier. All contributing sources are on the overall-
    pick scale, so the raw mean is meaningful; the simple resolver still offers the
    scale-invariant rank blend for the plain {id: adp} contract.
    """
    if fmt is None:
        fmt = AdpFormat.from_league(
            is_sf=bool(is_sf), scoring_type=scoring_type,
            ppr=ppr, te_premium=te_premium, num_teams=num_teams,
        )
    axis = fmt.axis
    valid = ADP_SOURCES.get(axis, ("sleeper",))
    ranked = rank_sources_by_match(fmt, valid)  # [(src, quality)], best first, excluded dropped
    min_idx = MATCH_QUALITY_ORDER.index(min_quality)

    _restrict = set(restrict_ids) if restrict_ids is not None else None
    per_source: List[tuple] = []  # (name, quality, {id: adp})
    for name, quality in ranked:
        if MATCH_QUALITY_ORDER.index(quality) > min_idx:
            continue
        m = _raw_source_map(name, int(season), fmt.is_superflex, axis, league_id, token)
        if _restrict is not None:
            m = {k: v for k, v in m.items() if k in _restrict}
        if m:
            per_source.append((name, quality, m))

    out: Dict[str, dict] = {}
    all_ids = set().union(*[m.keys() for _n, _q, m in per_source]) if per_source else set()
    for pid in all_ids:
        contribs = [(n, q, m[pid]) for n, q, m in per_source if pid in m]
        vals = [v for _n, _q, v in contribs]
        weights = [_TIER_WEIGHT.get(q, 1.0) for _n, q, _v in contribs]
        wsum = sum(weights) or 1.0
        consensus = sum(v * w for v, w in zip(vals, weights)) / wsum
        best_q = min((q for _n, q, _v in contribs),
                     key=lambda q: MATCH_QUALITY_ORDER.index(q))
        out[str(pid)] = {
            "consensus_adp": round(consensus, 2),
            "source_count": len(contribs),
            "exact_source_count": sum(1 for _n, q, _v in contribs if q == EXACT),
            "min_adp": round(min(vals), 2),
            "max_adp": round(max(vals), 2),
            "spread": round(max(vals) - min(vals), 2),
            "sources": {n: round(v, 2) for n, _q, v in contribs},
            "match_quality": best_q,
            "confidence": _confidence(len(contribs)),
        }
    return out
