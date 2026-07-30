"""Shared ADP fetching logic used by both the draft-grades endpoint and the prospects page."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

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
    ("rookie", False):  ("adp_dynasty_rookie", "adp_rookie"),
    ("rookie", True):   ("adp_dynasty_rookie", "adp_rookie"),
}

# Which market sources are valid per scoring axis. Yahoo publishes redraft ADP
# only (it is a seasonal platform), so it is offered for redraft alone. The
# "brfantasy" source is our own draft-crawler feed. It sees dynasty startup,
# rookie, and keeper/redraft drafts, so it is offered on all three axes.
ADP_SOURCES = {
    "redraft": ("sleeper", "yahoo", "brfantasy"),
    "dynasty": ("sleeper", "brfantasy"),
    "rookie":  ("sleeper", "brfantasy"),
}

# Human labels for the ADP sources, for source-selector UIs.
ADP_SOURCE_LABELS = {
    "sleeper":   "Sleeper",
    "yahoo":     "Yahoo",
    "brfantasy": "BR Fantasy",
    "consensus": "Consensus",
}

# resolver scoring axis -> draft_adp.draft_type produced by the BR Fantasy crawler.
_CRAWLER_DRAFT_TYPE = {"dynasty": "startup", "redraft": "redraft", "rookie": "rookie"}

# Reference league size the crawler's size-normalized ADP is rescaled onto, so
# the output reads as an overall pick in a standard 12-team draft.
_CRAWLER_REF_SIZE = 12


def adp_source_options(scoring_type: str):
    """[(value, label)] of the sources valid for a scoring axis, plus Consensus.

    Drives the source-selector dropdowns so each surface offers exactly the
    sources that make sense for what is being drafted (Yahoo only for redraft,
    BR Fantasy only for dynasty/rookie)."""
    st = scoring_type if scoring_type in ADP_SOURCES else "redraft"
    values = ["consensus", *ADP_SOURCES[st]]
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
    # Yahoo ADP is redraft-only and needs a user token; other axes yield nothing.
    if scoring_type != "redraft" or not (league_id and token):
        return {}
    try:
        return fetch_yahoo_adp(league_id, token, int(season), is_sf) or {}
    except Exception:
        return {}


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
                       as_rank: bool = False, fallback: bool = True) -> Dict[str, float]:
    """canonical player_id -> overall market ADP for a scoring axis and source.

    scoring_type: ``redraft`` | ``dynasty`` | ``rookie``.
    source:       ``sleeper`` | ``yahoo`` | ``brfantasy`` | ``consensus``.
                  ``yahoo`` is redraft-only and ``brfantasy`` is dynasty/rookie
                  only; requesting a source off its axis yields nothing and the
                  resolver falls back. Empty result means no data (the caller can
                  apply its own fallback, e.g. value rank).
    as_rank:      when True the result is re-ranked to contiguous 1..N draft
                  order (see ``ordinal_rank_adp``) for a clean board display."""
    scoring_type = scoring_type if scoring_type in ("redraft", "dynasty", "rookie") else "redraft"
    valid = ADP_SOURCES.get(scoring_type, ("sleeper",))

    def _src(name: str) -> Dict[str, float]:
        if name == "sleeper":
            return _sleeper_adp_source(season, is_sf, scoring_type)
        if name == "brfantasy":
            return _crawler_adp_source(season, is_sf, scoring_type)
        if name == "yahoo":
            return _yahoo_adp_source(season, is_sf, scoring_type, league_id, token)
        return {}

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
    fallback_m = (_sleeper_adp_source(season, is_sf, scoring_type)
                  or _crawler_adp_source(season, is_sf, scoring_type))
    return _finish(fallback_m)
