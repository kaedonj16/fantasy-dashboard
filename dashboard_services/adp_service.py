"""Shared ADP fetching logic used by both the draft-grades endpoint and the prospects page."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

logger = logging.getLogger(__name__)


def _atomic_json_write(path, data) -> None:
    """Write JSON to a temp file then rename — prevents partial reads on crash."""
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
            rows = conn.execute(
                """
                SELECT
                    da.player_id,
                    SUM(da.avg_pick * da.sample_size) / SUM(da.sample_size) AS avg_pick,
                    SUM(da.sample_size) AS sample_size
                FROM draft_adp da
                WHERE da.draft_type   = %s
                  AND da.season       = %s
                  AND da.is_superflex = %s
                  AND da.num_teams BETWEEN 8 AND 16
                GROUP BY da.player_id
                HAVING SUM(da.sample_size) >= %s
                ORDER BY avg_pick ASC
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
            result[pid] = {
                "adp_rank":    rank,
                "avg_pick":    float(row["avg_pick"] or rank),
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


def fetch_fc_startup_adp(is_sf: bool) -> dict:
    """
    Fetch dynasty startup ADP from FantasyCalc for all players (not just rookies).

    Returns sleeper_id -> {adp_rank, pos_rank, position, avg_pick} where avg_pick
    equals the FantasyCalc overall dynasty rank (1 = consensus #1 startup pick).
    Caches per league type per day.
    """
    import json as _json
    from datetime import date
    from utils.paths import DATA_DIR

    key = f"fc_startup_adp_{'sf' if is_sf else '1qb'}_{date.today().isoformat()}.json"
    cache_path = DATA_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path) as _f:
                return _json.load(_f)
        except Exception:
            logger.warning("adp_service: corrupt startup ADP cache at %s, rebuilding", cache_path)

    num_qbs = 2 if is_sf else 1
    # No type= filter → all dynasty players (startup pool)
    url = f"https://fantasycalc.com/api/values/current?numQbs={num_qbs}&ppr=0.5"
    try:
        import requests as _req
        resp = _req.get(url, timeout=15, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        if not resp.text.strip():
            logger.info("adp_service: FantasyCalc startup returned empty body (sf=%s) — skipping", is_sf)
            return {}
        fc_data = resp.json()
    except Exception as _exc:
        logger.warning("adp_service: FantasyCalc startup fetch failed (sf=%s): %s", is_sf, _exc)
        fc_data = []

    if not fc_data:
        return {}

    # Sort by overallRank ascending so rank 1 = pick 1
    sorted_entries = sorted(
        [e for e in fc_data if isinstance(e, dict) and e.get("overallRank")],
        key=lambda e: e["overallRank"],
    )

    result: dict = {}
    pos_counters: dict = {}
    for rank, entry in enumerate(sorted_entries, start=1):
        p = entry.get("player") or {}
        sid = str(p.get("sleeperId") or "")
        if not sid or sid == "None":
            continue
        pos = str(p.get("position") or "").upper()
        pos_counters[pos] = pos_counters.get(pos, 0) + 1
        result[sid] = {
            "adp_rank":  rank,
            "avg_pick":  float(entry["overallRank"]),
            "pos_rank":  pos_counters[pos],
            "position":  pos,
        }

    try:
        _atomic_json_write(cache_path, result)
    except Exception:
        logger.warning("adp_service: failed to write startup ADP cache to %s", cache_path, exc_info=True)
    return result


def fetch_fc_rookie_adp(is_sf: bool, season: int) -> dict:
    """
    Fetch dynasty rookie ADP from FantasyCalc and return a map of
    sleeper_id -> {adp_rank, pos_rank, position}.
    Caches per league type per day.
    """
    import json as _json
    from datetime import date
    from utils.paths import DATA_DIR

    key = f"fc_rookie_adp_{'sf' if is_sf else '1qb'}_{date.today().isoformat()}.json"
    cache_path = DATA_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path) as _f:
                return _json.load(_f)
        except Exception:
            logger.warning("adp_service: corrupt rookie ADP cache at %s, rebuilding", cache_path)

    num_qbs = 2 if is_sf else 1
    url = f"https://fantasycalc.com/api/values/current?numQbs={num_qbs}&type=1&ppr=0.5"
    try:
        import requests as _req
        resp = _req.get(url, timeout=10, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        fc_data = resp.json()
    except Exception:
        logger.warning("adp_service: FantasyCalc fetch failed (sf=%s)", is_sf, exc_info=True)
        fc_data = []

    fc_by_sleeper: dict = {}
    for entry in (fc_data or []):
        p = entry.get("player") or {}
        sid = str(p.get("sleeperId") or "")
        if sid and sid != "None":
            fc_by_sleeper[sid] = {
                "overall_rank": entry.get("overallRank"),
                "pos_rank":     entry.get("positionalRank"),
                "position":     str(p.get("position") or "").upper(),
                "name":         p.get("name") or "",
            }

    result: dict = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as _conn:
            all_rows = _conn.execute(
                "SELECT sleeper_id, name, position FROM rookie_prospects "
                "WHERE draft_class_year = %s",
                (season,)
            ).fetchall()

        our_sids = {str(r["sleeper_id"]) for r in all_rows if r["sleeper_id"]}
        sid_matched = sorted(
            [(sid, fc_by_sleeper[sid]) for sid in our_sids if sid in fc_by_sleeper],
            key=lambda x: (x[1]["overall_rank"] or 9999)
        )
        for rookie_rank, (sid, info) in enumerate(sid_matched, start=1):
            result[sid] = {
                "adp_rank": rookie_rank,
                "fc_overall": info["overall_rank"],
                "pos_rank":   info["pos_rank"],
                "position":   info["position"],
            }

        our_names = {str(r["name"]).lower() for r in all_rows}
        name_matched = sorted(
            [entry for entry in (fc_data or [])
             if (entry.get("player") or {}).get("name", "").lower() in our_names],
            key=lambda e: (e.get("overallRank") or 9999)
        )
        for entry in name_matched:
            p = entry.get("player") or {}
            sid = str(p.get("sleeperId") or "")
            if not sid or sid in result:
                continue
            result[sid] = {
                "adp_rank": len(result) + 1,
                "fc_overall": entry.get("overallRank"),
                "pos_rank":   entry.get("positionalRank"),
                "position":   str(p.get("position") or "").upper(),
            }

        all_entries = sorted(result.items(), key=lambda kv: kv[1].get("fc_overall") or 9999)
        result = {sid: {**info, "adp_rank": rank}
                  for rank, (sid, info) in enumerate(all_entries, start=1)}

    except Exception:
        logger.exception("adp_service: rookie ADP matching failed (sf=%s, season=%s)", is_sf, season)

    try:
        _atomic_json_write(cache_path, result)
    except Exception:
        logger.warning("adp_service: failed to write rookie ADP cache to %s", cache_path, exc_info=True)
    return result


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
