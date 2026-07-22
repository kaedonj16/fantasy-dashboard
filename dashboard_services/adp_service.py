"""Shared ADP fetching logic used by both the draft-grades endpoint and the prospects page."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


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
            logger.info("adp_service: FantasyCalc startup returned empty body (sf=%s) - skipping", is_sf)
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


def fetch_fc_redraft_adp(is_sf: bool) -> dict:
    """
    Fetch REDRAFT market ADP from FantasyCalc (isDynasty=false) for all players.

    Returns sleeper_id -> {adp_rank, pos_rank, position, avg_pick} where avg_pick
    is the redraft overall rank (1 = consensus #1 redraft pick). Cached per league
    type per day. Empty dict on any failure (callers fall back).
    """
    import json as _json
    from datetime import date
    from utils.paths import DATA_DIR

    key = f"fc_redraft_adp_{'sf' if is_sf else '1qb'}_{date.today().isoformat()}.json"
    cache_path = DATA_DIR / key
    if cache_path.exists():
        try:
            with open(cache_path) as _f:
                return _json.load(_f)
        except Exception:
            logger.warning("adp_service: corrupt redraft ADP cache at %s, rebuilding", cache_path)

    num_qbs = 2 if is_sf else 1
    url = f"https://fantasycalc.com/api/values/current?numQbs={num_qbs}&ppr=1&isDynasty=false"
    try:
        import requests as _req
        resp = _req.get(url, timeout=15, headers={"User-Agent": "fantasy-dashboard/1.0"})
        resp.raise_for_status()
        if not resp.text.strip():
            logger.info("adp_service: FantasyCalc redraft returned empty body (sf=%s)", is_sf)
            return {}
        fc_data = resp.json()
    except Exception as _exc:
        logger.warning("adp_service: FantasyCalc redraft fetch failed (sf=%s): %s", is_sf, _exc)
        return {}

    if not fc_data:
        return {}

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
        logger.warning("adp_service: failed to write redraft ADP cache to %s", cache_path, exc_info=True)
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
# only (it is a seasonal platform), so it is offered for redraft alone.
ADP_SOURCES = {
    "redraft": ("sleeper", "yahoo", "fc"),
    "dynasty": ("sleeper", "fc"),
    "rookie":  ("sleeper", "fc"),
}


def _adp_overall_from_row(row: dict, fields) -> Optional[float]:
    for f in fields:
        v = (row or {}).get(f)
        try:
            if v is not None and float(v) > 0:
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


def _fc_adp_source(season: int, is_sf: bool, scoring_type: str) -> Dict[str, float]:
    try:
        if scoring_type == "rookie":
            raw = fetch_fc_rookie_adp(is_sf, int(season)) or {}
        elif scoring_type == "dynasty":
            raw = fetch_fc_startup_adp(is_sf) or {}
        else:
            raw = fetch_fc_redraft_adp(is_sf) or {}
    except Exception:
        logger.debug("adp_service: FC source failed", exc_info=True)
        return {}
    out: Dict[str, float] = {}
    for pid, info in raw.items():
        try:
            ov = float((info or {}).get("avg_pick") or (info or {}).get("adp_rank") or 0)
        except (TypeError, ValueError):
            ov = 0.0
        if ov > 0:
            out[str(pid)] = ov
    return out


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


def resolve_market_adp(season: int, is_sf: bool, scoring_type: str = "redraft",
                       source: str = "consensus", league_id=None, token=None) -> Dict[str, float]:
    """canonical player_id -> overall market ADP for a scoring axis and source.

    scoring_type: ``redraft`` | ``dynasty`` | ``rookie``.
    source:       ``sleeper`` | ``yahoo`` | ``fc`` | ``consensus``. ``yahoo`` is
                  redraft-only; requesting it on another axis yields nothing and
                  the resolver falls back. Empty result means no source had data
                  (the caller can apply its own fallback, e.g. value rank)."""
    scoring_type = scoring_type if scoring_type in ("redraft", "dynasty", "rookie") else "redraft"
    valid = ADP_SOURCES.get(scoring_type, ("sleeper", "fc"))

    def _src(name: str) -> Dict[str, float]:
        if name == "sleeper":
            return _sleeper_adp_source(season, is_sf, scoring_type)
        if name == "fc":
            return _fc_adp_source(season, is_sf, scoring_type)
        if name == "yahoo":
            return _yahoo_adp_source(season, is_sf, scoring_type, league_id, token)
        return {}

    if source == "consensus":
        blended = consensus_adp([_src(n) for n in valid])
        if blended:
            return blended
    elif source in valid:
        got = _src(source)
        if got:
            return got
    # Fallback: sleeper, then fc.
    return _sleeper_adp_source(season, is_sf, scoring_type) or _fc_adp_source(season, is_sf, scoring_type)
