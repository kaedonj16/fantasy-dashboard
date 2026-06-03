"""
External projection fetchers.

Preseason  — FantasyPros consensus season projections (week=draft)
In-season  — Sleeper weekly projections for the upcoming week

Both are disk-cached so the first hit per cache window does the network
work and all subsequent calls within the TTL are instant reads.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE_DIR   = Path(__file__).parent.parent / "cache"
_FP_BASE     = "https://www.fantasypros.com"
_SLEEPER_BASE = "https://api.sleeper.app/v1"

_FP_POSITIONS    = ["qb", "rb", "wr", "te", "k", "dst"]
_FP_SCORING_PARAM = {"ppr": "PPR", "half_ppr": "HALF", "std": "STD"}
_FP_CACHE_HOURS  = 6        # re-fetch FP projections every 6 h
_SLEEPER_CACHE_HOURS = 1    # re-fetch Sleeper weekly projections every hour

# FantasyPros team abbreviation → Sleeper abbreviation
_TEAM_MAP = {
    "GBP": "GB",  "KCC": "KC",  "NOR": "NO",  "TBB": "TB",
    "NEP": "NE",  "LVR": "LV",  "SFO": "SF",  "JAC": "JAX",
    "WSH": "WAS", "LAR": "LAR", "CLV": "CLE", "HST": "HOU",
    "ARZ": "ARI", "BLT": "BAL",
}

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.fantasypros.com/",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase ASCII, strip punctuation and common suffixes."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _cache_fresh(path: Path, hours: float) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < hours * 3600


def _build_name_index(players_index: dict) -> dict[str, dict[str, str]]:
    """
    Build {normalized_name: {TEAM: sleeper_id, "_any": sleeper_id}}.
    "_any" is a fallback when the team abbreviation doesn't match exactly.
    """
    idx: dict[str, dict[str, str]] = {}
    for sid, p in players_index.items():
        nm   = _normalize_name(str(p.get("name") or ""))
        team = str(p.get("team") or "").upper()
        if not nm:
            continue
        entry = idx.setdefault(nm, {})
        if team:
            entry[team] = sid
        entry["_any"] = sid
    return idx


def _lookup_sleeper_id(fp_name: str, fp_team: str, name_index: dict) -> Optional[str]:
    nm   = _normalize_name(fp_name)
    team = _TEAM_MAP.get(fp_team.upper(), fp_team.upper())
    entry = name_index.get(nm)
    if not entry:
        return None
    return entry.get(team) or entry.get("_any")


# ---------------------------------------------------------------------------
# FantasyPros preseason projections
# ---------------------------------------------------------------------------

def _fp_cache_path(year: int, scoring: str) -> Path:
    return _CACHE_DIR / f"fp_projections_{year}_{scoring}.json"


def _fetch_fp_position(pos: str, scoring: str) -> list[dict]:
    """
    Scrape one FantasyPros projection page.
    Returns [{"name": str, "team": str, "fpts": float}, ...]
    """
    scoring_param = _FP_SCORING_PARAM.get(scoring, "PPR")
    url = f"{_FP_BASE}/nfl/projections/{pos}.php"
    try:
        resp = _SESSION.get(url, params={"week": "draft", "scoring": scoring_param}, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[fp_proj] Fetch failed %s %s: %s", pos, scoring_param, exc)
        return []

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        tbl  = soup.find("table", id="data")
        if not tbl:
            logger.warning("[fp_proj] No #data table for %s", pos)
            return []

        headers = [th.get_text(strip=True).upper() for th in tbl.select("thead th")]
        # FPTS is the last column labelled FPTS / PTS; fall back to last column
        fpts_idx = next(
            (i for i, h in reversed(list(enumerate(headers))) if "FPTS" in h or h == "PTS"),
            len(headers) - 1,
        )

        players = []
        for tr in tbl.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            # Player name from first cell anchor
            name_el = tds[0].find("a", class_="player-name") or tds[0].find("a")
            if not name_el:
                continue
            player_name = name_el.get_text(strip=True)

            # Team abbreviation from <small> inside first cell
            small = tds[0].find("small")
            team  = ""
            if small:
                raw = small.get_text(strip=True)          # e.g. "ATL - RB" or "ATL"
                team = re.split(r"[\-–]|\s", raw)[0].strip()

            try:
                fpts = float(tds[fpts_idx].get_text(strip=True).replace(",", ""))
            except (ValueError, IndexError):
                continue

            if fpts > 0:
                players.append({"name": player_name, "team": team, "fpts": fpts})

        return players

    except Exception as exc:
        logger.warning("[fp_proj] Parse error for %s: %s", pos, exc)
        return []


def fetch_fp_season_projections(
    year: int,
    scoring: str = "ppr",
    players_index: Optional[dict] = None,
    force_refresh: bool = False,
) -> dict[str, dict]:
    """
    Fetch FantasyPros preseason season projections, mapped to Sleeper player IDs.

    Returns {sleeper_player_id: {"pos": str, "season_pts": float, "ppg": float}}
    Disk-cached for _FP_CACHE_HOURS hours.
    """
    cache_path = _fp_cache_path(year, scoring)
    if not force_refresh and _cache_fresh(cache_path, _FP_CACHE_HOURS):
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if data:
                logger.debug("[fp_proj] Returning %d cached players", len(data))
                return data
        except Exception:
            pass

    # Build Sleeper name index for ID mapping
    if players_index is None:
        try:
            from utils.utils import load_players_index
            players_index = load_players_index() or {}
        except Exception:
            players_index = {}

    name_index = _build_name_index(players_index)
    result: dict[str, dict] = {}
    unmatched = 0

    for pos in _FP_POSITIONS:
        rows = _fetch_fp_position(pos, scoring)
        for row in rows:
            sid = _lookup_sleeper_id(row["name"], row["team"], name_index)
            if not sid:
                unmatched += 1
                continue
            season_pts = row["fpts"]
            result[sid] = {
                "pos":        pos.upper() if pos != "dst" else "DEF",
                "season_pts": round(season_pts, 1),
                "ppg":        round(season_pts / 17.0, 2),
            }
        time.sleep(0.4)  # polite rate limit between position pages

    if result:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
        logger.info(
            "[fp_proj] Cached %d players for %d/%s (%d unmatched)",
            len(result), year, scoring, unmatched,
        )
    else:
        logger.warning("[fp_proj] No projections fetched for %d/%s", year, scoring)

    return result


# ---------------------------------------------------------------------------
# Sleeper weekly projections (in-season)
# ---------------------------------------------------------------------------

def _sleeper_proj_cache_path(year: int, week: int) -> Path:
    return _CACHE_DIR / f"sleeper_proj_{year}_w{week:02d}.json"


def _raw_to_pts(st: dict, pos: str, scoring: str) -> float:
    """Compute fantasy points for one player from Sleeper raw projected stats."""
    pass_yd  = float(st.get("pass_yd")  or 0)
    pass_td  = float(st.get("pass_td")  or 0)
    pass_int = float(st.get("pass_int") or 0)
    rush_yd  = float(st.get("rush_yd")  or 0)
    rush_td  = float(st.get("rush_td")  or 0)
    rec      = float(st.get("rec")      or 0)
    rec_yd   = float(st.get("rec_yd")   or 0)
    rec_td   = float(st.get("rec_td")   or 0)
    fum_lost = float(st.get("fum_lost") or 0)

    if (pass_yd + pass_td + rush_yd + rush_td + rec + rec_yd + rec_td) == 0:
        return 0.0

    base = (
        pass_yd  * 0.04  + pass_td  * 4.0  + pass_int * -2.0
        + rush_yd * 0.1  + rush_td  * 6.0
        + rec_yd  * 0.1  + rec_td   * 6.0
        + fum_lost * -2.0
    )
    rec_pts  = 1.0 if scoring == "ppr" else (0.5 if scoring == "half_ppr" else 0.0)
    tep_pts  = 0.5 if scoring == "tep" and pos == "TE" else 0.0
    td6_pts  = pass_td * 2.0 if scoring in ("6pt_ppr", "6pt_half", "6pt_tep") else 0.0
    return round(base + rec * rec_pts + rec * tep_pts + td6_pts, 2)


def fetch_sleeper_week_projections(
    year: int,
    week: int,
    scoring: str = "ppr",
    force_refresh: bool = False,
) -> dict[str, float]:
    """
    Fetch Sleeper weekly projections for a single week.
    Returns {sleeper_player_id: projected_pts_for_that_week}

    Sleeper's projections endpoint returns raw stat projections (pass_yd,
    pass_td, rec, ...) — fantasy points are computed here from those stats
    so the numbers match Sleeper's own scoring. Disk-cached 1 h.
    """
    cache_path = _sleeper_proj_cache_path(year, week)

    def _parse_raw(raw: dict) -> dict[str, float]:
        try:
            from utils.utils import load_players_index as _lpi
            players_index = _lpi() or {}
        except Exception:
            players_index = {}
        out: dict[str, float] = {}
        for pid, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            st  = entry.get("stats") if isinstance(entry.get("stats"), dict) else entry
            pos = players_index.get(str(pid), {}).get("pos", "")
            pts = _raw_to_pts(st, pos, scoring)
            if pts > 0:
                out[str(pid)] = pts
        return out

    if not force_refresh and _cache_fresh(cache_path, _SLEEPER_CACHE_HOURS):
        try:
            with open(cache_path) as f:
                raw = json.load(f)
            if raw:
                return _parse_raw(raw)
        except Exception:
            pass

    try:
        url  = f"{_SLEEPER_BASE}/projections/nfl/{year}/{week}"
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
        raw  = resp.json() or {}
    except Exception as exc:
        logger.warning("[sleeper_proj] Week %d/%d fetch failed: %s", year, week, exc)
        return {}

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(raw, f)

    result = _parse_raw(raw)
    logger.info("[sleeper_proj] %d player projections for %d/w%d (scoring=%s)",
                len(result), year, week, scoring)
    return result
