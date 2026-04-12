"""
Sportradar NCAAFB player stats source.

API chain:
  1. GET /teams/hierarchy.json            → all FBS team IDs
  2. GET /teams/{team_id}/roster.json     → player IDs per team
  3. GET /players/{player_id}/profile.json → per-season stats

Env vars:
    SPORTRADAR_API_KEY       – required (shared with the draft API key)
    SPORTRADAR_ACCESS_LEVEL  – default "trial"

Public API:
    build_sportradar_ncaa_index(names)
        → SportradarNCAAIndex  (passed to SportradarNCAAFBSource.__init__)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from data_building.paths import DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_SR_BASE    = "https://api.sportradar.com/ncaafb"
_SR_LANG    = "en"
_THROTTLE_S = 0.5   # trial tier: faster rate to reduce delays

_CACHE_ROOT = DATA_DIR / "cache" / "sportradar_ncaa"

_SUFFIX_RE = re.compile(r'\b(jr|sr|ii|iii|iv|v)\.?\s*$', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
    return _CACHE_ROOT / f"{safe}.json"


def _cache_read(key: str) -> Optional[Any]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _cache_write(key: str, data: Any) -> None:
    _cache_path(key).write_text(json.dumps(data))


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def _sr_get(path: str, retries: int = 2) -> Optional[Any]:
    """Rate-limited GET against the Sportradar NCAAFB v7 API."""
    api_key = os.getenv("SPORTRADAR_API_KEY", "")
    if not api_key:
        return None

    access = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")
    url = f"{_SR_BASE}/{access}/v7/{_SR_LANG}/{path}"
    headers = {"accept": "application/json", "x-api-key": api_key}

    time.sleep(_THROTTLE_S)

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=25)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1) * 2
                print(f"[sr_ncaa] rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (401, 403):
                print(
                    f"[sr_ncaa] HTTP {resp.status_code} for {path} — "
                    f"NCAAFB product may not be enabled for this API key "
                    f"(key prefix: {api_key[:8]}..., access={access}). "
                    f"The NFL Draft and NCAAFB APIs are separate Sportradar products."
                )
                return None
            print(f"[sr_ncaa] HTTP {resp.status_code} for {path} — body: {resp.text[:200]}")
            return None
        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"[sr_ncaa] timeout attempt {attempt+1}/{retries} — retry in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            print(f"[sr_ncaa] error {path}: {exc}")
            break

    print(f"[sr_ncaa] FAILED after {retries} attempts: {path}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Team hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_teams() -> List[Dict]:
    """Return [{id, name, market, alias, conference}] for all FBS teams."""
    cached = _cache_read("teams_hierarchy")
    if cached is not None:
        print(f"[sr_ncaa] Teams loaded from cache: {len(cached)} teams")
        return cached

    data = _sr_get("league/hierarchy.json")
    if not data:
        print("[sr_ncaa] No data returned from API")
        return []

    divisions = data.get("divisions", [])
    if not divisions:
        print("[sr_ncaa] No divisions found in API response")
        return []

    teams: List[Dict] = []

    for division in data.get("divisions", []):
        div_name = division.get("name", "")
        # Only process I-A (FBS) and I-AA (FCS) divisions
        if div_name not in ["I-A", "I-AA"]:
            continue
            
        # Handle both structures: conferences directly under division OR under subdivisions
        conferences = []
        if division.get("conferences"):
            # Conferences are directly under division (like I-A)
            conferences = division.get("conferences", [])
        elif division.get("subdivisions"):
            # Conferences are under subdivisions (like I-AA)
            for subdivision in division.get("subdivisions", []):
                conferences.extend(subdivision.get("conferences", []))
        
        for conf in conferences:
            conf_name = conf.get("name", "")
            for team in conf.get("teams", []):
                tid = team.get("id")
                if tid:
                    teams.append({
                        "id":         tid,
                        "name":       team.get("name", ""),
                        "market":     team.get("market", ""),
                        "alias":      team.get("alias", ""),
                        "conference": conf_name,
                    })


    print(f"[sr_ncaa] teams_hierarchy: {len(teams)} teams")
    _cache_write("teams_hierarchy", teams)
    return teams


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Team rosters → name → player_id index
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_team_roster(team_id: str) -> List[Dict]:
    """Return [{id, name, position}] for one team's roster."""
    cached = _cache_read(f"roster_{team_id}")
    if cached is not None:
        return cached

    data = _sr_get(f"teams/{team_id}/full_roster.json")
    if not data:
        return []

    team_node = data.get("team", data)
    players = [
        {"id": p.get("id"), "name": p.get("name", ""), "position": p.get("position", "")}
        for p in team_node.get("players", [])
        if p.get("id")
    ]
    _cache_write(f"roster_{team_id}", players)
    return players


def _normalize_name(name: str) -> str:
    n = name.strip().lower()
    n = _SUFFIX_RE.sub("", n).strip()
    n = re.sub(r"['\u2019\u2018]", "", n)
    n = re.sub(r"\.", "", n)
    return n.strip()


def build_roster_index() -> Dict[str, str]:
    """
    Scan all FBS team rosters and return {name_lower → sportradar_player_id}.

    The full scan hits ~130 endpoints; result is cached so it only runs once.
    """
    # cached = _cache_read("roster_index")
    # if cached is not None:
    #     print(f"[sr_ncaa] Roster index loaded from cache ({len(cached)} players)")
    #     return cached

    teams = _fetch_teams()
    if not teams:

        print(f"[sr_ncaa] No teams found during _fetch_teams()")
        return {}
    
    index: Dict[str, str] = {}
    
    for i, team in enumerate(teams):
        tid = team["id"]
        roster = _fetch_team_roster(tid)
        if not roster:
            continue

        for player in roster:
            raw_name = player.get("name", "")
            pid = player.get("id")
            if not raw_name or not pid:
                continue

            key = _normalize_name(raw_name)
            index[key] = pid

            # If Sportradar uses "Last, First" format, also index "First Last"
            if "," in raw_name:
                parts = [p.strip() for p in raw_name.split(",", 1)]
                alt = _normalize_name(f"{parts[1]} {parts[0]}")
                index[alt] = pid

        if (i + 1) % 25 == 0:
            print(f"[sr_ncaa] Roster scan {i+1}/{len(teams)} — {len(index)} players indexed")

    print(f"[sr_ncaa] Roster index complete: {len(index)} players")
    _cache_write("roster_index", index)
    return index


def lookup_player_id(name: str, index: Dict[str, str]) -> Optional[str]:
    """Match a prospect name to a Sportradar player_id."""
    key = _normalize_name(name)
    if key in index:
        return index[key]

    # Try without middle name: "A.J. Brown" → "aj brown"
    parts = key.split()
    if len(parts) > 2:
        short = f"{parts[0]} {parts[-1]}"
        if short in index:
            return index[short]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Player profile
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_player_profile(player_id: str) -> Optional[Dict]:
    cached = _cache_read(f"profile_{player_id}")
    if cached is not None:
        return cached

    data = _sr_get(f"players/{player_id}/profile.json")
    if data:
        _cache_write(f"profile_{player_id}", data)
    return data


def normalize_profile(raw: Dict) -> Dict[int, Dict[str, Any]]:
    """
    Convert a Sportradar player profile to {year: normalized_stats_dict}.

    Output fields match the season_record schema used by ProspectSeasonStatsSource
    and the derivation functions, so callers can merge this dict over season_record
    to give derivations access to real target counts.
    """
    result: Dict[int, Dict[str, Any]] = {}

    for season in raw.get("seasons", []):
        if season.get("type") != "REG":
            continue
        year = season.get("year")
        if not year:
            continue

        teams = season.get("teams", [])
        if not teams:
            continue
        # If a player transferred, sum stats across teams for that year
        rush_att = rush_yds = rush_tds = 0
        rec_rec = rec_yds = rec_tds = 0
        rec_yac_total = 0
        rec_tgt_total = 0
        pass_att = pass_yds = pass_tds = completions = interceptions = 0
        games_played = 0
        games_started = 0
        has_tgt = False
        has_yac = False

        for team_entry in teams:
            s = team_entry.get("statistics", {})
            rush = s.get("rushing", {})
            recv = s.get("receiving", {})
            pass_ = s.get("passing", {})

            rush_att  += rush.get("attempts", 0) or 0
            rush_yds  += rush.get("yards", 0) or 0
            rush_tds  += rush.get("touchdowns", 0) or 0

            rec_rec += recv.get("receptions", 0) or 0
            rec_yds += recv.get("yards", 0) or 0
            rec_tds += recv.get("touchdowns", 0) or 0

            tgt = recv.get("targets")
            if tgt is not None:
                rec_tgt_total += tgt
                has_tgt = True

            yac = recv.get("yards_after_catch")
            if yac is not None:
                rec_yac_total += yac
                has_yac = True

            pass_att      += pass_.get("attempts", 0) or 0
            pass_yds      += pass_.get("yards", 0) or 0
            pass_tds      += pass_.get("touchdowns", 0) or 0
            completions   += pass_.get("completions", 0) or 0
            interceptions += pass_.get("interceptions", 0) or 0

            games_played  += s.get("games_played", 0) or 0
            games_started += s.get("games_started", 0) or 0

        row: Dict[str, Any] = {
            "season":          year,
            "games_played":    games_played or None,
            "games_started":   games_started or None,
            # Rushing
            "rush_attempts":   rush_att,
            "rush_yards":      rush_yds,
            "rush_tds":        rush_tds,
            "yds_per_carry":   round(rush_yds / rush_att, 3) if rush_att > 0 else None,
            # Receiving
            "receptions":      rec_rec,
            "targets":         rec_tgt_total if has_tgt else None,
            "receiving_yards": rec_yds,
            "receiving_tds":   rec_tds,
            "receiving_yac":   rec_yac_total if has_yac else None,
            "yds_per_reception": round(rec_yds / rec_rec, 3) if rec_rec > 0 else None,
            # Passing
            "pass_attempts":   pass_att,
            "pass_yards":      pass_yds,
            "pass_tds":        pass_tds,
            "completions":     completions,
            "interceptions":   interceptions,
            "completion_pct":  round(completions / pass_att * 100, 1) if pass_att > 0 else None,
            "td_int_ratio":    round(pass_tds / max(interceptions, 1), 2) if pass_tds else None,
            "yds_per_attempt": round(pass_yds / pass_att, 2) if pass_att > 0 else None,
        }
        result[year] = row

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public batch entry point
# ─────────────────────────────────────────────────────────────────────────────

class SportradarNCAAIndex:
    """
    Pre-built lookup table of {name_lower → {year → normalized_stats}}.

    Constructed once per pipeline run by build_sportradar_ncaa_index() and
    passed into SportradarNCAAFBSource.
    """

    def __init__(
        self,
        data: Dict[str, Dict[int, Dict[str, Any]]],
        bio: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._data = data
        self._bio = bio or {}

    def get_season_stats(self, name: str, year: int) -> Optional[Dict[str, Any]]:
        return self._data.get(_normalize_name(name), {}).get(year)

    def get_bio(self, name: str) -> Optional[Dict[str, Any]]:
        """Return {height_inches, weight_lbs} from the player's Sportradar profile, or None."""
        return self._bio.get(_normalize_name(name))

    def get_all_seasons(self, name: str) -> Dict[int, Dict[str, Any]]:
        """Return all available {year: season_record} dicts for this player, or {}."""
        return self._data.get(_normalize_name(name), {})

    def __len__(self) -> int:
        return len(self._data)


def build_sportradar_ncaa_index(names: List[str]) -> SportradarNCAAIndex:
    """
    Fetch Sportradar NCAAFB stats for a list of prospect names.

    Performs the full 3-step chain (teams → rosters → profiles) with disk
    caching at each step.  Returns a SportradarNCAAIndex for use by
    SportradarNCAAFBSource.
    """
    api_key = os.getenv("SPORTRADAR_API_KEY", "")
    if not api_key:
        print("[sr_ncaa] SPORTRADAR_API_KEY not set — Sportradar NCAAFB stats skipped")
        return SportradarNCAAIndex({})

    access = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")
    print(f"[sr_ncaa] Starting index build: key={api_key[:8]}... access={access} prospects={len(names)}")

    roster_index = build_roster_index()
    if not roster_index:
        print("[sr_ncaa] Roster index empty — Sportradar NCAAFB unavailable")
        return SportradarNCAAIndex({})

    data: Dict[str, Dict[int, Dict[str, Any]]] = {}
    bio: Dict[str, Dict[str, Any]] = {}
    found = not_found = 0

    for name in names:
        player_id = lookup_player_id(name, roster_index)
        if not player_id:
            not_found += 1
            continue

        raw = _fetch_player_profile(player_id)
        if not raw:
            print(f"[sr_ncaa] profile_failed name={name!r} id={player_id}")
            not_found += 1
            continue

        seasons = normalize_profile(raw)
        data[_normalize_name(name)] = seasons

        h = raw.get("height")
        w = raw.get("weight")
        if h is not None or w is not None:
            bio[_normalize_name(name)] = {
                "height_inches": int(h) if h is not None else None,
                "weight_lbs":    int(w) if w is not None else None,
            }

        found += 1

    print(f"[sr_ncaa] index built: {found} found, {not_found} not found")
    return SportradarNCAAIndex(data, bio)
