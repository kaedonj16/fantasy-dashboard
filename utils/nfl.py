"""Consolidated utils module: nfl.

NFL team data, stadiums, defense-vs-position, schedules

Merged from: utils/nfl_teams.py, utils/nfl_stadiums.py, utils/nfl_context.py, utils/defense_vs_position.py, utils/defensive_matchup_ratings.py, utils/team_offense_ranks.py, utils/schedule_ease.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING


# ======================================================================
# From utils/nfl_teams.py
# ======================================================================

"""NFL team abbreviation helpers.

Extracted from app.py. The abbreviation->name map is hoisted to module level so
it is built once at import instead of on every call, and the mapping can be
unit-tested without the pandas/DB stack.
"""

# Abbreviation -> full team name. WSH is a lookup alias only; the site
# displays and stores Washington as WAS.
TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
}


def get_team_full_name(abbreviation: str) -> str:
    """Map a team abbreviation to its full team name.

    Case-insensitive. Unknown abbreviations pass through unchanged.
    WSH resolves to the Commanders the same as WAS.
    """
    return TEAM_FULL_NAMES.get(str(abbreviation).upper(), abbreviation)


# ======================================================================
# From utils/nfl_stadiums.py
# ======================================================================

"""Static NFL stadium / game-environment metadata.

Fantasy output is meaningfully shaped by where a game is played: indoor
(dome or fixed/closed roof) games are weather-proof and slightly friendlier to
passing and kicking, while late-season games at cold-weather outdoor sites carry
real downside for passers, receivers, and kickers. Those are *structural* facts
about the venue - they need no live feed, so this module ships them offline.

We deliberately do NOT fabricate live weather, wind, or betting totals here: a
"Dome" / "Cold" tag we can always stand behind beats a wind speed we'd have to
guess. If a real weather or odds source is wired up later, ``game_environment``
is the single place to enrich the returned tag.

``dome`` covers true domes and retractable/fixed roofs that play climate-
controlled in practice (ATL, DAL, HOU, ARI, IND, plus SoFi's fixed canopy).
``climate`` is the outdoor-weather profile used for the late-season cold flag:
"dome" (n/a), "cold", "mild", or "warm".
"""


# team abbr -> stadium name, dome?, outdoor climate profile, and lat/lon (for
# weather lookups; domes carry coords too but weather is skipped for them).
STADIUMS: dict[str, dict] = {
    "ARI": {"name": "State Farm Stadium", "dome": True,  "climate": "dome", "lat": 33.5276, "lon": -112.2626},
    "ATL": {"name": "Mercedes-Benz Stadium", "dome": True, "climate": "dome", "lat": 33.7554, "lon": -84.4009},
    "BAL": {"name": "M&T Bank Stadium", "dome": False, "climate": "cold", "lat": 39.2780, "lon": -76.6227},
    "BUF": {"name": "Highmark Stadium", "dome": False, "climate": "cold", "lat": 42.7738, "lon": -78.7870},
    "CAR": {"name": "Bank of America Stadium", "dome": False, "climate": "mild", "lat": 35.2258, "lon": -80.8528},
    "CHI": {"name": "Soldier Field", "dome": False, "climate": "cold", "lat": 41.8623, "lon": -87.6167},
    "CIN": {"name": "Paycor Stadium", "dome": False, "climate": "cold", "lat": 39.0955, "lon": -84.5161},
    "CLE": {"name": "Huntington Bank Field", "dome": False, "climate": "cold", "lat": 41.5061, "lon": -81.6995},
    "DAL": {"name": "AT&T Stadium", "dome": True, "climate": "dome", "lat": 32.7473, "lon": -97.0945},
    "DEN": {"name": "Empower Field", "dome": False, "climate": "cold", "lat": 39.7439, "lon": -105.0201},
    "DET": {"name": "Ford Field", "dome": True, "climate": "dome", "lat": 42.3400, "lon": -83.0456},
    "GB":  {"name": "Lambeau Field", "dome": False, "climate": "cold", "lat": 44.5013, "lon": -88.0622},
    "HOU": {"name": "NRG Stadium", "dome": True, "climate": "dome", "lat": 29.6847, "lon": -95.4107},
    "IND": {"name": "Lucas Oil Stadium", "dome": True, "climate": "dome", "lat": 39.7601, "lon": -86.1639},
    "JAX": {"name": "EverBank Stadium", "dome": False, "climate": "warm", "lat": 30.3239, "lon": -81.6373},
    "KC":  {"name": "Arrowhead Stadium", "dome": False, "climate": "cold", "lat": 39.0489, "lon": -94.4839},
    "LV":  {"name": "Allegiant Stadium", "dome": True, "climate": "dome", "lat": 36.0909, "lon": -115.1833},
    "LAC": {"name": "SoFi Stadium", "dome": True, "climate": "dome", "lat": 33.9535, "lon": -118.3392},
    "LAR": {"name": "SoFi Stadium", "dome": True, "climate": "dome", "lat": 33.9535, "lon": -118.3392},
    "MIA": {"name": "Hard Rock Stadium", "dome": False, "climate": "warm", "lat": 25.9580, "lon": -80.2389},
    "MIN": {"name": "U.S. Bank Stadium", "dome": True, "climate": "dome", "lat": 44.9736, "lon": -93.2575},
    "NE":  {"name": "Gillette Stadium", "dome": False, "climate": "cold", "lat": 42.0909, "lon": -71.2643},
    "NO":  {"name": "Caesars Superdome", "dome": True, "climate": "dome", "lat": 29.9511, "lon": -90.0812},
    "NYG": {"name": "MetLife Stadium", "dome": False, "climate": "cold", "lat": 40.8135, "lon": -74.0745},
    "NYJ": {"name": "MetLife Stadium", "dome": False, "climate": "cold", "lat": 40.8135, "lon": -74.0745},
    "PHI": {"name": "Lincoln Financial Field", "dome": False, "climate": "cold", "lat": 39.9008, "lon": -75.1675},
    "PIT": {"name": "Acrisure Stadium", "dome": False, "climate": "cold", "lat": 40.4468, "lon": -80.0158},
    "SEA": {"name": "Lumen Field", "dome": False, "climate": "mild", "lat": 47.5952, "lon": -122.3316},
    "SF":  {"name": "Levi's Stadium", "dome": False, "climate": "mild", "lat": 37.4030, "lon": -121.9700},
    "TB":  {"name": "Raymond James Stadium", "dome": False, "climate": "warm", "lat": 27.9759, "lon": -82.5033},
    "TEN": {"name": "Nissan Stadium", "dome": False, "climate": "mild", "lat": 36.1665, "lon": -86.7713},
    "WAS": {"name": "Northwest Stadium", "dome": False, "climate": "cold", "lat": 38.9076, "lon": -76.8645},
}

# Common alternate abbreviations seen across Sleeper / Tank01 / ESPN feeds.
ALIASES: dict[str, str] = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV",
    "WSH": "WAS", "WFT": "WAS", "LVR": "LV", "SFO": "SF", "TAM": "TB",
    "GNB": "GB", "GBP": "GB", "KAN": "KC", "KCC": "KC", "NWE": "NE",
    "NEP": "NE", "NOR": "NO", "TBB": "TB",
}

_FULL_NAMES: dict[str, str] = {
    "ARIZONACARDINALS": "ARI", "ATLANTAFALCONS": "ATL", "BALTIMORERAVENS": "BAL",
    "BUFFALOBILLS": "BUF", "CAROLINAPANTHERS": "CAR", "CHICAGOBEARS": "CHI",
    "CINCINNATIBENGALS": "CIN", "CLEVELANDBROWNS": "CLE", "DALLASCOWBOYS": "DAL",
    "DENVERBRONCOS": "DEN", "DETROITLIONS": "DET", "GREENBAYPACKERS": "GB",
    "HOUSTONTEXANS": "HOU", "INDIANAPOLISCOLTS": "IND", "JACKSONVILLEJAGUARS": "JAX",
    "KANSASCITYCHIEFS": "KC", "LASVEGASRAIDERS": "LV", "OAKLANDRAIDERS": "LV",
    "LOSANGELESCHARGERS": "LAC", "SANDIEGOCHARGERS": "LAC",
    "LOSANGELESRAMS": "LAR", "STLOUISRAMS": "LAR", "MIAMIDOLPHINS": "MIA",
    "MINNESOTAVIKINGS": "MIN", "NEWENGLANDPATRIOTS": "NE", "NEWORLEANSSAINTS": "NO",
    "NEWYORKGIANTS": "NYG", "NEWYORKJETS": "NYJ", "PHILADELPHIAEAGLES": "PHI",
    "PITTSBURGHSTEELERS": "PIT", "SEATTLESEAHAWKS": "SEA", "SANFRANCISCO49ERS": "SF",
    "TAMPABAYBUCCANEERS": "TB", "TENNESSEETITANS": "TEN",
    "WASHINGTONCOMMANDERS": "WAS", "WASHINGTONFOOTBALLTEAM": "WAS",
    "WASHINGTONREDSKINS": "WAS",
}

# v2 provider team objects may expose a shortName/nickname instead of an
# abbreviation. NFL nicknames are unique, so these remain unambiguous; city-only
# "New York"/"Los Angeles" values are intentionally not accepted.
_NICKNAMES: dict[str, str] = {
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "CHIEFS": "KC",
    "RAIDERS": "LV", "CHARGERS": "LAC", "RAMS": "LAR", "DOLPHINS": "MIA",
    "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO", "GIANTS": "NYG",
    "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT", "SEAHAWKS": "SEA",
    "49ERS": "SF", "BUCCANEERS": "TB", "TITANS": "TEN", "COMMANDERS": "WAS",
}

# NFL weeks from ~mid-December on, when cold-weather sites actually play cold.
_COLD_WEEK_START = 14


def normalize_team(team: str) -> str:
    """Uppercase and de-alias an NFL team abbreviation."""
    t = str(team or "").strip().upper()
    return ALIASES.get(t, t)


def normalize_nfl_team(team: str) -> str:
    """Return the site's canonical NFL abbreviation, or ``""`` if unknown.

    Unlike the older permissive ``normalize_team()``, this strict boundary is
    safe for joins: an opaque provider ID cannot accidentally become a new team.
    """
    token = "".join(ch for ch in str(team or "").upper() if ch.isalnum())
    canonical = ALIASES.get(token, token)
    if canonical in STADIUMS:
        return canonical
    return _FULL_NAMES.get(token) or _NICKNAMES.get(token, "")


def stadium_coords(team: str) -> Optional[tuple]:
    """(lat, lon) for a team's home stadium, or None for an unknown team."""
    st = STADIUMS.get(normalize_team(team))
    if not st:
        return None
    return (st["lat"], st["lon"])


def game_environment(home_team: str, week: Optional[int] = None) -> Optional[dict]:
    """Environment tag for a game hosted by ``home_team``.

    Returns ``None`` for unknown teams (e.g. a bye or bad abbr). Otherwise a
    dict: ``env`` ("dome"/"outdoor"), ``label``, ``dome`` (bool), ``cold``
    (bool - only for cold-climate outdoor sites in the late-season window),
    ``stadium``, and a short human ``note``. Warm/mild outdoor games return a
    tag with no ``cold`` flag so the UI can leave them unmarked.
    """
    st = STADIUMS.get(normalize_team(home_team))
    if not st:
        return None
    if st["dome"]:
        return {
            "env": "dome", "label": "Dome", "dome": True, "cold": False,
            "stadium": st["name"], "note": "Indoor - weather-proof",
        }
    cold = st["climate"] == "cold" and week is not None and int(week) >= _COLD_WEEK_START
    return {
        "env": "outdoor", "label": "Outdoor", "dome": False, "cold": bool(cold),
        "stadium": st["name"],
        "note": "Cold-weather site, late season" if cold else "Outdoor",
    }


# ======================================================================
# From utils/nfl_context.py
# ======================================================================

"""Authoritative NFL season/week context and cache dimensions.

The provider state is authoritative.  Calendar inference exists only as an
observable availability fallback; importantly, January and February still
belong to the season which began in the previous calendar year.
"""

from typing import Mapping


VALID_PHASES = frozenset({"off", "pre", "reg", "post"})


def calendar_nfl_season(on_date: Optional[date] = None) -> int:
    """Return the NFL season containing *on_date* when provider state is absent."""
    day = on_date or datetime.now(timezone.utc).date()
    return day.year - 1 if day.month <= 2 else day.year


def _calendar_phase(day: date) -> str:
    if day.month <= 2:
        return "post"
    if day.month <= 7:
        return "off"
    if day.month == 8:
        return "pre"
    return "reg"


def normalize_nfl_state(
    state: Optional[Mapping[str, Any]], *, on_date: Optional[date] = None,
    provider: str = "sleeper",
) -> dict:
    """Normalize provider state and attach lightweight freshness provenance."""
    raw = dict(state or {})
    day = on_date or datetime.now(timezone.utc).date()
    try:
        season = int(raw.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    fallback_reason = None
    if season < 2000:
        season = calendar_nfl_season(day)
        fallback_reason = "provider season unavailable"
    try:
        week = max(0, int(raw.get("week") or raw.get("display_week") or 0))
    except (TypeError, ValueError):
        week = 0
    phase = str(raw.get("season_type") or "").strip().lower()
    if phase not in VALID_PHASES:
        phase = _calendar_phase(day)
        fallback_reason = fallback_reason or "provider phase unavailable"
    raw.update({"season": season, "week": week, "season_type": phase})
    raw["freshness"] = {
        "source_season": season,
        "source_week": week,
        "provider": provider,
        "classification": "live" if fallback_reason is None else "fallback",
        "fallback_reason": fallback_reason,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    return raw


def nfl_state_is_stale(state: Optional[Mapping[str, Any]]) -> bool:
    """True when get_nfl_state() served a last-good fallback after a failed fetch.

    Fresh successful fetches never set the flag; only the stale-fallback path
    in dashboard_services.api does. Consumers (nav chrome, API payloads) can
    branch on this instead of trusting the week blindly.
    """
    fresh = (state or {}).get("freshness")
    return bool(isinstance(fresh, dict) and fresh.get("stale"))


def nfl_state_last_good_at(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    """ISO timestamp of the last successful NFL state fetch, if the state is stale."""
    fresh = (state or {}).get("freshness")
    if isinstance(fresh, dict):
        return fresh.get("last_good_at")
    return None


def season_cache_key(namespace: str, *, season: int, week: Optional[int] = None,
                     league_id: Optional[str] = None, scoring: Optional[str] = None,
                     provider: Optional[str] = None) -> str:
    """Build a stable key which cannot accidentally cross season boundaries."""
    parts = [namespace, f"s{int(season)}"]
    if week is not None:
        parts.append(f"w{int(week)}")
    if provider:
        parts.append(f"p:{provider}")
    if league_id:
        parts.append(f"l:{league_id}")
    if scoring:
        parts.append(f"sc:{scoring}")
    return "|".join(parts)


def current_sample_weight(games: int, *, full_weight_at: int = 6) -> float:
    """Intentional early-season blend weight for observed current-year data."""
    return min(1.0, max(0.0, float(games) / max(1, int(full_weight_at))))


# ======================================================================
# From utils/defense_vs_position.py
# ======================================================================

"""Defense-vs-position matchup stats from Sleeper weekly player stats.

Simple, legible per-team numbers: how many fantasy points (and how many
yards per opportunity) each NFL defense allows to each skill position,
ranked 1-32 where 1 = most allowed = easiest matchup (the same convention
as ``utils/schedule_ease.py``'s ``sched_rank_color``).

This is intentionally NOT the opponent-adjusted multiplier system in
``utils/defensive_matchup_ratings.py`` (which feeds the Schedule
Assistant). That system answers "how much better/worse than expectation";
this one answers "how many points does this defense give up to WRs", which
is what start/sit matchup chips and the player modal show.

Data sources (all injected, so unit tests never touch the network):

- ``cache/sleeper_stats/sleeper_stats_s{season}_w{N}.json``: Sleeper weekly
  player stats. Rows are keyed by player id and carry precomputed
  ``pts_ppr`` / ``pts_half_ppr`` / ``pts_std`` plus raw categories
  (``rec_tgt``, ``rec_yd``, ``rush_att``, ``rush_yd``, ``pass_att``,
  ``pass_yd``). All three scoring formats are served because Sleeper
  precomputes them; no rescoring needed.
- nflverse-style schedule rows (``season``/``week``/``game_type``/
  ``home_team``/``away_team``/``home_score``/``away_score``) for opponent
  mapping. Only games with both final scores count, attributed per team,
  so a Thursday final lands on Friday while Sunday's games wait.

Known limitations, stated plainly:

- Players are mapped to teams via the current players index, so a player
  traded mid-season has all of his weeks attributed to his current team
  (same treatment as the existing points-allowed path).
- Defenses with no completed games yet (preseason, or a team on an early
  bye with no finals) are omitted entirely rather than ranked on nothing.

Stdlib only.
"""


import hashlib

POSITIONS = ("QB", "RB", "WR", "TE")

#: Regular-season weeks only. Postseason is excluded from these ranks.
REG_WEEKS = range(1, 19)

#: Efficiency stat shown per position, from categories Sleeper reports.
EFF_LABELS = {
    "QB": "yards per attempt",
    "RB": "yards per carry",
    "WR": "yards per target",
    "TE": "yards per target",
}





def _as_float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _row_scores(row: dict):
    """(home_points, away_points) as floats, or (None, None) when unscored."""
    try:
        home = float(str(row.get("home_score") or "").strip())
        away = float(str(row.get("away_score") or "").strip())
    except (TypeError, ValueError):
        return None, None
    return home, away


def completed_defense_games(schedule_rows, season: int) -> list:
    """One entry per team per completed regular-season game.

    Returns ``[{"team": "DAL", "week": 3, "opponent": "GB"}, ...]`` where
    ``team`` is the defense and ``opponent`` is the offense whose players'
    stats count as allowed by that defense. Only games with both final
    scores count; attribution is per team, so a Thursday final is usable
    on Friday while the rest of the week is still pending.
    """
    season = int(season)
    out = []
    for row in schedule_rows or []:
        if not isinstance(row, dict):
            continue
        try:
            rseason = int(float(str(row.get("season") or 0)))
            week = int(float(str(row.get("week") or 0)))
        except (TypeError, ValueError):
            continue
        if rseason != season or week not in REG_WEEKS:
            continue
        game_type = str(row.get("game_type") or "REG").upper()
        if game_type != "REG":
            continue
        home_score, away_score = _row_scores(row)
        if home_score is None or away_score is None:
            continue
        home = canon_team(row.get("home_team"))
        away = canon_team(row.get("away_team"))
        if not home or not away:
            continue
        out.append({"team": home, "week": week, "opponent": away})
        out.append({"team": away, "week": week, "opponent": home})
    return out


def table_fingerprint(completed_games) -> str:
    """Stable id for the set of completed games; changes when one goes final."""
    parts = sorted(
        f"{g.get('team')}:{int(g.get('week') or 0)}:{g.get('opponent')}"
        for g in completed_games or []
        if isinstance(g, dict)
    )
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _blank_pos_totals() -> dict:
    return {
        "fpts_ppr": 0.0,
        "fpts_half_ppr": 0.0,
        "fpts_std": 0.0,
        "rec_tgt": 0.0,
        "rec_yd": 0.0,
        "rush_att": 0.0,
        "rush_yd": 0.0,
        "pass_att": 0.0,
        "pass_yd": 0.0,
    }


def aggregate_defense_stats(completed_games, get_week_stats, players_index):
    """Sum allowed stats per (defense, position) over completed games.

    ``get_week_stats(week)`` returns ``{player_id: stat_row}`` for that
    week (``{}`` when the file is missing). ``players_index`` maps
    ``str(player_id)`` to ``{"team": ..., "pos": ...}``.

    Returns ``(allowed, games)`` where ``allowed[def][pos]`` is the totals
    dict and ``games[def]`` is the number of completed games.
    """
    allowed: dict = {}
    games: dict = {}
    week_cache: dict = {}
    for game in completed_games or []:
        if not isinstance(game, dict):
            continue
        defense = canon_team(game.get("team"))
        offense = canon_team(game.get("opponent"))
        try:
            week = int(game.get("week") or 0)
        except (TypeError, ValueError):
            continue
        if not defense or not offense or week < 1:
            continue
        games[defense] = games.get(defense, 0) + 1
        if week not in week_cache:
            try:
                week_cache[week] = get_week_stats(week) or {}
            except Exception:
                week_cache[week] = {}
        stats = week_cache[week]
        if not isinstance(stats, dict):
            continue
        bucket = allowed.setdefault(defense, {})
        for pid, row in stats.items():
            if not isinstance(row, dict):
                continue
            if str(pid).upper().startswith("TEAM_"):
                continue
            info = (players_index or {}).get(str(pid)) or {}
            if canon_team(info.get("team")) != offense:
                continue
            pos = str(info.get("pos") or "").upper()
            if pos not in POSITIONS:
                continue
            totals = bucket.setdefault(pos, _blank_pos_totals())
            totals["fpts_ppr"] += _as_float(row.get("pts_ppr"))
            totals["fpts_half_ppr"] += _as_float(row.get("pts_half_ppr"))
            totals["fpts_std"] += _as_float(row.get("pts_std"))
            totals["rec_tgt"] += _as_float(row.get("rec_tgt"))
            totals["rec_yd"] += _as_float(row.get("rec_yd"))
            totals["rush_att"] += _as_float(row.get("rush_att"))
            totals["rush_yd"] += _as_float(row.get("rush_yd"))
            totals["pass_att"] += _as_float(row.get("pass_att"))
            totals["pass_yd"] += _as_float(row.get("pass_yd"))
    return allowed, games


def _efficiency(pos: str, totals: dict):
    """Yards per opportunity allowed; None when there were no opportunities."""
    if pos in ("WR", "TE"):
        denom = totals["rec_tgt"]
        return (totals["rec_yd"] / denom) if denom > 0 else None
    if pos == "RB":
        denom = totals["rush_att"]
        return (totals["rush_yd"] / denom) if denom > 0 else None
    denom = totals["pass_att"]  # QB
    return (totals["pass_yd"] / denom) if denom > 0 else None


def _competition_ranks(values: dict) -> dict:
    """Rank 1 = highest value (most allowed = easiest). Ties share a rank
    and the next rank skips (1, 2, 2, 4)."""
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks = {}
    last_value = None
    last_rank = 0
    for i, (team, value) in enumerate(ordered):
        if last_value is None or value != last_value:
            last_rank = i + 1
            last_value = value
        ranks[team] = last_rank
    return ranks


def build_defense_vs_position(season, schedule_rows, get_week_stats, players_index) -> dict:
    """Full per-team per-position table for one season.

    Shape::

        {
          "season": 2026,
          "completed_games": 48,
          "fingerprint": "abc123...",
          "teams": {
            "DAL": {
              "games": 3,
              "QB": {"fpts_ppr_pg": 18.2, "fpts_half_ppr_pg": 17.9,
                     "fpts_std_pg": 17.5, "eff": 7.1,
                     "eff_label": "yards per attempt",
                     "rank": 8, "total": 32},
              "RB": {...}, "WR": {...}, "TE": {...},
            }, ...
          }
        }

    Teams with no completed games are omitted. Ranks use competition
    ranking with 1 = most fantasy points allowed = easiest matchup.
    """
    season = int(season)
    completed = completed_defense_games(schedule_rows, season)
    allowed, games = aggregate_defense_stats(completed, get_week_stats, players_index)

    per_game: dict = {}
    for defense, pos_map in allowed.items():
        n = max(int(games.get(defense) or 0), 1)
        for pos in POSITIONS:
            totals = pos_map.get(pos) or _blank_pos_totals()
            per_game.setdefault(defense, {})[pos] = {
                "fpts_ppr_pg": round(totals["fpts_ppr"] / n, 1),
                "fpts_half_ppr_pg": round(totals["fpts_half_ppr"] / n, 1),
                "fpts_std_pg": round(totals["fpts_std"] / n, 1),
                "eff": (round(_efficiency(pos, totals), 1)
                        if _efficiency(pos, totals) is not None else None),
                "eff_label": EFF_LABELS[pos],
            }

    teams: dict = {}
    for pos in POSITIONS:
        values = {
            d: per_game[d][pos]["fpts_ppr_pg"]
            for d in per_game
            if per_game[d].get(pos)
        }
        ranks = _competition_ranks(values)
        total = len(values)
        for defense in per_game:
            entry = per_game[defense].get(pos)
            if entry is None:
                continue
            entry["rank"] = ranks.get(defense)
            entry["total"] = total
            teams.setdefault(defense, {"games": int(games.get(defense) or 0)})[pos] = entry

    return {
        "season": season,
        "completed_games": len(completed) // 2,
        "fingerprint": table_fingerprint(completed),
        "teams": teams,
    }


# ======================================================================
# From utils/defensive_matchup_ratings.py
# ======================================================================

"""Authoritative opponent-adjusted defense-vs-position calculations.

All Schedule Assistant consumers use the multiplier produced here.  A value of
1.10 means a defense increased its opponents' pre-game expectation by 10%; a
value of .90 means it suppressed it by 10%.  Functions are deliberately pure
so rebuilds and corrected-stat replays are deterministic and idempotent.
"""

import math

BASELINE_WEIGHTS = {"recent": .50, "earlier": .30, "prior": .20}
MIN_PARTICIPATION = {"QB": {"attempts": 8, "snaps": 15},
                     "RB": {"touches": 3, "snaps": 8},
                     "WR": {"targets": 2, "routes": 5, "snaps": 8},
                     "TE": {"targets": 1, "routes": 5, "snaps": 8}}
WINSOR_MULTIPLIER = (.50, 1.50)
EARLY_SEASON_CURRENT_WEIGHTS = {0: 0., 1: .25, 2: .40, 3: .55, 4: .70, 5: .85}


def season_weights(completed_through_week: int, *, blend: bool = True):
    if not blend:
        return 0., 1.
    current = EARLY_SEASON_CURRENT_WEIGHTS.get(max(0, int(completed_through_week)), 1.)
    return 1. - current, current


def scoring_profile_hash(settings):
    canonical = json.dumps(settings or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def rating_cache_key(season, completed_week, settings, position):
    return f"{int(season)}:{int(completed_week)}:{scoring_profile_hash(settings)}:{position.upper()}"


def blend_value(previous, current, completed_week, *, blend=True):
    if previous is None:
        return (current, "current") if current is not None else (None, "unavailable")
    if current is None:
        return previous, "prior"
    pw, cw = season_weights(completed_week, blend=blend)
    return pw * float(previous) + cw * float(current), "blended" if pw else "current"


def rank_values(values):
    ordered = sorted(((t, v) for t, v in values.items() if v is not None),
                     key=lambda x: (-float(x[1]), x[0]))
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}, len(ordered)


def normalize_schedule(values):
    """Return 0..100 scores from full-precision schedule multipliers."""
    usable = [float(v) for v in values.values() if v is not None]
    if not usable:
        return {}
    lo, hi = min(usable), max(usable)
    if math.isclose(lo, hi):
        return {k: 50. for k, v in values.items() if v is not None}
    return {k: 100. * (float(v) - lo) / (hi - lo) for k, v in values.items() if v is not None}


def rank_team_schedules(team_opponents, values):
    averages = {team: sum(vs) / len(vs) for team, opponents in team_opponents.items()
                if (vs := [float(values[o]) for o in opponents if o and values.get(o) is not None])}
    ranks, total = rank_values(averages)
    return {t: (ranks[t], total, round(v, 4)) for t, v in averages.items()}


def meaningful_participation(row):
    """Conservative OR thresholds; unavailable opportunity falls back to points."""
    pos = str(row.get("position") or row.get("pos") or "").upper()
    observed = False
    for field, minimum in MIN_PARTICIPATION.get(pos, {}).items():
        value = row.get(field)
        if value is not None:
            observed = True
            if float(value or 0) >= minimum:
                return True
    return (not observed) and float(row.get("fantasy_points") or row.get("pts") or 0) > 0


def pregame_baseline(history, position_baseline, weights=None):
    """Leak-free stabilized expectation from records strictly before a game.

    ``history`` is already chronological and contains only qualifying games.
    The prior/replacement component is never zero; uncertainty reports how much
    of the estimate rests on fewer than four current-season games.
    """
    weights = weights or BASELINE_WEIGHTS
    current = [r for r in history if r.get("is_current_season", True)]
    prior = [r for r in history if not r.get("is_current_season", True)]
    recent, earlier = current[-4:], current[:-4]
    replacement = max(float(position_baseline or 0), .1)
    components = {
        "recent": sum(float(r["fantasy_points"]) for r in recent) / len(recent) if recent else None,
        "earlier": sum(float(r["fantasy_points"]) for r in earlier) / len(earlier) if earlier else None,
        "prior": sum(float(r["fantasy_points"]) for r in prior[-8:]) / len(prior[-8:]) if prior else replacement,
    }
    # Shift missing component weight to the role/position replacement rather
    # than silently returning zero. Current data progressively displaces prior.
    n = len(current)
    prior_scale = max(0., 1. - n / 8.)
    configured = dict(weights)
    configured["prior"] *= prior_scale
    numerator = denominator = 0.
    for key, weight in configured.items():
        value = components[key] if components[key] is not None else replacement
        numerator += weight * value
        denominator += weight
    expected = numerator / denominator if denominator else replacement
    reliability = min(1., n / 6.)
    return expected, reliability


def game_adjustment(actual, expected):
    expected = float(expected)
    if expected <= 0:
        return None
    multiplier = float(actual) / expected
    return {"actual": float(actual), "expected": expected,
            "points_over_expected": float(actual) - expected,
            "multiplier": multiplier, "adjusted_percent": (multiplier - 1.) * 100.}


def aggregate_defense_games(games, prior_multiplier=1., prior_weight=4.):
    """Opportunity weighted, winsorized and shrunk positional defense result."""
    valid = [g for g in games if float(g.get("expected") or 0) > 0]
    if not valid:
        return None
    weighted = weight_sum = opps = 0.
    for index, g in enumerate(valid):
        raw = float(g["actual"]) / float(g["expected"])
        clipped = min(WINSOR_MULTIPLIER[1], max(WINSOR_MULTIPLIER[0], raw))
        reliability = float(g.get("reliability", 1.))
        opportunity = max(1., float(g.get("opportunities") or g["expected"]))
        recency = .9 + .1 * (index + 1) / len(valid)
        weight = float(g["expected"]) * reliability * recency
        weighted += clipped * weight
        weight_sum += weight
        opps += opportunity
    observed = weighted / weight_sum
    shrunk = (weight_sum * observed + prior_weight * float(prior_multiplier)) / (weight_sum + prior_weight)
    actual = sum(float(g["actual"]) for g in valid)
    expected = sum(float(g["expected"]) for g in valid)
    confidence = "high" if len(valid) >= 8 and weight_sum >= 80 else "medium" if len(valid) >= 4 else "low"
    return {"raw_allowed_per_game": actual / len(valid),
            "expected_opponent_points": expected / len(valid),
            "adjusted_points_over_expected": (actual - expected) / len(valid),
            "observed_multiplier": observed, "adjusted_multiplier": shrunk,
            "adjusted_percent": (shrunk - 1.) * 100., "sample_size": len(valid),
            "opportunity_count": opps, "reliable_sample_weight": weight_sum,
            "prior_weight": prior_weight, "confidence": confidence}


# ======================================================================
# From utils/team_offense_ranks.py
# ======================================================================

"""Honest per-team NFL offense table.

Shared by the player-modal Team tab and the public NFL Teams page. Every
number here is either a real measured value or a labeled projection:

- ``points_pg``: real NFL points per game from completed games (nflverse
  schedule with final scores). Never the old fantasy proxy
  (``TDs * 6 + yards / 20``).
- ``plays_pg``: real offensive plays per game from the team_play_volume
  service, when available. Never pass attempts + rush attempts relabeled.
- Yard / attempt / TD numbers: per-game, divided by each team's ACTUAL
  games played (bye weeks excluded), not by a hardcoded 17.
- Ranks use competition ranking (1, 2, 2, 4). Legitimate zeroes are
  ranked; only truly-missing values are unranked (``None``).

Data modes:
- ``actual``: at least one completed game exists. Team totals come from
  the season CSV when present (past seasons), else from Sleeper weekly
  TEAM rows for fully-completed weeks (in-progress season).
- ``projection``: no completed games yet (preseason / future season).
  Per-game values are projected totals / 17 and ``points_pg`` is None:
  projecting NFL points from offensive projections alone would be
  fabrication, so the Scoring row is omitted instead.

Stdlib only. All data arrives via injected callables so unit tests never
touch the network, the database, or ``utils.utils`` (which needs
``requests``, absent from the slim lint CI job).
"""


import csv

STAT_KEYS = ("pass_yds", "pass_att", "rush_yds", "rush_att", "pass_tds", "rush_tds")

#: Per-game denominator for preseason/future projections. Projections cover a
#: full season, so 17 is the honest divisor; in-progress seasons always use
#: actual completed games instead.
PROJECTION_DIVISOR = 17

#: Sleeper weekly TEAM rows use singular stat names (pass_yd, rush_td);
#: the season CSV path uses plural. Accept both.
_STAT_KEY_ALIASES = {
    "pass_yds": ("pass_yds", "pass_yd"),
    "pass_att": ("pass_att",),
    "rush_yds": ("rush_yds", "rush_yd"),
    "rush_att": ("rush_att",),
    "pass_tds": ("pass_tds", "pass_td"),
    "rush_tds": ("rush_tds", "rush_td"),
}


def _stat_value(row: dict, key: str) -> float:
    for alias in _STAT_KEY_ALIASES[key]:
        if row.get(alias) is not None:
            try:
                return float(row.get(alias) or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0

#: Regular-season weeks only. Postseason is excluded from these ranks.

#: Canonical team abbreviations. nflverse/ESPN/Sleeper mostly agree, but older
#: rows and some feeds still send legacy codes.




def _row_season_week(row: dict):
    try:
        season = int(float(str(row.get("season") or 0)))
    except (TypeError, ValueError):
        season = 0
    try:
        week = int(float(str(row.get("week") or 0)))
    except (TypeError, ValueError):
        week = 0
    return season, week




def aggregate_completed_games(season: int, rows) -> dict:
    """Per-team points and completed games from nflverse-style schedule rows.

    Only regular-season games with both final scores count. Returns
    ``{team: {"points": float, "games": int, "weeks": [int]}}``.
    """
    season = int(season)
    out: dict = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rseason, week = _row_season_week(row)
        if rseason != season or week not in REG_WEEKS:
            continue
        home_pts, away_pts = _row_scores(row)
        if home_pts is None or away_pts is None:
            continue
        home = canon_team(row.get("home_team"))
        away = canon_team(row.get("away_team"))
        if not home or not away:
            continue
        for team, pts in ((home, home_pts), (away, away_pts)):
            entry = out.setdefault(team, {"points": 0.0, "games": 0, "weeks": []})
            entry["points"] += pts
            entry["games"] += 1
            if week not in entry["weeks"]:
                entry["weeks"].append(week)
    for entry in out.values():
        entry["weeks"].sort()
    return out


def fully_completed_weeks(season: int, rows) -> list:
    """Weeks where every scheduled regular-season game has a final score.

    Team stat totals are only aggregated over these weeks so that per-game
    denominators stay consistent mid-week (a Sunday slate without Monday
    night is not a completed week).
    """
    season = int(season)
    by_week: dict = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rseason, week = _row_season_week(row)
        if rseason != season or week not in REG_WEEKS:
            continue
        by_week.setdefault(week, []).append(row)
    complete = []
    for week in sorted(by_week):
        games = by_week[week]
        if games and all(_row_scores(g)[0] is not None for g in games):
            complete.append(week)
    return complete


def aggregate_sleeper_team_weeks(get_week_teams, weeks) -> dict:
    """Sum Sleeper weekly TEAM rows into per-team stat totals.

    ``get_week_teams(week)`` returns ``{team_abbr: {stat_key: value}}``.
    Handles mid-season trades correctly: TEAM rows are already by team.
    """
    totals: dict = {}
    for week in weeks or []:
        try:
            week_teams = get_week_teams(week) or {}
        except Exception:
            continue
        for team, row in week_teams.items():
            team = canon_team(team)
            if not team or not isinstance(row, dict):
                continue
            bucket = totals.setdefault(team, {k: 0.0 for k in STAT_KEYS})
            for key in STAT_KEYS:
                bucket[key] += _stat_value(row, key)
    return totals


def aggregate_sleeper_team_weeks_for_teams(get_week_teams, completed) -> dict:
    """Sum Sleeper weekly TEAM rows, attributing each week only to the teams
    whose game that week has a final score.

    ``completed`` is the ``aggregate_completed_games`` mapping
    (``{team: {"weeks": [...]}}``). A Thursday final is included for those
    two teams while the rest of the week is still pending; teams that have
    not played yet contribute nothing for that week.
    """
    need: dict = {}
    for team, entry in (completed or {}).items():
        team = canon_team(team)
        if not team:
            continue
        for w in (entry or {}).get("weeks") or []:
            try:
                wi = int(w)
            except (TypeError, ValueError):
                continue
            need.setdefault(wi, set()).add(team)
    totals: dict = {}
    for week in sorted(need):
        try:
            week_teams = get_week_teams(week) or {}
        except Exception:
            continue
        canon_rows = {}
        for key, row in week_teams.items():
            ct = canon_team(key)
            if ct and isinstance(row, dict):
                canon_rows[ct] = row
        for team in need[week]:
            row = canon_rows.get(team)
            if not isinstance(row, dict):
                continue
            bucket = totals.setdefault(team, {k: 0.0 for k in STAT_KEYS})
            for key in STAT_KEYS:
                bucket[key] += _stat_value(row, key)
    return totals


def read_csv_team_totals(csv_path: str) -> dict:
    """Sum a stats_player_reg CSV into per-team stat totals (past seasons)."""
    totals: dict = {}
    if not csv_path or not os.path.exists(csv_path):
        return totals
    col_map = {
        "pass_yds": "passing_yards",
        "pass_att": "attempts",
        "rush_yds": "rushing_yards",
        "rush_att": "carries",
        "pass_tds": "passing_tds",
        "rush_tds": "rushing_tds",
    }
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                team = canon_team(row.get("recent_team"))
                if not team:
                    continue
                bucket = totals.setdefault(team, {k: 0.0 for k in STAT_KEYS})
                for key, col in col_map.items():
                    try:
                        bucket[key] += float(row.get(col) or 0)
                    except (TypeError, ValueError):
                        continue
    except Exception:
        return {}
    return totals


def _per_game(value: float, games: int):
    if not games or games <= 0:
        return None
    return value / games


def build_offense_table(season: int, *, completed: dict, totals: dict,
                        plays_pg_map: dict, data_mode: str,
                        completed_weeks: list) -> dict:
    """Per-team per-game metrics. The single place per-game math happens."""
    teams: dict = {}
    all_teams = set(completed) | set(totals) | set(plays_pg_map or ())
    for team in sorted(all_teams):
        games = int((completed.get(team) or {}).get("games") or 0)
        bucket = totals.get(team) or {}
        if data_mode == "actual":
            divisor = games
        else:
            divisor = PROJECTION_DIVISOR
        pass_yds = _per_game(float(bucket.get("pass_yds") or 0), divisor)
        pass_att = _per_game(float(bucket.get("pass_att") or 0), divisor)
        rush_yds = _per_game(float(bucket.get("rush_yds") or 0), divisor)
        rush_att = _per_game(float(bucket.get("rush_att") or 0), divisor)
        pass_tds = _per_game(float(bucket.get("pass_tds") or 0), divisor)
        rush_tds = _per_game(float(bucket.get("rush_tds") or 0), divisor)
        comp = completed.get(team) or {}
        points_pg = _per_game(float(comp.get("points") or 0), games) if data_mode == "actual" else None
        plays = (plays_pg_map or {}).get(team)
        try:
            plays_pg = float(plays) if plays is not None else None
        except (TypeError, ValueError):
            plays_pg = None
        if data_mode == "projection" and plays_pg is None and pass_att is not None and rush_att is not None:
            # Projected plays from projected attempts; labeled projection,
            # never presented as a measured value.
            plays_pg = pass_att + rush_att
        total_yds = (pass_yds or 0) + (rush_yds or 0) if pass_yds is not None else None
        att_sum = (pass_att or 0) + (rush_att or 0)
        pass_rate = (pass_att / att_sum) if pass_att is not None and att_sum > 0 else None
        teams[team] = {
            "games": games,
            "points_pg": points_pg,
            "plays_pg": plays_pg,
            "pass_yds_pg": pass_yds,
            "pass_att_pg": pass_att,
            "rush_yds_pg": rush_yds,
            "rush_att_pg": rush_att,
            "total_yds_pg": total_yds,
            "pass_tds_pg": pass_tds,
            "rush_tds_pg": rush_tds,
            "pass_rate": pass_rate,
        }
    return {
        "season": int(season),
        "data_mode": data_mode,
        "completed_weeks": list(completed_weeks or []),
        "teams": teams,
    }


def compute_team_offense(season: int, *, games_rows=None, get_week_teams=None,
                         csv_path: str = None, projected_totals: dict = None,
                         plays_pg_map: dict = None) -> dict:
    """Build the honest offense table for one season.

    ``games_rows``: nflverse-style schedule rows (season/week/home_team/
    away_team/home_score/away_score). ``get_week_teams(week)``: Sleeper
    weekly TEAM rows. ``csv_path``: stats_player_reg CSV for past seasons.
    ``projected_totals``: per-team projected season totals (preseason).
    ``plays_pg_map``: ``{team: plays/game}`` from the team_play_volume
    service.

    Points, games, and stat totals are attributed per team: every game with
    a final score counts for the two teams that played it, even mid-week
    (a Thursday final is included for those teams while the rest of the
    week is pending, mirroring the defense-vs-position table). Per-game
    denominators are each team's own completed games.
    """
    season = int(season)
    rows = games_rows or []
    weeks = fully_completed_weeks(season, rows)
    completed = aggregate_completed_games(season, rows)
    has_actuals = any(e["games"] > 0 for e in completed.values())
    if has_actuals:
        data_mode = "actual"
        if csv_path and os.path.exists(csv_path):
            totals = read_csv_team_totals(csv_path)
        else:
            totals = aggregate_sleeper_team_weeks_for_teams(get_week_teams, completed)
    else:
        data_mode = "projection"
        totals = dict(projected_totals or {})
    table = build_offense_table(
        season,
        completed=completed,
        totals=totals,
        plays_pg_map=plays_pg_map or {},
        data_mode=data_mode,
        completed_weeks=weeks,
    )
    any_weeks = sorted(
        {w for e in completed.values() for w in (e.get("weeks") or [])}
    )
    full = set(weeks)
    table["in_progress_weeks"] = [w for w in any_weeks if w not in full]
    return table


def competition_ranks(values: dict) -> dict:
    """Rank teams best-first with competition ranking (1, 2, 2, 4).

    ``values`` maps team -> number or None. None means truly missing and is
    left unranked; legitimate zeroes ARE ranked. Ties share a rank and the
    next rank skips accordingly. Sort is by value desc, then team asc, so
    output is deterministic.
    """
    items = [(t, float(v)) for t, v in (values or {}).items() if v is not None]
    items.sort(key=lambda x: (-x[1], x[0]))
    total = len(items)
    out: dict = {}
    for team in (values or {}):
        out[team] = None
    last_val = None
    rank = 0
    for i, (team, val) in enumerate(items, 1):
        if last_val is None or val != last_val:
            rank = i
            last_val = val
        rounded = round(val, 2) if val == round(val, 2) else round(val, 3)
        out[team] = {"rank": rank, "value": rounded, "total": total}
    return out


#: Table metric -> rank-table key, in the order the Team tab renders them.
RANK_METRICS = (
    "points_pg",
    "pass_yds_pg",
    "pass_att_pg",
    "rush_yds_pg",
    "rush_att_pg",
    "total_yds_pg",
    "pass_tds_pg",
    "rush_tds_pg",
    "plays_pg",
    "pass_rate",
)


def rank_offense_table(table: dict) -> dict:
    """``{metric: {team: rank-entry | None}}`` for every RANK_METRICS metric."""
    teams = (table or {}).get("teams") or {}
    ranks = {}
    for metric in RANK_METRICS:
        ranks[metric] = competition_ranks({t: r.get(metric) for t, r in teams.items()})
    return ranks


def ranked_metric(values: dict, higher_better: bool = True) -> dict:
    """Competition-rank ``{team: value}`` honoring sort direction.

    ``values`` maps team -> number or None (None = truly missing, unranked).
    ``higher_better=False`` ranks lower values first (pressure/sack rates)
    while keeping the original values in the entries. Same contract as
    :func:`competition_ranks`: ties share a rank, zeroes are ranked.
    """
    if higher_better:
        return competition_ranks(values)
    negated = {t: (-v if v is not None else None) for t, v in (values or {}).items()}
    ranked = competition_ranks(negated)
    out = {}
    for team, entry in ranked.items():
        if entry is None:
            out[team] = None
        else:
            out[team] = {
                "rank": entry["rank"],
                "value": (values or {}).get(team),
                "total": entry["total"],
            }
    return out


# ======================================================================
# From utils/schedule_ease.py
# ======================================================================

"""Pure schedule-difficulty presentation helpers.

Extracted from app.py's schedule assistant: NFL team-code normalization across
data sources, and the color/ease mappings for matchup difficulty cells.
"""

# Alternate team codes used by various stat/schedule feeds -> canonical code.
SCHED_TEAM_ALIAS = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR", "OAK": "LV",
                    "SD": "LAC", "STL": "LAR", "ARZ": "ARI", "BLT": "BAL",
                    "CLV": "CLE", "HST": "HOU"}


def norm_sched_team(t) -> str:
    """Canonical NFL team code ('' stays '')."""
    t = (t or "").upper().strip()
    return SCHED_TEAM_ALIAS.get(t, t)


def sched_rank_color(rank, total):
    """(text_color, background) for a matchup by opponent's fpts-allowed rank
    (1 = most points allowed = easiest)."""
    if not rank or not total:
        return "#6b7280", "transparent"
    pct = rank / total
    if pct <= 0.25:
        return "#22c55e", "#22c55e18"   # elite (most pts allowed)
    if pct <= 0.50:
        return "#84cc16", "#84cc1618"
    if pct <= 0.75:
        return "#f59e0b", "#f59e0b18"
    return "#ef4444", "#ef444418"        # brutal (fewest pts allowed)


def matchup_cell_ease(rank, total, info):
    """Per-cell ease (0-100). Prefer the z-derived ease from the precomputed
    ratings table; fall back to rank percentile. Missing data returns ``None``
    so callers can exclude it rather than treating unknown as maximally hard."""
    if info and info.get("ease") is not None:
        return float(info["ease"])
    if rank and total and total > 1:
        return round((total - rank) / (total - 1) * 100, 1)
    return None


# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L92 ---
TEAM_ALIASES = {
    "jax": "JAX", "jac": "JAX", "jacksonville": "JAX", "gb": "GB", "gnb": "GB", "nwe": "NE", "ne": "NE",
    "sfo": "SF", "sf": "SF", "kan": "KC", "kc": "KC", "tam": "TB", "tb": "TB",
    "was": "WAS", "was football team": "WAS", "wsh": "WAS",
    "lv": "LV", "oak": "LV", "sd": "LAC", "lac": "LAC", "la chargers": "LAC",
    "stl": "LAR", "lar": "LAR", "la": "LAR", "la rams": "LAR", "no": "NO", "nor": "NO",
    "bal": "BAL", "cin": "CIN", "pit": "PIT", "cle": "CLE", "buf": "BUF", "mia": "MIA",
    "blt": "BAL", "clv": "CLE", "hst": "HOU", "arz": "ARI",  # legacy codes (from team files)
    "nyj": "NYJ", "nyg": "NYG", "phi": "PHI", "dal": "DAL", "wasdc": "WAS",
    "min": "MIN", "chi": "CHI", "det": "DET", "atl": "ATL", "car": "CAR", "norleans": "NO",
    "sea": "SEA", "den": "DEN", "ari": "ARI", "hou": "HOU", "ten": "TEN", "ind": "IND",
    "philadelphia": "PHI", "philadelphia eagles": "PHI", "eagles": "PHI",
}

# --- utils/utils.py L107 ---
TEAM_ABBR_ALIASES = {
    "WAS": "WSH",
    "WSH": "WAS",
    "JAC": "JAX",
    "JAX": "JAC",
    "LA": "LAR",
    "LAR": "LA",
}

# --- utils/utils.py L117 ---
def team_abbr_keys(team: str) -> tuple[str, ...]:
    """Return the team code plus its schedule/index alias, if any."""
    t = (team or "").strip().upper()
    if not t:
        return ()
    alt = TEAM_ABBR_ALIASES.get(t)
    return (t, alt) if alt else (t,)

# --- utils/utils.py L126 ---
def lookup_team_map(mapping: Optional[dict], team: str):
    """Lookup ``mapping[team]``, trying WAS/WSH (and other abbr aliases)."""
    if not mapping or not team:
        return None
    for key in team_abbr_keys(team):
        if key in mapping:
            return mapping[key]
    return None

# --- utils/utils.py L136 ---
def canonical_teams_index(teams_index: Optional[dict]) -> dict:
    """Merge alias keys (WSH into WAS) so each franchise appears once.

    Incoming feeds still use WSH; the site stores and displays WAS. When both
    keys exist, non-null fields from either copy are kept under WAS.
    """
    out: dict = {}
    for abv, meta in (teams_index or {}).items():
        if not isinstance(meta, dict):
            continue
        canon = canon_team(abv) or str(abv or "").strip().upper()
        if not canon:
            continue
        cur = out.get(canon)
        if cur is None:
            out[canon] = dict(meta)
            continue
        for k, v in meta.items():
            if cur.get(k) is None and v is not None:
                cur[k] = v
    return out

# --- utils/utils.py L159 ---
def canonicalize_game_teams(game: Optional[dict]) -> dict:
    """Rewrite a schedule game's home/away codes to site canonical form (WAS)."""
    if not isinstance(game, dict):
        return {}
    out = dict(game)
    for field in ("home", "away"):
        val = out.get(field)
        if not val:
            continue
        canon = canon_team(val)
        if canon:
            out[field] = canon
    return out

# --- utils/utils.py L174 ---
def canonicalize_schedule(data):
    """Normalize home/away on a week schedule list (or pass other shapes through)."""
    if isinstance(data, list):
        return [
            canonicalize_game_teams(g) if isinstance(g, dict) else g
            for g in data
        ]
    return data

# --- utils/utils.py L183 ---
DST_CANON = {
    "49ers": "SF",
    "Patriots": "NE",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Commanders": "WAS",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Raiders": "LV",
    "Saints": "NO",
    # ... extend as needed
}

# --- utils/utils.py L199 ---
_NFL_FRANCHISES = {
    "ARI": ("Arizona", "Arizona Cardinals", "Cardinals"), "ATL": ("Atlanta", "Atlanta Falcons", "Falcons"),
    "BAL": ("Baltimore", "Baltimore Ravens", "Ravens"), "BUF": ("Buffalo", "Buffalo Bills", "Bills"),
    "CAR": ("Carolina", "Carolina Panthers", "Panthers"), "CHI": ("Chicago", "Chicago Bears", "Bears"),
    "CIN": ("Cincinnati", "Cincinnati Bengals", "Bengals"), "CLE": ("Cleveland", "Cleveland Browns", "Browns"),
    "DAL": ("Dallas", "Dallas Cowboys", "Cowboys"), "DEN": ("Denver", "Denver Broncos", "Broncos"),
    "DET": ("Detroit", "Detroit Lions", "Lions"), "GB": ("Green Bay", "Green Bay Packers", "Packers"),
    "HOU": ("Houston", "Houston Texans", "Texans"), "IND": ("Indianapolis", "Indianapolis Colts", "Colts"),
    "JAX": ("Jacksonville", "Jacksonville Jaguars", "Jaguars"), "KC": ("Kansas City", "Kansas City Chiefs", "Chiefs"),
    "LV": ("Las Vegas", "Las Vegas Raiders", "Raiders"), "LAC": ("Los Angeles Chargers", "Chargers"),
    "LAR": ("Los Angeles Rams", "Rams"), "MIA": ("Miami", "Miami Dolphins", "Dolphins"),
    "MIN": ("Minnesota", "Minnesota Vikings", "Vikings"), "NE": ("New England", "New England Patriots", "Patriots"),
    "NO": ("New Orleans", "New Orleans Saints", "Saints"), "NYG": ("New York Giants", "Giants"),
    "NYJ": ("New York Jets", "Jets"), "PHI": ("Philadelphia", "Philadelphia Eagles", "Eagles"),
    "PIT": ("Pittsburgh", "Pittsburgh Steelers", "Steelers"), "SEA": ("Seattle", "Seattle Seahawks", "Seahawks"),
    "SF": ("San Francisco", "San Francisco 49ers", "49ers"), "TB": ("Tampa Bay", "Tampa Bay Buccaneers", "Buccaneers"),
    "TEN": ("Tennessee", "Tennessee Titans", "Titans"), "WAS": ("Washington", "Washington Commanders", "Commanders"),
}

# --- utils/utils.py L217 ---
for _abbr, _names in _NFL_FRANCHISES.items():
    for _name in _names:
        TEAM_ALIASES.setdefault(_name.lower(), _abbr)

# --- utils/utils.py L960 ---
def canon_team(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t0 = str(t).strip().replace("_", " ")
    t0 = re.sub(r"\s+NFL$", "", t0, flags=re.IGNORECASE).strip()
    t0 = re.sub(r"\s+DEFENSE$", "", t0, flags=re.IGNORECASE).strip()
    # e.g., "49ers D/ST" => "49ers"
    if "D/ST" in t0 or "DST" in t0:
        t0 = t0.replace("D/ST", "").replace("DST", "").strip()
    up = TEAM_ALIASES.get(t0.lower(), t0.upper())
    # If still not a 2–3 letter code but a nickname like "49ers", map to code
    return DST_CANON.get(up, up)

# --- utils/utils.py L1296 ---
def _espn_logo_slug(team_abv: str) -> str:
    """ESPN CDN team-logo slug (WAS → wsh; site-canonical otherwise)."""
    t = (canon_team(team_abv) or str(team_abv or "")).strip().upper()
    if t == "WAS":
        return "wsh"
    return t.lower()

# --- utils/utils.py L1304 ---
def _espn_logo_url(team_abv: str) -> str:
    # ESPN logo fallback (500px). Prefer teams_index.Logo when available.
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{_espn_logo_slug(team_abv)}.png"

# --- utils/utils.py L1309 ---
def def_team_logo_urls(team_abv: str) -> tuple[str, str]:
    from utils.data_cache import load_teams_index
    """Local team-logo path + ESPN CDN URL for a DEF/DST (WAS-canonical).

    Local files live at ``/static/images/team_logos/{ABBR}.png`` (WAS, not WSH).
    ESPN uses ``wsh.png`` for Washington — prefer ``teams_index[team]["Logo"]``
    when present so the CDN slug stays correct.
    """
    team = (canon_team(team_abv) or str(team_abv or "")).strip().upper()
    if not team:
        return ("", "")
    local = f"/static/images/team_logos/{team}.png"
    ti = (load_teams_index() or {}).get(team) or {}
    espn = str(ti.get("Logo") or "").strip() or _espn_logo_url(team)
    return (local, espn)

# --- utils/utils.py L1325 ---
def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# --- utils/utils.py L1334 ---
def get_bye_week(team: dict, season: int) -> Optional[int]:
    """
    team: one object from payload['body']
    season: e.g., 2025
    returns: bye week as int, or None if missing
    """
    bye_map = team.get("byeWeeks") or {}
    # keys may be strings ("2025") and values may be ["8"] (strings)
    weeks = bye_map.get(str(season)) or bye_map.get(season)
    if not weeks:
        return None
    # Some seasons can list multiple byes; take the first valid int
    for w in weeks:
        try:
            return int(w)
        except (TypeError, ValueError):
            continue
    return None

# --- utils/utils.py L1354 ---
def byes_for_season(payload: dict, season: int) -> dict[str, Optional[int]]:
    """
    payload: the full response with a 'body' list
    returns: { teamAbv: bye_week_or_None }
    """
    out = {}
    for team in payload.get("body", []):
        abv = canon_team(team.get("teamAbv")) or team.get("teamAbv")
        out[abv] = get_bye_week(team, season)
    return out
