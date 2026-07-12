"""Live game-condition signals for start/sit: Vegas totals and weather.

Two enrichments layered on top of the static venue tags (utils.nfl_stadiums):

  * **Vegas implied team total** - from Tank01's betting-odds endpoint (the same
    RapidAPI key the app already uses for projections). The game total plus a
    team's spread give its *implied team total*, the single best one-number read
    on how much scoring the market expects from that team this week.

  * **Weather** - from Open-Meteo (a keyless public forecast API), looked up by
    stadium coordinates for the game date. Only meaningful for outdoor venues, so
    domes are skipped. We surface a tag only when conditions are actually
    notable (strong wind, hard cold, real precipitation) - a mild, dry day gets
    no chip.

Design rules that mirror the rest of the app:
  * The pure parsing / math / tagging helpers take plain data and are fully unit
    tested; the network fetchers wrap them with a short on-disk + in-memory TTL
    cache and *never raise* - any failure degrades to None so the caller falls
    back to the static dome/cold tags.
  * Forecasts firm up only a few days out, and betting totals move all week, so
    both are cached briefly and re-fetched, not stored long term.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (no network) - unit tested directly
# ---------------------------------------------------------------------------

def implied_team_total(game_total: float, team_spread: float) -> Optional[float]:
    """Implied points for a team given the game total and that team's spread.

    A favorite laying 6 in a 44.5-point game is implied for
    44.5/2 - (-6)/2 = 22.25 + 3 = 25.25. Returns None on bad inputs.
    """
    try:
        gt = float(game_total)
        sp = float(team_spread)
    except (TypeError, ValueError):
        return None
    if gt <= 0:
        return None
    return round(gt / 2.0 - sp / 2.0, 1)


def total_tag(implied: Optional[float]) -> Optional[dict]:
    """Chip for an implied team total, or None if unremarkable/unknown.

    Thresholds are league-average anchored (~22-23 implied is a normal team
    total): >= 26 is a strong spot, <= 18 is a dud, the wide middle is unmarked.
    """
    if implied is None:
        return None
    label = f"{implied:g} implied"
    if implied >= 26:
        return {"label": label, "kind": "high", "note": "High team total (Vegas)"}
    if implied <= 18:
        return {"label": label, "kind": "low", "note": "Low team total (Vegas)"}
    return {"label": label, "kind": "mid", "note": "Vegas implied team total"}


def weather_tag(
    dome: bool,
    temp_f: Optional[float],
    wind_mph: Optional[float],
    precip_pct: Optional[float],
) -> Optional[dict]:
    """Notable-weather chip for an outdoor game, or None.

    Domes and benign conditions return None. Priority: wind (most impactful on
    passing/kicking) > precipitation > hard cold. Thresholds are the points where
    fantasy production is actually affected.
    """
    if dome:
        return None
    parts = []
    kind = "weather"
    if wind_mph is not None and wind_mph >= 15:
        parts.append(f"{round(wind_mph)} mph wind")
        kind = "wind"
    if precip_pct is not None and precip_pct >= 60:
        parts.append("rain/snow")
        if kind != "wind":
            kind = "precip"
    if temp_f is not None and temp_f <= 25:
        parts.append(f"{round(temp_f)}°")
        if kind == "weather":
            kind = "cold"
    if not parts:
        return None
    return {"label": " · ".join(parts), "kind": kind, "note": "Notable weather"}


def parse_tank01_odds(body) -> dict:
    """Parse a Tank01 getNFLBettingOdds `body` into per-team totals/spreads.

    Tank01 keys the body by gameID; each game carries team abbreviations and,
    either at the top level or under a sportsbook sub-dict, a total and home/away
    spreads. We read defensively across the field names Tank01 has used
    (totalUnder/totalOver/total, homeTeamSpread/awayTeamSpread) and take the first
    sportsbook when a `sportsBookOdds` list/dict is present.

    Returns ``{TEAM: {"total": float, "spread": float, "implied": float}}``.
    """
    out: dict = {}
    if not isinstance(body, dict):
        return out
    for _gid, game in body.items():
        if not isinstance(game, dict):
            continue
        home = _norm(game.get("homeTeam") or game.get("home") or game.get("teamAbvHome"))
        away = _norm(game.get("awayTeam") or game.get("away") or game.get("teamAbvAway"))
        if not home or not away:
            continue
        odds = _first_book(game)
        total = _to_float(
            odds.get("totalOver") or odds.get("total") or odds.get("totalUnder")
        )
        home_sp = _to_float(odds.get("homeTeamSpread") or odds.get("homeSpread"))
        away_sp = _to_float(odds.get("awayTeamSpread") or odds.get("awaySpread"))
        # If only one spread is present, the other is its negation.
        if home_sp is None and away_sp is not None:
            home_sp = -away_sp
        if away_sp is None and home_sp is not None:
            away_sp = -home_sp
        if total is None:
            continue
        for team, sp in ((home, home_sp), (away, away_sp)):
            if sp is None:
                continue
            out[team] = {
                "total": total,
                "spread": sp,
                "implied": implied_team_total(total, sp),
            }
    return out


def parse_open_meteo_daily(payload, index: int = 0) -> Optional[dict]:
    """Extract {temp_f, wind_mph, precip_pct} from an Open-Meteo daily forecast.

    ``index`` selects the day offset in the returned arrays. Returns None if the
    payload is missing the expected fields.
    """
    if not isinstance(payload, dict):
        return None
    daily = payload.get("daily") or {}
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    winds = daily.get("wind_speed_10m_max") or []
    precip = daily.get("precipitation_probability_max") or []
    if index >= len(highs) or index >= len(winds):
        return None
    hi = _to_float(highs[index]) if index < len(highs) else None
    lo = _to_float(lows[index]) if index < len(lows) else None
    # Game-relevant temp: afternoon/evening runs closer to the daily high; use a
    # high-low blend so a frigid low doesn't overstate a mild-afternoon game.
    temp = None
    if hi is not None and lo is not None:
        temp = round(0.65 * hi + 0.35 * lo, 1)
    elif hi is not None:
        temp = hi
    return {
        "temp_f": temp,
        "wind_mph": _to_float(winds[index]) if index < len(winds) else None,
        "precip_pct": _to_float(precip[index]) if index < len(precip) else None,
    }


# ---------------------------------------------------------------------------
# small internal utils
# ---------------------------------------------------------------------------

def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm(team) -> str:
    from utils.nfl_stadiums import normalize_team
    return normalize_team(team) if team else ""


def _first_book(game: dict) -> dict:
    """Return the odds dict for a game, unwrapping a sportsBookOdds list/dict."""
    sbo = game.get("sportsBookOdds")
    if isinstance(sbo, list) and sbo and isinstance(sbo[0], dict):
        return sbo[0]
    if isinstance(sbo, dict) and sbo:
        first = next(iter(sbo.values()))
        if isinstance(first, dict):
            return first
    return game  # odds live at the top level


# ---------------------------------------------------------------------------
# Network fetchers (cached, never raise)
# ---------------------------------------------------------------------------

_ODDS_CACHE: dict = {}
_ODDS_TTL = 60 * 60          # betting lines move all week; refresh hourly
_WEATHER_CACHE: dict = {}
_WEATHER_TTL = 60 * 60 * 3   # forecasts change slowly; refresh every few hours


def fetch_week_odds(season: int, week: int, game_dates: "list[str]") -> dict:
    """Per-team Vegas totals for a week, keyed by team abbr. Never raises.

    ``game_dates`` are the Tank01 ``gameDate`` strings (YYYYMMDD) for the week's
    games; Tank01's odds endpoint is queried per date. Returns {} on any failure
    or when no API key is configured.
    """
    key = (int(season), int(week))
    hit = _ODDS_CACHE.get(key)
    if hit and time.time() - hit[0] < _ODDS_TTL:
        return hit[1]
    result: dict = {}
    try:
        import requests
        from utils.utils import TANK01_API_HOST, TANK01_API_KEY
        if not TANK01_API_KEY:
            return {}
        headers = {"x-rapidapi-host": TANK01_API_HOST, "x-rapidapi-key": TANK01_API_KEY}
        url = f"https://{TANK01_API_HOST}/getNFLBettingOdds"
        for gd in sorted(set(game_dates or [])):
            try:
                resp = requests.get(url, headers=headers, params={"gameDate": gd}, timeout=15)
                if resp.status_code != 200:
                    continue
                body = (resp.json() or {}).get("body")
                result.update(parse_tank01_odds(body))
            except Exception:
                logger.debug("[game_conditions] odds fetch failed for %s", gd, exc_info=True)
    except Exception:
        logger.debug("[game_conditions] odds fetch unavailable", exc_info=True)
        return {}
    _ODDS_CACHE[key] = (time.time(), result)
    return result


def fetch_game_weather(lat: float, lon: float, game_date: str, today: "Optional[str]" = None) -> Optional[dict]:
    """Weather for a stadium on a game date via Open-Meteo. Never raises.

    ``game_date`` / ``today`` are YYYYMMDD strings; the day offset picks the right
    entry from the daily forecast. Returns parsed {temp_f, wind_mph, precip_pct}
    or None (past date, out of forecast range, or any failure).
    """
    ck = (round(float(lat), 3), round(float(lon), 3), str(game_date))
    hit = _WEATHER_CACHE.get(ck)
    if hit and time.time() - hit[0] < _WEATHER_TTL:
        return hit[1]
    offset = _day_offset(game_date, today)
    if offset is None or offset < 0 or offset > 15:
        return None  # only within Open-Meteo's ~16-day forecast horizon
    parsed = None
    try:
        import requests
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_probability_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "forecast_days": min(offset + 1, 16), "timezone": "America/New_York",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            parsed = parse_open_meteo_daily(resp.json(), offset)
    except Exception:
        logger.debug("[game_conditions] weather fetch failed", exc_info=True)
        return None
    _WEATHER_CACHE[ck] = (time.time(), parsed)
    return parsed


def build_week_conditions(
    season: int,
    week: int,
    week_games: "list[tuple]",
    *,
    today: "Optional[str]" = None,
    fetch_weather: bool = True,
) -> dict:
    """Per-team Vegas + weather conditions for a week's games. Never raises.

    ``week_games`` is a list of ``(home_team, away_team, game_date)`` tuples
    (game_date = YYYYMMDD). Returns ``{TEAM: {"implied_total": float|None,
    "weather": tag|None}}`` for every team with a game. Weather is looked up once
    per outdoor venue (domes skipped) and shared by both teams in the game;
    lookups run concurrently and are cached, so a warmed request is instant.
    """
    from utils.nfl_stadiums import game_environment, normalize_team, stadium_coords

    out: dict = {}
    try:
        game_dates = [gd for _h, _a, gd in week_games if gd]
        odds = fetch_week_odds(season, week, game_dates)

        # One weather lookup per distinct outdoor home venue.
        venue_weather: dict = {}
        if fetch_weather:
            jobs = {}
            for home, _away, gd in week_games:
                hn = normalize_team(home)
                if hn in jobs:
                    continue
                env = game_environment(hn)
                coords = stadium_coords(hn)
                if env and not env.get("dome") and coords:
                    jobs[hn] = (coords[0], coords[1], gd)
            if jobs:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
                    futs = {
                        ex.submit(fetch_game_weather, lat, lon, gd, today): hn
                        for hn, (lat, lon, gd) in jobs.items()
                    }
                    for fut in futs:
                        hn = futs[fut]
                        try:
                            venue_weather[hn] = fut.result()
                        except Exception:
                            venue_weather[hn] = None

        for home, away, _gd in week_games:
            hn, an = normalize_team(home), normalize_team(away)
            wx = venue_weather.get(hn)  # both teams share the host venue's weather
            wx_tag = None
            if wx:
                wx_tag = weather_tag(False, wx.get("temp_f"), wx.get("wind_mph"), wx.get("precip_pct"))
            for team in (hn, an):
                if not team:
                    continue
                out[team] = {
                    "implied_total": (odds.get(team) or {}).get("implied"),
                    "weather": wx_tag,
                }
    except Exception:
        logger.debug("[game_conditions] build_week_conditions failed", exc_info=True)
        return out
    return out


def _day_offset(game_date: "Optional[str]", today: "Optional[str]") -> Optional[int]:
    """Whole-day offset between two YYYYMMDD strings (game_date - today)."""
    from datetime import datetime
    if not game_date:
        return None
    try:
        gd = datetime.strptime(str(game_date), "%Y%m%d").date()
        if today:
            td = datetime.strptime(str(today), "%Y%m%d").date()
        else:
            td = datetime.now().date()
        return (gd - td).days
    except (ValueError, TypeError):
        return None
