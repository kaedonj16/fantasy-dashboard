"""Shared, provider-neutral NFL game data backed only by public ESPN feeds.

Fantasy points are deliberately *not* calculated here.  A connected fantasy
provider remains authoritative for league scoring; this module supplies game
state, cumulative NFL statistics, and play data only.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

from email.utils import parsedate_to_datetime

import requests

log = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"
CDN_SUMMARY_URL = "https://cdn.espn.com/core/nfl/playbyplay"
UA = "BRFantasy/1.0 (+https://brfantasyfootball.com)"
TEAM_ALIASES = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR"}
# Fallback final-score source when ESPN blocks the scoreboard (403). nflverse
# publishes one games.csv for all seasons; it carries home/away final scores
# for completed games, so a finished week still renders "Final 24-31 @ LV"
# even with ESPN down. Live games have blank scores and are skipped.
_NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
_NFLVERSE_GAMES_TTL = 6 * 3600.0
_nflverse_games_lock = threading.Lock()

_session = requests.Session()
_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_last_good: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_locks: dict[str, threading.Lock] = {}
_failures: OrderedDict[str, tuple[float, str]] = OrderedDict()
_guard = threading.Lock()
_MAX = 96
_FAILURE_COOLDOWN = 60.0
_LAST_GOOD_MAX_AGE = 3600.0


class NFLDataUnavailable(RuntimeError):
    """A classified public-feed failure safe for consumers to inspect."""
    def __init__(self, kind: str, message: str = "NFL data unavailable"):
        super().__init__(message)
        self.kind = kind


class ScoreboardResult(dict):
    """Mapping-compatible scoreboard with availability metadata.

    An unavailable result and a successfully fetched empty date are both empty
    mappings, but are no longer semantically indistinguishable.
    """
    def __init__(self, *args, availability="available", stale=False,
                 source="espn_nfl", fetched_at=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.availability, self.stale, self.source = availability, stale, source
        self.fetched_at = fetched_at


def _failure_kind(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if code in (401, 403): return "forbidden"
    if code == 429: return "rate_limited"
    if code is not None and code >= 500: return "server"
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)): return "transport"
    if isinstance(exc, (ValueError, TypeError)): return "invalid_response"
    return "transport"


def _retry_after(response) -> float:
    raw = (getattr(response, "headers", {}) or {}).get("Retry-After")
    if not raw: return 0.0
    try: return min(5.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        try: return min(5.0, max(0.0, parsedate_to_datetime(raw).timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError): return 0.0


def _team(value: Any) -> str:
    value = str(value or "").upper().strip()
    return TEAM_ALIASES.get(value, value)


def _request_json(url: str, *, params: dict, timeout: float = 10.0) -> dict:
    """Bounded public request. Authentication failures are never retried."""
    for attempt in range(2):
        try:
            response = _session.get(
                url, params=params, headers={"User-Agent": UA, "Accept": "application/json"},
                timeout=(3.05, min(float(timeout), 12.0)),
            )
            if response.status_code in (401, 403):
                response.raise_for_status()
            if response.status_code == 429 and attempt == 0:
                time.sleep(_retry_after(response) or 1.0); continue
            if response.status_code >= 500 and attempt == 0:
                time.sleep(.2); continue
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise NFLDataUnavailable("invalid_response")
            return value
        except requests.RequestException as exc:
            if attempt == 0 and _failure_kind(exc) in {"transport", "server"}:
                time.sleep(.2)
                continue
            raise
    raise NFLDataUnavailable("transport")


def _single_flight(key: str, fetch, ttl: float) -> tuple[dict, bool]:
    now = time.time()
    with _guard:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            _cache.move_to_end(key)
            return hit[1], False
        failed = _failures.get(key)
        if failed and now < failed[0]:
            old = _last_good.get(key)
            if old and now - old[0] <= _LAST_GOOD_MAX_AGE:
                return old[1], True
            return {}, True
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        now = time.time()
        with _guard:
            hit = _cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1], False
        try:
            value = fetch()
            if not isinstance(value, dict):
                raise NFLDataUnavailable("invalid_response")
        except Exception as exc:
            kind = getattr(exc, "kind", None) or _failure_kind(exc)
            with _guard:
                old = _last_good.get(key)
                entering = key not in _failures or now >= _failures[key][0]
                _failures[key] = (now + _FAILURE_COOLDOWN, kind)
            usable = old and now - old[0] <= _LAST_GOOD_MAX_AGE
            if entering:
                log.warning("ESPN NFL %s failure for %s%s", kind, key,
                            "; using compatible last-good data" if usable else "; data unavailable")
                log.debug("ESPN NFL diagnostic for %s", key, exc_info=True)
            return (old[1], True) if usable else ({}, True)
        with _guard:
            recovered = key in _failures
            _failures.pop(key, None)
            _cache[key] = (now, value)
            _last_good[key] = (now, value)
            for store in (_cache, _last_good):
                while len(store) > _MAX:
                    store.popitem(last=False)
            _locks.pop(key, None)
        if recovered: log.info("ESPN NFL feed recovered for %s", key)
        return value, False


def fetch_scoreboard(*, dates: str = "", season: int | str = "", week: int | str = "",
                     season_type: int | str = "", ttl: float = 30, timeout: float = 10) -> tuple[dict, bool]:
    params: dict[str, str] = {}
    if dates:
        params["dates"] = str(dates)
    if season:
        params["dates"] = str(season)
    if week:
        params["week"] = str(week)
    if season_type:
        params["seasontype"] = str(season_type)
    key = "scoreboard:" + ":".join(f"{k}={v}" for k, v in sorted(params.items()))
    return _single_flight(key, lambda: _request_json(SCOREBOARD_URL, params=params, timeout=timeout), ttl)


def fetch_summary(event_id: str, *, ttl: float = 15, timeout: float = 12) -> tuple[dict, bool]:
    eid = str(event_id or "").strip()
    if not eid:
        return {}, False
    key = f"event:{eid}"

    def load():
        primary = _request_json(SUMMARY_URL, params={"event": eid}, timeout=timeout)
        if primary:
            return primary
        return _request_json(CDN_SUMMARY_URL, params={"xhr": "1", "gameId": eid}, timeout=timeout)
    return _single_flight(key, load, ttl)


def _status(raw: dict) -> tuple[str, str]:
    typ = (raw or {}).get("type") or {}
    state = str(typ.get("state") or "pre").lower()
    name = str(typ.get("name") or typ.get("description") or "").lower()
    if "cancel" in name: return "canceled", "0"
    if "postpon" in name: return "postponed", "0"
    if typ.get("completed") or state == "post": return "final", "2"
    if state == "in": return "live", "1"
    return "pregame", "0"


def _event_sides(event: dict) -> tuple[dict, str, str, str, str]:
    comp = ((event.get("competitions") or [{}])[0])
    home = away = home_score = away_score = ""
    for entry in comp.get("competitors") or []:
        abbr = _team((entry.get("team") or {}).get("abbreviation"))
        if entry.get("homeAway") == "home": home, home_score = abbr, str(entry.get("score") or "")
        elif entry.get("homeAway") == "away": away, away_score = abbr, str(entry.get("score") or "")
    return comp, home, away, home_score, away_score


def normalize_scoreboard(payload: dict, *, stale: bool = False) -> list[dict]:
    events = payload.get("events") or ((payload.get("content") or {}).get("sbData") or {}).get("events") or []
    out = []
    for event in events:
        if not isinstance(event, dict): continue
        comp, home, away, hp, ap = _event_sides(event)
        if not home or not away: continue
        raw_status = event.get("status") or comp.get("status") or {}
        status, code = _status(raw_status)
        iso = str(event.get("date") or comp.get("date") or "")
        day = iso[:10].replace("-", "") if len(iso) >= 10 else ""
        typ = raw_status.get("type") or {}
        season = event.get("season") or {}
        week = event.get("week") or {}
        game_id = f"{day}_{away}@{home}" if day else f"espn:{event.get('id')}"
        out.append({
            "gameID": game_id, "internal_game_id": game_id, "espn_event_id": str(event.get("id") or ""),
            "season": season.get("year"), "season_type": season.get("type"), "week": week.get("number"),
            "gameTime": iso, "gameTime_epoch": _iso_epoch(iso), "home": home, "away": away,
            "homePts": hp, "awayPts": ap, "gameStatusCode": code,
            "gameStatus": typ.get("shortDetail") or typ.get("description") or status,
            "normalized_status": status, "gameClock": raw_status.get("displayClock") or "",
            "lineScore": {"period": raw_status.get("period") or ""}, "source": "espn_nfl",
            "fetched_at": datetime.utcnow().isoformat() + "Z", "stale": stale,
            "availability": {"score": True, "game_state": True, "player_stats": False,
                             "team_stats": False, "plays": False},
        })
    return out


def _iso_epoch(value: str) -> int:
    try: return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError): return 0


def scoreboard_for_date(game_date: str, *, timeout: float = 10) -> dict[str, dict]:
    payload, stale = fetch_scoreboard(dates=game_date, ttl=20, timeout=timeout)
    key = "scoreboard:dates=" + str(game_date)
    with _guard:
        successful = _last_good.get(key)
    usable_success = bool(successful and time.time() - successful[0] <= _LAST_GOOD_MAX_AGE)
    availability = "stale" if stale and usable_success else ("unavailable" if stale else "available")
    fetched_at = datetime.utcfromtimestamp(successful[0]).isoformat() + "Z" if usable_success else None
    games = {game["gameID"]: game for game in normalize_scoreboard(payload, stale=stale)}
    if not games and availability == "unavailable":
        # ESPN is refusing the scoreboard (403). Fall back to nflverse final
        # scores so finished weeks still show "Final 24-31 @ LV" instead of a
        # bare "Final". Stale-but-usable ESPN data above still wins when it
        # exists, since it also covers live games.
        nv_games = _nflverse_scoreboard_for_date(game_date)
        if nv_games:
            return ScoreboardResult(
                {game["gameID"]: game for game in nv_games},
                availability="available", stale=False, source="nflverse",
                fetched_at=datetime.utcnow().isoformat() + "Z",
            )
    return ScoreboardResult(
        games,
        availability=availability, stale=bool(stale and usable_success), fetched_at=fetched_at,
    )


def _nflverse_games_rows() -> list[dict]:
    """All nflverse schedule rows, cached on disk and refreshed at most every 6h.

    A stale on-disk copy is preferred over nothing, so a transient download
    failure doesn't wipe final scores. Returns [] when no copy is available.
    """
    import csv

    from utils.utils import CACHE_DIR

    path = CACHE_DIR / "nflverse_games.csv"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < _NFLVERSE_GAMES_TTL
    if not fresh:
        with _nflverse_games_lock:
            fresh = path.exists() and (time.time() - path.stat().st_mtime) < _NFLVERSE_GAMES_TTL
            if not fresh:
                try:
                    resp = _session.get(_NFLVERSE_GAMES_URL, timeout=(3.05, 20.0))
                    resp.raise_for_status()
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_bytes(resp.content)
                    tmp.replace(path)
                except Exception:
                    log.warning("nflverse games.csv download failed", exc_info=True)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        log.warning("nflverse games.csv unreadable", exc_info=True)
        return []


def _nflverse_scoreboard_for_date(game_date: str) -> list[dict]:
    """Final-score game dicts for one YYYYMMDD date, in the normalized
    scoreboard shape, sourced from the nflverse schedule.

    Only rows with both scores present are returned -- scheduled/live games
    have no final score yet and stay on the schedule/game-status path.
    """
    want = str(game_date or "")
    games: list[dict] = []
    for row in _nflverse_games_rows():
        gameday = str(row.get("gameday") or "").replace("-", "")
        if gameday != want:
            continue
        home = _team(row.get("home_team"))
        away = _team(row.get("away_team"))
        home_pts = str(row.get("home_score") or "").strip()
        away_pts = str(row.get("away_score") or "").strip()
        if not home or not away or not home_pts or not away_pts:
            continue
        try:
            week = int(float(str(row.get("week") or "0")))
        except (TypeError, ValueError):
            week = 0
        try:
            season = int(float(str(row.get("season") or "0")))
        except (TypeError, ValueError):
            season = 0
        game_id = f"{want}_{away}@{home}"
        games.append({
            "gameID": game_id, "internal_game_id": game_id,
            "espn_event_id": str(row.get("espn") or ""),
            "season": season or None, "season_type": 2, "week": week or None,
            "gameTime": str(row.get("gameday") or ""), "gameTime_epoch": 0,
            "home": home, "away": away,
            "homePts": home_pts, "awayPts": away_pts,
            "gameStatusCode": "2", "gameStatus": "Final",
            "normalized_status": "final", "gameClock": "",
            "lineScore": {"period": ""}, "source": "nflverse",
            "fetched_at": datetime.utcnow().isoformat() + "Z", "stale": False,
            "availability": {"score": True, "game_state": True, "player_stats": False,
                             "team_stats": False, "plays": False},
        })
    return games


def games_for_week(week: int, season: int, season_type: str = "reg") -> list[dict]:
    type_id = {"pre": 1, "reg": 2, "post": 3}.get(str(season_type).lower(), season_type)
    payload, stale = fetch_scoreboard(season=season, week=week, season_type=type_id, ttl=300)
    return normalize_scoreboard(payload, stale=stale)


def find_event(game_id: str) -> tuple[str, dict]:
    try:
        date, matchup = game_id.split("_", 1); away, home = matchup.split("@", 1)
    except ValueError:
        return "", {}
    for delta in (0, -1, 1):
        try: day = (datetime.strptime(date, "%Y%m%d") + timedelta(days=delta)).strftime("%Y%m%d")
        except ValueError: day = date
        games = scoreboard_for_date(day)
        for game in games.values():
            if game["away"] == _team(away) and game["home"] == _team(home):
                return game.get("espn_event_id", ""), game
    return "", {}


_CATEGORY_MAP = {
    "passing": {"C/ATT": ("Passing", "passCompletionsAndAttempts"), "YDS": ("Passing", "passYds"), "TD": ("Passing", "passTD"), "INT": ("Passing", "int")},
    "rushing": {"CAR": ("Rushing", "carries"), "YDS": ("Rushing", "rushYds"), "TD": ("Rushing", "rushTD")},
    "receiving": {"REC": ("Receiving", "receptions"), "TGTS": ("Receiving", "targets"), "YDS": ("Receiving", "recYds"), "TD": ("Receiving", "recTD")},
    "fumbles": {"LOST": ("Fumbles", "fumblesLost")},
    "kicking": {"FG": ("Kicking", "fgMadeAndAttempts"), "XP": ("Kicking", "xpMadeAndAttempts"), "LNG": ("Kicking", "fgLong")},
    "defensive": {"TOT": ("Defense", "totalTackles"), "SACKS": ("Defense", "sacks"), "INT": ("Defense", "defensiveInterceptions"), "PD": ("Defense", "passesDefended"), "FF": ("Defense", "forcedFumbles"), "FR": ("Defense", "fumblesRecovered"), "TD": ("Defense", "defTD")},
}


# ESPN's summary boxscore ships each column twice: a machine key
# (``passingYards``) and a display label (``YDS``). ``_CATEGORY_MAP`` is written
# against the labels, but real payloads line the two arrays up positionally --
# the old code preferred keys whenever the arrays had equal length, so almost
# every offensive field silently failed to map (only ``sacks`` survived, which
# is why box scores showed dashes for offense and lone zero-sack defensive
# rows). Resolve each column through the label first, then through the key.
_ESPN_KEY_TO_LABEL = {
    "passing": {
        "completions/passingattempts": "C/ATT",
        "passingyards": "YDS",
        "passingtouchdowns": "TD",
        "interceptions": "INT",
    },
    "rushing": {
        "rushingattempts": "CAR",
        "rushingyards": "YDS",
        "rushingtouchdowns": "TD",
    },
    "receiving": {
        "receptions": "REC",
        "receivingtargets": "TGTS",
        "receivingyards": "YDS",
        "receivingtouchdowns": "TD",
    },
    "fumbles": {"fumbleslost": "LOST"},
    "kicking": {
        "fieldgoalsmade-fieldgoalsattempted": "FG",
        "extrapointsmade-extrapointsattempted": "XP",
        "longestfieldgoalmade": "LNG",
    },
    "defensive": {
        "totaltackles": "TOT",
        "sacks": "SACKS",
        "passesdefended": "PD",
        "forcedfumbles": "FF",
        "fumblesrecovered": "FR",
        "defensivetouchdowns": "TD",
    },
}


def _resolve_boxscore_target(cat_name: str, label: str, key: str):
    """Map one boxscore column to its (block, stat) target via label or key."""
    cat = _CATEGORY_MAP.get(cat_name) or {}
    target = cat.get((label or "").upper())
    if target:
        return target, (label or "").upper()
    alias = (_ESPN_KEY_TO_LABEL.get(cat_name) or {}).get((key or "").lower())
    if alias:
        target = cat.get(alias)
        if target:
            return target, alias
    return None, ""


def _number(value: Any) -> Any:
    text = str(value or "").strip()
    if not text: return None
    if "/" in text:
        vals = text.split("/", 1)
        try: return [float(v) for v in vals]
        except ValueError: return None
    try:
        number = float(text.replace(",", "")); return int(number) if number.is_integer() else number
    except ValueError: return None


def summary_to_legacy(summary: dict, game: dict, *, stale: bool = False) -> dict:
    """Translate ESPN's labelled boxscore into the established UI contract.

    Labels/keys are paired before values are read; no statistic depends on a
    positional offset. Unknown fields remain absent rather than becoming zero.
    """
    box = summary.get("boxscore") or {}
    player_stats: dict[str, dict] = {}
    available: set[str] = set()
    for side in box.get("players") or []:
        team = _team((side.get("team") or {}).get("abbreviation"))
        for category in side.get("statistics") or []:
            cat_name = str(category.get("name") or "").lower()
            keys = [str(x) for x in (category.get("keys") or [])]
            labels = [str(x) for x in (category.get("labels") or [])]
            # Pair keys and labels positionally; either may be absent.
            columns = max(len(keys), len(labels))
            for row in category.get("athletes") or []:
                athlete = row.get("athlete") or {}
                espn_id = str(athlete.get("id") or "")
                entry = player_stats.setdefault(espn_id or f"name:{athlete.get('displayName')}", {
                    "playerID": espn_id, "espnID": espn_id, "longName": athlete.get("displayName") or "",
                    "teamAbv": team, "pos": (athlete.get("position") or {}).get("abbreviation") or "",
                })
                for idx, raw in enumerate(row.get("stats") or []):
                    if idx >= columns:
                        continue
                    label = labels[idx] if idx < len(labels) else ""
                    col_key = keys[idx] if idx < len(keys) else ""
                    target, resolved = _resolve_boxscore_target(cat_name, label, col_key)
                    if not target: continue
                    block, stat_key = target; parsed = _number(raw)
                    if parsed is None: continue
                    dest = entry.setdefault(block, {})
                    if isinstance(parsed, list):
                        if stat_key == "passCompletionsAndAttempts": dest.update(passCompletions=parsed[0], passAttempts=parsed[1])
                        elif stat_key == "fgMadeAndAttempts": dest.update(fgMade=parsed[0], fgAttempts=parsed[1])
                        elif stat_key == "xpMadeAndAttempts": dest.update(xpMade=parsed[0], xpAttempts=parsed[1])
                    else: dest[stat_key] = parsed
                    available.add(f"{cat_name}.{resolved.lower()}")
    result = dict(game)
    result.update({"playerStats": player_stats, "teamStats": {}, "source": "espn_nfl",
                   "stale": stale, "fetched_at": datetime.utcnow().isoformat() + "Z",
                   "field_availability": sorted(available), "breakdown_complete": False})
    return result


def boxscore_for_game(game_id: str, *, play_by_play: bool = False) -> dict:
    event_id, game = find_event(game_id)
    if not event_id: return {}
    ttl = 15 if game.get("gameStatusCode") == "1" else (300 if game.get("gameStatusCode") == "0" else 900)
    summary, stale = fetch_summary(event_id, ttl=ttl)
    if not summary: return dict(game)
    result = summary_to_legacy(summary, game, stale=stale)
    if play_by_play:
        gp = summary.get("gamepackageJSON") or summary
        result["espn_summary"] = gp
    return result


def get_game(game_id: str) -> dict:
    """Return the provider-neutral complete game contract for one event."""
    event_id, game = find_event(game_id)
    if not event_id:
        return {}
    ttl = 15 if game.get("gameStatusCode") == "1" else (300 if game.get("gameStatusCode") == "0" else 900)
    summary, stale = fetch_summary(event_id, ttl=ttl)
    legacy = summary_to_legacy(summary, game, stale=stale) if summary else dict(game)
    header_comp = ((((summary.get("header") or {}).get("competitions") or [{}])[0])
                   if isinstance(summary, dict) else {})
    situation = header_comp.get("situation") or {}
    drives = summary.get("drives") or {} if isinstance(summary, dict) else {}
    if isinstance(drives, dict):
        drive_rows = list(drives.get("previous") or [])
        if isinstance(drives.get("current"), dict):
            drive_rows.append(drives["current"])
    else:
        drive_rows = drives if isinstance(drives, list) else []
    plays = []
    for drive in drive_rows:
        for play in (drive.get("plays") or []) if isinstance(drive, dict) else []:
            if not isinstance(play, dict):
                continue
            plays.append({
                "id": str(play.get("id") or ""), "text": play.get("text") or "",
                "period": (play.get("period") or {}).get("number"),
                "clock": (play.get("clock") or {}).get("displayValue"),
                "scoring": bool(play.get("scoringPlay")),
                "no_play": "no play" in str(play.get("text") or "").lower(),
            })
    return {
        "internal_game_id": game.get("gameID"), "espn_event_id": event_id,
        "season": game.get("season"), "season_type": game.get("season_type"), "week": game.get("week"),
        "kickoff_timestamp": game.get("gameTime"), "home_team": game.get("home"),
        "away_team": game.get("away"), "status": game.get("normalized_status"),
        "period": (game.get("lineScore") or {}).get("period"), "clock": game.get("gameClock"),
        "home_score": game.get("homePts"), "away_score": game.get("awayPts"),
        "possession": _team(situation.get("possession")), "down": situation.get("down"),
        "distance": situation.get("distance"), "field_position": situation.get("possessionText"),
        "cumulative_player_statistics": legacy.get("playerStats") or {},
        "cumulative_team_statistics": legacy.get("teamStats") or {}, "normalized_plays": plays,
        "source": "espn_nfl", "fetched_at": legacy.get("fetched_at") or game.get("fetched_at"),
        "stale": stale, "field_availability": legacy.get("field_availability") or [],
        "completeness": {"nfl_statistics": "partial", "fantasy_points": "provider_only"},
    }
