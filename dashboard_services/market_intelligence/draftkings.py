"""DraftKings sportsbook season-long NFL player futures (free, unofficial).

SportsGameOdds carries no season-long NFL player markets, but DraftKings does —
its NFL "Futures › Player Stats O/U" section posts season passing/rushing/
receiving yard (and TD) over/unders for marquee players all offseason, i.e. right
at draft time. This module reads them from DraftKings' internal ``sportscontent``
JSON API and emits season-context MarketRecords for the same projection/ADP
pipeline the (weekly) SportsGameOdds feed uses.

The NFL leagueId (88808), site (US-SB), and full stat-to-subcategory map (the
"Futures > Player Stats O/U" tabs) are retained below for a future access-model
change. The source is OFF
by default because the undocumented endpoint denies production datacenter traffic.
Every value is an optional override; enabling it is a deliberate operator action:

    DRAFTKINGS_SEASON_ENABLED     "1"/"true" to enable (default off)
    DRAFTKINGS_NFL_SEASON_MARKETS "passing_yards=17147,rushing_yards=17223,..."
                                  replaces the baked-in map; stat_type must be a
                                  key of projection.STAT_KEYS
    DRAFTKINGS_NFL_LEAGUE_ID      default "88808"
    DRAFTKINGS_SITE               default "US-SB" (region path segment)

The ids come from the stat tabs behind
https://sportsbook.draftkings.com/leagues/football/nfl?category=futures&subcategory=player-stats-o-u
(each tab's data-entity-id is its subCategoryId).

The response shape (confirmed from a live NFL response) is::

    {"events":     [{"id": E, "name": "NFL 2026/27 - Mike Evans",
                     "startEventDate": "...", "participants": [{"name": "Mike Evans"}, ...]}],
     "markets":    [{"id": M, "eventId": E, "name": "... Regular Season Receiving TDs",
                     "subcategoryId": "17315"}],
     "selections": [{"marketId": M, "outcomeType": "Over", "label": "Over 6.5",
                     "displayOdds": {"american": "−105"}}]}

The player is on the EVENT (a participant with no team id, or the event name after
" - "); the line is the number in the Over selection's ``label``; the side is
``outcomeType``; and negative odds use a Unicode minus (U+2212), normalized before
parsing.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .models import MarketRecord
from .projection import STAT_KEYS

_DEFAULT_LEAGUE_ID = "88808"
_DEFAULT_SITE = "US-SB"
# NFL "Futures > Player Stats O/U" tabs (each tab's data-entity-id). Baked in so
# the source needs no configuration; override via DRAFTKINGS_NFL_SEASON_MARKETS.
_DEFAULT_SEASON_MARKETS = {
    "passing_yards": "17147",
    "passing_touchdowns": "17148",
    "rushing_yards": "17223",
    "rushing_touchdowns": "17224",
    "receiving_yards": "17314",
    "receiving_touchdowns": "17315",
    "receptions": "20168",
}
_LINE_KEYS = ("points", "line", "handicap", "number", "trueLine")
_PLAYER_KEYS = ("participant", "playerName", "name")
# Trailing number in a label/name like "Over 4500.5" or "Passing Yards 4500.5".
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*$")


def _env_enabled(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _parse_market_map(raw: str) -> dict[str, str]:
    """"passing_yards=4501,rushing_yards=4502" -> {stat_type: subCategoryId}."""
    out: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        stat, sub = pair.split("=", 1)
        stat, sub = stat.strip(), sub.strip()
        if stat in STAT_KEYS and sub:
            out[stat] = sub
    return out


def _american_from_selection(selection: dict) -> float | None:
    odds = selection.get("displayOdds")
    raw = odds.get("american") if isinstance(odds, dict) else selection.get("oddsAmerican")
    if raw in (None, ""):
        return None
    # DraftKings renders negatives with a Unicode minus (U+2212), not ASCII "-".
    text = str(raw).replace("−", "-").replace("+", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _num_in(text) -> float | None:
    match = _NUM_RE.search(str(text or ""))
    return float(match.group(1)) if match else None


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    text = re.sub(r"(\.\d{6})\d+", r"\1", str(value).replace("Z", "+00:00"))
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _player_from_event(event: dict) -> str:
    """The player is the event participant with no team id; the event name
    ("NFL 2026/27 - Mike Evans") is the fallback."""
    for part in event.get("participants", []):
        if isinstance(part, dict) and not part.get("id") and part.get("name"):
            name = str(part["name"])
            return "" if _num_in(name) is not None and re.fullmatch(r"[\d.]+", name.strip()) else name
    pieces = re.split(r"\s[–-]\s", str(event.get("name") or ""))
    name = pieces[-1].strip() if len(pieces) > 1 else ""
    return "" if re.fullmatch(r"[\d.]+", name) else name


def _line_from_selection(selection: dict, market: dict | None = None) -> float | None:
    # Prefer a dedicated numeric field on the selection or its market.
    for source in (selection, market or {}):
        for key in _LINE_KEYS:
            val = source.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    # Fall back to a number embedded in the selection label or the market name
    # ("Over 4500.5", "Patrick Mahomes Passing Yards 4500.5").
    return _num_in(selection.get("label")) or _num_in((market or {}).get("name"))


def _player_name(market: dict) -> str:
    for key in _PLAYER_KEYS:
        val = market.get(key)
        if isinstance(val, dict):
            val = val.get("name") or val.get("fullName")
        if val:
            return str(val)
    return ""


class DraftKingsClient:
    """Reads DraftKings' internal sportscontent markets feed. No API key."""

    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://sportsbook.draftkings.com/",
        "Origin": "https://sportsbook.draftkings.com",
    }

    def __init__(self, session=None, timeout: float = 15.0):
        self.base_url = os.getenv("DRAFTKINGS_BASE_URL", "https://sportsbook-nash.draftkings.com").rstrip("/")
        self.site = (os.getenv("DRAFTKINGS_SITE", _DEFAULT_SITE) or _DEFAULT_SITE).strip()
        self.league_id = (os.getenv("DRAFTKINGS_NFL_LEAGUE_ID", "").strip() or _DEFAULT_LEAGUE_ID)
        # Env override replaces the baked-in map; otherwise use the defaults.
        self.market_map = (_parse_market_map(os.getenv("DRAFTKINGS_NFL_SEASON_MARKETS", ""))
                           or dict(_DEFAULT_SEASON_MARKETS))
        # This undocumented endpoint persistently denies production datacenter
        # traffic. Keep the adapter for a future access-model change, but require a
        # deliberate opt-in; an absent flag must never cause a network request.
        self.enabled = _env_enabled("DRAFTKINGS_SEASON_ENABLED", default=False)
        self.timeout = timeout
        self.last_error: str | None = None  # why the most recent fetch returned {}
        self._access_denied = False
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.league_id and self.market_map)

    def _markets_url(self) -> str:
        return (f"{self.base_url}/sites/{self.site}/api/sportscontent/controldata/"
                "league/leagueSubcategory/v1/markets")

    def _params(self, sub_category_id: str) -> dict:
        events_q = (f"$filter=leagueId eq '{self.league_id}' AND "
                    f"clientMetadata/Subcategories/any(s: s/Id eq '{sub_category_id}')")
        markets_q = (f"$filter=clientMetadata/subCategoryId eq '{sub_category_id}' AND "
                     "tags/all(t: t ne 'SportcastBetBuilder')")
        # templateVars carries both the league and the subcategory, comma-joined.
        return {"isBatchable": "false", "templateVars": f"{self.league_id},{sub_category_id}",
                "eventsQuery": events_q, "marketsQuery": markets_q,
                "include": "Events", "entity": "events"}

    def fetch_subcategory(self, sub_category_id: str) -> dict:
        if self._access_denied:
            return {}
        try:
            response = self.session.get(self._markets_url(), params=self._params(sub_category_id),
                                        headers=self._headers, timeout=self.timeout)
        except Exception as exc:
            self.last_error = f"request error: {type(exc).__name__}: {exc}"
            return {}
        if response.status_code >= 400:
            # Do not retain/log provider response bodies. Edge-denial pages can be
            # large and may contain request identifiers that are not operationally
            # useful to this job.
            self.last_error = f"HTTP {response.status_code}"
            self._access_denied = response.status_code in (401, 403)
            return {}
        try:
            payload = response.json()
        except ValueError:
            self.last_error = f"non-JSON response (HTTP {response.status_code}, {len(response.content or b'')} bytes)"
            return {}
        if not isinstance(payload, dict):
            self.last_error = f"unexpected JSON type: {type(payload).__name__}"
            return {}
        self.last_error = None
        return payload

    def iter_season_markets(self) -> Iterator[tuple[str, dict]]:
        """Yield (stat_type, payload) for each configured season stat market."""
        if not self.configured:
            return
        for stat_type, sub_id in self.market_map.items():
            payload = self.fetch_subcategory(sub_id)
            if payload:
                yield stat_type, payload


def _selection_side(selection: dict) -> str:
    side = str(selection.get("outcomeType") or "").lower()
    if side in ("over", "under"):
        return side
    label = str(selection.get("label") or "").lower()  # "Over 6.5"
    return "over" if label.startswith("over") else "under" if label.startswith("under") else label


def season_records_from_payload(payload: dict, stat_type: str,
                                observed_at: datetime | None = None,
                                default_start: datetime | None = None) -> list[MarketRecord]:
    """Flatten one DraftKings sportscontent payload into season MarketRecords.

    Joins selections -> markets -> events: the player is on the event, the line is
    in the Over selection's label ("Over 6.5"), the side is ``outcomeType``. One
    record per player Over (the line is symmetric; consensus needs the line and
    the two prices)."""
    observed_at = observed_at or datetime.now(timezone.utc)
    # Season futures have no single game date; keep the record "future" so the
    # consensus freshness/settlement guards accept it through the season.
    default_start = default_start or (observed_at + timedelta(days=120))
    events = {str(e.get("id")): e for e in payload.get("events", []) if isinstance(e, dict)}
    markets = {str(m.get("id")): m for m in payload.get("markets", []) if isinstance(m, dict)}
    by_market: dict[str, dict] = {}
    for sel in payload.get("selections", []):
        if isinstance(sel, dict):
            by_market.setdefault(str(sel.get("marketId")), {})[_selection_side(sel)] = sel

    out: list[MarketRecord] = []
    for market_id, sides in by_market.items():
        over, under = sides.get("over"), sides.get("under")
        if not over:
            continue
        market = markets.get(market_id, {})
        line = _line_from_selection(over, market)
        if line is None:
            continue
        event = events.get(str(market.get("eventId")), {})
        name = _player_from_event(event) or _player_name(market)
        if not name:
            continue
        out.append(MarketRecord(
            provider_event_id=str(market.get("eventId") or market_id),
            provider_player_id=f"dk:{name}",
            sportsbook="draftkings", market_type="ou", stat_type=stat_type,
            period="season", line=line,
            event_start_time=_parse_dt(event.get("startEventDate")) or default_start,
            observed_at=observed_at, side="over",
            over_price=_american_from_selection(over),
            under_price=_american_from_selection(under) if under else None,
            context="season",
        ))
    return out
