"""DraftKings sportsbook season-long NFL player futures (free, unofficial).

SportsGameOdds carries no season-long NFL player markets, but DraftKings does —
its NFL "Futures › Player Stats O/U" section posts season passing/rushing/
receiving yard (and TD) over/unders for marquee players all offseason, i.e. right
at draft time. This module reads them from DraftKings' internal ``sportscontent``
JSON API and emits season-context MarketRecords for the same projection/ADP
pipeline the (weekly) SportsGameOdds feed uses.

CONFIGURED VIA ENV, off by default. DraftKings does not publish its numeric
``leagueId``/``subCategoryId`` values and rotates them each season, so they are
configuration, not constants:

    DRAFTKINGS_NFL_LEAGUE_ID   confirmed "88808"  (templateVars in the request)
    DRAFTKINGS_NFL_SEASON_MARKETS  "passing_yards=17314,rushing_yards=...,..."
                               stat_type=subCategoryId pairs; stat_type must be a
                               key of projection.STAT_KEYS. "17314" is a confirmed
                               NFL player-stat O/U subcategory; capture the rest of
                               the stat tabs the same way.
    DRAFTKINGS_SITE            optional, default "US-SB" (region path segment)

The ids come from the network request behind
https://sportsbook.draftkings.com/leagues/football/nfl?category=futures&subcategory=player-stats-o-u
— confirmed leagueId=88808, and e.g. subCategoryId=17314.

The response shape (from the public ``sportscontent`` API) is::

    {"markets": [{"id": ..., "name": "Patrick Mahomes Passing Yards"}],
     "selections": [{"marketId": ..., "label": "Over"|"Over 4500.5", "points": 4500.5,
                     "displayOdds": {"american": "-110"}}]}

IMPORTANT — the line field is not yet confirmed. The ``controldata`` endpoint
lists every player market but may return the number under ``points``/``line``, in
the selection ``label`` ("Over 4500.5"), in the market ``name``, or only in a
sibling live-odds feed. The parser tries all of the in-response locations and
skips a market when it can find no line, so a wrong guess yields no records rather
than bad ones. Nothing here runs until the env above is set. Re-verify against one
real response body before trusting the output.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .models import MarketRecord
from .projection import STAT_KEYS

_DEFAULT_SITE = "US-SB"
_LINE_KEYS = ("points", "line", "handicap", "number", "trueLine")
_PLAYER_KEYS = ("participant", "playerName", "name")
# Trailing number in a label/name like "Over 4500.5" or "Passing Yards 4500.5".
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*$")


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
    try:
        return float(str(raw).replace("+", "")) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _num_in(text) -> float | None:
    match = _NUM_RE.search(str(text or ""))
    return float(match.group(1)) if match else None


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
        self.league_id = os.getenv("DRAFTKINGS_NFL_LEAGUE_ID", "").strip()
        self.market_map = _parse_market_map(os.getenv("DRAFTKINGS_NFL_SEASON_MARKETS", ""))
        self.timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    @property
    def configured(self) -> bool:
        return bool(self.league_id and self.market_map)

    def _markets_url(self) -> str:
        return (f"{self.base_url}/sites/{self.site}/api/sportscontent/controldata/"
                "league/leagueSubcategory/v1/markets")

    def _params(self, sub_category_id: str) -> dict:
        events_q = (f"$filter=leagueId eq '{self.league_id}' AND "
                    f"clientMetadata/Subcategories/any(s: s/Id eq '{sub_category_id}')")
        markets_q = (f"$filter=clientMetadata/subCategoryId eq '{sub_category_id}' AND "
                     "tags/all(t: t ne 'SportcastBetBuilder')")
        return {"isBatchable": "false", "templateVars": self.league_id,
                "eventsQuery": events_q, "marketsQuery": markets_q,
                "include": "Events", "entity": "events"}

    def fetch_subcategory(self, sub_category_id: str) -> dict:
        response = self.session.get(self._markets_url(), params=self._params(sub_category_id),
                                    headers=self._headers, timeout=self.timeout)
        if response.status_code >= 400:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def iter_season_markets(self) -> Iterator[tuple[str, dict]]:
        """Yield (stat_type, payload) for each configured season stat market."""
        for stat_type, sub_id in self.market_map.items():
            payload = self.fetch_subcategory(sub_id)
            if payload:
                yield stat_type, payload


def season_records_from_payload(payload: dict, stat_type: str, event_id: str,
                                observed_at: datetime | None = None,
                                event_start: datetime | None = None) -> list[MarketRecord]:
    """Flatten one DraftKings sportscontent payload into season MarketRecords.

    One record per player Over selection (the line is symmetric, so the Over
    carries the number; consensus only needs the line and the two prices)."""
    observed_at = observed_at or datetime.now(timezone.utc)
    # Season futures have no single game date; keep the record "future" so the
    # consensus freshness/settlement guards accept it through the season.
    event_start = event_start or (observed_at + timedelta(days=120))
    markets = {str(m.get("id")): m for m in payload.get("markets", []) if isinstance(m, dict)}
    by_market: dict[str, dict] = {}
    for sel in payload.get("selections", []):
        if not isinstance(sel, dict):
            continue
        label = str(sel.get("label") or "").lower()
        # Label may be "Over" or carry the line ("Over 4500.5"); key by side.
        side = "over" if label.startswith("over") else "under" if label.startswith("under") else label
        by_market.setdefault(str(sel.get("marketId")), {})[side] = sel

    out: list[MarketRecord] = []
    for market_id, sides in by_market.items():
        over, under = sides.get("over"), sides.get("under")
        if not over:
            continue
        market = markets.get(market_id, {})
        line = _line_from_selection(over, market)
        if line is None:
            continue
        name = _player_name(market) or str(market.get("name") or "")
        if not name:
            continue
        out.append(MarketRecord(
            provider_event_id=event_id, provider_player_id=f"dk:{name}",
            sportsbook="draftkings", market_type="ou", stat_type=stat_type,
            period="season", line=line, event_start_time=event_start,
            observed_at=observed_at, side="over",
            over_price=_american_from_selection(over),
            under_price=_american_from_selection(under) if under else None,
            context="season",
        ))
    return out
