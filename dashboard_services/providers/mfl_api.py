"""Read-only MyFantasyLeague export API provider.

Raw MFL names remain in this module. Public methods return the existing
Sleeper-compatible dictionaries consumed by BR Fantasy.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from .base import (
    DRAFTS, DRAFT_RESULTS, FUTURE_PICKS, HISTORY, LEAGUE, MATCHUPS, ROSTERS,
    ROSTER_SETTINGS, SCORING_SETTINGS, STANDINGS, STARTERS, TRADED_PICKS,
    TRANSACTIONS, TRADES, USERS, LeagueNotFoundError, ProviderAdapter,
    ProviderAuthenticationError, ProviderMetadata, ProviderUnavailableError,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)
BASE_URL = "https://api.myfantasyleague.com/{season}/export"
TIMEOUT = (4, 15)
_CACHE: dict[tuple, tuple[float, dict]] = {}


def _request_get(url: str, **kwargs):
    """Import requests only when performing I/O.

    The repository's lightweight CI job intentionally installs only pytest.
    Keeping the optional HTTP dependency out of module import time lets mocked
    provider and registry tests run there, while the full application job uses
    requests from requirements.txt.
    """
    import requests
    try:
        return requests.get(url, **kwargs)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ProviderUnavailableError("MyFantasyLeague is temporarily unavailable.") from exc
    except requests.RequestException as exc:
        raise ProviderUnavailableError("MyFantasyLeague returned an invalid response.") from exc


def _raise_for_status(response) -> None:
    """Translate checked HTTP failures without importing requests at collection."""
    import requests
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderUnavailableError("MyFantasyLeague is temporarily unavailable.") from exc


def _items(value: Any, singular: str) -> list[dict]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        nested = value.get(singular)
        if isinstance(nested, list): return [x for x in nested if isinstance(x, dict)]
        if isinstance(nested, dict): return [nested]
        return [value] if singular in value else []
    return []


def _num(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default


def _int(value, default=0):
    try: return int(value)
    except (TypeError, ValueError): return default


class MFLProvider(ProviderAdapter):
    metadata = ProviderMetadata(
        "mfl", "MyFantasyLeague", "league_id", capabilities=frozenset({
            LEAGUE, USERS, ROSTERS, MATCHUPS, STANDINGS,
            TRANSACTIONS, TRADES, DRAFTS, DRAFT_RESULTS, TRADED_PICKS,
            FUTURE_PICKS, HISTORY, SCORING_SETTINGS, ROSTER_SETTINGS,
        }),
    )

    def _export(self, export_type: str, league_id: str, season: int, *, ttl=300, **params) -> dict:
        league_id = str(league_id).strip()
        if not league_id.isdigit() or not (2000 <= _int(season) <= 2100):
            raise LeagueNotFoundError("Invalid MFL league ID or season.")
        key = (export_type, league_id, int(season), tuple(sorted(params.items())))
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        query = {"TYPE": export_type, "L": league_id, "JSON": 1, **params}
        try:
            response = _request_get(BASE_URL.format(season=int(season)), params=query,
                                    timeout=TIMEOUT, headers={"User-Agent": "BR-Fantasy/1.0"})
            if response.status_code in (401, 403):
                raise ProviderAuthenticationError("This MFL league is private or requires authentication.")
            if response.status_code == 404:
                raise LeagueNotFoundError("No MFL league was found for that ID and season.")
            _raise_for_status(response)
            payload = response.json()
        except (ProviderAuthenticationError, LeagueNotFoundError):
            raise
        except ProviderUnavailableError:
            raise
        except ValueError as exc:
            logger.warning("MFL export failed type=%s league=%s error=%s", export_type, league_id, type(exc).__name__)
            raise ProviderUnavailableError("MyFantasyLeague returned an invalid response.") from exc
        if not isinstance(payload, dict):
            raise ProviderUnavailableError("MyFantasyLeague returned an invalid response.")
        error = payload.get("error") or payload.get("errors")
        if error:
            message = str(error).lower()
            if "private" in message or "login" in message or "password" in message:
                raise ProviderAuthenticationError("This MFL league is private or requires authentication.")
            raise LeagueNotFoundError("No MFL league was found for that ID and season.")
        _CACHE[key] = (time.monotonic(), payload)
        return payload

    def get_league(self, league_id, season):
        raw = self._export("league", league_id, season, ttl=1800)
        lg = raw.get("league") or {}
        if not isinstance(lg, dict) or not lg:
            raise LeagueNotFoundError("No MFL league was found for that ID and season.")
        franchises = _items((lg.get("franchises") or {}).get("franchise", []), "franchise")
        return {
            "league_id": str(lg.get("id") or league_id), "season": int(season),
            "name": lg.get("name") or "MyFantasyLeague League",
            "total_rosters": _int(lg.get("size") or len(franchises)),
            "settings": {"playoff_week_start": _int(lg.get("lastRegularSeasonWeek"), 14) + 1,
                         "league_type": lg.get("type") or "redraft"},
            "scoring_settings": self._scoring(lg),
            "roster_positions": self._positions(lg),
            "metadata": {"divisions": lg.get("divisions"), "conferences": lg.get("conferences")},
        }

    def _franchises(self, league_id, season):
        lg = (self._export("league", league_id, season, ttl=1800).get("league") or {})
        return _items((lg.get("franchises") or {}).get("franchise", []), "franchise")

    def get_users(self, league_id, season):
        return [{"user_id": str(f.get("id")), "roster_id": _int(f.get("id")),
                 "display_name": f.get("owner_name") or f.get("name") or f"Team {f.get('id')}",
                 "avatar": f.get("icon") or f.get("logo"), "league_id": str(league_id),
                 "metadata": {"team_name": f.get("name") or f"Team {f.get('id')}",
                              "provider_franchise_id": str(f.get("id"))}}
                for f in self._franchises(league_id, season) if f.get("id") is not None]

    @lru_cache(maxsize=8)
    def _player_map(self, season: int) -> dict[str, str]:
        raw = self._export("players", "0", season, ttl=86400) if False else {}
        # Populated lazily per league below because MFL requires L on some hosts.
        return raw

    def _canonical_map(self, league_id, season):
        raw = self._export("players", league_id, season, ttl=86400, DETAILS=1)
        players = _items((raw.get("players") or {}).get("player", []), "player")
        try:
            from utils.utils import load_players_index, normalize_name
            index = load_players_index() or {}
            by_name = {}
            for canonical, info in index.items():
                name = normalize_name(info.get("full_name") or info.get("name") or "")
                pos = str(info.get("position") or "").upper()
                if name: by_name[(name, pos)] = str(canonical)
            out = {}
            for p in players:
                name = normalize_name(p.get("name") or "")
                pos = str(p.get("position") or "").upper()
                canonical = by_name.get((name, pos)) or next((v for (n, _), v in by_name.items() if n == name), None)
                if canonical: out[str(p.get("id"))] = canonical
            return out
        except Exception as exc:
            logger.warning("MFL player crosswalk unavailable error=%s", type(exc).__name__)
            return {}

    def get_rosters(self, league_id, season):
        raw = self._export("rosters", league_id, season, ttl=300)
        rosters = _items((raw.get("rosters") or {}).get("franchise", []), "franchise")
        xwalk = self._canonical_map(league_id, season)
        out = []
        for r in rosters:
            entries = _items(r.get("player", []), "player")
            mapped = [(xwalk.get(str(p.get("id"))), str(p.get("status") or "")) for p in entries]
            players = [p for p, _ in mapped if p]
            reserve = [p for p, status in mapped if p and status.upper() in {"INJURED_RESERVE", "TAXI_SQUAD"}]
            out.append({"league_id": str(league_id), "roster_id": _int(r.get("id")),
                        "owner_id": str(r.get("id")), "players": players,
                        "starters": [], "reserve": reserve, "taxi": None,
                        "settings": {}, "metadata": {"unmapped_player_count": len(entries)-len(players)}})
        return out

    def get_matchups(self, league_id, season, week):
        raw = self._export("weeklyResults", league_id, season, ttl=600, W=int(week))
        block = raw.get("weeklyResults") or {}
        matchups = _items(block.get("matchup", []), "matchup")
        out = []
        for mid, matchup in enumerate(matchups, 1):
            franchises = _items(matchup.get("franchise", []), "franchise")
            for team in franchises:
                out.append({"matchup_id": mid, "roster_id": _int(team.get("id")),
                            "points": _num(team.get("score")), "players": [], "starters": [],
                            "starters_points": [], "players_points": {}, "week": int(week),
                            "custom_points": None})
        return out

    def get_transactions(self, league_id, season, week):
        raw = self._export("transactions", league_id, season, ttl=300)
        txs = _items((raw.get("transactions") or {}).get("transaction", []), "transaction")
        out = []
        for i, tx in enumerate(txs):
            kind = str(tx.get("type") or "").upper()
            normalized = "trade" if "TRADE" in kind else ("waiver" if "WAIVER" in kind else "free_agent")
            out.append({"transaction_id": str(tx.get("id") or f"mfl-{season}-{i}"),
                        "type": normalized, "status": "complete", "created": _int(tx.get("timestamp")),
                        "roster_ids": [str(x) for x in str(tx.get("franchise") or "").split(",") if x],
                        "adds": {}, "drops": {}, "draft_picks": [],
                        "metadata": {"provider_type": kind, "provider_data": tx.get("transaction")}})
        return out

    def get_drafts(self, league_id, season):
        raw = self._export("draftResults", league_id, season, ttl=3600)
        picks = _items((raw.get("draftResults") or {}).get("draftUnit", []), "draftUnit")
        return [{"draft_id": f"mfl:{season}:{league_id}", "league_id": str(league_id),
                 "season": str(season), "status": "complete", "type": "snake",
                 "metadata": {"name": "MFL Draft"}, "settings": {},
                 "draft_order": {}, "slot_to_roster_id": {}, "last_picked": 0,
                 "picks": [{"round": _int(p.get("round")), "pick_no": _int(p.get("pick")),
                            "roster_id": _int(p.get("franchise")),
                            "player_id": str(p.get("player") or ""), "picked_by": str(p.get("franchise") or ""),
                            "metadata": {"timestamp": p.get("timestamp"), "auction_amount": p.get("amount")}}
                           for p in picks]}]

    def get_traded_picks(self, league_id, season):
        raw = self._export("futureDraftPicks", league_id, season, ttl=1800)
        picks = _items((raw.get("futureDraftPicks") or {}).get("futureDraftPick", []), "futureDraftPick")
        return [{"season": str(p.get("year") or season), "round": _int(p.get("round")),
                 "roster_id": _int(p.get("originalPickFor") or p.get("original_franchise")),
                 "owner_id": _int(p.get("currentPickFor") or p.get("franchise")),
                 "previous_owner_id": _int(p.get("previousPickFor")) or None}
                for p in picks]

    def get_bracket(self, league_id, season, kind):
        raise UnsupportedCapabilityError("MFL playoff brackets are not reliably available through the export API.")

    @staticmethod
    def _positions(lg):
        raw = str(lg.get("starters") or lg.get("rosterSize") or "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    @staticmethod
    def _scoring(lg):
        rules = _items((lg.get("rules") or {}).get("positionRule", []), "positionRule")
        return {str(r.get("event") or r.get("id")): _num(r.get("points")) for r in rules if r.get("event") or r.get("id")}

    def get_league_globals(self, league_id, season):
        league = self.get_league(league_id, season)
        return {"scoring_settings": league.get("scoring_settings"),
                "roster_positions": league.get("roster_positions"),
                "league_settings": league.get("settings"), "total_rosters": league.get("total_rosters")}
