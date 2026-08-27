"""Read-only Fleaflicker API provider.

Raw upstream names stay in this module. Public methods return the existing
Sleeper-compatible dictionaries consumed by BR Fantasy.

Public leagues need no auth. Private leagues use the undocumented ``/api/Login``
token, passed in the ``Authorization`` header. Passwords are never stored —
only the returned token is retained (encrypted by the accounts layer).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .base import (
    DRAFTS, DRAFT_RESULTS, FUTURE_PICKS, HISTORY, LEAGUE, MATCHUPS, ROSTERS,
    ROSTER_SETTINGS, SCORING_SETTINGS, STANDINGS, STARTERS, TRADED_PICKS,
    TRANSACTIONS, TRADES, USERS, LeagueNotFoundError, ProviderAdapter,
    ProviderAuthenticationError, ProviderMetadata, ProviderUnavailableError,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)
BASE_URL = "https://www.fleaflicker.com/api"
SPORT = "NFL"
TIMEOUT = (5, 20)
# Browser-like UA: some Fleaflicker edge configs are picky about obvious bots.
_UA = (
    "Mozilla/5.0 (compatible; BR-Fantasy/1.0; +https://www.brfantasyfootball.com) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fleaflicker.com/",
}
# Endpoints whose OpenAPI signatures do not take ``season``.
_NO_SEASON = frozenset({"FetchLeagueRules"})
_CACHE: dict[tuple, tuple[float, dict]] = {}


def _request_get(url: str, **kwargs):
    import requests
    try:
        return requests.get(url, **kwargs)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.") from exc
    except requests.RequestException as exc:
        raise ProviderUnavailableError("Fleaflicker returned an invalid response.") from exc


def _request_post(url: str, **kwargs):
    import requests
    try:
        return requests.post(url, **kwargs)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.") from exc
    except requests.RequestException as exc:
        raise ProviderUnavailableError("Fleaflicker returned an invalid response.") from exc


def _raise_for_status(response) -> None:
    """Translate checked HTTP failures without importing requests at collection.

    Successful mocked responses must not require ``requests`` so the lightweight
    unit-test job (pytest only) can exercise provider auth paths.
    """
    status = getattr(response, "status_code", None)
    try:
        if status is not None and 200 <= int(status) < 400:
            return
    except (TypeError, ValueError):
        pass
    import requests
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.") from exc


def _response_looks_like_html(response) -> bool:
    content_type = str((getattr(response, "headers", None) or {}).get("Content-Type") or "").lower()
    if "text/html" in content_type or "application/xhtml" in content_type:
        return True
    text = (getattr(response, "text", None) or "")[:200].lstrip().lower()
    return text.startswith("<!doctype html") or text.startswith("<html")


def _classify_http_error(response, method: str, league_id: str):
    """Map upstream HTTP failures to provider errors without leaking bodies."""
    status = getattr(response, "status_code", None)
    logger.warning(
        "Fleaflicker HTTP failure method=%s league=%s status=%s content_type=%s html=%s",
        method, league_id, status,
        (getattr(response, "headers", None) or {}).get("Content-Type"),
        _response_looks_like_html(response),
    )
    # Edge/WAF HTML blocks are infrastructure, not "private league" auth.
    if status in (401, 403) and _response_looks_like_html(response):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
    if status in (401, 403):
        raise ProviderAuthenticationError(
            "This Fleaflicker league is private or requires authentication."
        )
    if status == 404:
        raise LeagueNotFoundError("No Fleaflicker league was found for that ID and season.")
    raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")


def _num(value, default=0.0):
    """Read a number that may be absent (omitted zero) or wrapped as {value}."""
    if isinstance(value, dict):
        value = value.get("value")
        if isinstance(value, dict):
            value = value.get("value")
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(_num(value, default))
    except (TypeError, ValueError):
        return default


def _get(data: dict, *keys, default=None):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def normalize_auth_token(token: Optional[str]) -> str:
    """Return a clean Authorization token; strip an optional Bearer prefix."""
    value = str(token or "").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def login(email: str, password: str) -> str:
    """Exchange email/password for a session token. Does not persist the password."""
    email = str(email or "").strip()
    password = str(password or "")
    if not email or not password:
        raise ProviderAuthenticationError("Fleaflicker email and password are required.")
    try:
        response = _request_post(
            f"{BASE_URL}/Login",
            json={"email": email, "password": password},
            timeout=TIMEOUT,
            headers={**_HEADERS, "Content-Type": "application/json"},
        )
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError("Fleaflicker rejected that email or password.")
        _raise_for_status(response)
        payload = response.json()
    except (ProviderAuthenticationError, ProviderUnavailableError):
        raise
    except ValueError as exc:
        raise ProviderUnavailableError("Fleaflicker returned an invalid login response.") from exc
    if not isinstance(payload, dict):
        raise ProviderUnavailableError("Fleaflicker returned an invalid login response.")
    failure = payload.get("failure")
    if failure:
        raise ProviderAuthenticationError("Fleaflicker rejected that email or password.")
    user = payload.get("user") or {}
    token = normalize_auth_token(user.get("token") if isinstance(user, dict) else "")
    if not token:
        raise ProviderAuthenticationError("Fleaflicker login did not return a session token.")
    return token


def resolve_credentials(
    league_id: str, season: int, *, token: Optional[str] = None,
) -> Optional[str]:
    """Prefer an explicit token, then account-stored / staged private credentials."""
    explicit = normalize_auth_token(token)
    if explicit:
        return explicit
    try:
        from flask import has_request_context, session
        if not has_request_context():
            return None
        account_id = session.get("account_id")
        if account_id:
            from dashboard_services.accounts import get_provider_league_credentials
            stored = get_provider_league_credentials(
                int(account_id), "fleaflicker", league_id, season,
            ) or {}
            return normalize_auth_token(stored.get("token"))
        pending = session.get("pending_provider_connection_token")
        if pending:
            from dashboard_services.accounts import peek_private_provider_connection
            staged = peek_private_provider_connection(
                pending, "fleaflicker", league_id, season,
            ) or {}
            return normalize_auth_token(staged.get("token"))
    except Exception:
        logger.debug("Fleaflicker credential lookup failed", exc_info=True)
    return None


class FleaflickerProvider(ProviderAdapter):
    metadata = ProviderMetadata(
        "fleaflicker", "Fleaflicker", "league_id", capabilities=frozenset({
            LEAGUE, USERS, ROSTERS, STARTERS, MATCHUPS, STANDINGS,
            TRANSACTIONS, TRADES, DRAFTS, DRAFT_RESULTS, TRADED_PICKS,
            FUTURE_PICKS, HISTORY, SCORING_SETTINGS, ROSTER_SETTINGS,
        }),
    )

    def _call(
        self, method: str, league_id: str, season: int, *, ttl=300,
        token: Optional[str] = None, include_season: Optional[bool] = None, **params,
    ) -> dict:
        league_id = str(league_id).strip()
        if not league_id.isdigit() or not (2000 <= _int(season) <= 2100):
            raise LeagueNotFoundError("Invalid Fleaflicker league ID or season.")
        auth = resolve_credentials(league_id, season, token=token)
        if include_season is None:
            include_season = method not in _NO_SEASON
        key = (
            method, league_id, int(season) if include_season else 0,
            auth or "", include_season, tuple(sorted(params.items())),
        )
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        query = {"sport": SPORT, "league_id": int(league_id), **params}
        if include_season:
            query["season"] = int(season)
        headers = dict(_HEADERS)
        if auth:
            headers["Authorization"] = auth
        try:
            response = _request_get(
                f"{BASE_URL}/{method}", params=query, timeout=TIMEOUT, headers=headers,
            )
            if response.status_code >= 400:
                _classify_http_error(response, method, league_id)
            if _response_looks_like_html(response):
                logger.warning(
                    "Fleaflicker non-JSON body method=%s league=%s status=%s",
                    method, league_id, response.status_code,
                )
                raise ProviderUnavailableError("Fleaflicker returned an invalid response.")
            payload = response.json()
        except (ProviderAuthenticationError, LeagueNotFoundError, ProviderUnavailableError):
            raise
        except ValueError as exc:
            logger.warning(
                "Fleaflicker call failed method=%s league=%s error=%s",
                method, league_id, type(exc).__name__,
            )
            raise ProviderUnavailableError("Fleaflicker returned an invalid response.") from exc
        if not isinstance(payload, dict):
            raise ProviderUnavailableError("Fleaflicker returned an invalid response.")
        error = payload.get("error")
        if error:
            message = str(error.get("message") if isinstance(error, dict) else error).lower()
            if any(x in message for x in ("private", "auth", "login", "permission", "forbidden")):
                raise ProviderAuthenticationError(
                    "This Fleaflicker league is private or requires authentication."
                )
            if "not found" in message:
                raise LeagueNotFoundError("No Fleaflicker league was found for that ID and season.")
            logger.warning(
                "Fleaflicker API error method=%s league=%s message=%s",
                method, league_id, message[:120],
            )
            raise ProviderUnavailableError("Fleaflicker returned an invalid response.")
        _CACHE[key] = (time.monotonic(), payload)
        return payload

    def connect_league(self, league_id: str, season: int, *, token: Optional[str] = None) -> dict:
        """Validate access and return a minimal league summary for connect flows."""
        league = self.get_league(league_id, season, token=token)
        return {"name": league.get("name"), "league_id": league.get("league_id"),
                "season": league.get("season"), "total_rosters": league.get("total_rosters")}

    def get_league(self, league_id, season, *, token: Optional[str] = None):
        standings = None
        try:
            standings = self._call(
                "FetchLeagueStandings", league_id, season, ttl=1800, token=token,
            )
        except LeagueNotFoundError:
            standings = None
        league = (standings or {}).get("league") or {}
        teams = self._teams_from_standings(standings or {})
        # Early / rolled season: season-specific standings can 404 or come back
        # empty before Fleaflicker opens that year — retry without season.
        if not league and not teams:
            standings = self._call(
                "FetchLeagueStandings", league_id, season, ttl=1800, token=token,
                include_season=False,
            )
            league = standings.get("league") or {}
            teams = self._teams_from_standings(standings)
        if not league and not teams:
            raise LeagueNotFoundError("No Fleaflicker league was found for that ID and season.")
        # Rules are optional for connect/preview: standings alone identifies the league.
        # FetchLeagueRules does not take season; never fail the whole connect on rules.
        rules: dict = {}
        try:
            rules = self._call(
                "FetchLeagueRules", league_id, season, ttl=3600, token=token,
                include_season=False,
            )
        except Exception:
            logger.warning(
                "Fleaflicker rules unavailable league=%s season=%s", league_id, season,
                exc_info=True,
            )
        return {
            "league_id": str(_get(league, "id") or league_id),
            "season": int(_get(standings or {}, "season") or season),
            "name": _get(league, "name") or f"Fleaflicker League {league_id}",
            "total_rosters": _int(_get(league, "size")) or len(teams),
            "settings": {"league_type": "redraft"},
            "scoring_settings": self._scoring(rules),
            "roster_positions": self._positions(rules),
            "metadata": {"provider": "fleaflicker"},
        }

    def get_users(self, league_id, season, *, token: Optional[str] = None):
        standings = self._call("FetchLeagueStandings", league_id, season, ttl=1800, token=token)
        out = []
        for team in self._teams_from_standings(standings):
            owners = team.get("owners") or []
            owner = owners[0] if owners else {}
            team_id = _get(team, "id")
            if team_id is None:
                continue
            display = (
                _get(owner, "displayName", "display_name")
                or _get(team, "name")
                or f"Team {team_id}"
            )
            out.append({
                "user_id": str(_get(owner, "id") or team_id),
                "roster_id": _int(team_id),
                "display_name": display,
                "avatar": _get(owner, "avatarUrl", "avatar_url"),
                "league_id": str(league_id),
                "metadata": {
                    "team_name": _get(team, "name") or f"Team {team_id}",
                    "provider_team_id": str(team_id),
                },
            })
        return out

    def _canonical_map(self, league_id, season, *, token: Optional[str] = None):
        try:
            from utils.utils import load_players_index, normalize_name
            index = load_players_index() or {}
            by_name = {}
            for canonical, info in index.items():
                name = normalize_name(info.get("full_name") or info.get("name") or "")
                pos = str(info.get("position") or "").upper()
                if name:
                    by_name[(name, pos)] = str(canonical)
            out = {}
            raw = self._call("FetchLeagueRosters", league_id, season, ttl=300, token=token)
            for roster in raw.get("rosters") or []:
                for player in roster.get("players") or []:
                    pro = player.get("proPlayer") or player.get("pro_player") or {}
                    name = normalize_name(_get(pro, "nameFull", "name_full") or "")
                    pos = str(_get(pro, "position") or "").upper()
                    pid = _get(pro, "id")
                    if pid is None or not name:
                        continue
                    canonical = by_name.get((name, pos)) or next(
                        (v for (n, _), v in by_name.items() if n == name), None
                    )
                    if canonical:
                        out[str(pid)] = canonical
            return out
        except Exception as exc:
            logger.warning("Fleaflicker player crosswalk unavailable error=%s", type(exc).__name__)
            return {}

    def get_rosters(self, league_id, season, *, token: Optional[str] = None):
        raw = self._call("FetchLeagueRosters", league_id, season, ttl=300, token=token)
        xwalk = self._canonical_map(league_id, season, token=token)
        out = []
        for roster in raw.get("rosters") or []:
            team = roster.get("team") or {}
            team_id = _get(team, "id")
            if team_id is None:
                continue
            players, starters, reserve = [], [], []
            # Prefer the flat players list; fall back to FetchRoster groups when absent.
            entries = roster.get("players") or []
            if not entries:
                try:
                    detail = self._call(
                        "FetchRoster", league_id, season, ttl=300, token=token,
                        team_id=int(team_id),
                    )
                    for group in detail.get("groups") or []:
                        group_label = str(group.get("group") or "").upper()
                        for slot in group.get("slots") or []:
                            lp = slot.get("leaguePlayer") or slot.get("league_player") or {}
                            pro = lp.get("proPlayer") or lp.get("pro_player") or {}
                            pid = _get(pro, "id")
                            if pid is None:
                                continue
                            cid = xwalk.get(str(pid))
                            if not cid:
                                continue
                            players.append(cid)
                            if group_label == "START":
                                starters.append(cid)
                            elif group_label in {"IR", "INJURED_RESERVE"}:
                                reserve.append(cid)
                    entries = None
                except Exception:
                    logger.debug("Fleaflicker FetchRoster fallback failed", exc_info=True)
            if entries is not None:
                for player in entries:
                    pro = player.get("proPlayer") or player.get("pro_player") or {}
                    pid = _get(pro, "id")
                    if pid is None:
                        continue
                    cid = xwalk.get(str(pid))
                    if cid:
                        players.append(cid)
            out.append({
                "league_id": str(league_id), "roster_id": _int(team_id),
                "owner_id": str(team_id), "players": players,
                "starters": starters, "reserve": reserve, "taxi": None,
                "settings": {}, "metadata": {
                    "unmapped_player_count": max(0, len(entries or []) - len(players))
                    if entries is not None else 0,
                    "provider_team_id": str(team_id),
                },
            })
        return out

    def get_matchups(self, league_id, season, week, *, token: Optional[str] = None):
        raw = self._call(
            "FetchLeagueScoreboard", league_id, season, ttl=600, token=token,
            scoring_period=int(week),
        )
        out = []
        for mid, game in enumerate(raw.get("games") or [], 1):
            for side, score_key in (("home", "homeScore"), ("away", "awayScore")):
                team = game.get(side) or {}
                team_id = _get(team, "id")
                if team_id is None:
                    continue
                score_block = game.get(score_key) or game.get(
                    "home_score" if side == "home" else "away_score"
                ) or {}
                points = _num(score_block.get("score") if isinstance(score_block, dict) else score_block)
                out.append({
                    "matchup_id": mid, "roster_id": _int(team_id),
                    "points": points, "players": [], "starters": [],
                    "starters_points": [], "players_points": {},
                    "week": int(week), "custom_points": None,
                    "metadata": {"provider_game_id": str(_get(game, "id") or "")},
                })
        return out

    def get_transactions(self, league_id, season, week, *, token: Optional[str] = None):
        raw = self._call("FetchLeagueTransactions", league_id, season, ttl=300, token=token)
        out = []
        for i, tx in enumerate(raw.get("transactions") or raw.get("items") or []):
            kind = str(_get(tx, "type", "transactionType", "transaction_type") or "").upper()
            normalized = "trade" if "TRADE" in kind else (
                "waiver" if "WAIVER" in kind else "free_agent"
            )
            out.append({
                "transaction_id": str(_get(tx, "id") or f"flea-{season}-{i}"),
                "type": normalized, "status": "complete",
                "created": _int(_get(tx, "timestamp", "timeEpochMilli", "time_epoch_milli")),
                "roster_ids": [], "adds": {}, "drops": {}, "draft_picks": [],
                "metadata": {"provider_type": kind},
            })
        return out

    def get_drafts(self, league_id, season, *, token: Optional[str] = None):
        raw = self._call("FetchLeagueDraftBoard", league_id, season, ttl=3600, token=token)
        picks = []
        overall = 0
        for row in raw.get("rows") or []:
            for cell in row.get("cells") or []:
                overall += 1
                team = cell.get("team") or {}
                drafted = cell.get("player") or {}
                pro = drafted.get("proPlayer") or drafted.get("pro_player") or drafted
                picks.append({
                    "round": _int(row.get("round")),
                    "pick_no": overall,
                    "roster_id": _int(_get(team, "id")),
                    "player_id": str(_get(pro, "id") or ""),
                    "picked_by": str(_get(team, "id") or ""),
                    "metadata": {
                        "player_name": _get(pro, "nameFull", "name_full"),
                        "position": _get(pro, "position"),
                    },
                })
        return [{
            "draft_id": f"fleaflicker:{season}:{league_id}",
            "league_id": str(league_id), "season": str(season),
            "status": "complete", "type": "snake",
            "metadata": {"name": "Fleaflicker Draft"}, "settings": {},
            "draft_order": {}, "slot_to_roster_id": {}, "last_picked": 0,
            "picks": picks,
        }]

    def get_traded_picks(self, league_id, season, *, token: Optional[str] = None):
        # Future picks are per-team; aggregate from standings team ids.
        standings = self._call("FetchLeagueStandings", league_id, season, ttl=1800, token=token)
        out = []
        for team in self._teams_from_standings(standings):
            team_id = _get(team, "id")
            if team_id is None:
                continue
            try:
                raw = self._call(
                    "FetchTeamPicks", league_id, season, ttl=1800, token=token,
                    team_id=int(team_id),
                )
            except Exception:
                continue
            for pick in raw.get("picks") or raw.get("futurePicks") or raw.get("future_picks") or []:
                out.append({
                    "season": str(_get(pick, "season", "year") or season),
                    "round": _int(_get(pick, "round")),
                    "roster_id": _int(_get(pick, "originalOwner", "original_owner", "ownedBy", "owned_by")
                                      or team_id),
                    "owner_id": _int(_get(pick, "owner", "currentOwner", "current_owner") or team_id),
                    "previous_owner_id": None,
                })
        return out

    def get_bracket(self, league_id, season, kind):
        raise UnsupportedCapabilityError(
            "Fleaflicker playoff brackets are not exposed through the public API."
        )

    @staticmethod
    def _teams_from_standings(standings: dict) -> list[dict]:
        teams = []
        for division in standings.get("divisions") or []:
            for team in division.get("teams") or []:
                if isinstance(team, dict):
                    teams.append(team)
        if not teams:
            for team in standings.get("teams") or []:
                if isinstance(team, dict):
                    teams.append(team)
        return teams

    @staticmethod
    def _positions(rules: dict) -> list[str]:
        positions = rules.get("rosterPositions") or rules.get("roster_positions") or []
        out = []
        for pos in positions:
            label = str(pos.get("label") or "").strip()
            starts = _int(pos.get("start"))
            if not label or starts <= 0:
                continue
            out.extend([label] * starts)
        return out

    @staticmethod
    def _scoring(rules: dict) -> dict:
        out = {}
        for group in rules.get("groups") or []:
            for rule in group.get("scoringRules") or group.get("scoring_rules") or []:
                cat = rule.get("category") or {}
                key = str(_get(cat, "abbreviation", "nameSingular", "name_singular") or "").strip()
                if key:
                    out[key] = _num(_get(rule, "points", "pointsPer", "points_per"))
        return out

    def get_league_globals(self, league_id, season, *, token: Optional[str] = None):
        league = self.get_league(league_id, season, token=token)
        return {
            "scoring_settings": league.get("scoring_settings"),
            "roster_positions": league.get("roster_positions"),
            "league_settings": league.get("settings"),
            "total_rosters": league.get("total_rosters"),
        }
