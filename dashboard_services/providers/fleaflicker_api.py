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

# players_index uses ``pos``; a few overlays still use ``position``.
_SKILL_POS = frozenset({"QB", "RB", "WR", "TE", "K", "DEF", "DST", "PK"})
_IDP_POS = frozenset({
    "CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB",
    "DL", "DE", "DT", "EDGE", "IDP",
})
_START_GROUPS = frozenset({"START", "STARTERS", "STARTING"})
# Keep local so this module still collects in the slim CI job (no utils.utils).
_TEAM_ABBR_ALIASES = {
    "WAS": "WSH", "WSH": "WAS",
    "JAC": "JAX", "JAX": "JAC",
    "LA": "LAR", "LAR": "LA",
}
# (group label, category abbreviation) -> Sleeper scoring key.
# Bare TD/Yd without a group is ambiguous (pass/rush/rec) and must not map.
_FLEA_STAT_BY_GROUP_ABBREV = {
    ("passing", "yd"): "pass_yd",
    ("passing", "py"): "pass_yd",
    ("passing", "td"): "pass_td",
    ("passing", "int"): "pass_int",
    ("passing", "2pc"): "pass_2pt",
    ("passing", "fd"): "pass_fd",
    ("rushing", "yd"): "rush_yd",
    ("rushing", "ry"): "rush_yd",
    ("rushing", "td"): "rush_td",
    ("rushing", "2pc"): "rush_2pt",
    ("rushing", "fd"): "rush_fd",
    ("receiving", "yd"): "rec_yd",
    ("receiving", "rey"): "rec_yd",
    ("receiving", "td"): "rec_td",
    ("receiving", "rec"): "rec",
    ("receiving", "catch"): "rec",
    ("receiving", "2pc"): "rec_2pt",
    ("receiving", "fd"): "rec_fd",
    ("misc", "fum"): "fum_lost",
    ("misc", "lost"): "fum_lost",
    ("defense", "int"): "def_int",
    ("defense", "sack"): "sack",
    ("kicking", "fg"): "fgm",
    ("kicking", "xpt"): "xpm",
    ("kicking", "xp"): "xpm",
    ("kicking", "pat"): "xpm",
}
_FLEA_STAT_BY_NAME = {
    "passing yard": "pass_yd",
    "passing td": "pass_td",
    "2 pt conversion passing": "pass_2pt",
    "passing first down": "pass_fd",
    "rushing yard": "rush_yd",
    "rushing td": "rush_td",
    "2 pt conversion rushing": "rush_2pt",
    "rushing first down": "rush_fd",
    "receiving yard": "rec_yd",
    "receiving td": "rec_td",
    "catch": "rec",
    "reception": "rec",
    "2 pt conversion receiving": "rec_2pt",
    "receiving first down": "rec_fd",
    "fumble": "fum_lost",
}


def _index_pos(info: dict) -> str:
    return str((info or {}).get("pos") or (info or {}).get("position") or "").upper()


def _index_team(info: dict) -> str:
    return str((info or {}).get("team") or "").upper()


def _team_keys(team: str) -> set[str]:
    t = (team or "").strip().upper()
    if not t:
        return set()
    alt = _TEAM_ABBR_ALIASES.get(t)
    return {t, alt} if alt else {t}


def _flea_pro_team(pro: dict) -> str:
    """NFL team from a Fleaflicker ``proPlayer`` (abbrev or nested proTeam)."""
    if not isinstance(pro, dict):
        return ""
    direct = _get(pro, "proTeamAbbreviation", "pro_team_abbreviation")
    if direct:
        return str(direct).upper()
    team = _get(pro, "proTeam", "pro_team")
    if isinstance(team, dict):
        return str(_get(team, "abbreviation") or "").upper()
    if isinstance(team, str):
        return team.upper()
    return ""


def _flea_stat_key(group_label: str, category: dict) -> Optional[str]:
    """Sleeper key for one Fleaflicker scoring category, or None if ambiguous."""
    cat = category if isinstance(category, dict) else {}
    group = str(group_label or "").strip().lower()
    abbrev = str(_get(cat, "abbreviation") or "").strip().lower()
    name = str(
        _get(cat, "nameSingular", "name_singular") or ""
    ).strip().lower()
    if group and abbrev:
        mapped = _FLEA_STAT_BY_GROUP_ABBREV.get((group, abbrev))
        if mapped:
            return mapped
    if name:
        mapped = _FLEA_STAT_BY_NAME.get(name)
        if mapped:
            return mapped
        if "interception" in name:
            if "pass" in group:
                return "pass_int"
            if "defense" in group or group in {"dst", "d/st"}:
                return "def_int"
            return None
    return None


def _flea_is_threshold_bonus(rule: dict) -> bool:
    """True for milestone extras, not per-stat rates.

    Fleaflicker lists both ``1 point for every 10 Receiving Yards`` and
    ``1 extra point when total Receiving Yards >= 150`` under the same
    ``Yd`` abbreviation. Treating the extra as a rate scored 83-yard
    games as 83 points and 9-catch bonuses as 2-PPR.
    """
    if not isinstance(rule, dict):
        return False
    if _get(rule, "boundLower", "bound_lower") is not None:
        return True
    if _get(rule, "boundUpper", "bound_upper") is not None:
        return True
    text = f"{rule.get('description') or ''} {rule.get('template') or ''}".lower()
    if "extra point" in text or "when total" in text:
        return True
    return False


def _flea_points_per(rule: dict) -> Optional[float]:
    """Per-stat rate. Prefer ``pointsPer`` (already 0.04 / yard) over raw points."""
    if not isinstance(rule, dict):
        return None
    pper = _get(rule, "pointsPer", "points_per")
    if pper is not None:
        return _num(pper)
    pts_raw = _get(rule, "points")
    if pts_raw is None:
        return None
    pts = _num(pts_raw)
    every = _get(rule, "forEvery", "for_every")
    if every is not None:
        ev = _num(every)
        if ev and ev != 1.0:
            return pts / ev
    return pts


def _name_index_from_players(index: dict, normalize_name) -> dict:
    """name -> [(pos, team, canonical_id), ...] so namesakes stay distinct."""
    by_name: dict[str, list[tuple[str, str, str]]] = {}
    for canonical, info in (index or {}).items():
        if not isinstance(info, dict):
            continue
        name = normalize_name(info.get("full_name") or info.get("name") or "")
        if not name:
            continue
        entry = (_index_pos(info), _index_team(info), str(canonical))
        bucket = by_name.setdefault(name, [])
        if entry not in bucket:
            bucket.append(entry)
    return by_name


def _norm_name_candidate(cand) -> tuple[str, str, Any]:
    """Accept (pos, id), (pos, team, id), or a bare id."""
    if isinstance(cand, (list, tuple)):
        if len(cand) >= 3:
            return str(cand[0] or "").upper(), str(cand[1] or "").upper(), cand[2]
        if len(cand) == 2:
            return str(cand[0] or "").upper(), "", cand[1]
        if len(cand) == 1:
            return "", "", cand[0]
    return "", "", cand


def _pick_canonical(
    by_name: dict, name: str, flea_pos: str = "", flea_team: str = "",
) -> Optional[str]:
    """Resolve a Fleaflicker name/pos/team onto a Sleeper id.

    ``players_index`` has duplicate names (Lamar Jackson QB BAL vs CB ATL,
    DeVonta Smith WR PHI vs CB CAR, Josh Allen QB BUF vs DE JAX). Prefer
    exact position + NFL team, then position, then the matching team —
    never the IDP namesake that used to paint Lamar as IDP / FA.
    """
    if not name:
        return None
    # Legacy {(name, pos): id} map from older tests / callers.
    if by_name and isinstance(next(iter(by_name.keys()), None), tuple):
        pos = (flea_pos or "").upper()
        hit = by_name.get((name, pos))
        if hit:
            return hit
        return next((v for (n, _), v in by_name.items() if n == name), None)

    candidates = [_norm_name_candidate(c) for c in (by_name.get(name) or [])]
    candidates = [c for c in candidates if c[2] is not None]
    if not candidates:
        return None
    pos = (flea_pos or "").upper()
    team_keys = _team_keys(flea_team)

    def team_ok(cand_team: str) -> bool:
        return bool(team_keys and (_team_keys(cand_team) & team_keys))

    def first(rows: list[tuple[str, str, Any]]) -> Optional[str]:
        if not rows:
            return None
        if team_keys:
            matched = [c for c in rows if team_ok(c[1])]
            if matched:
                return matched[0][2]
        return rows[0][2]

    if pos:
        exact = [c for c in candidates if c[0] == pos]
        hit = first(exact)
        if hit:
            return hit
    if team_keys:
        skill_team = [c for c in candidates if c[0] in _SKILL_POS and team_ok(c[1])]
        if skill_team:
            return skill_team[0][2]
        any_team = [c for c in candidates if team_ok(c[1])]
        if any_team:
            return any_team[0][2]
    if pos in _SKILL_POS or not pos:
        skill = [c for c in candidates if c[0] in _SKILL_POS]
        hit = first(skill)
        if hit:
            return hit
    if pos in _SKILL_POS:
        non_idp = [c for c in candidates if c[0] not in _IDP_POS]
        hit = first(non_idp)
        if hit:
            return hit
    return candidates[0][2]


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
    """Exchange email/password for a session token. Does not persist the password.

    Fleaflicker's undocumented ``/api/Login`` expects ``loginId`` (typically the
    account email), not ``email``. Sending ``email`` returns an HTML 400.
    """
    email = str(email or "").strip()
    password = str(password or "")
    if not email or not password:
        raise ProviderAuthenticationError("Fleaflicker email and password are required.")
    try:
        response = _request_post(
            f"{BASE_URL}/Login",
            params={"sport": SPORT},
            json={"loginId": email, "password": password},
            timeout=TIMEOUT,
            headers={**_HEADERS, "Content-Type": "application/json"},
        )
        # Bad request bodies used to be HTML 400 and looked like an outage.
        if response.status_code >= 400:
            if response.status_code in (401, 403) and not _response_looks_like_html(response):
                raise ProviderAuthenticationError("Fleaflicker rejected that email or password.")
            if _response_looks_like_html(response) or response.status_code >= 500:
                logger.warning(
                    "Fleaflicker Login HTTP failure status=%s content_type=%s html=%s",
                    response.status_code,
                    (getattr(response, "headers", None) or {}).get("Content-Type"),
                    _response_looks_like_html(response),
                )
                raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")
            raise ProviderAuthenticationError("Fleaflicker rejected that email or password.")
        if _response_looks_like_html(response):
            raise ProviderUnavailableError("Fleaflicker returned an invalid login response.")
        payload = response.json()
    except (ProviderAuthenticationError, ProviderUnavailableError):
        raise
    except ValueError as exc:
        raise ProviderUnavailableError("Fleaflicker returned an invalid login response.") from exc
    if not isinstance(payload, dict):
        raise ProviderUnavailableError("Fleaflicker returned an invalid login response.")
    failure = payload.get("failure")
    if failure:
        # LOGIN_CAPTCHA_REQUIRED / unknown ids still mean the user cannot proceed
        # with password auth — surface as auth, not as a generic outage.
        raise ProviderAuthenticationError("Fleaflicker rejected that email or password.")
    user = payload.get("user") or {}
    token = normalize_auth_token(user.get("token") if isinstance(user, dict) else "")
    if not token:
        raise ProviderAuthenticationError("Fleaflicker login did not return a session token.")
    owner_id = _get(user, "id") if isinstance(user, dict) else None
    return {
        "token": token,
        "user_id": str(owner_id) if owner_id is not None else None,
    }


def _epoch_ms(value) -> Optional[int]:
    """Parse Fleaflicker int64 epoch-milli fields (often returned as strings)."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _draft_time_ms(league: dict) -> Optional[int]:
    """Scheduled live-draft start from the league payload, if set."""
    if not isinstance(league, dict):
        return None
    return _epoch_ms(
        _get(
            league,
            "draft_live_time_epoch_milli",
            "draftLiveTimeEpochMilli",
        )
    )


def _fleaflicker_sleeper_league_type(max_keepers: int, team_count: int) -> tuple[int, str]:
    """Map Fleaflicker keeper limits to Sleeper settings.type + league_type.

    Fleaflicker publishes ``max_keepers`` on the league object from
    FetchLeagueStandings. ffscrapr treats ``max_keepers * team_count > 250`` as
    dynasty (whole-roster retention); smaller positive limits are keeper leagues.
    """
    keepers = max(0, int(max_keepers or 0))
    teams = max(1, int(team_count or 1))
    if keepers <= 0:
        return 0, "redraft"
    if keepers * teams > 250:
        return 2, "dynasty"
    return 1, "keeper"


def _fleaflicker_draft_status(league: Optional[dict]) -> str:
    """Read FetchLeagueStandings ``draft_status``.

    The protobuf default is ``NOT_YET_DRAFTED`` and is omitted from JSON, so a
    league that has not drafted yet arrives with no field. Treat that as
    pre-draft — never as an empty string that later looks like ``complete``.
    """
    raw = _get(league or {}, "draft_status", "draftStatus")
    status = str(raw or "").strip().upper()
    return status or "NOT_YET_DRAFTED"


def _normalize_fleaflicker_draft_status(
    flea_status: Optional[str],
    *,
    is_in_progress: bool = False,
    pick_count: int = 0,
) -> str:
    """Map Fleaflicker draft state to Sleeper-compatible status strings."""
    status = str(flea_status or "").strip().upper() or "NOT_YET_DRAFTED"
    if is_in_progress or status == "DRAFT_IN_PROGRESS":
        return "drafting"
    if status == "POST_DRAFT":
        return "complete"
    if status == "NOT_YET_DRAFTED":
        return "pre_draft"
    # Unknown leftover labels only. Omitted / pre-draft status used to fall
    # through to this and treat last year's draft-board rows as complete.
    if pick_count > 0:
        return "complete"
    return "pre_draft"


def resolve_fleaflicker_team_id(
    users: list[dict],
    *,
    team_id: Optional[str] = None,
    flea_user_id: Optional[str] = None,
) -> Optional[str]:
    """Map an explicit team id or Fleaflicker owner id to a league roster id."""
    if team_id:
        wanted = str(team_id)
        for user in users or []:
            roster_id = str(user.get("roster_id") or user.get("user_id") or "")
            if roster_id == wanted or str(user.get("user_id") or "") == wanted:
                return roster_id or wanted
        return wanted
    if not flea_user_id:
        return None
    wanted_owner = str(flea_user_id)
    for user in users or []:
        meta = user.get("metadata") or {}
        if str(meta.get("flea_owner_id") or "") == wanted_owner:
            return str(user.get("roster_id") or user.get("user_id") or "") or None
    return None


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
        total_rosters = _int(_get(league, "size")) or len(teams)
        max_keepers = _int(_get(league, "max_keepers", "maxKeepers"))
        sleeper_type, league_type = _fleaflicker_sleeper_league_type(
            max_keepers, total_rosters,
        )
        settings: dict[str, Any] = {
            "type": sleeper_type,
            "league_type": league_type,
            "draft_status": _fleaflicker_draft_status(league),
        }
        if max_keepers > 0:
            settings["max_keepers"] = max_keepers
        return {
            "league_id": str(_get(league, "id") or league_id),
            "season": int(_get(standings or {}, "season") or season),
            "name": _get(league, "name") or f"Fleaflicker League {league_id}",
            "total_rosters": total_rosters,
            "draft_day": _draft_time_ms(league),
            "settings": settings,
            "scoring_settings": self._scoring(rules),
            "roster_positions": self._positions(rules, league=league),
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
            team_name = _get(team, "name") or f"Team {team_id}"
            owner_id = _get(owner, "id")
            out.append({
                # Align with rosters.owner_id (team id), same pattern as MFL franchises.
                "user_id": str(team_id),
                "roster_id": _int(team_id),
                "display_name": display,
                "avatar": _get(owner, "avatarUrl", "avatar_url"),
                "league_id": str(league_id),
                "metadata": {
                    "team_name": team_name,
                    "provider_team_id": str(team_id),
                    "flea_owner_id": str(owner_id) if owner_id is not None else None,
                },
            })
        return out

    @staticmethod
    def _build_name_index() -> dict:
        # utils.utils imports requests at module load. The unit-test CI job does
        # not install that stack, so fail soft and let xwalk handle IDs.
        try:
            from utils.utils import load_players_index, normalize_name
            return _name_index_from_players(load_players_index() or {}, normalize_name)
        except Exception as exc:
            logger.debug("Fleaflicker name index unavailable error=%s", type(exc).__name__)
            return {}

    @staticmethod
    def _canonical_lookup(pro: dict, xwalk: dict, by_name: dict) -> Optional[str]:
        pid = _get(pro, "id")
        if pid is not None:
            cached = xwalk.get(str(pid))
            if cached:
                return cached
        try:
            from utils.utils import normalize_name
            name = normalize_name(_get(pro, "nameFull", "name_full") or "")
        except Exception:
            return None
        pos = str(_get(pro, "position") or "").upper()
        if not name:
            return None
        return _pick_canonical(by_name, name, pos, _flea_pro_team(pro))

    @staticmethod
    def _slot_group_label(group_label: str, slot: dict) -> str:
        pos = slot.get("position") or {}
        return str(_get(pos, "group") or group_label or "").upper()

    def _parse_fetch_roster_groups(
        self, detail: dict, xwalk: dict, by_name: dict,
    ) -> tuple[list[str], list[str], list[str], set[str]]:
        players, starters, reserve = [], [], []
        seen: set[str] = set()
        for group in detail.get("groups") or []:
            group_label = str(group.get("group") or "").upper()
            for slot in group.get("slots") or []:
                lp = slot.get("leaguePlayer") or slot.get("league_player") or {}
                pro = lp.get("proPlayer") or lp.get("pro_player") or {}
                if _get(pro, "id") is None:
                    continue
                cid = self._canonical_lookup(pro, xwalk, by_name)
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                players.append(cid)
                bucket = self._slot_group_label(group_label, slot)
                if bucket in _START_GROUPS:
                    starters.append(cid)
                elif bucket in {"INJURED", "IR", "INJURED_RESERVE", "TAXI"}:
                    reserve.append(cid)
        return players, starters, reserve, seen

    def _canonical_map(self, league_id, season, *, token: Optional[str] = None):
        try:
            from utils.utils import load_players_index, normalize_name
            by_name = _name_index_from_players(load_players_index() or {}, normalize_name)
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
                    canonical = _pick_canonical(by_name, name, pos, _flea_pro_team(pro))
                    if canonical:
                        out[str(pid)] = canonical
            return out
        except Exception as exc:
            logger.warning("Fleaflicker player crosswalk unavailable error=%s", type(exc).__name__)
            return {}

    def get_rosters(self, league_id, season, *, token: Optional[str] = None):
        raw = self._call("FetchLeagueRosters", league_id, season, ttl=300, token=token)
        by_name = self._build_name_index()
        xwalk = self._canonical_map(league_id, season, token=token)
        team_names = {}
        try:
            standings = self._call(
                "FetchLeagueStandings", league_id, season, ttl=1800, token=token,
            )
            for team in self._teams_from_standings(standings):
                tid = _get(team, "id")
                if tid is not None:
                    team_names[str(tid)] = _get(team, "name") or f"Team {tid}"
        except Exception:
            logger.debug("Fleaflicker standings unavailable for roster names", exc_info=True)
        out = []
        for roster in raw.get("rosters") or []:
            team = roster.get("team") or {}
            team_id = _get(team, "id")
            if team_id is None:
                continue
            team_name = team_names.get(str(team_id)) or _get(team, "name") or f"Team {team_id}"
            entries = roster.get("players") or []
            players, starters, reserve = [], [], []
            seen: set[str] = set()
            mapped_flat = 0
            # FetchRoster carries START/BENCH groups; the bulk roster list does not.
            try:
                detail = self._call(
                    "FetchRoster", league_id, season, ttl=300, token=token,
                    team_id=int(team_id),
                )
                players, starters, reserve, seen = self._parse_fetch_roster_groups(
                    detail, xwalk, by_name,
                )
            except Exception:
                logger.debug("Fleaflicker FetchRoster lineup parse failed", exc_info=True)
            starter_set = set(starters)
            for player in entries:
                pro = player.get("proPlayer") or player.get("pro_player") or {}
                if _get(pro, "id") is None:
                    continue
                cid = self._canonical_lookup(pro, xwalk, by_name)
                if not cid:
                    continue
                mapped_flat += 1
                if cid not in seen:
                    seen.add(cid)
                    players.append(cid)
                display = str(_get(player, "display_group", "displayGroup") or "").lower()
                if cid not in starter_set and "start" in display:
                    starters.append(cid)
                    starter_set.add(cid)
            out.append({
                "league_id": str(league_id), "roster_id": _int(team_id),
                "owner_id": str(team_id), "players": players,
                "starters": starters, "reserve": reserve, "taxi": None,
                "settings": {}, "metadata": {
                    "unmapped_player_count": max(0, len(entries) - mapped_flat),
                    "provider_team_id": str(team_id),
                    "team_name": team_name,
                },
            })
        return out

    def _starters_from_boxscore(
        self, lineups: list, side: str, xwalk: dict, by_name: dict,
    ) -> tuple[list[str], dict[str, float]]:
        """Slot-ordered START players for home/away from FetchLeagueBoxscore."""
        starters: list[str] = []
        points: dict[str, float] = {}
        seen: set[str] = set()
        for group in lineups or []:
            group_label = str(group.get("group") or "").upper()
            for slot in group.get("slots") or []:
                bucket = self._slot_group_label(group_label, slot)
                if bucket not in _START_GROUPS:
                    continue
                player = slot.get(side) or {}
                if not isinstance(player, dict):
                    continue
                pro = player.get("proPlayer") or player.get("pro_player") or {}
                if _get(pro, "id") is None:
                    continue
                cid = self._canonical_lookup(pro, xwalk, by_name)
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                starters.append(cid)
                raw_pts = (
                    player.get("viewingActualPoints")
                    or player.get("viewing_actual_points")
                    or {}
                )
                points[cid] = _num(raw_pts)
        return starters, points

    def get_matchups(self, league_id, season, week, *, token: Optional[str] = None):
        raw = self._call(
            "FetchLeagueScoreboard", league_id, season, ttl=600, token=token,
            scoring_period=int(week),
        )
        by_name = self._build_name_index()
        xwalk = {}
        try:
            xwalk = self._canonical_map(league_id, season, token=token)
        except Exception:
            logger.debug("Fleaflicker matchup crosswalk unavailable", exc_info=True)
        out = []
        for mid, game in enumerate(raw.get("games") or [], 1):
            game_id = _get(game, "id")
            lineups: list = []
            if game_id is not None:
                try:
                    box = self._call(
                        "FetchLeagueBoxscore", league_id, season, ttl=300,
                        token=token,
                        scoring_period=int(week),
                        fantasy_game_id=int(game_id),
                    )
                    lineups = box.get("lineups") or []
                except Exception:
                    logger.debug(
                        "Fleaflicker FetchLeagueBoxscore failed game=%s",
                        game_id, exc_info=True,
                    )
            for side, score_key in (("home", "homeScore"), ("away", "awayScore")):
                team = game.get(side) or {}
                team_id = _get(team, "id")
                if team_id is None:
                    continue
                score_block = game.get(score_key) or game.get(
                    "home_score" if side == "home" else "away_score"
                ) or {}
                points = _num(score_block.get("score") if isinstance(score_block, dict) else score_block)
                starters, players_points = self._starters_from_boxscore(
                    lineups, side, xwalk, by_name,
                )
                out.append({
                    "matchup_id": mid, "roster_id": _int(team_id),
                    "points": points, "players": list(starters),
                    "starters": starters,
                    "starters_points": [players_points.get(pid, 0.0) for pid in starters],
                    "players_points": players_points,
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
        standings = self._call(
            "FetchLeagueStandings", league_id, season, ttl=1800, token=token,
        )
        league = (standings or {}).get("league") or {}
        draft_ts_ms = _draft_time_ms(league)
        flea_status = _fleaflicker_draft_status(league)
        raw = self._call("FetchLeagueDraftBoard", league_id, season, ttl=3600, token=token)
        picks = []
        overall = 0
        for row in raw.get("rows") or []:
            for cell in row.get("cells") or []:
                overall += 1
                team = cell.get("team") or {}
                drafted = cell.get("player") or {}
                pro = drafted.get("proPlayer") or drafted.get("pro_player") or drafted
                if not _get(pro, "id"):
                    continue
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
        status = _normalize_fleaflicker_draft_status(
            flea_status,
            is_in_progress=bool(raw.get("is_in_progress") or raw.get("isInProgress")),
            pick_count=len(picks),
        )
        # Last year's board can still have cells after the season rolls. An
        # official pre-draft status means this year's draft has no picks yet.
        if status == "pre_draft":
            picks = []
        return [{
            "draft_id": f"fleaflicker:{season}:{league_id}",
            "league_id": str(league_id), "season": str(season),
            "status": status,
            "type": "snake",
            "start_time": draft_ts_ms,
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
    def _roster_position_rows(rules: dict, league: Optional[dict] = None) -> list:
        """Rules first; standings ``rosterRequirements.positions`` as fallback."""
        blobs = [rules or {}]
        if isinstance(league, dict):
            blobs.append(league)
        for blob in blobs:
            for key in ("rosterPositions", "roster_positions"):
                rows = blob.get(key)
                if isinstance(rows, list) and rows:
                    return rows
            req = blob.get("rosterRequirements") or blob.get("roster_requirements") or {}
            if isinstance(req, dict):
                rows = req.get("positions") or []
                if isinstance(rows, list) and rows:
                    return rows
        return []

    @staticmethod
    def _fleaflicker_slot_name(pos: dict) -> str:
        """Map a Fleaflicker roster-position row onto a Sleeper-style slot.

        Fleaflicker labels flex ``RB/WR/TE``, superflex ``QB/RB/WR/TE``,
        restricted flex ``RB/WR`` / ``WR/TE`` / ``RB/TE``, and defense ``D/ST``.
        Downstream draft-room / lineup math only counts canonical names, so
        those must be mapped here (same idea as MFL). A generic ``FLEX`` label
        uses ``eligibility`` so WR/RB-only spots stay distinct from RB/WR/TE.
        """
        from utils.lineup_slots import canonicalize_slot, normalize_slot_name

        label = str(_get(pos, "label") or "").strip()
        mapped = canonicalize_slot(label)
        elig = {
            normalize_slot_name(e)
            for e in (pos.get("eligibility") or [])
            if e is not None and str(e).strip()
        }
        refined = FleaflickerProvider._slot_from_eligibility(elig)
        if mapped == "FLEX":
            if refined:
                return refined
            return "FLEX"
        if mapped:
            return mapped
        return refined

    @staticmethod
    def _slot_from_eligibility(elig: set) -> str:
        """Map a Fleaflicker eligibility set onto a canonical flex slot."""
        skill = {"QB", "RB", "WR", "TE"}
        if not elig:
            return ""
        if "QB" in elig and (elig & {"RB", "WR", "TE"}):
            return "SUPER_FLEX"
        if elig >= skill:
            return "SUPER_FLEX"
        if elig == {"RB", "WR"}:
            return "RB_WR"
        if elig == {"WR", "TE"}:
            return "WR_TE"
        if elig == {"RB", "TE"}:
            return "RB_TE"
        if elig == {"RB", "WR", "TE"}:
            return "FLEX"
        if elig <= {"RB", "WR", "TE"} and len(elig) >= 2:
            return "FLEX"
        return ""

    @staticmethod
    def _positions(rules: dict, league: Optional[dict] = None) -> list[str]:
        """Expand Fleaflicker roster rules into Sleeper-style starting slots.

        Only START-group rows (or rows with a positive ``start`` count) become
        lineup slots. Bench / IR / taxi are dropped. Protobuf omits zero-valued
        ``start``; a START-group row with no start is one slot.
        """
        from utils.lineup_slots import BENCH_SLOT_NAMES

        positions = FleaflickerProvider._roster_position_rows(rules, league)
        out: list[str] = []
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            group = str(_get(pos, "group") or "").upper()
            if group in {"BENCH", "INJURED", "IR", "INJURED_RESERVE", "TAXI"}:
                continue
            mapped = FleaflickerProvider._fleaflicker_slot_name(pos)
            if not mapped or mapped in BENCH_SLOT_NAMES:
                continue
            raw_start = _get(pos, "start")
            if raw_start is None:
                starts = 1 if group == "START" else 0
            else:
                starts = _int(raw_start)
            if starts <= 0:
                continue
            out.extend([mapped] * starts)
        if not out:
            logger.warning(
                "[fleaflicker-roster] empty roster_positions platform=fleaflicker"
            )
        return out

    @staticmethod
    def _scoring(rules: dict) -> dict:
        """Map Fleaflicker group+abbrev rules onto Sleeper scoring keys.

        Passing/Rushing/Receiving all use ``TD`` / ``Yd``. Storing those
        abbreviations last-write-wins, so ``projection_points`` never saw
        ``pass_yd`` / ``rec`` and every starter painted 0.0.
        """
        from utils.league_scoring import assign_scoring_rate

        out = {}
        for group in rules.get("groups") or []:
            group_label = str(group.get("label") or group.get("name") or "").strip()
            for rule in group.get("scoringRules") or group.get("scoring_rules") or []:
                if not isinstance(rule, dict):
                    continue
                if _flea_is_threshold_bonus(rule):
                    continue
                cat = rule.get("category") or {}
                key = _flea_stat_key(group_label, cat)
                if not key:
                    continue
                rate = _flea_points_per(rule)
                if rate is None:
                    continue
                assign_scoring_rate(out, key, rate)
        return out

    def get_league_globals(self, league_id, season, *, token: Optional[str] = None):
        league = self.get_league(league_id, season, token=token)
        return {
            "scoring_settings": league.get("scoring_settings"),
            "roster_positions": league.get("roster_positions"),
            "league_settings": league.get("settings"),
            "total_rosters": league.get("total_rosters"),
        }
