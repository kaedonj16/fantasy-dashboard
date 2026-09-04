"""Read-only Fleaflicker API provider.

Raw upstream names stay in this module. Public methods return the existing
Sleeper-compatible dictionaries consumed by BR Fantasy.

Public leagues need no auth. Private leagues use the undocumented ``/api/Login``
token, passed in the ``Authorization`` header. Passwords are never stored —
only the returned token is retained (encrypted by the accounts layer).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from .base import (
    BRACKET, DRAFTS, DRAFT_RESULTS, FUTURE_PICKS, HISTORY, LEAGUE, MATCHUPS,
    ROSTERS, ROSTER_SETTINGS, SCORING_SETTINGS, STANDINGS, STARTERS,
    TRADED_PICKS, TRANSACTIONS, TRADES, USERS, LeagueNotFoundError,
    ProviderAdapter, ProviderAuthenticationError, ProviderMetadata,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)
BASE_URL = "https://www.fleaflicker.com/api"
SPORT = "NFL"
TIMEOUT = (5, 20)
# Draft board, per-team picks, and lineup detail are optional for a 200 page.
# A hung FetchLeagueDraftBoard used to stall /teams for the full 20s read
# timeout, then cache that failure for the success TTL (up to an hour).
OPTIONAL_TIMEOUT = (3, 6)
_FAIL_TTL = 30
_OPTIONAL_FAIL_TTL = 30
_OPTIONAL_METHODS = frozenset({
    "FetchLeagueDraftBoard",
    "FetchTeamPicks",
    "FetchLeagueBoxscore",
    "FetchRoster",
})
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
# Endpoints whose OpenAPI signatures do not take ``season``. Passing one
# returns HTML 400, which we used to surface as a generic outage — and
# ``get_transactions_by_week`` then printed that error 18 times.
_NO_SEASON = frozenset({
    "FetchLeagueRules",
    "FetchLeagueTransactions",
    "FetchTrades",
    "FetchTeamPicks",
    "FetchLeagueActivity",
    "FetchLeagueBoxscore",
})
_CACHE: dict[tuple, tuple[float, dict]] = {}
_FAIL_CACHE: dict[tuple, tuple[float, Exception]] = {}
_OPTIONAL_FAIL: dict[str, tuple[float, Exception]] = {}
_KEY_LOCKS: dict[tuple, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()
_KEY_LOCKS_MAX = 256
_TX_PAGE_CAP = 25
_TRADE_PAGE_CAP = 15
_TX_BY_WEEK: dict[tuple, tuple[float, dict]] = {}
_TX_BY_WEEK_TTL = 300
_TX_BY_WEEK_MAX = 64
_ET = ZoneInfo("America/New_York")

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


def _lock_for(key: tuple) -> threading.Lock:
    """Serialize identical in-flight Fleaflicker GETs (18 weeks used to stampede)."""
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            if len(_KEY_LOCKS) >= _KEY_LOCKS_MAX:
                _KEY_LOCKS.clear()
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def _cached_failure(cache: dict, key, ttl: float):
    hit = cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    return None


def _remember_failure(key: tuple, exc: Exception) -> None:
    now = time.monotonic()
    _FAIL_CACHE[key] = (now, exc)
    if len(_FAIL_CACHE) > 256:
        cutoff = now - _FAIL_TTL
        for stale, (ts, _) in list(_FAIL_CACHE.items()):
            if ts < cutoff:
                _FAIL_CACHE.pop(stale, None)


def _optional_fail_key(auth: Optional[str]) -> str:
    return auth or ""


def _raise_if_optional_down(auth: Optional[str]) -> None:
    exc = _cached_failure(_OPTIONAL_FAIL, _optional_fail_key(auth), _OPTIONAL_FAIL_TTL)
    if exc is not None:
        raise exc


def _trip_optional_down(auth: Optional[str], exc: Exception) -> None:
    _OPTIONAL_FAIL[_optional_fail_key(auth)] = (time.monotonic(), exc)


def _fantasy_week_from_ms(ts_ms: Optional[int], season: int) -> Optional[int]:
    """Map an epoch-milli timestamp onto NFL regular-season week 1-18.

    Fantasy weeks start the Tuesday after Labor Day (first Monday in September).
    Activity from the two months before week 1 lands in week 1 so waiver/FA
    adds still show on the activity page. Older seasons return None.
    """
    if not ts_ms:
        return None
    try:
        when = datetime.fromtimestamp(ts_ms / 1000.0, tz=_ET).date()
        year = int(season)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    sep1 = date(year, 9, 1)
    labor = sep1 + timedelta(days=(0 - sep1.weekday()) % 7)
    week1_tue = labor + timedelta(days=1)
    delta = (when - week1_tue).days
    if delta < -90:
        return None
    if delta < 0:
        return 1
    week = delta // 7 + 1
    if week > 22:
        return None
    return min(18, week)


def _activity_tx_kind(tx: dict) -> str:
    """Protobuf omits the default ``TRANSACTION_ADD`` from JSON."""
    kind = str(_get(tx or {}, "type", "transactionType", "transaction_type") or "").upper()
    if kind:
        return kind
    return "TRANSACTION_ADD" if tx else ""


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
    log = logger.debug if status == 400 else logger.warning
    log(
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
            FUTURE_PICKS, HISTORY, SCORING_SETTINGS, ROSTER_SETTINGS, BRACKET,
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
        optional = method in _OPTIONAL_METHODS
        if optional:
            _raise_if_optional_down(auth)
        cached = _CACHE.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        failed = _cached_failure(_FAIL_CACHE, key, _FAIL_TTL)
        if failed is not None:
            raise failed
        with _lock_for(key):
            if optional:
                _raise_if_optional_down(auth)
            cached = _CACHE.get(key)
            if cached and time.monotonic() - cached[0] < ttl:
                return cached[1]
            failed = _cached_failure(_FAIL_CACHE, key, _FAIL_TTL)
            if failed is not None:
                raise failed
            query = {"sport": SPORT, "league_id": int(league_id), **params}
            if include_season:
                query["season"] = int(season)
            headers = dict(_HEADERS)
            if auth:
                headers["Authorization"] = auth
            try:
                response = _request_get(
                    f"{BASE_URL}/{method}",
                    params=query,
                    timeout=OPTIONAL_TIMEOUT if optional else TIMEOUT,
                    headers=headers,
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
            except (ProviderAuthenticationError, LeagueNotFoundError, ProviderUnavailableError) as exc:
                _remember_failure(key, exc)
                if isinstance(exc, ProviderUnavailableError):
                    _trip_optional_down(auth, exc)
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
        playoff_week_start = (
            _int(_get(league, "playoff_week_start", "playoffStartWeek", "playoff_start_week"))
            or _int(_get(rules, "playoff_week_start", "playoffStartWeek"))
            or 15
        )
        playoff_teams = (
            _int(_get(league, "playoff_teams", "playoffTeams", "num_playoff_teams"))
            or _int(_get(rules, "playoff_teams", "playoffTeams"))
            or 6
        )
        settings: dict[str, Any] = {
            "type": sleeper_type,
            "league_type": league_type,
            "draft_status": _fleaflicker_draft_status(league),
            "playoff_week_start": playoff_week_start,
            "playoff_teams": playoff_teams,
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
        roster_detail_unavailable = False
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
            # One upstream outage used to sequentially timeout every team.
            if not roster_detail_unavailable:
                try:
                    detail = self._call(
                        "FetchRoster", league_id, season, ttl=300, token=token,
                        team_id=int(team_id),
                    )
                    players, starters, reserve, seen = self._parse_fetch_roster_groups(
                        detail, xwalk, by_name,
                    )
                except ProviderUnavailableError:
                    roster_detail_unavailable = True
                    logger.debug(
                        "Fleaflicker FetchRoster unavailable league=%s; "
                        "using bulk roster list for remaining teams",
                        league_id,
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
        boxscore_failed = False
        for mid, game in enumerate(raw.get("games") or [], 1):
            game_id = _get(game, "id")
            lineups: list = []
            if game_id is not None and not boxscore_failed:
                try:
                    box = self._call(
                        "FetchLeagueBoxscore", league_id, season, ttl=300,
                        token=token,
                        scoring_period=int(week),
                        fantasy_game_id=int(game_id),
                    )
                    lineups = box.get("lineups") or []
                except Exception:
                    boxscore_failed = True
                    logger.debug(
                        "Fleaflicker FetchLeagueBoxscore failed game=%s; "
                        "skipping remaining boxscores for week %s",
                        game_id, week, exc_info=True,
                    )
            for side, score_key in (("home", "homeScore"), ("away", "awayScore")):
                team = game.get(side) or {}
                team_id = _get(team, "id")
                if team_id is None:
                    continue
                score_block = game.get(score_key) or game.get(
                    "home_score" if side == "home" else "away_score"
                ) or {}
                if not isinstance(score_block, dict):
                    score_block = {"score": score_block}
                points = _num(score_block.get("score"))
                # Scoreboard publishes team projected totals (same role as Yahoo
                # team_projected_points). Boxscore slots are often empty before
                # kickoff, so Sleeper player joins miss and headers painted 0.0
                # unless we carry this through as projected_points / proj_total.
                projected_points = None
                if score_block.get("projected") is not None:
                    projected_points = _num(score_block.get("projected"))
                starters, players_points = self._starters_from_boxscore(
                    lineups, side, xwalk, by_name,
                )
                out.append({
                    "matchup_id": mid, "roster_id": _int(team_id),
                    "points": points,
                    "projected_points": projected_points,
                    "players": list(starters),
                    "starters": starters,
                    "starters_points": [players_points.get(pid, 0.0) for pid in starters],
                    "players_points": players_points,
                    "week": int(week), "custom_points": None,
                    "metadata": {"provider_game_id": str(_get(game, "id") or "")},
                })
        return out

    def get_transactions(self, league_id, season, week, *, token: Optional[str] = None):
        """Return this week's adds/claims/drops/trades.

        ``FetchLeagueTransactions`` has no season or scoring_period. The old
        path sent ``season`` (HTML 400) and ignored ``week``, so
        ``get_transactions_by_week`` fired 18 identical failing requests.
        Fetch once, paginate, and bucket by fantasy week.
        """
        try:
            want = int(week)
        except (TypeError, ValueError):
            want = 1
        by_week = self._transactions_by_week(league_id, season, token=token)
        return list(by_week.get(want) or [])

    def _paginate_items(
        self, method: str, league_id, season, *, token: Optional[str] = None,
        list_keys: tuple[str, ...] = ("items",), max_pages: int = _TX_PAGE_CAP,
        extra: Optional[dict] = None,
    ) -> list:
        items: list = []
        offset = None
        params = dict(extra or {})
        for _ in range(max(1, int(max_pages))):
            page_params = dict(params)
            if offset is not None:
                page_params["result_offset"] = int(offset)
            raw = self._call(
                method, league_id, season, ttl=300, token=token, **page_params,
            )
            chunk = []
            for key in list_keys:
                found = raw.get(key) if isinstance(raw, dict) else None
                if isinstance(found, list) and found:
                    chunk = found
                    break
            if not chunk:
                break
            items.extend(chunk)
            nxt = _get(raw, "resultOffsetNext", "result_offset_next")
            if nxt is None:
                break
            try:
                nxt_i = int(nxt)
            except (TypeError, ValueError):
                break
            if offset is not None and nxt_i == int(offset):
                break
            offset = nxt_i
        return items

    def _tx_canonical_player(self, blob, xwalk: dict, by_name: dict) -> Optional[str]:
        if not isinstance(blob, dict):
            return None
        pro = blob.get("proPlayer") or blob.get("pro_player") or blob
        if not isinstance(pro, dict) or _get(pro, "id") is None:
            return None
        cid = self._canonical_lookup(pro, xwalk, by_name)
        if cid:
            return cid
        return str(_get(pro, "id"))

    def _transactions_by_week(
        self, league_id, season, *, token: Optional[str] = None,
    ) -> dict[int, list]:
        auth = resolve_credentials(str(league_id), season, token=token) or ""
        cache_key = (str(league_id), int(season), auth)
        hit = _TX_BY_WEEK.get(cache_key)
        if hit and time.monotonic() - hit[0] < _TX_BY_WEEK_TTL:
            return hit[1]
        with _lock_for(("tx-by-week",) + cache_key):
            hit = _TX_BY_WEEK.get(cache_key)
            if hit and time.monotonic() - hit[0] < _TX_BY_WEEK_TTL:
                return hit[1]
            assembled = self._assemble_transactions_by_week(
                league_id, season, token=token,
            )
            if len(_TX_BY_WEEK) >= _TX_BY_WEEK_MAX:
                _TX_BY_WEEK.clear()
            _TX_BY_WEEK[cache_key] = (time.monotonic(), assembled)
            return assembled

    def _assemble_transactions_by_week(
        self, league_id, season, *, token: Optional[str] = None,
    ) -> dict[int, list]:
        xwalk = {}
        try:
            xwalk = self._canonical_map(league_id, season, token=token)
        except Exception:
            logger.debug("Fleaflicker transaction crosswalk unavailable", exc_info=True)
        by_name = self._build_name_index()
        by_week: dict[int, list] = {}
        seen: set[str] = set()

        items = self._paginate_items(
            "FetchLeagueTransactions", league_id, season, token=token,
            list_keys=("items", "transactions"), max_pages=_TX_PAGE_CAP,
        )
        for item in items:
            row = self._normalize_activity_transaction(item, season, xwalk, by_name)
            if not row:
                continue
            tid = str(row.get("transaction_id") or "")
            if tid and tid in seen:
                continue
            if tid:
                seen.add(tid)
            by_week.setdefault(int(row["leg"]), []).append(row)

        try:
            for row in self._completed_trades(
                league_id, season, token=token, xwalk=xwalk, by_name=by_name,
            ):
                tid = str(row.get("transaction_id") or "")
                if tid and tid in seen:
                    continue
                if tid:
                    seen.add(tid)
                by_week.setdefault(int(row["leg"]), []).append(row)
        except Exception:
            logger.debug("Fleaflicker completed trades unavailable", exc_info=True)
        return by_week

    def _normalize_activity_transaction(
        self, item, season, xwalk: dict, by_name: dict,
    ) -> Optional[dict]:
        if not isinstance(item, dict):
            return None
        tx = item.get("transaction") or {}
        if not isinstance(tx, dict) or not tx:
            return None
        kind = _activity_tx_kind(tx)
        if kind in {"TRANSACTION_DRAFT", "TRANSACTION_IMPORT"} or "TRADE" in kind:
            return None
        ts = _epoch_ms(
            _get(item, "timeEpochMilli", "time_epoch_milli", "timestamp")
        ) or _epoch_ms(_get(tx, "timestamp", "timeEpochMilli", "time_epoch_milli"))
        week = _fantasy_week_from_ms(ts, season)
        if week is None:
            return None
        team = tx.get("team") or {}
        team_id = _int(_get(team, "id")) if isinstance(team, dict) else 0
        cid = self._tx_canonical_player(tx.get("player") or {}, xwalk, by_name)
        adds: dict[str, int] = {}
        drops: dict[str, int] = {}
        if cid and team_id:
            if "DROP" in kind:
                drops[cid] = team_id
            else:
                adds[cid] = team_id
        if not adds and not drops:
            return None
        normalized = "waiver" if ("CLAIM" in kind or "WAIVER" in kind) else "free_agent"
        return {
            "transaction_id": str(
                _get(tx, "id") or f"flea-{season}-{ts}-{team_id}-{cid}"
            ),
            "type": normalized,
            "status": "complete",
            "created": ts or 0,
            "roster_ids": [team_id] if team_id else [],
            "adds": adds,
            "drops": drops,
            "draft_picks": [],
            "leg": week,
            "metadata": {"provider_type": kind},
        }

    def _completed_trades(
        self, league_id, season, *, token, xwalk: dict, by_name: dict,
    ) -> list[dict]:
        out: list[dict] = []
        offset = None
        for _ in range(_TRADE_PAGE_CAP):
            params: dict[str, Any] = {"filter": "TRADES_COMPLETED"}
            if offset is not None:
                params["result_offset"] = int(offset)
            raw = self._call(
                "FetchTrades", league_id, season, ttl=300, token=token, **params,
            )
            page = raw.get("trades") if isinstance(raw, dict) else None
            if not isinstance(page, list) or not page:
                break
            hit_older = False
            for trade in page:
                row = self._normalize_completed_trade(trade, season, xwalk, by_name)
                if row:
                    out.append(row)
                    continue
                ts = _epoch_ms(_get(
                    trade if isinstance(trade, dict) else {},
                    "approvedOn", "approved_on", "proposedOn", "proposed_on",
                ))
                if ts and _fantasy_week_from_ms(ts, season) is None:
                    hit_older = True
            if hit_older:
                break
            nxt = _get(raw, "resultOffsetNext", "result_offset_next")
            if nxt is None:
                break
            try:
                nxt_i = int(nxt)
            except (TypeError, ValueError):
                break
            if offset is not None and nxt_i == int(offset):
                break
            offset = nxt_i
        return out

    def _normalize_completed_trade(
        self, trade, season, xwalk: dict, by_name: dict,
    ) -> Optional[dict]:
        if not isinstance(trade, dict):
            return None
        status = str(_get(trade, "status") or "").upper()
        if any(x in status for x in ("VETO", "REJECT", "CANCEL", "OPEN", "REVIEW", "PROPOSE")):
            return None
        ts = _epoch_ms(_get(
            trade, "approvedOn", "approved_on",
            "tentativeExecutionTime", "tentative_execution_time",
            "proposedOn", "proposed_on",
        ))
        week = _fantasy_week_from_ms(ts, season)
        if week is None:
            return None
        adds: dict[str, int] = {}
        roster_ids: list[int] = []
        draft_picks: list[dict] = []
        for side in trade.get("teams") or []:
            if not isinstance(side, dict):
                continue
            team = side.get("team") or {}
            team_id = _int(_get(team, "id")) if isinstance(team, dict) else 0
            if not team_id:
                continue
            roster_ids.append(team_id)
            for player in side.get("playersObtained") or side.get("players_obtained") or []:
                cid = self._tx_canonical_player(player, xwalk, by_name)
                if cid:
                    adds[cid] = team_id
            for pick in side.get("picksObtained") or side.get("picks_obtained") or []:
                if not isinstance(pick, dict):
                    continue
                slot = pick.get("slot") or {}
                rnd = _int(_get(slot, "round") if isinstance(slot, dict) else None) or _int(
                    _get(pick, "round")
                )
                orig = pick.get("originalOwner") or pick.get("original_owner") or {}
                orig_id = _int(_get(orig, "id")) if isinstance(orig, dict) else 0
                draft_picks.append({
                    "season": str(_get(pick, "season", "year") or season),
                    "round": rnd,
                    "roster_id": orig_id or team_id,
                    "owner_id": team_id,
                    "previous_owner_id": orig_id or None,
                })
        drops: dict[str, int] = {}
        if len(roster_ids) == 2:
            a, b = roster_ids[0], roster_ids[1]
            for pid, to in adds.items():
                drops[pid] = b if int(to) == int(a) else a
        if not adds and not draft_picks:
            return None
        trade_id = _get(trade, "id", "tradeId", "trade_id")
        return {
            "transaction_id": str(trade_id or f"flea-trade-{season}-{ts}"),
            "type": "trade",
            "status": "complete",
            "created": ts or 0,
            "roster_ids": roster_ids,
            "adds": adds,
            "drops": drops,
            "draft_picks": draft_picks,
            "leg": week,
            "metadata": {"provider_type": "TRADE"},
        }

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
        picks_unavailable = False
        for team in self._teams_from_standings(standings):
            team_id = _get(team, "id")
            if team_id is None:
                continue
            if picks_unavailable:
                break
            try:
                raw = self._call(
                    "FetchTeamPicks", league_id, season, ttl=1800, token=token,
                    team_id=int(team_id),
                )
            except ProviderUnavailableError:
                # Same host outage will fail every team; don't pay N timeouts.
                picks_unavailable = True
                continue
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

    def _playoff_seeds(self, league_id, season, *, token: Optional[str] = None) -> list[int]:
        """Team ids in standings order (best first)."""
        try:
            standings = self._call(
                "FetchLeagueStandings", league_id, season, ttl=1800, token=token,
            )
        except Exception:
            return []
        ranked = []
        for i, team in enumerate(self._teams_from_standings(standings), 1):
            team_id = _get(team, "id")
            if team_id is None:
                continue
            rec = team.get("recordOverall") or team.get("record_overall") or team.get("record") or {}
            if not isinstance(rec, dict):
                rec = {}
            try:
                rank = int(_get(team, "rank", "seed") or _get(rec, "rank", "seed") or i)
            except (TypeError, ValueError):
                rank = i
            ranked.append((rank, _int(team_id)))
        ranked.sort(key=lambda x: x[0])
        return [tid for _, tid in ranked if tid]

    def get_bracket(self, league_id, season, kind):
        """Derive a winners bracket from playoff-week scoreboards, or project seeds."""
        from utils.playoff_bracket import derive_or_project_bracket

        if str(kind or "winners").lower() != "winners":
            return []
        try:
            league = self.get_league(league_id, season)
        except Exception:
            league = {}
        settings = (league or {}).get("settings") or {}
        start = _int(settings.get("playoff_week_start"), 15)
        playoff_teams = _int(settings.get("playoff_teams"), 6)
        by_week: dict[int, list] = {}
        for week in range(start, start + 4):
            try:
                rows = self.get_matchups(league_id, season, week) or []
            except Exception:
                rows = []
            if rows:
                by_week[week] = rows
        return derive_or_project_bracket(
            matchups_by_week=by_week,
            playoff_week_start=start,
            seed_roster_ids=self._playoff_seeds(league_id, season),
            playoff_teams=playoff_teams,
            kind=kind,
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
