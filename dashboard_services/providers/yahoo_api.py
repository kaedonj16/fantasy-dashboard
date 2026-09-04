"""
Yahoo Fantasy Sports API provider.

Authentication: OAuth 2.0 Authorization Code flow.
Required env vars:
    YAHOO_CLIENT_ID      – Consumer Key from developer.yahoo.com
    YAHOO_CLIENT_SECRET  – Consumer Secret
    YAHOO_REDIRECT_URI   – Callback URL (e.g. https://yourdomain.com/auth/yahoo/callback)

All public functions return normalized dicts that match the Sleeper / ESPN
provider shapes so the rest of the app needs no platform-specific branching.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from dashboard_services.display_names import public_owner_label
from utils.utils import load_players_index
from utils.coerce import safe_float as _safe_float, safe_int as _safe_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAHOO_AUTH_URL  = "https://api.login.yahoo.com/oauth2/request_auth"
YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
YAHOO_API_BASE  = "https://fantasysports.yahooapis.com/fantasy/v2"
# Historical note: Yahoo Fantasy read used to be requested as the OAuth 1.0a
# permission "fspt-r". Under OAuth 2.0 that is NOT a request scope — Fantasy
# access comes from the app's API Permissions (Fantasy Sports - Read). Sending
# scope=fspt-r produced a token with no Fantasy permission (403 "not authorized"
# on every call), so the authorize request now sends no scope at all. Kept only
# for reference; get_authorization_url no longer uses it.
YAHOO_SCOPE     = "fspt-r"  # unused; see get_authorization_url

# Only NFL for now; extend as needed.
_GAME_CODE = "nfl"
# Yahoo's numeric game key changes every season. 461 is 2025; 2026 is 470.
# Used when the games collection can't be parsed (list-of-fragments JSON)
# so we never reuse last year's key and 403 a member as "not in this league".
_NFL_GAME_KEYS_FALLBACK = {
    2024: "449",
    2025: "461",
    2026: "470",
}

# Per-process cache: (league_key, endpoint) -> (fetched_at, data)
_api_cache: Dict[str, tuple[float, Any]] = {}
_api_cache_lock = threading.Lock()
_API_CACHE_TTL = 300  # 5 minutes

# Session Yahoo accounts that are not members of a league 403 every
# scoreboard/transaction week. Remember that for a short TTL so a dashboard
# load does not fire 18× identical "not in this league" requests.
_yahoo_no_access: Dict[tuple, float] = {}
_yahoo_no_access_lock = threading.Lock()
_YAHOO_NO_ACCESS_TTL = 180
_LEAGUE_PATH_RE = re.compile(r"(?:^|/)league/([^/;]+)")


class YahooLeagueAccessDenied(Exception):
    """This Yahoo token is not a member of the requested league."""

# Per-process map of a bare numeric league_id -> its full Yahoo league key
# (e.g. "461.l.123456"). Yahoo's "nfl" game code only resolves to the CURRENT
# season, so a league from any other season needs its real, season-specific game
# key. resolve_league_key() looks the user's leagues up and fills this in; once
# set, every _league_key() call for that id uses the correct key. The mapping is
# permanent per league so it's safe to cache for the life of the process.
_league_key_map: Dict[str, str] = {}
_league_key_lock = threading.Lock()
# (bare_id, season) -> "461.l.123456" for historical seasons. Separate from
# _league_key_map so a prior-year lookup never overwrites the current-season key.
_season_key_map: Dict[tuple, str] = {}
_season_key_lock = threading.Lock()


def yahoo_enabled() -> bool:
    """Whether the Yahoo connect flow is offered to users.

    Enabled by default now that Yahoo Fantasy API access is granted. Set
    YAHOO_ENABLED=0 (or false/no/off) on the host to turn it off without a
    deploy-time code change.
    """
    raw = (os.environ.get("YAHOO_ENABLED") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def yahoo_api_debug_enabled() -> bool:
    """Verbose Yahoo parse/API diagnostics for production troubleshooting.

  Set YAHOO_API_DEBUG=1 on the host (or hit /api/yahoo-debug while signed in)
  to capture response shapes and parsed team/roster counts in logs.
    """
    return (os.environ.get("YAHOO_API_DEBUG") or "").strip().lower() in ("1", "true", "yes", "on")


def _yahoo_debug(msg: str, *args, **kwargs) -> None:
    if yahoo_api_debug_enabled():
        logger.info("[yahoo-debug] " + msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _env_clean(key: str) -> str:
    """Read an OAuth env var, trimming surrounding whitespace/newlines.

    Credentials pasted into a hosting dashboard often pick up a trailing newline
    or stray space; an un-trimmed client_id / redirect_uri then no longer matches
    what's registered on the Yahoo app, and Yahoo answers the authorization
    request with its generic "uh-oh" page instead of redirecting back."""
    return (os.environ.get(key) or "").strip()


def get_authorization_url(state: str, force_login: bool = False) -> str:
    """Return the Yahoo OAuth 2.0 authorization URL the user should be redirected to.

    force_login=True adds prompt=login so Yahoo shows its account chooser instead of
    silently re-authorizing whatever account is already signed into the browser. Use
    it when recovering from a 403 (wrong account) — otherwise the user loops back to
    the same wrong account. Leave it off for normal sign-in to keep that one-tap.
    """
    client_id    = _env_clean("YAHOO_CLIENT_ID")
    redirect_uri = _env_clean("YAHOO_REDIRECT_URI")
    # NOTE: do NOT send a `scope` parameter. "fspt-r" is Yahoo's OAuth 1.0a
    # permission name; in OAuth 2.0 Yahoo grants Fantasy access from the app's
    # API Permissions (Fantasy Sports - Read), not from a request scope. Passing
    # scope=fspt-r made Yahoo mint a token with NO Fantasy permission, so every
    # Fantasy API call came back 403 "This application is not authorized to
    # perform this action" and the token response omitted xoauth_yahoo_guid.
    # Omitting scope lets the app's permission govern (the working OAuth2 flow).
    logger.info(
        "[yahoo-auth] building auth request: client_id=%s… redirect_uri=%r force_login=%s",
        client_id[:10], redirect_uri, force_login,
    )
    params = {
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "state":         state,
    }
    if force_login:
        params["prompt"] = "login"
    return f"{YAHOO_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """
    Exchange an authorization code for access + refresh tokens.
    Returns the full token response dict including xoauth_yahoo_guid.
    """
    client_id     = _env_clean("YAHOO_CLIENT_ID")
    client_secret = _env_clean("YAHOO_CLIENT_SECRET")
    redirect_uri  = _env_clean("YAHOO_REDIRECT_URI")

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode()
    ).decode()

    resp = requests.post(
        YAHOO_TOKEN_URL,
        data={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": redirect_uri,
        },
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        timeout=15,
    )
    resp.raise_for_status()
    tok = resp.json()
    # Non-sensitive diagnostics: which fields Yahoo returned and any granted scope.
    # A missing xoauth_yahoo_guid and/or a scope that isn't fspt-r is the tell for
    # a token that authenticated but carries no Fantasy permission.
    try:
        logger.info(
            "[yahoo] token exchange ok: keys=%s token_type=%s expires_in=%s scope=%r guid_present=%s",
            sorted(tok.keys()), tok.get("token_type"), tok.get("expires_in"),
            tok.get("scope"), bool(tok.get("xoauth_yahoo_guid")),
        )
    except Exception:
        pass
    return tok


def yahoo_oauth_start_url(
    *,
    league_id: str = "",
    next_url: str = "/portfolio",
    reauth: bool = False,
) -> str:
    """Build a local /auth/yahoo URL for the link modal or home connect flow."""
    from urllib.parse import urlencode

    params: Dict[str, str] = {"next": next_url or "/"}
    if league_id:
        params["league_id"] = str(league_id)
    if reauth:
        params["reauth"] = "1"
    return "/auth/yahoo?" + urlencode(params)


def resolve_session_yahoo_token(session) -> tuple[str, str]:
    """Return (guid, access_token) from the Flask session, refreshing from DB if needed.

    Prefer the DB-backed, expiry-aware accessor whenever we have a Yahoo GUID.
    A stale ``session["yahoo_access_token"]`` is common after ~1 hour and must
    not short-circuit refresh — Yahoo then answers ``token_expired`` 401s.
    """
    guid = str(session.get("yahoo_guid") or "")
    if guid:
        token = get_valid_access_token(guid) or ""
        if token:
            session["yahoo_access_token"] = token
            return guid, token
        # Refresh failed or no DB row — drop the stale session bearer so callers
        # re-offer OAuth instead of retrying an expired token.
        session.pop("yahoo_access_token", None)
        return guid, ""
    return "", str(session.get("yahoo_access_token") or "")


def yahoo_auth_error_kind(exc: BaseException | str) -> str:
    """Classify Yahoo HTTP failures for link/validate recovery.

    Returns ``expired`` (needs refresh/reauth), ``forbidden`` (wrong account /
    no league access), or ``""`` (other).
    """
    msg = str(exc or "").lower()
    if "token_expired" in msg or "oauth_problem" in msg or "401" in msg:
        return "expired"
    if "not in this league" in msg or "not allowed to view this page" in msg:
        return "forbidden"
    if "403" in msg or "forbidden" in msg:
        return "forbidden"
    return ""


def get_login_guid(access_token: str, league_id: str = "") -> str:
    """Return the logged-in user's Yahoo GUID via the Fantasy API.

    Fallback for when the OAuth token response omits ``xoauth_yahoo_guid`` (it is
    not guaranteed with the ``fspt-r`` scope). Tries two sources and returns "" if
    both fail:

      1. ``users;use_login=1`` — direct, but the fspt-r token is often forbidden
         (403) from the user-identity resource.
      2. the league's teams (when ``league_id`` is given) — Yahoo flags the
         authenticated user's own team manager with ``is_current_login=1``, which
         carries the real guid and only needs the league read fspt-r does allow.
    """
    def _dig_guid(container) -> str:
        stack = list(container) if isinstance(container, (list, tuple)) else [container]
        while stack:
            part = stack.pop(0)
            if isinstance(part, dict) and part.get("guid"):
                return str(part["guid"])
            if isinstance(part, list):
                stack.extend(part)
        return ""

    try:
        raw   = _yahoo_get(access_token, "users;use_login=1")
        users = (raw.get("fantasy_content", {}) or {}).get("users", {}) or {}
        user  = (users.get("0", {}) or {}).get("user", []) or []
        guid  = _dig_guid(user)
        if guid:
            return guid
    except Exception as exc:
        logger.warning("[yahoo] get_login_guid (users) failed: %s", exc)

    if league_id:
        try:
            raw   = _yahoo_get(access_token, f"league/{_league_key(league_id)}/teams")
            for t in _extract_teams(raw):
                mgr = _yahoo_primary_manager(t)
                if str(mgr.get("is_current_login")) == "1" and mgr.get("guid"):
                    return str(mgr["guid"])
        except Exception as exc:
            logger.warning("[yahoo] get_login_guid (league teams) failed: %s", exc)

    return ""


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh an expired access token using the stored refresh token."""
    client_id     = _env_clean("YAHOO_CLIENT_ID")
    client_secret = _env_clean("YAHOO_CLIENT_SECRET")

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode()
    ).decode()

    resp = requests.post(
        YAHOO_TOKEN_URL,
        data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Token storage (DB-backed)
# ---------------------------------------------------------------------------

def _init_token_table() -> None:
    """Create the yahoo_oauth_tokens table if it doesn't exist yet."""
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS yahoo_oauth_tokens (
                guid          TEXT        NOT NULL PRIMARY KEY,
                access_token  TEXT        NOT NULL,
                refresh_token TEXT        NOT NULL,
                expires_at    TIMESTAMPTZ NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)


def save_tokens(guid: str, access_token: str, refresh_token: str, expires_in: int) -> None:
    from dashboard_services.db import get_conn
    try:
        _init_token_table()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO yahoo_oauth_tokens (guid, access_token, refresh_token, expires_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (guid) DO UPDATE SET
                access_token  = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                expires_at    = EXCLUDED.expires_at,
                updated_at    = NOW()
        """, (guid, access_token, refresh_token, expires_at))


def load_tokens(guid: str) -> Optional[Dict[str, Any]]:
    from dashboard_services.db import get_conn
    try:
        _init_token_table()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT access_token, refresh_token, expires_at FROM yahoo_oauth_tokens WHERE guid = %s",
                (guid,)
            ).fetchone()
        if row:
            return {
                "access_token":  row["access_token"],
                "refresh_token": row["refresh_token"],
                "expires_at":    row["expires_at"],
            }
    except Exception as exc:
        logger.warning("[yahoo] load_tokens failed: %s", exc)
    return None


def get_valid_access_token(guid: str, *, force_refresh: bool = False) -> Optional[str]:
    """Return a valid access token for the given Yahoo GUID, refreshing if needed.

    ``force_refresh=True`` always hits Yahoo's token endpoint (used after a
    ``token_expired`` API response when our stored ``expires_at`` was still in
    the future — clock skew / Yahoo revoked early).
    """
    tokens = load_tokens(guid)
    if not tokens:
        return None

    expires_at = tokens["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    needs_refresh = force_refresh or (
        datetime.now(timezone.utc) >= expires_at - timedelta(minutes=5)
    )
    if needs_refresh:
        try:
            new_tok = refresh_access_token(tokens["refresh_token"])
            save_tokens(
                guid,
                new_tok["access_token"],
                new_tok.get("refresh_token", tokens["refresh_token"]),
                new_tok.get("expires_in", 3600),
            )
            return new_tok["access_token"]
        except Exception as exc:
            logger.warning("[yahoo] token refresh failed for %s: %s", guid, exc)
            return None

    return tokens["access_token"]


# ---------------------------------------------------------------------------
# League → owner mapping
#
# Yahoo OAuth tokens are account-level, so any league member who has authorized
# can read the whole league. Recording which guid(s) authorized while viewing a
# league lets a *different* viewer (a public share) or a background/cron job —
# neither of which has the owner's session — fetch the league on their behalf.
# ---------------------------------------------------------------------------

def _init_league_owner_table() -> None:
    """Create the yahoo_league_owners table if it doesn't exist yet."""
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS yahoo_league_owners (
                league_id  TEXT        NOT NULL,
                season     INTEGER     NOT NULL,
                guid       TEXT        NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (league_id, season, guid)
            )
        """)


def save_league_owner(league_id: str, season: int, guid: str) -> None:
    """Record that ``guid``'s stored token can read ``(league_id, season)``."""
    if not (league_id and guid):
        return
    from dashboard_services.db import get_conn
    try:
        _init_league_owner_table()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    try:
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO yahoo_league_owners (league_id, season, guid, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (league_id, season, guid) DO UPDATE SET updated_at = NOW()
            """, (str(league_id), int(season), guid))
    except Exception as exc:
        logger.warning("[yahoo] save_league_owner failed: %s", exc)


def get_league_token(league_id: str, season: int) -> Optional[str]:
    """Return a valid (auto-refreshed) access token from any authorized owner of
    this league, or None. Prefers an owner who authorized for the exact season,
    then the most recently updated — since the token is account-level, any of
    them can read it. This is the path background jobs and non-owner viewers use.
    """
    if not league_id:
        return None
    from dashboard_services.db import get_conn
    try:
        _init_league_owner_table()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT guid FROM yahoo_league_owners
                WHERE league_id = %s
                ORDER BY (season = %s) DESC, updated_at DESC
                """,
                (str(league_id), int(season or 0)),
            ).fetchall()
    except Exception as exc:
        logger.warning("[yahoo] get_league_token lookup failed: %s", exc)
        return None

    for row in rows or []:
        guid = row["guid"] if isinstance(row, dict) else row[0]
        tok = get_valid_access_token(guid)
        if tok:
            return tok
    return None


# ---------------------------------------------------------------------------
# API request helper
# ---------------------------------------------------------------------------

def _clear_yahoo_request_state() -> None:
    """Reset HTTP caches. Tests only."""
    with _api_cache_lock:
        _api_cache.clear()
    with _yahoo_no_access_lock:
        _yahoo_no_access.clear()
    with _league_key_lock:
        _league_key_map.clear()
    with _season_key_lock:
        _season_key_map.clear()


def _league_key_from_yahoo_path(path: str) -> str:
    hit = _LEAGUE_PATH_RE.search(path or "")
    return hit.group(1) if hit else ""


def _token_fingerprint(access_token: str) -> str:
    return (access_token or "")[:16]


def _yahoo_access_blocked(league_key: str, access_token: str) -> bool:
    if not league_key:
        return False
    key = (league_key, _token_fingerprint(access_token))
    with _yahoo_no_access_lock:
        until = _yahoo_no_access.get(key)
    return bool(until and until > time.time())


def _mark_yahoo_access_denied(league_key: str, access_token: str) -> None:
    if not league_key:
        return
    key = (league_key, _token_fingerprint(access_token))
    with _yahoo_no_access_lock:
        _yahoo_no_access[key] = time.time() + _YAHOO_NO_ACCESS_TTL


def _yahoo_not_in_league(status_code: int, body: str) -> bool:
    if int(status_code or 0) != 403:
        return False
    text = (body or "").lower()
    return "not in this league" in text or "not allowed to view this page" in text


def _owner_token_for_league_path(path: str, used_token: str) -> str:
    """Stored league-owner token when the session Yahoo account is not a member."""
    league_key = _league_key_from_yahoo_path(path)
    if not league_key:
        return ""
    owner = get_league_token(_bare_yahoo_id(league_key), 0) or ""
    if owner and owner != used_token:
        return owner
    return ""


def _alternate_league_paths(path: str) -> List[str]:
    """Same league id under the current-season game key (470 / nfl), not last year's 461."""
    hit = re.match(r"league/(\d+|nfl)\.l\.(\d+)(.*)$", path or "")
    if not hit:
        return []
    used, lid, rest = hit.group(1), hit.group(2), hit.group(3)
    year = datetime.now().year
    candidates: List[str] = ["nfl"]
    current = _NFL_GAME_KEYS_FALLBACK.get(year)
    if current:
        candidates.append(str(current))
    for gk in _NFL_GAME_KEYS_FALLBACK.values():
        candidates.append(str(gk))
    out: List[str] = []
    seen = {str(used)}
    for gk in candidates:
        if not gk or gk in seen:
            continue
        seen.add(gk)
        out.append(f"league/{gk}.l.{lid}{rest}")
    return out


def _remember_full_league_key(full: str) -> None:
    """Cache a league key that actually returned 200, by id and by season."""
    if not full or ".l." not in full:
        return
    lid = _bare_yahoo_id(full)
    gk = str(full).split(".l.", 1)[0]
    with _league_key_lock:
        _league_key_map[lid] = full
    year = None
    if gk == "nfl":
        year = datetime.now().year
    else:
        for season, key in _NFL_GAME_KEYS_FALLBACK.items():
            if str(key) == gk:
                year = season
                break
    if year:
        with _season_key_lock:
            _season_key_map[(lid, int(year))] = full


def _yahoo_get(
    access_token: str,
    path: str,
    params: Optional[Dict] = None,
    *,
    _retried: bool = False,
    _retried_key: bool = False,
) -> Any:
    """Make a GET request to the Yahoo Fantasy API, returning parsed JSON."""
    league_key = _league_key_from_yahoo_path(path)
    if _yahoo_access_blocked(league_key, access_token):
        if not _retried_key:
            for alt_path in _alternate_league_paths(path):
                alt_key = _league_key_from_yahoo_path(alt_path)
                if alt_key and not _yahoo_access_blocked(alt_key, access_token):
                    try:
                        return _yahoo_get(
                            access_token, alt_path, params,
                            _retried=_retried, _retried_key=True,
                        )
                    except YahooLeagueAccessDenied:
                        continue
        # Try every game key with this token before switching accounts.
        if not _retried and not _retried_key:
            alt = _owner_token_for_league_path(path, access_token)
            if alt and not _yahoo_access_blocked(league_key, alt):
                return _yahoo_get(alt, path, params, _retried=True)
        raise YahooLeagueAccessDenied(f"not a member of {league_key}")

    url = f"{YAHOO_API_BASE}/{path.lstrip('/')}"
    p = {"format": "json"}
    if params:
        p.update(params)

    cache_key = f"{url}?{urlencode(sorted(p.items()))}"
    now = time.time()

    with _api_cache_lock:
        hit = _api_cache.get(cache_key)
        if hit and (now - hit[0]) < _API_CACHE_TTL:
            return hit[1]

    resp = requests.get(
        url,
        params=p,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if resp.status_code >= 400:
        # Yahoo puts the real reason in the body (e.g. "token_expired",
        # "insufficient scope", "not in this league"). raise_for_status() drops
        # it, so surface it here — this is what tells a scope/permission problem
        # apart from a genuine league-membership 403.
        body = (resp.text or "")[:500].replace("\n", " ")
        logger.warning("[yahoo] %s %s -> %s body=%s", "GET", path, resp.status_code, body)
        if _yahoo_not_in_league(resp.status_code, body):
            _mark_yahoo_access_denied(league_key, access_token)
            if not _retried_key:
                for alt_path in _alternate_league_paths(path):
                    try:
                        logger.info("[yahoo] retrying %s as %s", path, alt_path)
                        data = _yahoo_get(
                            access_token, alt_path, params,
                            _retried=_retried, _retried_key=True,
                        )
                        _remember_full_league_key(_league_key_from_yahoo_path(alt_path))
                        return data
                    except YahooLeagueAccessDenied:
                        continue
            if not _retried and not _retried_key:
                alt = _owner_token_for_league_path(path, access_token)
                if alt and not _yahoo_access_blocked(league_key, alt):
                    logger.info(
                        "[yahoo] retrying %s with stored league-owner token", path,
                    )
                    return _yahoo_get(alt, path, params, _retried=True)
            raise YahooLeagueAccessDenied(
                f"not a member of {league_key or path}"
            )
    resp.raise_for_status()
    data = resp.json()

    if yahoo_api_debug_enabled():
        _yahoo_debug(
            "GET %s -> %s shape=%s",
            path, resp.status_code, _summarize_fantasy_response(data),
        )

    with _api_cache_lock:
        _api_cache[cache_key] = (time.time(), data)

    if league_key and ".l." in league_key:
        _remember_full_league_key(league_key)
    return data


def _league_key(league_id: str) -> str:
    # If the entered id already looks like a full key ("461.l.123456"), trust it.
    lid = str(league_id)
    if ".l." in lid:
        return lid
    with _league_key_lock:
        full = _league_key_map.get(lid)
    if full:
        return full
    # Fall back to the current-season game code. Correct when the league IS this
    # season's; resolve_league_key() supplies the right key for any other season.
    return f"{_GAME_CODE}.l.{lid}"


def _bare_yahoo_id(league_id: str) -> str:
    lid = str(league_id).strip()
    return lid.split(".l.")[-1] if ".l." in lid else lid


def _league_key_for_season(league_id: str, season: int, access_token: str = "") -> str:
    """Season-specific Yahoo league key (``<game_key>.l.<id>``).

    ``_league_key`` caches one key — the current (or resolved) season. Historical
    fetches must use that year's NFL game key or they silently re-read this
    season. Falls back to ``_league_key`` when game keys aren't available.
    """
    lid = _bare_yahoo_id(league_id)
    try:
        season_i = int(season)
    except (TypeError, ValueError):
        return _league_key(league_id)

    cache_key = (lid, season_i)
    with _season_key_lock:
        hit = _season_key_map.get(cache_key)
        if hit:
            return hit

    game_keys = _nfl_game_keys(access_token) if access_token else []
    for year, gk in game_keys:
        try:
            if int(year) != season_i:
                continue
        except (TypeError, ValueError):
            continue
        full = f"{gk}.l.{lid}"
        with _season_key_lock:
            _season_key_map[cache_key] = full
        return full
    gk = _NFL_GAME_KEYS_FALLBACK.get(season_i)
    if gk:
        full = f"{gk}.l.{lid}"
        with _season_key_lock:
            _season_key_map[cache_key] = full
        return full
    # Current season: ``nfl.l.<id>`` works for members. Do not reuse a
    # prior-year numeric key from ``_league_key_map`` (461 is 2025).
    if season_i == datetime.now().year:
        return f"{_GAME_CODE}.l.{lid}"
    return _league_key(league_id)


def yahoo_league_exists_for_season(access_token: str, league_id: str, season: int) -> bool:
    """True when this Yahoo league has a readable record for ``season``."""
    if not access_token:
        return False
    lid = _bare_yahoo_id(league_id)
    try:
        season_i = int(season)
    except (TypeError, ValueError):
        return False
    game_keys = _nfl_game_keys(access_token) or []
    gk = None
    for year, key in game_keys:
        try:
            if int(year) == season_i:
                gk = key
                break
        except (TypeError, ValueError):
            continue
    if not gk:
        return False
    full = f"{gk}.l.{lid}"
    try:
        raw = _yahoo_get(access_token, f"league/{full}")
    except Exception:
        return False
    if not _extract_league_meta(raw):
        return False
    with _season_key_lock:
        _season_key_map[(lid, season_i)] = full
    return True


def _flatten_yahoo_game_entry(entry: Any) -> Dict[str, Any]:
    node = entry
    if isinstance(entry, dict) and "game" in entry:
        node = entry.get("game")
    if isinstance(node, dict):
        return dict(node)
    if isinstance(node, list):
        flat: Dict[str, Any] = {}
        _merge_yahoo_dict_parts(node, flat)
        return flat
    return {}


def _parse_nfl_game_keys(raw: Any) -> Dict[int, str]:
    """Pair Yahoo game_key/season fragments into {season: game_key}.

    ``format=json`` often returns games as a list of one-key dicts
    (``[{game_key: "470"}, {season: "2026"}]``) instead of one dict per season.
    Walking those nodes independently never joins the pair, so 2026 used to
    miss ``470`` and fall back to last year's ``461``.
    """
    parsed: Dict[int, str] = {}
    if not isinstance(raw, dict):
        return parsed
    games = ((raw.get("fantasy_content") or {}).get("games"))
    rows = _yahoo_collection_rows(games, "game")
    pending: Dict[str, Any] = {}

    def _take(flat: Dict[str, Any]) -> None:
        if not flat:
            return
        gk = str(flat.get("game_key") or flat.get("game_id") or "")
        se = flat.get("season")
        if gk and se is not None:
            try:
                parsed[int(se)] = gk
            except (TypeError, ValueError):
                pass
            return
        if gk:
            pending["game_key"] = gk
        if se is not None:
            pending["season"] = se
        pk = str(pending.get("game_key") or pending.get("game_id") or "")
        ps = pending.get("season")
        if pk and ps is not None:
            try:
                parsed[int(ps)] = pk
            except (TypeError, ValueError):
                pass
            pending.clear()

    for entry in rows:
        _take(_flatten_yahoo_game_entry(entry))
    return parsed


def _nfl_game_keys(access_token: str) -> List[tuple]:
    """Return [(season:int, game_key:str), ...] for recent NFL seasons, newest
    first. Yahoo's game key changes every season and a league key is
    "<game_key>.l.<id>", so we need the actual keys to reach a specific season's
    league. This reads the games collection (not the user resource, so fspt-r is
    fine) for the last several seasons. Returns known fallbacks if it can't be read."""
    parsed: Dict[int, str] = {}
    yr = datetime.now().year
    seasons = [yr - i for i in range(0, 8)]
    try:
        raw = _yahoo_get(
            access_token,
            "games;game_codes=nfl;seasons=" + ",".join(str(s) for s in seasons),
        )
    except Exception as exc:
        logger.warning("[yahoo] _nfl_game_keys failed: %s", exc)
        raw = None

    parsed.update(_parse_nfl_game_keys(raw))

    merged = dict(_NFL_GAME_KEYS_FALLBACK)
    merged.update(parsed)
    return sorted(merged.items(), reverse=True)


def resolve_league_key(access_token: str, league_id: str) -> Dict[str, Any]:
    """Find the real, season-specific league key for a bare numeric league_id.

    Yahoo's "nfl" shortcut only ever points at the current season's game, so a
    league from a prior (or not-yet-current) season can't be reached as
    "nfl.l.<id>" — the request 403s even for a member. This looks up the actual
    NFL game keys for recent seasons and probes "<game_key>.l.<id>" newest-first
    until one resolves (a league the account can read), then caches that full key
    so every downstream call uses it.

    Returns a status dict so callers can tell a genuine wrong-account from a
    can't-check (which must NOT be treated as denial, or a current-season league
    that works today would start being wrongly rejected):
      {"status": "found", "league_key", "league_id", "season", "name"}
      {"status": "absent"}   game keys enumerated, no season had that id -> wrong account/id
      {"status": "unknown"}  couldn't enumerate game keys -> caller falls back to nfl.l.<id>
    """
    lid = str(league_id).strip()
    if not lid:
        return {"status": "absent"}
    # A full key was pasted directly — accept and cache it.
    if ".l." in lid:
        bare = lid.split(".l.")[-1]
        with _league_key_lock:
            _league_key_map[bare] = lid
        return {"status": "found", "league_key": lid, "league_id": bare,
                "season": None, "name": None}

    game_keys = _nfl_game_keys(access_token)
    if not game_keys:
        return {"status": "unknown"}

    for season, gk in game_keys:
        full_key = f"{gk}.l.{lid}"
        try:
            raw  = _yahoo_get(access_token, f"league/{full_key}")
            meta = _extract_league_meta(raw)
        except Exception:
            # 403/404 for a season the account isn't in with this id — keep probing.
            continue
        if meta:
            with _league_key_lock:
                _league_key_map[lid] = full_key
            logger.info("[yahoo] resolved league %s -> %s (season %s)", lid, full_key, season)
            return {
                "status":     "found",
                "league_key": full_key,
                "league_id":  lid,
                "season":     int(season) or None,
                "name":       meta.get("name"),
            }
    return {"status": "absent"}


# ---------------------------------------------------------------------------
# Player ID mapping (name + position → canonical Sleeper ID)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _name_pos_to_canonical() -> Dict[str, str]:
    """Build a (normalized_name, pos) -> canonical_id lookup from players_index."""
    from utils.utils import normalize_name
    idx = load_players_index() or {}
    mapping: Dict[str, str] = {}
    for canonical_id, info in idx.items():
        name = normalize_name(info.get("name") or "")
        pos  = (info.get("pos") or "").upper()
        team = (info.get("team") or "").upper()
        if name:
            # Most-specific to least-specific keys so later entries don't clobber
            mapping.setdefault(f"{name}|{pos}|{team}", canonical_id)
            mapping.setdefault(f"{name}|{pos}|",       canonical_id)
            mapping.setdefault(f"{name}||",             canonical_id)
    return mapping


@lru_cache(maxsize=1)
def _yahoo_id_to_canonical() -> Dict[str, str]:
    """Exact yahoo_id -> canonical Sleeper id crosswalk.

    A Yahoo player's ``player_id`` equals Sleeper's ``yahoo_id``, which the full
    Sleeper players feed carries for (almost) every player. Building the map from
    that feed gives a precise id match and avoids the lossy name/pos/team fallback
    (which silently drops Jr./Sr., rookies, and stale-team players). Cached for the
    process; empty on any failure so callers fall back to name matching."""
    try:
        from dashboard_services.api import get_nfl_players
        feed = get_nfl_players() or {}
    except Exception:
        logger.debug("[yahoo] player feed load failed for crosswalk", exc_info=True)
        return {}
    out: Dict[str, str] = {}
    for sleeper_id, info in feed.items():
        yid = (info or {}).get("yahoo_id")
        if yid:
            out[str(yid)] = str(sleeper_id)
    return out


def _yahoo_pos(raw_pos: str) -> str:
    _MAP = {
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
        "K": "K", "DEF": "DEF", "D": "DEF", "DST": "DEF",
        "PK": "K",
    }
    return _MAP.get((raw_pos or "").upper(), (raw_pos or "").upper())


def _merge_yahoo_dict_parts(node: Any, into: Dict[str, Any]) -> None:
    """Recursively merge Yahoo's positional single-key dict fragments into ``into``."""
    if isinstance(node, dict):
        into.update(node)
    elif isinstance(node, list):
        for item in node:
            _merge_yahoo_dict_parts(item, into)


def _yahoo_position_from_selected(raw: Any) -> Optional[str]:
    """Parse Yahoo ``selected_position`` which is often ``[meta, {position: BN}]``."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        pos = raw.get("position")
        if isinstance(pos, str):
            return pos
        if isinstance(pos, (list, dict)):
            inner = _unwrap_yahoo_list_or_dict(pos).get("position")
            return str(inner) if inner else None
        return None
    if isinstance(raw, list):
        # Yahoo roster slots commonly use index 1 for the position string.
        for item in reversed(raw):
            if not isinstance(item, dict):
                continue
            if item.get("position"):
                pos = item.get("position")
                if isinstance(pos, str):
                    return pos
                if isinstance(pos, (list, dict)):
                    inner = _unwrap_yahoo_list_or_dict(pos).get("position")
                    if inner:
                        return str(inner)
            nested = _yahoo_position_from_selected(item.get("selected_position"))
            if nested:
                return nested
        flat: Dict[str, Any] = {}
        _merge_yahoo_dict_parts(raw, flat)
        pos = flat.get("position")
        return str(pos) if pos else None
    return None


def _yahoo_selected_position(slot_node: Any) -> Optional[str]:
    """Extract roster slot (QB/BN/IR) from a Yahoo player slot fragment."""
    slot = _unwrap_yahoo_list_or_dict(slot_node)
    if not slot:
        return None
    if "selected_position" in slot:
        return _yahoo_position_from_selected(slot.get("selected_position"))
    return _yahoo_position_from_selected(slot)


def _flatten_yahoo_player(rp: Any) -> tuple:
    """Normalize a Yahoo ``player`` entry to (meta_dict, selected_position).

    Yahoo returns a player as ``[[{k:v}, {k:v}, ...], {selected_position}]`` —
    the metadata is a positional list of single-key dicts. Merge it into one
    flat dict (also tolerating an already-flat dict), so callers can read
    ``name``/``player_id``/``editorial_team_abbr`` uniformly."""
    if isinstance(rp, dict):
        return rp, _yahoo_selected_position(rp)

    node = rp
    while isinstance(node, list) and len(node) == 1:
        node = node[0]

    flat: Dict[str, Any] = {}
    sel_pos = None
    if isinstance(node, list) and node:
        _merge_yahoo_dict_parts(node[0], flat)
        if len(node) > 1:
            sel_pos = _yahoo_selected_position(node[1])
    elif isinstance(node, dict):
        flat = node
        sel_pos = _yahoo_selected_position(node)

    if not sel_pos:
        sel_pos = _yahoo_selected_position(flat)
    return flat, sel_pos


def _resolve_player(
    yahoo_name: str, yahoo_pos: str, yahoo_team: str, yahoo_id: Optional[str] = None
) -> Optional[str]:
    """Map a Yahoo player to a canonical Sleeper id. Prefers the exact yahoo_id
    crosswalk; falls back to name/pos/team when the id is unknown."""
    if yahoo_id:
        hit = _yahoo_id_to_canonical().get(str(yahoo_id))
        if hit:
            return hit
    from utils.utils import normalize_name
    m    = _name_pos_to_canonical()
    name = normalize_name(yahoo_name or "")
    pos  = _yahoo_pos(yahoo_pos)
    team = (yahoo_team or "").upper()
    return (
        m.get(f"{name}|{pos}|{team}")
        or m.get(f"{name}|{pos}|")
        or m.get(f"{name}||")
    )


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _yahoo_league_nodes(raw: Dict) -> List[Any]:
    """Normalize ``fantasy_content.league`` to a list of nodes.

    Official ``format=json`` uses an array. Some payloads use a count-keyed
    object (``{"0": meta, "1": {scoreboard}}``) instead.
    """
    fc = (raw or {}).get("fantasy_content", {}) or {}
    lg = fc.get("league")
    if isinstance(lg, list):
        return lg
    if isinstance(lg, dict):
        rows = []
        count = _safe_int(lg.get("count")) or 0
        if count:
            for i in range(count):
                entry = lg.get(str(i))
                if entry is not None:
                    rows.append(entry)
        if not rows:
            rows = [v for k, v in lg.items() if str(k).isdigit() and v is not None]
        return rows
    return []


def _extract_league_meta(raw: Dict) -> Dict:
    league_list = _yahoo_league_nodes(raw)
    first = league_list[0] if league_list else {}
    return first if isinstance(first, dict) else {}


def _league_child_block(raw: Dict, child_key: str) -> Any:
    """Find a child collection (``teams``, ``settings``, ``scoreboard``, …).

    Yahoo usually nests these under ``league[1]``, but some sub-resource
    responses attach them to ``league[0]`` instead — scanning avoids empty
    extracts when the index shifts.
    """
    for item in _yahoo_league_nodes(raw):
        if isinstance(item, dict) and child_key in item:
            block = item.get(child_key)
            if block is not None:
                return block
    return {}


def _summarize_fantasy_response(raw: Dict) -> Dict[str, Any]:
    """Non-sensitive structural summary for debug logs / the debug endpoint."""
    fc = raw.get("fantasy_content", {}) or {}
    lg = fc.get("league") or []
    summary: Dict[str, Any] = {
        "league_nodes": len(lg) if isinstance(lg, list) else 0,
        "league_indices": [],
    }
    if not isinstance(lg, list):
        return summary
    for i, item in enumerate(lg):
        if not isinstance(item, dict):
            summary["league_indices"].append({"index": i, "type": type(item).__name__})
            continue
        entry: Dict[str, Any] = {"index": i, "keys": sorted(item.keys())}
        if "teams" in item and isinstance(item.get("teams"), dict):
            tb = item["teams"]
            entry["teams_count_field"] = _safe_int(tb.get("count"))
            entry["teams_numeric_keys"] = sorted(
                (k for k in tb if str(k).isdigit()), key=lambda x: int(x)
            )
        for meta_key in ("name", "league_key", "num_teams", "draft_status", "season"):
            if meta_key in item:
                entry[meta_key] = item.get(meta_key)
        summary["league_indices"].append(entry)
    return summary


def _summarize_team_entry(team_data: List) -> Dict[str, Any]:
    """Parsed team snapshot for debug output (no tokens / no full rosters)."""
    team_key = _team_attr(team_data, "team_key") or ""
    team_id = _team_attr(team_data, "team_id") or team_key.split(".")[-1]
    raw_players = _extract_roster_players(team_data)
    resolved = 0
    unmapped_samples: List[str] = []
    for rp in raw_players:
        p_meta, _ = _flatten_yahoo_player(rp)
        yid = str(p_meta.get("player_id") or "")
        name = (p_meta.get("name") or {}).get("full") if isinstance(p_meta.get("name"), dict) else p_meta.get("name")
        pos_list = p_meta.get("display_position") or ""
        if isinstance(pos_list, dict):
            pos_list = pos_list.get("position") or ""
        pos = (pos_list.split(",")[0] if isinstance(pos_list, str) else "") or ""
        team = (p_meta.get("editorial_team_abbr") or "").upper()
        if _resolve_player(name or "", pos, team, yahoo_id=yid):
            resolved += 1
        elif len(unmapped_samples) < 5:
            unmapped_samples.append(f"yid={yid} name={name!r} pos={pos}")
    standings = _team_field_dict(team_data, "team_standings")
    outcome = _unwrap_yahoo_list_or_dict(standings.get("outcome_totals"))
    roster_block = _team_attr(team_data, "roster")
    roster_keys = sorted(roster_block.keys()) if isinstance(roster_block, dict) else []
    players_block = _roster_players_block(roster_block) if isinstance(roster_block, dict) else {}
    return {
        "team_id":        team_id,
        "team_key":       team_key,
        "name":           _team_attr(team_data, "name"),
        "raw_players":    len(raw_players),
        "resolved_players": resolved,
        "unmapped_samples": unmapped_samples,
        "has_roster":     isinstance(roster_block, dict) and bool(roster_block),
        "roster_block_keys": roster_keys[:12],
        "players_block_keys": sorted(
            (k for k in (players_block or {}) if str(k).isdigit() or k == "count"),
            key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
        )[:6],
        "wins":           outcome.get("wins"),
        "points_for":     standings.get("points_for"),
    }


def diagnose_league(season: int, league_id: str, access_token: str) -> Dict[str, Any]:
    """Structured Yahoo parse diagnostics for support / ``/api/yahoo-debug``."""
    lk = _league_key_for_season(league_id, season, access_token)
    teams_path = f"league/{lk}/teams;out=roster,stats,standings"
    users_path = f"league/{lk}/teams"
    out: Dict[str, Any] = {
        "ok": True,
        "season": int(season),
        "league_id": str(league_id),
        "league_key": lk,
        "crosswalk_size": len(_yahoo_id_to_canonical()),
    }
    try:
        raw_rosters = _yahoo_get(access_token, teams_path)
        out["rosters_path"] = teams_path
        out["rosters_response_shape"] = _summarize_fantasy_response(raw_rosters)
        teams = _extract_teams(raw_rosters)
        out["extracted_team_count"] = len(teams)
        out["teams"] = [_summarize_team_entry(t) for t in teams]
        out["parsed_users_count"] = len(get_users(season, league_id, access_token))
        out["parsed_rosters_count"] = len(get_rosters(season, league_id, access_token))
        meta_raw = _yahoo_get(access_token, f"league/{lk}")
        meta = _extract_league_meta(meta_raw) or {}
        out["league_meta"] = {
            "name": meta.get("name"),
            "num_teams": meta.get("num_teams"),
            "draft_status": meta.get("draft_status"),
            "scoring_type": meta.get("scoring_type"),
            "current_week": meta.get("current_week"),
        }
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["users_path"] = users_path
    return out


def _extract_teams(raw: Dict) -> List[Dict]:
    """Extract the teams dict from a league+teams response."""
    teams_block = _league_child_block(raw, "teams") or {}
    out = []
    # Yahoo collections are 0-indexed ("0", "1", …, count-1). A 1-based loop
    # skipped team 0 and read past the end, so leagues looked empty or short.
    for entry in _yahoo_collection_rows(teams_block, "team"):
        if isinstance(entry, dict) and "team" in entry:
            out.append(entry["team"])
    return out


_TEAM_SUBRESOURCES = frozenset({
    "roster", "team_standings", "team_points", "team_projected_points",
    "team_stats", "matchups", "draft_results",
})


def _team_attr(team_list: List, key: str, default=None):
    """Find a team field in Yahoo's positional team array.

    With sub-resources (``;out=roster,stats,standings``) metadata lives in
    ``team[0]`` as a nested list of single-key dicts. Without sub-resources
    it's often a flat list of dicts. Walk both shapes."""
    def _walk(nodes):
        for node in nodes:
            if isinstance(node, dict):
                if key in node:
                    return node[key]
                if any(k in node for k in _TEAM_SUBRESOURCES):
                    continue
            elif isinstance(node, list):
                hit = _walk(node)
                if hit is not None:
                    return hit
        return None

    hit = _walk(team_list or [])
    return default if hit is None else hit


def _team_field_dict(team_list: List, key: str) -> Dict[str, Any]:
    """Read a team sub-resource and normalize Yahoo list wrappers to a dict."""
    return _unwrap_yahoo_list_or_dict(_team_attr(team_list, key))


def _yahoo_manager_entries(team_list: List) -> List[Dict[str, Any]]:
    """Normalize Yahoo ``managers`` nodes to flat manager dicts."""
    raw = _team_attr(team_list, "managers")
    if raw is None:
        return []
    rows = _yahoo_collection_rows(raw, "manager")
    if not rows and isinstance(raw, list):
        rows = raw
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mgr_node = row.get("manager") if "manager" in row else row
        flat: Dict[str, Any] = {}
        _merge_yahoo_dict_parts(mgr_node, flat)
        if flat:
            out.append(flat)
    return out


def _yahoo_primary_manager(team_list: List) -> Dict[str, Any]:
    entries = _yahoo_manager_entries(team_list)
    return entries[0] if entries else {}


def _yahoo_owner_id(team_list: List, team_id: Any) -> str:
    mgr = _yahoo_primary_manager(team_list)
    guid = mgr.get("guid") or mgr.get("manager_id")
    if guid:
        return str(guid)
    return str(team_id)


def _team_managers(team_list: List) -> List[Dict[str, Any]]:
    """Normalize Yahoo ``managers`` nodes to ``[{"manager": ...}, ...]``."""
    return [{"manager": m} for m in _yahoo_manager_entries(team_list)]


def _roster_players_block(roster_block: Any) -> Dict[str, Any]:
    """Locate the ``players`` collection inside a Yahoo roster blob.

    Bulk ``league/.../teams;out=roster`` often returns a roster shell without
    players; the team roster resource nests them under ``roster.players`` or an
    extra wrapper layer (see yahoo_fantasy_api's team.roster parser).
    """
    if not isinstance(roster_block, dict):
        return {}
    players = roster_block.get("players")
    if isinstance(players, dict):
        return players
    for value in roster_block.values():
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("players"), dict):
            return value["players"]
        inner = value.get("roster")
        if isinstance(inner, dict) and isinstance(inner.get("players"), dict):
            return inner["players"]
    return {}


def _extract_roster_players(team_data: List) -> List[Dict]:
    """Extract player list from the roster portion of a team entry."""
    roster_block = _team_attr(team_data, "roster")
    if not isinstance(roster_block, dict):
        for item in team_data or []:
            if isinstance(item, dict) and "roster" in item:
                roster_block = item["roster"]
                break
    if not isinstance(roster_block, dict):
        return []

    players_block = _roster_players_block(roster_block) or {}
    out = []
    for entry in _yahoo_collection_rows(players_block, "player"):
        if not isinstance(entry, dict) or "player" not in entry:
            continue
        rp = entry["player"]
        entry_slot = entry.get("selected_position")
        if entry_slot is not None and isinstance(rp, list):
            slot_node = rp[1] if len(rp) > 1 else None
            if not _yahoo_selected_position(slot_node):
                rp = list(rp) + [{"selected_position": entry_slot}]
        out.append(rp)
    return out


def _league_current_week(access_token: str, league_key: str) -> Optional[int]:
    try:
        meta = _extract_league_meta(_yahoo_get(access_token, f"league/{league_key}"))
        week = _safe_int(meta.get("current_week"))
        return week if week > 0 else None
    except Exception:
        return None


def _fetch_team_roster_players(
    access_token: str, team_key: str, week: Optional[int] = None,
) -> List[Dict]:
    """Fetch one team's roster via the team resource (includes players).

    Yahoo's bulk ``teams;out=roster`` attaches roster metadata but omits the
    players collection — this endpoint is the reliable source.
    """
    if not team_key:
        return []
    path = f"team/{team_key}/roster"
    if week and week > 0:
        path += f";week={int(week)}"
    try:
        raw = _yahoo_get(access_token, path)
    except Exception as exc:
        logger.warning("[yahoo] team roster %s failed: %s", team_key, exc)
        return []
    fc = raw.get("fantasy_content", {}) or {}
    team = fc.get("team")
    if isinstance(team, list):
        return _extract_roster_players(team)
    return []


def _prefetch_team_rosters(
    access_token: str,
    teams: List[List],
    week: Optional[int] = None,
    team_keys: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """Parallel per-team roster fetch keyed by team_key."""
    if team_keys is not None:
        keys = [k for k in team_keys if k]
    else:
        keys = []
        for t in teams:
            tk = _team_attr(t, "team_key") or ""
            if tk:
                keys.append(tk)
    if not keys:
        return {}

    out: Dict[str, List[Dict]] = {}
    workers = min(len(keys), 8)

    def _load(team_key: str) -> tuple[str, List[Dict]]:
        return team_key, _fetch_team_roster_players(access_token, team_key, week)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load, tk): tk for tk in keys}
        for fut in as_completed(futures):
            try:
                tk, players = fut.result()
                out[tk] = players
            except Exception as exc:
                logger.warning("[yahoo] prefetch roster failed: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Public API – normalized to match Sleeper/ESPN shapes
# ---------------------------------------------------------------------------

def get_league(season: int, league_id: str, access_token: str) -> Dict[str, Any]:
    raw  = _yahoo_get(access_token, f"league/{_league_key_for_season(league_id, season, access_token)}")
    meta = _extract_league_meta(raw)
    return {
        "league_id": str(league_id),
        "season":    int(season),
        "name":      meta.get("name") or "Yahoo League",
    }


def get_users(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    raw   = _yahoo_get(access_token, f"league/{_league_key_for_season(league_id, season, access_token)}/teams")
    teams = _extract_teams(raw)
    out: List[Dict[str, Any]] = []
    for t in teams:
        team_key  = _team_attr(t, "team_key") or ""
        team_id   = _team_attr(t, "team_id") or team_key.split(".")[-1]
        team_name = _team_attr(t, "name") or f"Team {team_id}"
        logo      = _team_attr(t, "team_logos", {})
        logo_url  = None
        if isinstance(logo, list) and logo:
            logo_url = (logo[0].get("team_logo") or {}).get("url")

        mgr   = _yahoo_primary_manager(t)
        guid  = _yahoo_owner_id(t, team_id)
        # Yahoo privacy mode returns nickname "--hidden--". Never treat that as
        # a display name — fall back to the public team name.
        nick  = public_owner_label(mgr.get("nickname"), team_name, fallback=team_name)

        out.append({
            "avatar":       logo_url,
            "display_name": nick,
            "username":     nick,
            "team_name":    team_name,
            "is_bot":       False,
            "is_owner":     None,
            "league_id":    str(league_id),
            "metadata":     {"team_name": team_name, "avatar": logo_url},
            "settings":     None,
            "user_id":      guid,
            "roster_id":    _safe_int(team_id),
        })
    _yahoo_debug(
        "get_users league=%s season=%s -> %s users (ids=%s)",
        league_id, season, len(out),
        [u.get("roster_id") for u in out[:20]],
    )
    return out


def _yahoo_players_need_hydration(raw_players: List[Dict]) -> bool:
    """True when roster players are missing or lack lineup slot info."""
    if not raw_players:
        return True
    return any(not _flatten_yahoo_player(rp)[1] for rp in raw_players)


# Bench stays on ``players`` only (dashboard bench list). IR-only goes to
# ``reserve``, matching ESPN/Sleeper. NA is Yahoo's inactive/bye slot.
_YAHOO_BENCH_SLOTS = frozenset({"BN", "NA"})
_YAHOO_IR_SLOTS = frozenset({"IR", "IR+"})


def _yahoo_player_canonical(rp: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (canonical_id, selected_position) for one Yahoo roster row."""
    p_meta, sel_pos = _flatten_yahoo_player(rp)
    name = (p_meta.get("name") or {}).get("full") or ""
    pos_list = p_meta.get("display_position") or p_meta.get("eligible_positions") or ""
    if isinstance(pos_list, dict):
        pos_list = pos_list.get("position") or ""
    pos = (pos_list.split(",")[0] if isinstance(pos_list, str) else "") or ""
    team = (p_meta.get("editorial_team_abbr") or "").upper()
    yid = str(p_meta.get("player_id") or "")
    return _resolve_player(name, pos, team, yahoo_id=yid), sel_pos


def _split_yahoo_lineup(raw_players: List[Any]) -> tuple[List[str], List[str], List[str]]:
    """Map Yahoo roster rows to (players, starters, reserve/IR).

    BN/NA are not starters and not IR — ``build_teams_overview`` puts those
    leftover ``players`` on the dashboard Bench list. Putting BN in
    ``reserve`` hid the rest of the roster because the teams-card does not
    render IR.
    """
    players: List[str] = []
    starters: List[str] = []
    reserve: List[str] = []
    for rp in raw_players:
        canon, sel_pos = _yahoo_player_canonical(rp)
        if not canon:
            continue
        players.append(canon)
        slot = (sel_pos or "").upper()
        if slot in _YAHOO_IR_SLOTS:
            reserve.append(canon)
        elif slot in _YAHOO_BENCH_SLOTS:
            continue
        else:
            starters.append(canon)
    if not starters and players:
        # Yahoo tags every player BN before a lineup is submitted.
        ir_set = set(reserve)
        starters = [p for p in players if p not in ir_set][:9]
    return players, starters, reserve


def get_rosters(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    lk = _league_key_for_season(league_id, season, access_token)
    raw = _yahoo_get(
        access_token,
        f"league/{lk}/teams;out=roster,stats,standings",
    )
    teams = _extract_teams(raw)
    out: List[Dict[str, Any]] = []

    # Bulk teams;out=roster omits or misplaces lineup slots. During an active
    # week, always hydrate from team/{team_key}/roster;week=N (reliable source).
    roster_by_key: Dict[str, List[Dict]] = {}
    week = _league_current_week(access_token, lk) or 1
    team_keys = [_team_attr(t, "team_key") or "" for t in teams]
    team_keys = [k for k in team_keys if k]
    if week > 0 and team_keys:
        roster_by_key = _prefetch_team_rosters(
            access_token, teams, week, team_keys=team_keys,
        )
        _yahoo_debug(
            "get_rosters prefetched %s team rosters week=%s counts=%s",
            len(roster_by_key), week,
            {k: len(v) for k, v in list(roster_by_key.items())[:5]},
        )

    for t in teams:
        team_key  = _team_attr(t, "team_key") or ""
        team_id   = _safe_int(_team_attr(t, "team_id") or team_key.split(".")[-1])
        team_name = _team_attr(t, "name") or f"Team {team_id}"
        owner_id = _yahoo_owner_id(t, team_id)
        logo      = _team_attr(t, "team_logos", {})
        logo_url  = None
        if isinstance(logo, list) and logo:
            logo_url = (logo[0].get("team_logo") or {}).get("url")
        elif isinstance(logo, dict):
            inner = _unwrap_yahoo_list_or_dict(logo.get("team_logo"))
            logo_url = inner.get("url")

        # Standings / record
        standings = _team_field_dict(t, "team_standings")
        outcome   = _unwrap_yahoo_list_or_dict(standings.get("outcome_totals"))
        wins      = _safe_int(outcome.get("wins"))
        losses    = _safe_int(outcome.get("losses"))
        ties      = _safe_int(outcome.get("ties"))
        pts_for   = _safe_float(standings.get("points_for"))
        pts_ag    = _safe_float(standings.get("points_against"))

        # Roster — prefer week-specific team resource (has lineup slots).
        raw_players = roster_by_key.get(team_key) or _extract_roster_players(t)
        if _yahoo_players_need_hydration(raw_players) and team_key:
            raw_players = _fetch_team_roster_players(access_token, team_key, week)
        players, starters, reserve = _split_yahoo_lineup(raw_players)

        # If every mapped player landed in starters, lineup slots are still missing.
        if (
            players
            and len(starters) == len(players)
            and len(players) > 9
            and team_key
        ):
            retry_players = _fetch_team_roster_players(access_token, team_key, week)
            if retry_players:
                players, starters, reserve = _split_yahoo_lineup(retry_players)

        fpts_whole = int(pts_for)
        fpts_dec   = int(round((pts_for - fpts_whole) * 100))
        fpa_whole  = int(pts_ag)
        fpa_dec    = int(round((pts_ag - fpa_whole) * 100))

        out.append({
            "co_owners":  None,
            "keepers":    None,
            "league_id":  str(league_id),
            "metadata":   {"team_name": team_name, "avatar": logo_url},
            "owner_id":   owner_id,
            "player_map": None,
            "players":    players,
            "reserve":    reserve,
            "roster_id":  team_id,
            "settings": {
                "wins":                 wins,
                "losses":               losses,
                "ties":                 ties,
                "fpts":                 fpts_whole,
                "fpts_decimal":         fpts_dec,
                "fpts_against":         fpa_whole,
                "fpts_against_decimal": fpa_dec,
                "ppts":                 0,
                "ppts_decimal":         0,
                "total_moves":          0,
                "waiver_budget_used":   0,
                "waiver_position":      0,
            },
            "starters": starters,
            "taxi":     None,
        })
    _yahoo_debug(
        "get_rosters league=%s season=%s -> %s rosters players_per_roster=%s",
        league_id, season, len(out),
        {r.get("roster_id"): len(r.get("players") or []) for r in out[:20]},
    )
    return out


def _yahoo_scoreboard_dict(raw: Dict) -> Dict[str, Any]:
    """Unwrap Yahoo ``scoreboard`` which is often a list of fragments."""
    block = _league_child_block(raw, "scoreboard")
    if isinstance(block, list):
        flat: Dict[str, Any] = {}
        _merge_yahoo_dict_parts(block, flat)
        return flat
    if isinstance(block, dict):
        if "matchups" in block:
            return block
        inner = block.get("0")
        if isinstance(inner, dict) and ("matchups" in inner or "week" in inner):
            merged = dict(block)
            merged.update(inner)
            return merged
        return block
    return {}


def _yahoo_teams_block(node: Any, *, _depth: int = 0) -> Any:
    """Find a ``teams`` collection inside Yahoo's matchup wrappers.

    Official ``format=json`` often keeps ``<teams>`` under a numeric key
    after the week/status scalars, e.g. ``{"0": {"teams": {...}}}`` or
    ``[{week…}, {"0": {"teams": ...}}]``. Reading only ``matchup["teams"]``
    then yields nothing and the dashboard prints "No matchup data found."
    """
    if _depth > 6 or node is None:
        return {}
    if isinstance(node, dict):
        teams = node.get("teams")
        if teams not in (None, "", [], {}):
            return teams
        for key, val in node.items():
            if str(key).isdigit() or key == "matchup":
                found = _yahoo_teams_block(val, _depth=_depth + 1)
                if found:
                    return found
        for val in node.values():
            if isinstance(val, (dict, list)):
                found = _yahoo_teams_block(val, _depth=_depth + 1)
                if found:
                    return found
    elif isinstance(node, list):
        for item in node:
            found = _yahoo_teams_block(item, _depth=_depth + 1)
            if found:
                return found
    return {}


def _flatten_yahoo_matchup(entry: Any) -> Dict[str, Any]:
    """Normalize a scoreboard matchup row to a dict with ``teams``.

    Yahoo commonly returns ``{"matchup": [{week/status…}, {teams: …}]}`` —
    a list of single-key fragments — not a flat dict. Treating a list as
    invalid dropped every pairing and painted "No matchups".
    """
    node = entry
    if isinstance(entry, dict) and "matchup" in entry:
        node = entry.get("matchup")
    if isinstance(node, dict):
        flat = dict(node)
    elif isinstance(node, list):
        flat = {}
        _merge_yahoo_dict_parts(node, flat)
    else:
        return {}
    if "teams" not in flat:
        teams = _yahoo_teams_block(flat) or _yahoo_teams_block(node)
        if teams:
            flat["teams"] = teams
    return flat


def _matchups_from_scoreboard_node(node: Any) -> Any:
    """Yahoo wraps scoreboard as ``{week, "0": {matchups}}`` more often than a
    flat ``{matchups}`` object. Look in both places."""
    if isinstance(node, list):
        for item in node:
            found = _matchups_from_scoreboard_node(item)
            if found:
                return found
        return {}
    if not isinstance(node, dict):
        return {}
    if "matchups" in node:
        return node.get("matchups") or {}
    inner = node.get("0")
    if isinstance(inner, dict) and "matchups" in inner:
        return inner.get("matchups") or {}
    return {}


def _as_yahoo_matchup(entry: Any) -> Dict[str, Any]:
    """Normalize a scoreboard row to a matchup dict with a ``teams`` block."""
    return _flatten_yahoo_matchup(entry)


def _yahoo_team_list_from_entry(tm_entry: Any) -> List:
    """Scoreboard teams are ``{"team": [...]}`` or the positional team array."""
    if isinstance(tm_entry, list):
        return tm_entry
    if not isinstance(tm_entry, dict):
        return []
    tm = tm_entry.get("team") if "team" in tm_entry else tm_entry
    if isinstance(tm, dict) and "team" in tm:
        tm = tm.get("team")
    if isinstance(tm, list):
        return tm
    return [tm] if tm else []


def _yahoo_roster_id_from_team(tm: List) -> int:
    return _safe_int(
        _team_attr(tm, "team_id")
        or (_team_attr(tm, "team_key") or "").split(".")[-1]
    )


def _matchup_rows_from_scoreboard(raw: Any, week: int) -> List[Dict[str, Any]]:
    """Parse one Yahoo scoreboard payload into Sleeper-shaped matchup rows."""
    if not isinstance(raw, dict):
        return []
    matchups = _matchups_from_scoreboard_node(_league_child_block(raw, "scoreboard"))
    if not matchups:
        matchups = _yahoo_scoreboard_dict(raw).get("matchups") or {}
    out: List[Dict[str, Any]] = []
    m_id = 0
    for entry in _yahoo_collection_rows(matchups, "matchup"):
        matchup = _as_yahoo_matchup(entry)
        if not matchup:
            continue
        teams_block = matchup.get("teams") or _yahoo_teams_block(matchup) or {}
        team_rows = _yahoo_collection_rows(teams_block, "team")
        if not team_rows and isinstance(teams_block, dict):
            for j in range(_safe_int(teams_block.get("count")) or 2):
                row = teams_block.get(str(j))
                if row:
                    team_rows.append(row)

        sides: List[Dict[str, Any]] = []
        for tm_entry in team_rows:
            tm = _yahoo_team_list_from_entry(tm_entry)
            roster_id = _yahoo_roster_id_from_team(tm)
            if not roster_id:
                continue
            pts_block = _team_field_dict(tm, "team_points")
            sides.append({
                "points":          _safe_float(pts_block.get("total")),
                "players":         [],
                "roster_id":       roster_id,
                "custom_points":   None,
                "starters":        [],
                "starters_points": [],
                "players_points":  {},
            })
        if len(sides) < 2:
            continue
        m_id += 1
        for side in sides[:2]:
            side["matchup_id"] = m_id
            side["week"] = int(week)
            out.append(side)
    return out


def get_matchups(season: int, league_id: str, week: int, access_token: str) -> List[Dict[str, Any]]:
    """Return Sleeper-shaped matchup rows for Yahoo's published week pairings.

    Yahoo's JSON scoreboard nests matchups under ``scoreboard["0"]["matchups"]``.
    Reading only ``scoreboard["matchups"]`` yields an empty list, and the
    Season Hub then invents round-robin opponents that do not match Yahoo.
    """
    lk = _league_key_for_season(league_id, season, access_token)
    try:
        raw = _yahoo_get(access_token, f"league/{lk}/scoreboard;week={week}")
    except YahooLeagueAccessDenied:
        return []
    except Exception as exc:
        logger.warning("[yahoo] get_matchups failed: %s", exc)
        return []

    out = _matchup_rows_from_scoreboard(raw, week)
    # Before kickoff Yahoo sometimes leaves ``;week=N`` empty while the
    # default scoreboard already has the current week's pairings. Do not
    # retry that path after an HTTP/auth failure — it just doubles 403s.
    if not out:
        try:
            raw_default = _yahoo_get(access_token, f"league/{lk}/scoreboard")
        except YahooLeagueAccessDenied:
            raw_default = None
        except Exception as exc:
            logger.warning("[yahoo] get_matchups default scoreboard failed: %s", exc)
            raw_default = None
        default_rows = _matchup_rows_from_scoreboard(raw_default, week)
        sb = _yahoo_scoreboard_dict(raw_default if isinstance(raw_default, dict) else {})
        sb_week = _safe_int(sb.get("week"))
        if default_rows and (sb_week == _safe_int(week) or _safe_int(week) <= 1):
            out = default_rows

    _yahoo_debug(
        "get_matchups league=%s season=%s week=%s -> %s rows pairings=%s",
        league_id, season, week, len(out),
        [(r.get("matchup_id"), r.get("roster_id")) for r in out],
    )
    return out


def get_transactions(season: int, league_id: str, week: int, access_token: str) -> List[Dict[str, Any]]:
    try:
        raw = _yahoo_get(
            access_token,
            f"league/{_league_key_for_season(league_id, season, access_token)}/transactions;types=add,drop,trade",
        )
    except YahooLeagueAccessDenied:
        return []
    except Exception as exc:
        logger.warning("[yahoo] get_transactions failed: %s", exc)
        return []

    fc    = raw.get("fantasy_content", {})
    lg    = fc.get("league") or []
    trans = (lg[1] if len(lg) > 1 else {}).get("transactions") or {}
    count = _safe_int(trans.get("count") or trans.get("0", {}).get("count")) or 0

    out: List[Dict[str, Any]] = []
    for i in range(count):
        entry = trans.get(str(i)) or {}
        tx    = entry.get("transaction") or []
        if not tx:
            continue
        meta   = tx[0] if isinstance(tx, list) else tx
        tx_type_raw = meta.get("type") or ""
        tx_type = (
            "trade"      if "trade"    in tx_type_raw else
            "waiver"     if "waiver"   in tx_type_raw else
            "free_agent" if "free_ag"  in tx_type_raw else
            tx_type_raw
        )
        ts_ms = _safe_int(meta.get("timestamp")) or 0
        if ts_ms and ts_ms < 1e12:
            ts_ms *= 1000  # Yahoo returns seconds, not ms

        players_block = (tx[1] if len(tx) > 1 else {}) if isinstance(tx, list) else {}
        adds:  Dict[str, int] = {}
        drops: Dict[str, int] = {}

        if isinstance(players_block, dict):
            p_data = players_block.get("players") or {}
            p_count = _safe_int(p_data.get("count") or p_data.get("0", {}).get("count")) or 0
            for pi in range(p_count):
                pe     = p_data.get(str(pi)) or {}
                player = pe.get("player") or []
                p_meta = player[0] if isinstance(player, list) and player else {}
                if isinstance(p_meta, list):
                    p_meta = p_meta[0] if p_meta else {}

                name  = (p_meta.get("name") or {}).get("full") or ""
                pos   = p_meta.get("display_position") or ""
                team  = (p_meta.get("editorial_team_abbr") or "").upper()
                canon = _resolve_player(name, pos, team)
                if not canon:
                    continue

                tx_data = (player[1] if len(player) > 1 else {}) if isinstance(player, list) else {}
                tx_type_p = (tx_data.get("transaction_data") or [{}])[0]
                if isinstance(tx_type_p, dict):
                    dest_type  = tx_type_p.get("destination_type") or ""
                    src_type   = tx_type_p.get("source_type") or ""
                    dest_team  = _safe_int(tx_type_p.get("destination_team_key", "").split(".")[-1])
                    src_team   = _safe_int(tx_type_p.get("source_team_key", "").split(".")[-1])

                    if dest_type == "team" and dest_team:
                        adds[canon] = dest_team
                    if src_type == "team" and src_team:
                        drops[canon] = src_team

        out.append({
            "type":           tx_type,
            "adds":           adds or None,
            "drops":          drops or None,
            "roster_ids":     list(set(list(adds.values()) + list(drops.values()))),
            "draft_picks":    [],
            "status":         "complete",
            "created":        ts_ms,
            "status_updated": ts_ms,
            "leg":            week,
            "transaction_id": meta.get("transaction_key") or str(i),
            "consenter_ids":  [],
            "metadata":       {},
        })
    return out


def _yahoo_draft_status_label(raw: Any) -> str:
    """Map Yahoo ``draft_status`` onto Sleeper-style pre_draft|drafting|complete."""
    text = str(raw or "").strip().lower().replace("-", "").replace("_", "")
    if text in ("postdraft", "complete", "finished"):
        return "complete"
    if text in ("draft", "drafting", "middraft", "livedraft", "inprogress"):
        return "drafting"
    if text in ("predraft", "pre", ""):
        return "pre_draft"
    return "pre_draft"


def get_drafts(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    """Return the league's draft record with a live-aware status when possible.

    Yahoo does not expose a Sleeper-like draft list; one synthetic draft per
    league/season is enough for Draft Room Connect. Status prefers league
    ``draft_status`` (predraft / draft / postdraft). Missing meta stays
    pre-draft so an undrafted keeper league is not marked complete.
    """
    from datetime import datetime as _dt
    start_ts_ms = int(_dt(int(season), 8, 1).timestamp() * 1000)
    # Missing meta is not a finished draft (same class as Fleaflicker omitting
    # NOT_YET_DRAFTED). Keepers/history still see a record; status stays pending.
    status = "pre_draft"
    teams = 0
    try:
        key = _league_key_for_season(league_id, season, access_token)
        raw = _yahoo_get(access_token, f"league/{key}")
        meta = _extract_league_meta(raw) or {}
        status = _yahoo_draft_status_label(meta.get("draft_status"))
        teams = _safe_int(meta.get("num_teams")) or 0
        # Settings sometimes carry the scheduled draft time (epoch seconds).
        settings = meta.get("settings") if isinstance(meta.get("settings"), dict) else {}
        draft_time = _safe_int((settings or {}).get("draft_time") or meta.get("draft_time"))
        if draft_time and draft_time > 10_000_000_000:  # already ms
            start_ts_ms = int(draft_time)
        elif draft_time and draft_time > 1_000_000_000:
            start_ts_ms = int(draft_time) * 1000
    except Exception as exc:
        logger.info("[yahoo] get_drafts meta skipped error_type=%s", type(exc).__name__)
    return [{
        "draft_id":   f"yahoo_{league_id}_{season}",
        "league_id":  str(league_id),
        "season":     int(season),
        "season_type": "regular",
        "start_time": start_ts_ms,
        "status":     status,
        "type":       "snake",
        "teams":      teams or None,
    }]


def _extract_draft_results_block(raw: Dict) -> Optional[Dict]:
    lg = (raw.get("fantasy_content", {}) or {}).get("league") or []
    for item in lg:
        if isinstance(item, dict) and "draft_results" in item:
            block = item["draft_results"]
            return block if isinstance(block, dict) else None
    return None


def get_draft_pick_rows(
    season: int, league_id: str, access_token: str
) -> List[Dict[str, Any]]:
    """Raw Yahoo draft_result rows for live sync (overall pick + yahoo player id).

    Unlike ``get_draft_results`` (round-only map for keepers), this keeps every
    pick with a player so Draft Room can paint the board mid-draft. Yahoo's
    draftresults resource updates during the live draft.
    """
    out: List[Dict[str, Any]] = []
    try:
        key = _league_key_for_season(league_id, season, access_token)
        raw = _yahoo_get(access_token, f"league/{key}/draftresults")
        block = _extract_draft_results_block(raw) or {}
        count = _safe_int(block.get("count")) or 0
        for i in range(count):
            entry = block.get(str(i)) or {}
            dr = entry.get("draft_result") or {}
            if not isinstance(dr, dict):
                continue
            overall = _safe_int(dr.get("pick"))
            rnd = _safe_int(dr.get("round"))
            pkey = str(dr.get("player_key") or "")
            tkey = str(dr.get("team_key") or "")
            if not overall or not pkey:
                continue
            yid = pkey.rsplit(".", 1)[-1]
            tid = tkey.rsplit(".", 1)[-1] if tkey else ""
            cost = dr.get("cost")
            out.append({
                "pick": overall,
                "round": rnd,
                "player_id": str(yid),
                "team_id": str(tid) if tid else None,
                "cost": cost,
                "player_key": pkey,
                "team_key": tkey or None,
            })
    except Exception as exc:
        logger.warning("[yahoo] get_draft_pick_rows failed: %s", exc)
        return out
    out.sort(key=lambda r: int(r.get("pick") or 0))
    return out


def get_draft_results(season: int, league_id: str, access_token: str) -> Dict[str, int]:
    """canonical Sleeper id -> the round the player was drafted, from Yahoo's
    league draftresults resource.

    Each draft_result carries ``pick``, ``round``, and a ``player_key`` of the
    form ``nfl.p.<player_id>`` where ``player_id`` is the Yahoo id (== Sleeper's
    ``yahoo_id``), so map it through the existing crosswalk. Empty on any failure
    (no draft yet, network, mapping) so callers fall back gracefully."""
    out: Dict[str, int] = {}
    try:
        xwalk = _yahoo_id_to_canonical()
        for row in get_draft_pick_rows(season, league_id, access_token):
            rnd = _safe_int(row.get("round"))
            yid = row.get("player_id")
            if not rnd or not yid:
                continue
            canon = xwalk.get(str(yid))
            if canon:
                out[str(canon)] = int(rnd)
    except Exception as exc:
        logger.warning("[yahoo] get_draft_results failed: %s", exc)
        return out
    return out


def _draft_analysis_block(player_entry: Any) -> Optional[Dict[str, Any]]:
    """Pull the draft_analysis dict out of a Yahoo players-collection entry.

    With the ``/draft_analysis`` sub-resource the player is
    ``[[meta...], {"draft_analysis": {...}}]`` (sometimes a further list), so scan
    every part for the draft_analysis key rather than assuming a fixed index."""
    parts = player_entry if isinstance(player_entry, list) else [player_entry]
    for part in parts:
        if isinstance(part, dict) and "draft_analysis" in part:
            da = part["draft_analysis"]
            if isinstance(da, list):  # occasionally wrapped as a positional list
                merged: Dict[str, Any] = {}
                for d in da:
                    if isinstance(d, dict):
                        merged.update(d)
                return merged
            if isinstance(da, dict):
                return da
    return None


def get_draft_analysis_adp(
    season: int, league_id: str, access_token: str, max_players: int = 300
) -> Dict[str, float]:
    """canonical Sleeper id -> average draft pick, from Yahoo's draft_analysis.

    Yahoo's ADP is league-format-aware (it reflects how this league's scoring
    drafts), redraft-only, and paginates 25 players at a time, so page through
    the players collection until a short page or ``max_players`` is reached.
    Returns overall ADP keyed by canonical id; players we can't map are skipped.
    Empty on any failure so the resolver falls back to another source."""
    out: Dict[str, float] = {}
    league_key = _league_key(league_id)
    start = 0
    page = 25
    try:
        while start < max_players:
            raw = _yahoo_get(
                access_token,
                f"league/{league_key}/players;start={start};count={page}/draft_analysis",
            )
            lg = (raw.get("fantasy_content", {}) or {}).get("league") or []
            players_block = None
            for item in lg:
                if isinstance(item, dict) and "players" in item:
                    players_block = item["players"]
                    break
            if not players_block:
                break
            count = _safe_int(players_block.get("count")) or 0
            if count <= 0:
                break
            for i in range(count):
                entry = (players_block.get(str(i)) or {}).get("player")
                if not entry:
                    continue
                flat, _ = _flatten_yahoo_player(entry)
                da = _draft_analysis_block(entry)
                if not da:
                    continue
                ap = _safe_float(da.get("average_pick"))
                if not ap or ap <= 0:
                    continue
                canon = _resolve_player(
                    flat.get("name", {}).get("full") if isinstance(flat.get("name"), dict) else flat.get("name"),
                    (flat.get("display_position") or flat.get("primary_position") or ""),
                    (flat.get("editorial_team_abbr") or ""),
                    yahoo_id=flat.get("player_id"),
                )
                if canon:
                    out[str(canon)] = float(ap)
            if count < page:
                break
            start += page
    except Exception as exc:
        logger.warning("[yahoo] get_draft_analysis_adp failed at start=%s: %s", start, exc)
        return out
    return out


def get_bracket_like(league_id: str, season: int, kind: str, access_token: str) -> List[Dict[str, Any]]:
    """Yahoo playoff bracket - returns an empty list; bracket rendering is best-effort."""
    return []


def get_league_globals(season: int, league_id: str, access_token: str) -> Dict[str, Any]:
    """
    Return league scoring settings and roster positions in Sleeper-compatible format.
    Called by platform_api.sync_league_globals().
    """
    try:
        raw  = _yahoo_get(access_token, f"league/{_league_key(league_id)}/settings")
        fc   = raw.get("fantasy_content", {})
        lg   = fc.get("league") or []
        meta = lg[0] if isinstance(lg, list) and lg else {}
        # Yahoo nests settings under league[1].settings[0] (same shape yahoo_fantasy_api
        # and our teams/scoreboard extractors use). league[0].settings is usually absent.
        settings = _yahoo_settings_dict(lg)
    except Exception as exc:
        logger.warning("[yahoo] get_league_globals failed: %s", exc)
        return {}

    scoring_settings = _yahoo_scoring_settings(meta, settings)
    from utils.league_scoring import normalize_league_scoring
    scoring_settings = normalize_league_scoring(
        "yahoo", scoring_settings, league_id=league_id, season=season)

    roster_positions = _yahoo_roster_positions(settings)
    if not roster_positions:
        logger.warning("[yahoo-roster] empty roster_positions platform=yahoo "
                       "league_id=%s season=%s", league_id, season)

    num_teams = _safe_int(meta.get("num_teams")) or 0
    playoff_teams = (
        _safe_int(settings.get("num_playoff_teams"))
        or _safe_int(meta.get("num_playoff_teams"))
        or 4
    )
    playoff_week_start = (
        _safe_int(settings.get("playoff_start_week"))
        or _safe_int(meta.get("playoff_start_week"))
    )
    _lt = _yahoo_sleeper_league_type(meta, settings, num_teams)
    league_settings: Dict[str, Any] = {
        "playoff_teams": playoff_teams,
        "num_teams":     num_teams,
        "type":          _lt,
        "league_type":   "dynasty" if _lt == 2 else ("keeper" if _lt == 1 else "redraft"),
    }
    if playoff_week_start:
        league_settings["playoff_week_start"] = playoff_week_start
    _mk = (
        _safe_int(settings.get("max_keepers"))
        or _safe_int(meta.get("max_keepers"))
        or 0
    )
    if _mk > 0:
        league_settings["max_keepers"] = _mk

    return {
        "scoring_settings": scoring_settings,
        "roster_positions": roster_positions,
        "league_settings":  league_settings,
        "total_rosters":    num_teams,
    }


def _yahoo_sleeper_league_type(
    meta: Dict[str, Any], settings: Dict[str, Any], num_teams: int
) -> int:
    """Map Yahoo keeper limits onto Sleeper settings.type (0/1/2).

    Yahoo does not publish an explicit dynasty flag. Mirror Fleaflicker's
    max_keepers × team_count heuristic (ffscrapr): 0 → redraft, small → keeper,
    whole-roster retention → dynasty. Also treat a non-``none`` cant_cut_list as
    at least keeper when max_keepers is absent.
    """
    max_keepers = (
        _safe_int(settings.get("max_keepers"))
        or _safe_int(meta.get("max_keepers"))
        or 0
    )
    teams = max(1, int(num_teams or 1))
    if max_keepers > 0:
        if max_keepers * teams > 250:
            return 2
        return 1
    cant = str(
        settings.get("cant_cut_list") or meta.get("cant_cut_list") or "none"
    ).strip().lower()
    if cant not in ("", "none", "false", "0"):
        return 1
    return 0


def _yahoo_settings_dict(league_list: Any) -> Dict[str, Any]:
    """Unwrap Yahoo ``fantasy_content.league[1].settings`` into a plain dict."""
    if not isinstance(league_list, list) or len(league_list) < 2:
        # Rare: some payloads put a settings blob on league[0].
        meta0 = league_list[0] if isinstance(league_list, list) and league_list else {}
        maybe = meta0.get("settings") if isinstance(meta0, dict) else None
        return _unwrap_yahoo_list_or_dict(maybe)

    block = league_list[1] if isinstance(league_list[1], dict) else {}
    return _unwrap_yahoo_list_or_dict(block.get("settings"))


def _unwrap_yahoo_list_or_dict(node: Any) -> Dict[str, Any]:
    """Yahoo often wraps a single object as ``[obj]`` or ``{"0": obj, "count": 1}``."""
    if isinstance(node, list):
        first = node[0] if node else {}
        return first if isinstance(first, dict) else {}
    if isinstance(node, dict):
        if any(k in node for k in ("roster_positions", "stat_modifiers", "num_playoff_teams")):
            return node
        zero = node.get("0")
        if isinstance(zero, dict):
            return zero
        return node
    return {}


# Yahoo NFL points-league stat_id → Sleeper-style scoring keys.
# 78 = TE reception bonus (TE premium) when present in some Yahoo configs.
_YAHOO_STAT_KEYS: Dict[int, str] = {
    4: "pass_yd", 5: "pass_td", 6: "pass_int",
    9: "rush_yd", 10: "rush_td",
    11: "rec", 12: "rec_yd", 13: "rec_td",
    18: "fum_lost",
    57: "pass_2pt", 58: "rush_2pt", 59: "rec_2pt",
    78: "bonus_rec_te",
}


def _yahoo_is_threshold_bonus(stat: dict) -> bool:
    """Skip a duplicate yardage row that is a 300-yard extra, not 0.04 / yard."""
    if not isinstance(stat, dict):
        return False
    if not (stat.get("bonuses") or stat.get("bonus")):
        return False
    try:
        return abs(float(stat.get("value"))) >= 1.0
    except (TypeError, ValueError):
        return False


def _yahoo_scoring_settings(meta: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build scoring from Yahoo ``stat_modifiers``, not competition ``scoring_type``.

    ``scoring_type`` is head/point/roto (matchup format). Reception PPR lives in
    ``stat_modifiers.stats``. Falling back to format labels previously forced H2H
    PPR leagues to ``rec=0`` and points leagues to full PPR.
    """
    from utils.league_scoring import assign_scoring_rate

    scoring: Dict[str, Any] = {}
    modifiers = settings.get("stat_modifiers") or {}
    stats_node = modifiers.get("stats") if isinstance(modifiers, dict) else None
    rows = _yahoo_collection_rows(stats_node, "stat")
    found_rec = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Rows are either {"stat": {...}} wrappers or flat stat dicts.
        stat = row.get("stat") if isinstance(row.get("stat"), dict) else row
        if _yahoo_is_threshold_bonus(stat):
            continue
        try:
            stat_id = int(stat.get("stat_id"))
        except (TypeError, ValueError):
            continue
        key = _YAHOO_STAT_KEYS.get(stat_id)
        if not key:
            continue
        try:
            value = float(stat.get("value"))
        except (TypeError, ValueError):
            continue
        assign_scoring_rate(scoring, key, value)
        if key == "rec":
            found_rec = True
    for key, default in (
        ("rec", 0.0), ("pass_yd", 0.04), ("pass_td", 4.0), ("pass_int", -2.0),
        ("rush_yd", 0.1), ("rush_td", 6.0), ("rec_yd", 0.1), ("rec_td", 6.0),
        ("fum_lost", -2.0), ("2pt", 2.0),
    ):
        if key not in scoring:
            scoring[key] = default
    if not found_rec:
        # Last-resort heuristic only when modifiers are missing entirely.
        scoring_type = str(meta.get("scoring_type") or settings.get("scoring_type") or "").lower()
        if "half" in scoring_type:
            scoring["rec"] = 0.5
        elif "ppr" in scoring_type:
            scoring["rec"] = 1.0
        logger.warning("[yahoo-scoring] reception modifier missing; "
                       "rec=%s scoring_type=%s", scoring["rec"], scoring_type or "unknown")
    return scoring


def _yahoo_collection_rows(node: Any, item_key: str) -> List[Any]:
    """Normalize Yahoo collection shapes into a list of row dicts."""
    if node is None:
        return []
    if isinstance(node, list):
        return node
    if not isinstance(node, dict):
        return []
    if item_key in node:
        inner = node.get(item_key)
        if isinstance(inner, list):
            return inner
        if isinstance(inner, dict):
            return [inner]
    rows: List[Any] = []
    count = _safe_int(node.get("count")) or 0
    if count:
        for i in range(count):
            entry = node.get(str(i))
            if entry is not None:
                rows.append(entry)
        if rows:
            return rows
    # Fall back to numeric-string keys.
    for key, value in node.items():
        if str(key).isdigit() and value is not None:
            rows.append(value)
    return rows


_YAHOO_SLOT = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "W/R/T": "FLEX", "W/R": "RB_WR", "W/T": "WR_TE", "R/T": "RB_TE",
    "RB/WR/TE": "FLEX", "WR/TE": "WR_TE", "RB/WR": "RB_WR", "RB/TE": "RB_TE",
    "Q/W/R/T": "SUPER_FLEX", "OP": "SUPER_FLEX",
    "K": "K", "DEF": "DEF", "D": "DEF",
    "BN": "BN", "IR": "IR",
}


def _yahoo_roster_positions(settings: Dict[str, Any]) -> List[str]:
    """Expand Yahoo roster_positions into a Sleeper-style slot list."""
    roster_positions_raw = settings.get("roster_positions") or {}
    pos_count = _yahoo_collection_rows(roster_positions_raw, "roster_position")
    if isinstance(roster_positions_raw, list):
        pos_count = roster_positions_raw
    elif isinstance(roster_positions_raw, dict) and "roster_position" in roster_positions_raw:
        inner = roster_positions_raw.get("roster_position")
        if isinstance(inner, dict):
            pos_count = [inner]
        elif isinstance(inner, list):
            pos_count = inner

    roster_positions: List[str] = []
    for slot in pos_count:
        if not isinstance(slot, dict):
            continue
        # Sometimes wrapped as {"roster_position": {...}}
        if "roster_position" in slot and isinstance(slot["roster_position"], dict):
            slot = slot["roster_position"]
        abbr = slot.get("position") or ""
        count = _safe_int(slot.get("count")) or 0
        if count <= 0 or not abbr:
            continue
        raw_abbr = str(abbr).upper()
        from utils.lineup_slots import canonicalize_slot
        norm = _YAHOO_SLOT.get(raw_abbr) or canonicalize_slot(raw_abbr) or raw_abbr
        roster_positions.extend([norm] * count)
    return roster_positions
