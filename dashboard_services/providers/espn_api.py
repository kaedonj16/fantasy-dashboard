# dashboard_services/espn_api.py
from __future__ import annotations

import os
import logging
import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from espn_api.football import League
import requests

from utils.utils import load_players_index
from utils.coerce import safe_int as _safe_int


# ============================================================
# Errors
# ============================================================

class ESPNError(Exception):
    pass


class ESPNRequestValidationError(ESPNError):
    pass


class ESPNAccessDenied(ESPNError):
    pass


class ESPNInvalidLeague(ESPNError):
    pass


class ESPNRateLimited(ESPNError):
    pass


class ESPNUnavailable(ESPNError):
    pass


class ESPNMalformedResponse(ESPNError):
    pass


ESPN_FFL_LEAGUE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
    "seasons/{season}/segments/0/leagues/{league_id}"
)
ESPN_REQUEST_TIMEOUT = 12
logger = logging.getLogger(__name__)


class ESPNFantasyClient:
    """Small credential-scoped ESPN client used by connection endpoints.

    Credentials are kept on this short-lived instance and are never included in
    errors or responses.  The rest of the provider continues to consume the
    normalized helpers below.
    """

    def __init__(self, swid: Optional[str] = None, espn_s2: Optional[str] = None):
        if bool(swid) != bool(espn_s2):
            raise ESPNRequestValidationError("Both SWID and ESPN_S2 are required.")
        self._swid = _normalize_swid(swid or "") or None
        self._espn_s2 = _clean_secret(espn_s2 or "") or None

    @property
    def authenticated(self) -> bool:
        return bool(self._swid and self._espn_s2)

    def get_league_client(self, league_id: str, season: int) -> League:
        kwargs: Dict[str, Any] = {
            "league_id": int(league_id),
            "year": int(season),
        }
        if self.authenticated:
            kwargs.update(swid=self._swid, espn_s2=self._espn_s2)
        return League(**kwargs)

    def get_league(self, league_id: str, season: int) -> Dict[str, Any]:
        """Fetch basic league metadata directly from ESPN's v3 FFL API.

        The third-party ``espn-api`` package assumes successful JSON and can
        turn an empty ESPN response into an opaque ``NoneType.get`` exception.
        Connection validation uses the HTTP response directly so users receive
        an accurate access/not-found/unavailable message instead.
        """
        cookies = None
        if self.authenticated:
            cookies = {"SWID": self._swid, "espn_s2": self._espn_s2}
        request_id = uuid.uuid4().hex[:12]
        try:
            response = requests.get(
                ESPN_FFL_LEAGUE_URL.format(season=int(season), league_id=int(league_id)),
                params=(("view", "mSettings"), ("view", "mTeam")),
                cookies=cookies,
                timeout=ESPN_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.warning(
                "[espn-connect] ref=%s outcome=request_error error_type=%s authenticated=%s league_id=%s season=%s",
                request_id, type(exc).__name__, self.authenticated, league_id, season,
            )
            raise ESPNUnavailable("ESPN is temporarily unavailable.") from exc
        content_type = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
        content_length = len(response.content) if hasattr(response, "content") else None
        logger.info(
            "[espn-connect] ref=%s outcome=response status=%s content_type=%r content_length=%s authenticated=%s league_id=%s season=%s",
            request_id, response.status_code, content_type, content_length,
            self.authenticated, league_id, season,
        )
        if response.status_code in (401, 403):
            raise ESPNAccessDenied("ESPN denied access to this league.")
        if response.status_code == 404:
            raise ESPNInvalidLeague("ESPN could not find this league and season.")
        if response.status_code == 429:
            raise ESPNRateLimited("ESPN is rate limiting requests.")
        if response.status_code >= 500:
            raise ESPNUnavailable("ESPN is temporarily unavailable.")
        if not response.ok:
            raise ESPNMalformedResponse("ESPN returned an unexpected response.")
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            logger.warning(
                "[espn-connect] ref=%s outcome=json_decode_failed status=%s content_type=%r content_length=%s authenticated=%s league_id=%s season=%s",
                request_id, response.status_code, content_type, content_length,
                self.authenticated, league_id, season,
            )
            error = ESPNMalformedResponse("ESPN returned an invalid response.")
            error.debug_reference = request_id
            raise error from exc
        if not isinstance(payload, dict):
            logger.warning(
                "[espn-connect] ref=%s outcome=invalid_payload payload_type=%s authenticated=%s league_id=%s season=%s",
                request_id, type(payload).__name__, self.authenticated, league_id, season,
            )
            error = ESPNMalformedResponse("ESPN returned an empty response.")
            error.debug_reference = request_id
            raise error
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            logger.warning(
                "[espn-connect] ref=%s outcome=missing_settings top_level_keys=%s authenticated=%s league_id=%s season=%s",
                request_id, sorted(str(key) for key in payload)[:30],
                self.authenticated, league_id, season,
            )
            error = ESPNMalformedResponse("ESPN returned incomplete league data.")
            error.debug_reference = request_id
            raise error
        teams = []
        for t in (payload.get("teams") or []):
            if not isinstance(t, dict):
                continue
            tid = t.get("id")
            if tid is None:
                continue
            name = (t.get("name")
                    or " ".join(part for part in (t.get("location"), t.get("nickname")) if part).strip()
                    or f"Team {tid}")
            teams.append({"id": str(tid), "name": str(name).strip()})
        return {
            "league_id": str(payload.get("id") or league_id),
            "season": int(payload.get("seasonId") or season),
            "name": settings.get("name") or f"ESPN League {league_id}",
            "teams": teams,
        }


# ============================================================
# Env + helpers
# ============================================================

def _clean_secret(raw: str) -> str:
    """Strip whitespace and one layer of matching surrounding quotes.

    Pasting a cookie into a hosting dashboard (Render, etc.) often wraps it in
    quotes (ESPN_S2="AEB...") or leaves a trailing newline; those characters ride
    along into os.getenv and silently break ESPN auth. Normalize them away.
    """
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _normalize_swid(swid: str) -> str:
    swid = _clean_secret(swid)
    if swid and not (swid.startswith("{") and swid.endswith("}")):
        swid = "{" + swid.strip("{}") + "}"
    return swid


def _espn_creds() -> Tuple[Optional[str], Optional[str]]:
    """(espn_s2, swid) from the environment, cleaned. Either may be None."""
    espn_s2 = _clean_secret(os.getenv("ESPN_S2", "")) or None
    swid = _normalize_swid(os.getenv("ESPN_SWID", "")) or None
    return espn_s2, swid


def espn_diagnostics() -> Dict[str, Any]:
    """Non-secret view of the configured ESPN credentials, for debugging a
    'my private league won't load' report. Never returns the actual values."""
    espn_s2, swid = _espn_creds()
    return {
        "espn_s2_present": bool(espn_s2),
        "espn_s2_len": len(espn_s2) if espn_s2 else 0,
        "espn_swid_present": bool(swid),
        "espn_swid_braced": bool(swid and swid.startswith("{") and swid.endswith("}")),
    }


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def _split_points(val: Any) -> tuple[int, int]:
    f = safe_float(val, 0.0)
    whole = int(f)
    dec = int(round((f - whole) * 100))
    return whole, dec


def _streak_from_outcomes(outcomes: Any) -> str:
    if not isinstance(outcomes, list):
        return ""

    cleaned = [x for x in (str(o).upper() for o in outcomes) if x in ("W", "L", "T")]
    if not cleaned:
        return ""

    last = cleaned[-1]
    n = 0
    for x in reversed(cleaned):
        if x != last:
            break
        n += 1
    return f"{n}{last}"


# ============================================================
# GLOBAL CACHES (CRITICAL)
# ============================================================

# Public League objects previously lived in @lru_cache for the whole process.
# That froze empty pre-/mid-draft rosters until redeploy — Teams stayed blank
# after ESPN finished the draft. Short TTL + explicit clear on refresh.
_PUBLIC_LEAGUE_TTL = 120  # seconds
_public_league_cache: Dict[Tuple[int, str], Tuple[float, Any]] = {}
_public_league_lock = threading.Lock()


def _public_league_cached(season: int, league_id: str) -> League:
    key = (int(season), str(league_id))
    now = time.time()
    with _public_league_lock:
        hit = _public_league_cache.get(key)
        if hit and (now - hit[0]) < _PUBLIC_LEAGUE_TTL:
            return hit[1]
    league = League(league_id=int(league_id), year=int(season))
    with _public_league_lock:
        _public_league_cache[key] = (time.time(), league)
        # Bound memory if many leagues are touched on one worker.
        if len(_public_league_cache) > 32:
            oldest = min(_public_league_cache.items(), key=lambda kv: kv[1][0])[0]
            _public_league_cache.pop(oldest, None)
    return league


def _is_espn_access_denied(exc: Exception) -> bool:
    """Recognize access denial, including espn-api 0.45's anonymous bug.

    In espn-api 0.45.1, a 401/403 anonymous request enters the access-denied
    formatter with ``self.cookies is None``. Its attempt to call
    ``self.cookies.get(...)`` raises AttributeError before ESPNAccessDenied can
    be constructed. This exact compatibility case should trigger our
    credential retry; unrelated AttributeErrors must still propagate.
    """
    if type(exc).__name__ == "ESPNAccessDenied":
        return True
    return (
        isinstance(exc, AttributeError)
        and str(exc) == "'NoneType' object has no attribute 'get'"
    )


def _league_cached(season: int, league_id: str) -> League:
    """Load a league anonymously when possible, then fall back to credentials.

    ESPN cookies identify one account.  Sending them on every request can make
    ESPN scope the request to that account, which prevents otherwise-public
    leagues belonging to other users from loading.  Anonymous-first therefore
    supports every public league; the configured cookies remain a fallback for
    private leagues that the configured account may access.

    ESPN does not offer OAuth for this API.  Consequently, an arbitrary private
    league still cannot be read unless its owner makes it public or supplies
    credentials for an account that belongs to it.
    """
    access_denied: Optional[Exception] = None
    try:
        return _public_league_cached(season, league_id)
    except Exception as exc:
        # Do not hide invalid IDs, bad seasons, or network/library errors behind
        # a second request. Only an access denial can be fixed by authentication.
        if not _is_espn_access_denied(exc):
            raise
        # Do not retain the third-party exception: some espn-api versions put
        # cookie values in ESPNAccessDenied messages. Keep all later logs safe.
        access_denied = ESPNAccessDenied("ESPN denied anonymous access to this league.")
        if isinstance(exc, AttributeError):
            logger.info(
                "[espn] treating espn-api anonymous None-cookies AttributeError as access denied league_id=%s season=%s",
                league_id, season,
            )

    espn_s2, swid = _espn_creds()
    try:
        from flask import has_request_context, session
        if has_request_context() and session.get("account_id"):
            from dashboard_services.accounts import (
                get_espn_league_credentials, get_any_espn_account_credentials,
            )
            stored = get_espn_league_credentials(session["account_id"], league_id, season) or {}
            # Prefer credentials attached to this saved league. If this league was
            # linked without cookies (or season-bumped away from the row that has
            # them), reuse any ESPN login on the same Google account — never env /
            # another account's cookies.
            espn_s2 = stored.get("espn_s2")
            swid = stored.get("swid")
            if not (espn_s2 and swid):
                any_creds = get_any_espn_account_credentials(session["account_id"]) or {}
                espn_s2 = any_creds.get("espn_s2")
                swid = any_creds.get("swid")
        elif has_request_context() and session.get("pending_provider_connection_token"):
            from dashboard_services.accounts import peek_private_espn_connection
            staged = peek_private_espn_connection(
                session["pending_provider_connection_token"], league_id, season,
            ) or {}
            espn_s2 = staged.get("espn_s2")
            swid = staged.get("swid")
    except Exception:
        # Database/configuration trouble must not expose credentials and the
        # original ESPN access-denied result remains the useful outcome.
        pass
    if not (espn_s2 and swid):
        raise access_denied
    try:
        return League(
            league_id=int(league_id),
            year=int(season),
            espn_s2=espn_s2,
            swid=swid,
        )
    except Exception as exc:
        if _is_espn_access_denied(exc):
            try:
                from flask import has_request_context, session
                if has_request_context() and session.get("account_id"):
                    from dashboard_services.accounts import mark_espn_connection_status
                    mark_espn_connection_status(
                        int(session["account_id"]), str(league_id), int(season),
                        "reauth_required", "espn_auth_rejected",
                    )
            except Exception:
                pass
            raise ESPNAccessDenied("ESPN denied authenticated access to this league.") from None
        raise


_league_cached.cache_clear = lambda: clear_espn_league_caches()  # type: ignore[attr-defined]


def _league(season: int, league_id: str) -> League:
    return _league_cached(season, league_id)


def connect_league(
    season: int,
    league_id: str,
    *,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and normalize a public or explicitly authenticated league."""
    return ESPNFantasyClient(swid=swid, espn_s2=espn_s2).get_league(league_id, season)


@lru_cache(maxsize=1)
def _players_index_cached() -> Dict[str, Dict[str, Any]]:
    return load_players_index()


@lru_cache(maxsize=1)
def _espn_to_canon_cached() -> Dict[str, str]:
    return build_espn_to_canonical(_players_index_cached())


# Box scores change during live games, so they must NOT be cached for the
# whole process lifetime (the old @lru_cache served stale scores until redeploy).
# Use a short TTL instead.
_BOX_SCORE_TTL = 90  # seconds
_box_score_cache: Dict[Tuple[int, str, int], Tuple[float, Any]] = {}
_box_score_lock = threading.Lock()


def _box_scores_cached(season: int, league_id: str, week: int):
    key = (int(season), str(league_id), int(week))
    now = time.time()
    with _box_score_lock:
        hit = _box_score_cache.get(key)
        if hit and (now - hit[0]) < _BOX_SCORE_TTL:
            return hit[1]
    # Fetch outside the lock so a slow ESPN call doesn't block other keys.
    scores = _league(season, league_id).box_scores(week)
    with _box_score_lock:
        _box_score_cache[key] = (time.time(), scores)
    return scores


@lru_cache(maxsize=16)
def _playoff_schedule_cached(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    data = lg.espn_request.league_get(params={"view": "mMatchup"})
    return data.get("schedule") or []


# ============================================================
# Public API
# ============================================================

def _normalize_league(lg: League, league_id: str, season: int) -> Dict[str, Any]:
    name = (
            getattr(getattr(lg, "settings", None), "name", None)
            or getattr(lg, "name", None)
            or "ESPN League"
    )
    return {
        "league_id": str(league_id),
        "season": int(season),
        "name": name,
    }


def get_league(season: int, league_id: str) -> Dict[str, Any]:
    return _normalize_league(_league(season, league_id), league_id, season)


def get_users(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    _, swid = _espn_creds()
    swid = swid or ""

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for t in lg.teams or []:
        team_name = getattr(t, "team_name", None) or getattr(t, "name", None)
        logo = getattr(t, "logo_url", None) or getattr(t, "logo", None)
        owners = getattr(t, "owners", None) or []

        for o in owners:
            oid = str(o.get("id") or "").strip()
            if not oid or oid in seen:
                continue
            seen.add(oid)

            display = (
                    o.get("displayName")
                    or o.get("firstName")
                    or f"User {oid}"
            )

            out.append({
                "avatar": logo,
                "display_name": display,
                "is_bot": False,
                "is_owner": (swid == oid) if swid else None,
                "league_id": str(league_id),
                "metadata": {"team_name": team_name} if team_name else {},
                "settings": None,
                "user_id": oid,
            })

    return out


def get_teams(season: int, league_id: str) -> List[Dict[str, Any]]:
    """Lightweight team list for the "which team is yours?" link picker.

    ESPN membership isn't tied to our identity, so linking an ESPN league means
    the user selects their team by hand; this returns the choices as
    ``{team_id, name}``. ``is_mine`` is set when the signed-in account's stored
    SWID owns the team (falling back to server-level cookies).
    """
    lg = _league(season, league_id)
    swid = ""
    try:
        from flask import has_request_context, session
        if has_request_context() and session.get("account_id"):
            from dashboard_services.accounts import get_espn_league_credentials
            stored = get_espn_league_credentials(session["account_id"], league_id, season) or {}
            swid = str(stored.get("swid") or "").strip()
    except Exception:
        swid = ""
    if not swid:
        _, env_swid = _espn_creds()
        swid = (env_swid or "").strip()
    from utils.redzone_user import owner_id_variants
    swid_ids = owner_id_variants(swid)
    out: List[Dict[str, Any]] = []
    for t in lg.teams or []:
        tid = _safe_int(getattr(t, "team_id", None) or getattr(t, "id", None))
        if tid is None:
            continue
        name = getattr(t, "team_name", None) or getattr(t, "name", None) or f"Team {tid}"
        owners = getattr(t, "owners", None) or []
        owner_ids = {str(o.get("id") or "").strip() for o in owners if isinstance(o, dict)}
        out.append({
            "team_id": str(tid),
            "name": str(name),
            "is_mine": bool(swid_ids and owner_ids & swid_ids),
        })
    return out


def get_rosters(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    espn_to_canon = _espn_to_canon_cached()

    rosters: List[Dict[str, Any]] = []

    for t in lg.teams or []:
        roster_id = _safe_int(getattr(t, "team_id", None))
        owners = getattr(t, "owners", None) or []
        owner_id = None
        if owners and isinstance(owners[0], dict):
            owner_id = str(owners[0].get("id") or "").strip() or None
        team_name = getattr(t, "team_name", None) or getattr(t, "name", None)

        wins = _safe_int(getattr(t, "wins", None))
        losses = _safe_int(getattr(t, "losses", None))
        ties = _safe_int(getattr(t, "ties", None))
        streak = _streak_from_outcomes(getattr(t, "outcomes", None))

        fpts, fpts_dec = _split_points(getattr(t, "points_for", None))
        fpa, fpa_dec = _split_points(getattr(t, "points_against", None))

        players: List[str] = []
        starters: List[str] = []
        reserve: List[str] = []

        for p in getattr(t, "roster", None) or []:
            pid = getattr(p, "playerId", None)
            if pid is None:
                continue
            # Same D/ST + skill-player path as matchups — negative ESPN defense
            # ids are never in the espnID crosswalk and must be team abbreviations.
            cp = resolve_espn_player_id(pid, espn_to_canon, player=p)
            if not cp:
                continue

            players.append(cp)
            slot = (
                    getattr(p, "slot_position", None)
                    or getattr(p, "slotPosition", None)
                    or getattr(p, "lineupSlot", None)
            )

            slot = (str(slot).strip().upper() if slot is not None else "")

            if slot in ("IR", "RES"):
                reserve.append(cp)
            elif slot in ("", "BE", "BENCH", "INACTIVE"):
                # IMPORTANT: if slot is missing/unknown, treat as bench
                pass
            else:
                starters.append(cp)

        meta: Dict[str, Any] = {"record": f"{wins}-{losses}", "streak": streak}
        if team_name:
            meta["team_name"] = team_name

        rosters.append({
            "co_owners": None,
            "keepers": None,
            "league_id": str(league_id),
            "metadata": meta,
            "owner_id": owner_id,
            "player_map": None,
            "players": players,
            "reserve": reserve,
            "roster_id": roster_id,
            "settings": {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "fpts": fpts,
                "fpts_decimal": fpts_dec,
                "fpts_against": fpa,
                "fpts_against_decimal": fpa_dec,
                "ppts": 0,
                "ppts_decimal": 0,
                "total_moves": 0,
                "waiver_budget_used": 0,
                "waiver_position": 0,
            },
            "starters": starters,
            "taxi": None,
        })

    return rosters


# Put this near the top of the module
STARTER_SLOT_ORDER = {
    "QB": 0,
    "RB": 1,
    "WR": 2,
    "TE": 3,
    "RB/WR/TE": 4,
    "FLEX": 4,  # some leagues label it this way
    "OP": 4,  # offensive player flex in some formats
    "K": 5,
    "D/ST": 6,
    "DST": 6,
    "DEF": 6,
}

# ESPN's common proTeamId -> abbrev mapping (covers modern NFL)
ESPN_PROTEAMID_TO_ABBR = {
    0: "FA",
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}


def _norm_slot(slot: Any) -> str:
    s = (str(slot or "").strip().upper())
    if s in ("D/ST", "D-ST", "DEFENSE"):
        return "D/ST"
    if s in ("DST",):
        return "DST"
    if s in ("FLEX", "RB/WR/TE", "RBWRTE"):
        return "RB/WR/TE"
    return s


def _slot_rank(slot: Any) -> int:
    return STARTER_SLOT_ORDER.get(_norm_slot(slot), 999)


def _dst_canonical_id(bp: Any, pid_raw: int) -> Optional[str]:
    """Map an ESPN D/ST player to a Sleeper-style team abbreviation (e.g. JAX).

    ESPN defenses use negative playerIds (``-16000 - proTeamId``). Roster and
    box-score objects may expose either a numeric ``proTeamId`` or a string
    ``proTeam`` abbreviation; accept both, then fall back to the -160xx math.
    """
    pro_raw = None
    if bp is not None:
        pro_raw = (
            getattr(bp, "proTeamId", None)
            or getattr(bp, "pro_team_id", None)
            or getattr(bp, "proTeam", None)
        )

    # String abbreviations from espn-api (e.g. "JAX", "WSH") are usable as-is.
    if isinstance(pro_raw, str):
        abbr = pro_raw.strip().upper()
        if abbr == "WSH":
            abbr = "WAS"
        if abbr and abbr != "FA" and abbr in ESPN_PROTEAMID_TO_ABBR.values():
            return abbr
        pro_raw = None

    pro_id = pro_raw
    # Fallback: derive from ESPN -1600x convention
    if pro_id is None and isinstance(pid_raw, int) and pid_raw < 0:
        pro_id = abs(pid_raw) - 16000  # -16009 -> 9

    try:
        pro_id = int(pro_id) if pro_id is not None else None
    except Exception:
        pro_id = None

    abbr = ESPN_PROTEAMID_TO_ABBR.get(pro_id) if pro_id is not None else None
    if not abbr or abbr == "FA":
        return None
    return abbr


def resolve_espn_player_id(
        espn_pid: Any,
        espn_to_canon: Dict[str, str],
        player: Any = None,
) -> Optional[str]:
    """Canonical sleeper/team id for an ESPN playerId (skill players + D/ST)."""
    if espn_pid is None:
        return None
    try:
        pid_int = int(espn_pid)
    except Exception:
        pid_int = None

    if pid_int is not None and pid_int < 0:
        return _dst_canonical_id(player, pid_int)
    return canon_pid(str(espn_pid), espn_to_canon)


def get_matchups(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    espn_to_canon = _espn_to_canon_cached()
    box_scores = _box_scores_cached(season, league_id, week)

    out: List[Dict[str, Any]] = []
    matchup_id = 0

    for bs in box_scores:
        home = getattr(bs, "home_team", None)
        away = getattr(bs, "away_team", None)
        if not home or not away:
            continue  # bye week

        matchup_id += 1

        def build_side(team, lineup, score):
            players: List[str] = []
            players_points: Dict[str, float] = {}

            starter_entries: List[Tuple[int, int, str, float]] = []
            # (rank, original_index, pid, pts) so ties preserve lineup order

            for i, bp in enumerate(lineup or []):
                pid_raw = getattr(bp, "playerId", None) or getattr(bp, "player_id", None)
                if pid_raw is None:
                    continue

                cp = resolve_espn_player_id(pid_raw, espn_to_canon, player=bp)
                if not cp:
                    continue

                pts = safe_float(getattr(bp, "points", None))
                slot = getattr(bp, "slot_position", None) or getattr(bp, "slotPosition", None) or getattr(bp,
                                                                                                          "lineupSlot",
                                                                                                          None)

                players.append(cp)
                players_points[cp] = pts

                # starter?
                if slot not in ("BE", "Bench", "IR", "RES", "Inactive"):
                    starter_entries.append((_slot_rank(slot), i, cp, pts))

            starter_entries.sort(key=lambda t: (t[0], t[1]))
            starters = [cp for _, _, cp, _ in starter_entries]
            starters_points = [pts for _, _, _, pts in starter_entries]

            return {
                "points": safe_float(score),
                "players": players,
                "roster_id": _safe_int(getattr(team, "team_id", None) or getattr(team, "id", None)),
                "custom_points": None,
                "matchup_id": matchup_id,
                "starters": starters,
                "starters_points": starters_points,
                "players_points": players_points,
            }

        out.append(build_side(home, getattr(bs, "home_lineup", None), getattr(bs, "home_score", None)))
        out.append(build_side(away, getattr(bs, "away_lineup", None), getattr(bs, "away_score", None)))

    return out


def espn_get_bracket_like(
        league_id: str,
        season: int,
        kind: str,
) -> List[Dict[str, Any]]:
    schedule = _playoff_schedule_cached(season, league_id)
    kind = kind.lower()

    def keep(g):
        p = (g.get("playoffTierType") or "").upper()
        return ("WINNERS" in p) if kind == "winners" else ("LOSERS" in p or "CONSOLATION" in p)

    games = [g for g in schedule if g.get("home") and g.get("away") and keep(g)]
    if not games:
        return []

    rounds = sorted({_safe_int(g.get("matchupPeriodId")) for g in games})
    rmap = {mp: i + 1 for i, mp in enumerate(rounds)}

    out = []
    for g in games:
        h, a = g["home"], g["away"]
        out.append({
            "r": rmap.get(_safe_int(g.get("matchupPeriodId")), 1),
            "m": _safe_int(g.get("id")),
            "t1": _safe_int(h.get("teamId")),
            "t2": _safe_int(a.get("teamId")),
            "t1_from": None,
            "t2_from": None,
            "w": None,
            "l": None,
        })

    return sorted(out, key=lambda x: (x["r"], x["m"]))


@lru_cache(maxsize=32)
def _all_transactions_cached(season: int, league_id: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch all transactions for a season from ESPN's mTransactions2 view and
    return them keyed by scoring-period (week) number.

    Cached per season+league so the 18 per-week calls in build_week_activity
    only hit the ESPN API once.
    """
    try:
        lg = _league(season, league_id)
        data = lg.espn_request.league_get(params={"view": "mTransactions2"})
    except Exception as exc:
        print(f"[ESPN] _all_transactions_cached failed: {exc}")
        return {}

    raw_txs = data.get("transactions") or []
    espn_to_canon = _espn_to_canon_cached()

    _TYPE_MAP = {
        "WAIVER": "waiver",
        "WAIVER_RESOLUTION": "waiver",
        "FREE_AGENT": "free_agent",
        "TRADE_ACCEPT": "trade",
        "TRADE_ACCEPTED": "trade",
    }

    by_week: Dict[int, List[Dict[str, Any]]] = {}

    for tx in raw_txs:
        tx_type_raw = (tx.get("type") or "").upper()
        tx_type = _TYPE_MAP.get(tx_type_raw)
        if not tx_type:
            continue

        # Only fully-executed transactions
        exec_type = (tx.get("executionType") or "").upper()
        if exec_type not in ("EXECUTE", "EXECUTED", "PROCESS"):
            continue

        scoring_period = int(tx.get("scoringPeriodId") or 0)
        process_ms = tx.get("processDate") or tx.get("proposedDate") or 0

        adds: Dict[str, int] = {}
        drops: Dict[str, int] = {}
        roster_ids: set = set()

        for item in tx.get("items") or []:
            item_type = (item.get("type") or "").upper()
            player_id = item.get("playerId")
            from_team = item.get("fromTeamId")
            to_team = item.get("toTeamId")

            if not player_id or int(player_id) <= 0:
                continue

            cp = canon_pid(str(player_id), espn_to_canon)
            if not cp:
                continue

            if item_type in ("ADDED", "WAIVER_ADDED", "PICKED_UP"):
                if to_team:
                    adds[cp] = int(to_team)
                    roster_ids.add(int(to_team))
            elif item_type in ("DROPPED", "WAIVER_DROPPED"):
                if from_team:
                    drops[cp] = int(from_team)
                    roster_ids.add(int(from_team))
            elif item_type == "TRADED_TO":
                if to_team:
                    adds[cp] = int(to_team)
                    roster_ids.add(int(to_team))
            elif item_type == "TRADED_FROM":
                if from_team:
                    drops[cp] = int(from_team)
                    roster_ids.add(int(from_team))

        if not adds and not drops:
            continue

        entry = {
            "type": tx_type,
            "adds": adds or None,
            "drops": drops or None,
            "roster_ids": sorted(roster_ids),
            "draft_picks": [],
            "status": "complete",
            "created": int(process_ms),
            "status_updated": int(process_ms),
            "leg": scoring_period,
            "transaction_id": str(tx.get("id") or ""),
            "consenter_ids": [],
            "metadata": {},
        }

        by_week.setdefault(scoring_period, []).append(entry)

    return by_week


def get_transactions(season: int, league_id: str, week: int) -> List[Dict[str, Any]]:
    """Return transactions for a specific week, fetching the full season once."""
    return _all_transactions_cached(season, league_id).get(week, [])


_draft_meta_cache: Dict[Tuple[int, str], Tuple[float, Tuple[Optional[int], Optional[bool]]]] = {}
_draft_meta_lock = threading.Lock()
_DRAFT_META_TTL = 300  # 5 minutes — the scheduled draft date barely changes.


def clear_espn_league_caches(league_id: Optional[str] = None, season: Optional[int] = None) -> None:
    """Drop cached ESPN League / draft-meta objects so the next fetch is live.

    Call from refresh-league / full page refresh after a draft completes so Teams
    pick up ESPN roster assignments instead of empty pre-draft shells.

    When ``league_id`` (and optionally ``season``) is provided, only that
    league's entries are evicted — switching leagues used to wipe every ESPN
    cache on the worker and force a cold rebuild of unrelated leagues.
    """
    with _public_league_lock:
        if league_id is None:
            _public_league_cache.clear()
        else:
            lid = str(league_id)
            drop = [
                key for key in _public_league_cache
                if str(key[1]) == lid and (season is None or int(key[0]) == int(season))
            ]
            for key in drop:
                _public_league_cache.pop(key, None)
    with _draft_meta_lock:
        if league_id is None:
            _draft_meta_cache.clear()
        else:
            lid = str(league_id)
            drop = [
                key for key in _draft_meta_cache
                if str(key[1]) == lid and (season is None or int(key[0]) == int(season))
            ]
            for key in drop:
                _draft_meta_cache.pop(key, None)
    # Box scores are week-scoped; drop matching league entries so live scores
    # still refresh without nuking every other league on the worker.
    with _box_score_lock:
        if league_id is None:
            _box_score_cache.clear()
        else:
            lid = str(league_id)
            drop = [
                key for key in _box_score_cache
                if str(key[1]) == lid and (season is None or int(key[0]) == int(season))
            ]
            for key in drop:
                _box_score_cache.pop(key, None)
    try:
        if league_id is None:
            _playoff_schedule_cached.cache_clear()
    except Exception:
        pass


def _espn_draft_meta(season: int, league_id: str) -> Tuple[Optional[int], Optional[bool]]:
    """Return (scheduled_draft_date_ms, drafted_bool) from ESPN, or (None, None).

    ESPN carries the scheduled draft time at settings.draftSettings.date (epoch
    ms) and whether it has happened at draftDetail.drafted. Reading these lets the
    dashboard show a real pre-draft countdown instead of assuming the draft is
    already done. Cached briefly so the 30s countdown poll doesn't hammer ESPN.
    """
    key = (int(season), str(league_id))
    now = time.time()
    with _draft_meta_lock:
        hit = _draft_meta_cache.get(key)
        if hit and (now - hit[0]) < _DRAFT_META_TTL:
            return hit[1]
    try:
        lg = _league(season, league_id)
        data = lg.espn_request.league_get(params={"view": "mSettings"})
    except Exception as e:
        print(f"[espn] draft meta fetch failed: {e}")
        return None, None
    date_ms: Optional[int] = None
    drafted: Optional[bool] = None
    if isinstance(data, dict):
        try:
            _d = ((data.get("settings") or {}).get("draftSettings") or {}).get("date")
            date_ms = int(_d) if _d else None
        except (TypeError, ValueError):
            date_ms = None
        _dd = data.get("draftDetail")
        if isinstance(_dd, dict) and "drafted" in _dd:
            drafted = bool(_dd.get("drafted"))
    with _draft_meta_lock:
        _draft_meta_cache[key] = (now, (date_ms, drafted))
    return date_ms, drafted


def get_drafts(season: int, league_id: str) -> List[Dict[str, Any]]:
    """Return a single draft record for the league.

    Uses ESPN's real scheduled draft date + drafted flag when available, so an
    upcoming ESPN draft counts down correctly and a completed one reads as done.
    Falls back to a conservative Aug-1 "complete" record when ESPN doesn't report
    a date, preserving has_draft_ended() behavior for historical seasons.
    """
    from datetime import datetime
    date_ms, drafted = _espn_draft_meta(season, league_id)
    if date_ms:
        if drafted is None:
            # No explicit flag — infer from whether the scheduled time has passed.
            drafted = date_ms <= int(datetime.now().timestamp() * 1000)
        return [{
            "draft_id": f"espn_{league_id}_{season}",
            "league_id": str(league_id),
            "season": int(season),
            "season_type": "regular",
            "start_time": date_ms,
            "status": "complete" if drafted else "pre_draft",
            "type": "snake",
        }]
    # Fallback: no date from ESPN — treat as a completed draft (Aug 1) so
    # has_draft_ended() and historical seasons behave as before.
    start_ts_ms = int(datetime(int(season), 8, 1).timestamp() * 1000)
    return [{
        "draft_id": f"espn_{league_id}_{season}",
        "league_id": str(league_id),
        "season": int(season),
        "season_type": "regular",
        "start_time": start_ts_ms,
        "status": "complete",
        "type": "snake",
    }]


def iter_draft_picks(season: int, league_id: str) -> List[Any]:
    """Raw ESPN draft picks (objects or dicts) for keeper-round detection.

    Prefers ``League.draft`` from espn_api; falls back to the ``mDraftDetail``
    view so a completed draft still yields player→round even when the helper
    list is empty.
    """
    try:
        lg = _league(int(season), str(league_id))
    except Exception as e:
        print(f"[espn] draft picks fetch failed: {e}")
        return []
    picks = list(getattr(lg, "draft", None) or [])
    if picks:
        return picks
    try:
        data = lg.espn_request.league_get(params={"view": "mDraftDetail"})
    except Exception as e:
        print(f"[espn] mDraftDetail fetch failed: {e}")
        return []
    if not isinstance(data, dict):
        return []
    return list(((data.get("draftDetail") or {}).get("picks") or []))


# ESPN slot name -> Sleeper roster position
_ESPN_SLOT_TO_SLEEPER: Dict[str, str] = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "FLEX": "FLEX", "RB/WR/TE": "FLEX", "RB/WR": "FLEX", "WR/TE": "FLEX",
    "OP": "SUPER_FLEX",
    "K": "K",
    "D/ST": "DEF", "DST": "DEF", "DEF": "DEF", "D-ST": "DEF",
    "BE": "BN", "BENCH": "BN",
    "IR": "IR",
}

# ESPN mSettings scoringItems statId -> the provider-agnostic keys consumed by
# score_stats, projections, Player Modal and Draft Room. ESPN's canonical
# reception scoring item is statId 53 ("Each reception"), not the league's
# coarse scoring_type label.
_ESPN_SCORING_STAT_KEYS: Dict[int, str] = {
    0: "pass_att", 1: "pass_cmp", 2: "pass_inc",
    3: "pass_yd", 4: "pass_td", 19: "pass_2pt", 20: "pass_int",
    23: "rush_att", 24: "rush_yd", 25: "rush_td", 26: "rush_2pt",
    42: "rec_yd", 43: "rec_td", 44: "rec_2pt", 53: "rec", 58: "rec_tgt",
    68: "fum", 72: "fum_lost",
    74: "fgm_50p", 77: "fgm_40_49", 80: "fgm_0_39", 83: "fgm",
    76: "fgmiss_50p", 79: "fgmiss_40_49", 82: "fgmiss_0_39", 85: "fgmiss",
    86: "xpm", 88: "xpmiss",
    89: "pts_allow_0", 90: "pts_allow_1_6", 91: "pts_allow_7_13",
    92: "pts_allow_14_17", 121: "pts_allow_18_21", 122: "pts_allow_22_27",
    123: "pts_allow_28_34", 124: "pts_allow_35_45", 125: "pts_allow_46p",
    95: "int", 96: "fum_rec", 97: "blk_kick", 98: "safe", 99: "sack",
    105: "def_st_td", 106: "fum_forced",
    211: "pass_fd", 212: "rush_fd", 213: "rec_fd",
}


def _espn_scoring_item_points(item: dict):
    """Read ESPN points without dropping an explicit zero override."""
    overrides = item.get("pointsOverrides") or {}
    if isinstance(overrides, dict) and "16" in overrides and overrides["16"] is not None:
        return overrides["16"]
    return item.get("points")


def normalize_espn_scoring_items(scoring_items: List[dict]) -> Dict[str, float]:
    """Normalize raw ESPN mSettings scoringItems into the shared scoring shape."""
    normalized: Dict[str, float] = {}
    for item in scoring_items or []:
        if not isinstance(item, dict):
            continue
        try:
            stat_id = int(item.get("statId"))
        except (TypeError, ValueError):
            continue
        key = _ESPN_SCORING_STAT_KEYS.get(stat_id)
        value = _espn_scoring_item_points(item)
        if key is None or value is None:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


# ESPN lineupSlotId → display/name used before _ESPN_SLOT_TO_SLEEPER mapping.
# Same ids as espn_draft._roster_positions_from_payload; kept here so league
# globals (Standings Proj%, playoff odds) don't depend on the draft module.
_ESPN_LINEUP_SLOT_IDS: Dict[int, str] = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "OP",
    16: "D/ST", 17: "K", 20: "BE", 21: "IR",
    23: "RB/WR/TE", 3: "RB/WR", 5: "WR/TE",
}


def _map_espn_slot_name(slot: Any) -> Optional[str]:
    """Map an ESPN slot label to the Sleeper-style name used by lineup math."""
    if slot is None:
        return None
    raw = str(slot).upper().strip()
    if not raw:
        return None
    return _ESPN_SLOT_TO_SLEEPER.get(raw, raw)


def expand_espn_lineup_slot_counts(slot_counts: Any) -> List[str]:
    """Expand ESPN ``lineupSlotCounts`` / name→count maps into a slot list.

    Accepts either ESPN numeric lineupSlotIds (``{"0": 1, "2": 2, ...}``) or
    already-labeled counts from ``settings.position_slot_counts`` (``{"QB": 1}``).
    Bench/IR are included so consumers that care about roster depth see them;
    starting-lineup helpers already skip BN/IR via ``_BENCH_SLOTS``.
    """
    if not isinstance(slot_counts, dict) or not slot_counts:
        return []
    out: List[str] = []
    for key, count in slot_counts.items():
        try:
            n = int(count)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        name: Optional[str] = None
        try:
            slot_id = int(key)
        except (TypeError, ValueError):
            slot_id = None
        if slot_id is not None and str(slot_id) == str(key).strip():
            name = _ESPN_LINEUP_SLOT_IDS.get(slot_id)
        if name is None:
            name = str(key)
        mapped = _map_espn_slot_name(name)
        if not mapped:
            continue
        out.extend([mapped] * n)
    return out


def _espn_roster_positions_from_settings(settings: Any, msettings_payload: Optional[dict] = None) -> List[str]:
    """Resolve ESPN roster slots for projections / standings / playoff odds.

    ``espn_api.football.settings.Settings`` exposes ``position_slot_counts``, not
    ``roster_slots``. Reading the missing attribute left ``roster_positions``
    empty after draft, so Value Rankings Proj% and preseason playoff sims
    treated every lineup as 0 projected points.
    """
    # 1) Authoritative: raw mSettings lineupSlotCounts (numeric ESPN slot ids).
    if isinstance(msettings_payload, dict):
        raw = (((msettings_payload.get("settings") or {}).get("rosterSettings") or {})
               .get("lineupSlotCounts"))
        expanded = expand_espn_lineup_slot_counts(raw)
        if expanded:
            return expanded

    # 2) Do NOT trust espn_api's position_slot_counts. The library zips
    # POSITION_MAP.values() (mixed int→name and name→int) against
    # lineupSlotCounts.values(), which invents slots like TQB and drops FLEX.
    # Prefer an explicit list shim (tests) or leave empty for the shared
    # default-lineup guard in simulate_playoff_odds.

    # 3) Legacy/test shim: a flat list on roster_slots.
    raw_slots = getattr(settings, "roster_slots", None) or []
    if isinstance(raw_slots, dict):
        return expand_espn_lineup_slot_counts(raw_slots)
    out: List[str] = []
    for slot in raw_slots:
        mapped = _map_espn_slot_name(slot)
        if mapped:
            out.append(mapped)
    return out


def _raw_espn_msettings(lg) -> dict:
    """Fetch the mSettings view from the already credential-scoped league."""
    try:
        payload = lg.espn_request.league_get(params={"view": "mSettings"})
    except Exception as exc:
        logger.warning("[espn-scoring] mSettings unavailable: %s", type(exc).__name__)
        return {}
    return payload if isinstance(payload, dict) else {}


def _raw_espn_scoring_items(lg) -> List[dict]:
    """Fetch mSettings scoringItems from the already credential-scoped league."""
    payload = _raw_espn_msettings(lg)
    items = (((payload.get("settings") or {}).get("scoringSettings") or {})
             .get("scoringItems") or [])
    return items if isinstance(items, list) else []


def get_league_globals(season: int, league_id: str) -> Dict[str, Any]:
    """
    Extract ESPN league settings in Sleeper-compatible format.
    Returns a dict with: scoring_settings, roster_positions, league_settings, total_rosters.
    Called by platform_api.sync_league_globals() to populate api.py module globals.
    """
    try:
        lg = _league(season, league_id)
    except Exception as e:
        print(f"[espn] get_league_globals failed: {e}")
        return {}

    settings = getattr(lg, "settings", None)

    # Use ESPN's raw scoring items, not espn_api's coarse/missing scoring_type.
    # Keep defaults only for categories ESPN omitted; explicit zeroes in the
    # raw list override them below.
    scoring_settings: Dict[str, Any] = {
        "rec": 0.0,
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "pass_int": -2.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "fum_lost": -2.0,
        "2pt": 2.0,
        "fg_0_19": 3.0,
        "fg_20_29": 3.0,
        "fg_30_39": 3.0,
        "fg_40_49": 4.0,
        "fg_50p": 5.0,
        "xpt": 1.0,
    }
    # One mSettings fetch feeds scoring + lineup slots (avoids a second round-trip).
    msettings_payload = _raw_espn_msettings(lg)
    scoring_items = (((msettings_payload.get("settings") or {}).get("scoringSettings") or {})
                     .get("scoringItems") or [])
    if not isinstance(scoring_items, list):
        scoring_items = []
    normalized_scoring = normalize_espn_scoring_items(scoring_items)
    scoring_settings.update(normalized_scoring)
    from utils.league_scoring import normalize_league_scoring
    scoring_settings = normalize_league_scoring(
        "espn", scoring_settings, league_id=league_id, season=season)
    # Transitional aliases for older consumers of dashboard_services.api's
    # provider-agnostic contract. Both families carry the same value; new point
    # math uses the Sleeper-style keys above.
    scoring_settings.update({
        "pointsPerReception": scoring_settings["rec"],
        "passYards": scoring_settings["pass_yd"],
        "passTD": scoring_settings["pass_td"],
        "passInterceptions": scoring_settings["pass_int"],
        "rushYards": scoring_settings["rush_yd"],
        "rushTD": scoring_settings["rush_td"],
        "receivingYards": scoring_settings["rec_yd"],
        "receivingTD": scoring_settings["rec_td"],
        "fumbles": scoring_settings["fum_lost"],
    })
    if "rec" not in normalized_scoring:
        logger.warning("[espn-scoring] reception item statId=53 missing "
                       "platform=espn league_id=%s season=%s scoring_items=%s",
                       league_id, season, len(scoring_items))
    else:
        logger.info("[espn-scoring] platform=espn league_id=%s season=%s "
                    "normalized_rec=%s normalized_pass_td=%s scoring_items=%s",
                    league_id, season, scoring_settings["rec"],
                    scoring_settings.get("pass_td"), len(scoring_items))

    roster_positions = _espn_roster_positions_from_settings(settings, msettings_payload)
    if not roster_positions:
        logger.warning("[espn-roster] empty roster_positions platform=espn "
                       "league_id=%s season=%s", league_id, season)
    else:
        logger.info("[espn-roster] platform=espn league_id=%s season=%s "
                    "slots=%s", league_id, season, len(roster_positions))

    # League settings
    total_rosters = len(getattr(lg, "teams", None) or [])
    # Sleeper-compatible playoff_week_start: first playoff week = regular-
    # season matchup periods + 1. Without this, sims default to week 14/15
    # even when ESPN runs a 13- or 15-week regular season.
    reg_season_count = _safe_int(getattr(settings, "reg_season_count", None))
    playoff_week_start = (reg_season_count + 1) if reg_season_count and reg_season_count > 0 else None
    league_settings: Dict[str, Any] = {
        "playoff_teams": _safe_int(getattr(settings, "playoff_team_count", 4)),
        "num_teams": total_rosters,
        "type": 0,
    }
    if playoff_week_start:
        league_settings["playoff_week_start"] = playoff_week_start

    # ESPN trade deadline is a calendar timestamp on tradeSettings.deadlineDate
    # (epoch ms). Map it so Season Hub can gate the trade-deadline card instead
    # of painting "Trade deadline: Playoff push" with no deadline context.
    trade_settings = ((msettings_payload.get("settings") or {}).get("tradeSettings") or {})
    deadline_raw = _safe_int(
        trade_settings.get("deadlineDate") if isinstance(trade_settings, dict) else None
    )
    if deadline_raw and deadline_raw > 0:
        # ESPN usually sends ms; accept seconds too.
        deadline_ts = int(deadline_raw / 1000) if deadline_raw > 10**12 else int(deadline_raw)
        league_settings["trade_deadline_ts"] = deadline_ts

    return {
        "scoring_settings": scoring_settings,
        "roster_positions": roster_positions,
        "league_settings": league_settings,
        "total_rosters": total_rosters,
    }


def build_espn_to_canonical(players_index: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    returns: espnId(str) -> canonical_id(str) where canonical_id is the dict key in your index
    Example index:
      "5938": {..., "espnID": "4039057"}
    => map["4039057"] = "5938"
    """
    out: Dict[str, str] = {}
    for canonical_id, info in (players_index or {}).items():
        espn_id = info.get("espnID") or info.get("espnId") or info.get("espn_id")
        if espn_id:
            out[str(espn_id)] = str(canonical_id)
    return out


def canon_pid(espn_pid: Any, espn_to_canon: Dict[str, str]) -> Optional[str]:
    if espn_pid is None:
        return None
    return espn_to_canon.get(str(espn_pid))
