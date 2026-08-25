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

@lru_cache(maxsize=16)
def _public_league_cached(season: int, league_id: str) -> League:
    return League(league_id=int(league_id), year=int(season))


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
            from dashboard_services.accounts import get_espn_league_credentials
            stored = get_espn_league_credentials(session["account_id"], league_id, season) or {}
            # An authenticated account may use only credentials attached to its
            # own saved league; never fall back to another account/server cookie.
            espn_s2 = stored.get("espn_s2")
            swid = stored.get("swid")
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


_league_cached.cache_clear = _public_league_cached.cache_clear  # type: ignore[attr-defined]


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
    ``{team_id, name}``. ``is_mine`` is set when the server's SWID owns the team
    (only useful when server-level cookies happen to be the user's).
    """
    lg = _league(season, league_id)
    _, swid = _espn_creds()
    swid = (swid or "").strip()
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
            "is_mine": bool(swid and swid in owner_ids),
        })
    return out


def get_rosters(season: int, league_id: str) -> List[Dict[str, Any]]:
    lg = _league(season, league_id)
    espn_to_canon = _espn_to_canon_cached()

    rosters: List[Dict[str, Any]] = []

    for t in lg.teams or []:
        roster_id = _safe_int(getattr(t, "team_id", None))
        owners = getattr(t, "owners", None) or []
        owner_id = str(owners[0].get("id")) if owners else None

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
            cp = canon_pid(str(pid), espn_to_canon)
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

        rosters.append({
            "co_owners": None,
            "keepers": None,
            "league_id": str(league_id),
            "metadata": {"record": f"{wins}-{losses}", "streak": streak},
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
    # Prefer proTeamId if present
    pro_id = (
            getattr(bp, "proTeamId", None)
            or getattr(bp, "pro_team_id", None)
            or getattr(bp, "proTeam", None)
    )
    if pro_id == "WSH":
        pro_id = "WAS"

    if isinstance(pro_id, str):
        pro_id = None

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

                # Canonicalize player id (special-case D/ST)
                cp: Optional[str] = None
                try:
                    pid_int = int(pid_raw)
                except Exception:
                    pid_int = None

                if pid_int is not None and pid_int < 0 and str(pid_int).startswith("-160"):
                    cp = _dst_canonical_id(bp, pid_int)
                else:
                    cp = canon_pid(str(pid_raw), espn_to_canon)

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
    "FLEX": "FLEX", "RB/WR/TE": "FLEX", "RB/WR": "FLEX",
    "OP": "SUPER_FLEX",
    "K": "K",
    "D/ST": "DEF", "DST": "DEF", "DEF": "DEF", "D-ST": "DEF",
    "BE": "BN", "BENCH": "BN",
    "IR": "IR",
}


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

    # Scoring type -> PPR value
    scoring_type = (getattr(settings, "scoring_type", None) or "standard").lower().replace(" ", "_")
    ppr_map = {"ppr": 1.0, "half_ppr": 0.5, "half-ppr": 0.5, "standard": 0.0}
    ppr = ppr_map.get(scoring_type, 0.0)

    scoring_settings: Dict[str, Any] = {
        "rec": ppr,
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

    # Roster positions
    raw_slots = getattr(settings, "roster_slots", None) or []
    roster_positions = [
        _ESPN_SLOT_TO_SLEEPER.get(str(s).upper().strip(), str(s).upper())
        for s in raw_slots
    ]

    # League settings
    total_rosters = len(getattr(lg, "teams", None) or [])
    league_settings: Dict[str, Any] = {
        "playoff_teams": _safe_int(getattr(settings, "playoff_team_count", 4)),
        "num_teams": total_rosters,
        "type": 0,
    }

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
