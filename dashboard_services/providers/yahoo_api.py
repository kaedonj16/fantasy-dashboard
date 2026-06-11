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
import threading
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from utils.utils import load_players_index
from utils.coerce import safe_int as _safe_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAHOO_AUTH_URL  = "https://api.login.yahoo.com/oauth2/request_auth"
YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
YAHOO_API_BASE  = "https://fantasysports.yahooapis.com/fantasy/v2"
YAHOO_SCOPE     = "openid fspt-r"

# Only NFL for now; extend as needed.
_GAME_CODE = "nfl"

# Per-process cache: (league_key, endpoint) -> (fetched_at, data)
_api_cache: Dict[str, tuple[float, Any]] = {}
_api_cache_lock = threading.Lock()
_API_CACHE_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def get_authorization_url(state: str) -> str:
    """Return the Yahoo OAuth 2.0 authorization URL the user should be redirected to."""
    params = {
        "client_id":     os.environ["YAHOO_CLIENT_ID"],
        "redirect_uri":  os.environ["YAHOO_REDIRECT_URI"],
        "response_type": "code",
        "scope":         YAHOO_SCOPE,
        "state":         state,
    }
    return f"{YAHOO_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """
    Exchange an authorization code for access + refresh tokens.
    Returns the full token response dict including xoauth_yahoo_guid.
    """
    client_id     = os.environ["YAHOO_CLIENT_ID"]
    client_secret = os.environ["YAHOO_CLIENT_SECRET"]
    redirect_uri  = os.environ["YAHOO_REDIRECT_URI"]

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
    return resp.json()


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh an expired access token using the stored refresh token."""
    client_id     = os.environ["YAHOO_CLIENT_ID"]
    client_secret = os.environ["YAHOO_CLIENT_SECRET"]

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
        pass
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
        pass
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


def get_valid_access_token(guid: str) -> Optional[str]:
    """Return a valid access token for the given Yahoo GUID, refreshing if needed."""
    tokens = load_tokens(guid)
    if not tokens:
        return None

    expires_at = tokens["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    # Refresh if expiring within 5 minutes
    if datetime.now(timezone.utc) >= expires_at - timedelta(minutes=5):
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
# API request helper
# ---------------------------------------------------------------------------

def _yahoo_get(access_token: str, path: str, params: Optional[Dict] = None) -> Any:
    """Make a GET request to the Yahoo Fantasy API, returning parsed JSON."""
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
    resp.raise_for_status()
    data = resp.json()

    with _api_cache_lock:
        _api_cache[cache_key] = (time.time(), data)

    return data


def _league_key(league_id: str) -> str:
    return f"{_GAME_CODE}.l.{league_id}"


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


def _yahoo_pos(raw_pos: str) -> str:
    _MAP = {
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
        "K": "K", "DEF": "DEF", "D": "DEF", "DST": "DEF",
        "PK": "K",
    }
    return _MAP.get((raw_pos or "").upper(), (raw_pos or "").upper())


def _resolve_player(yahoo_name: str, yahoo_pos: str, yahoo_team: str) -> Optional[str]:
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

def _extract_league_meta(raw: Dict) -> Dict:
    fc = raw.get("fantasy_content", {})
    league_list = fc.get("league") or []
    return league_list[0] if league_list else {}


def _extract_teams(raw: Dict) -> List[Dict]:
    """Extract the teams dict from a league+teams response."""
    fc   = raw.get("fantasy_content", {})
    lg   = fc.get("league") or []
    meta = lg[1] if len(lg) > 1 else {}
    teams_block = meta.get("teams") or {}
    count = _safe_int(teams_block.get("count") or teams_block.get("0", {}).get("count")) or 0
    out = []
    for i in range(1, count + 1):
        entry = teams_block.get(str(i))
        if entry and "team" in entry:
            out.append(entry["team"])
    return out


def _team_attr(team_list: List, key: str, default=None):
    """Yahoo returns team attributes as a list of dicts; find the one with `key`."""
    for item in team_list or []:
        if isinstance(item, dict) and key in item:
            return item[key]
    return default


def _extract_roster_players(team_data: List) -> List[Dict]:
    """Extract player list from the roster portion of a team entry."""
    roster_block = None
    for item in team_data or []:
        if isinstance(item, dict) and "roster" in item:
            roster_block = item["roster"]
            break
    if not roster_block:
        return []

    players_block = roster_block.get("players") or {}
    count = _safe_int(players_block.get("count") or players_block.get("0", {}).get("count")) or 0
    out = []
    for i in range(1, count + 1):
        entry = players_block.get(str(i))
        if entry and "player" in entry:
            out.append(entry["player"])
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Public API – normalized to match Sleeper/ESPN shapes
# ---------------------------------------------------------------------------

def get_league(season: int, league_id: str, access_token: str) -> Dict[str, Any]:
    raw  = _yahoo_get(access_token, f"league/{_league_key(league_id)}")
    meta = _extract_league_meta(raw)
    return {
        "league_id": str(league_id),
        "season":    int(season),
        "name":      meta.get("name") or "Yahoo League",
    }


def get_users(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    raw   = _yahoo_get(access_token, f"league/{_league_key(league_id)}/teams")
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

        managers = _team_attr(t, "managers") or []
        if isinstance(managers, dict):
            managers = [managers]
        mgr   = (managers[0].get("manager") or {}) if managers else {}
        guid  = mgr.get("guid") or str(team_id)
        nick  = mgr.get("nickname") or team_name

        out.append({
            "avatar":       logo_url,
            "display_name": nick,
            "is_bot":       False,
            "is_owner":     None,
            "league_id":    str(league_id),
            "metadata":     {"team_name": team_name},
            "settings":     None,
            "user_id":      guid,
            "roster_id":    _safe_int(team_id),
        })
    return out


def get_rosters(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    raw = _yahoo_get(
        access_token,
        f"league/{_league_key(league_id)}/teams;out=roster,stats,standings",
    )
    teams = _extract_teams(raw)
    out: List[Dict[str, Any]] = []

    for t in teams:
        team_id   = _safe_int(_team_attr(t, "team_id"))
        managers  = _team_attr(t, "managers") or []
        if isinstance(managers, dict):
            managers = [managers]
        mgr      = (managers[0].get("manager") or {}) if managers else {}
        owner_id = mgr.get("guid") or str(team_id)

        # Standings / record
        standings = _team_attr(t, "team_standings") or {}
        outcome   = standings.get("outcome_totals") or {}
        wins      = _safe_int(outcome.get("wins"))
        losses    = _safe_int(outcome.get("losses"))
        ties      = _safe_int(outcome.get("ties"))
        pts_for   = _safe_float(standings.get("points_for"))
        pts_ag    = _safe_float(standings.get("points_against"))

        # Roster
        raw_players = _extract_roster_players(t)
        players:  List[str] = []
        starters: List[str] = []
        reserve:  List[str] = []

        for rp in raw_players:
            p_meta  = rp[0] if isinstance(rp, list) else rp
            sel_pos = None
            if isinstance(rp, list) and len(rp) > 1:
                sel_pos = (rp[1].get("selected_position") or {}).get("position")

            name     = (p_meta.get("name") or {}).get("full") or ""
            pos_list = p_meta.get("display_position") or p_meta.get("eligible_positions") or ""
            if isinstance(pos_list, dict):
                pos_list = pos_list.get("position") or ""
            pos  = (pos_list.split(",")[0] if isinstance(pos_list, str) else "") or ""
            team = (p_meta.get("editorial_team_abbr") or "").upper()

            canon = _resolve_player(name, pos, team)
            if not canon:
                continue

            players.append(canon)
            if sel_pos in ("BN", "IR", "IR+"):
                reserve.append(canon)
            else:
                starters.append(canon)

        fpts_whole = int(pts_for)
        fpts_dec   = int(round((pts_for - fpts_whole) * 100))
        fpa_whole  = int(pts_ag)
        fpa_dec    = int(round((pts_ag - fpa_whole) * 100))

        out.append({
            "co_owners":  None,
            "keepers":    None,
            "league_id":  str(league_id),
            "metadata":   {},
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
    return out


def get_matchups(season: int, league_id: str, week: int, access_token: str) -> List[Dict[str, Any]]:
    raw        = _yahoo_get(access_token, f"league/{_league_key(league_id)}/scoreboard;week={week}")
    fc         = raw.get("fantasy_content", {})
    lg         = fc.get("league") or []
    scoreboard = (lg[1] if len(lg) > 1 else {}).get("scoreboard") or {}
    matchups   = scoreboard.get("matchups") or {}
    count      = _safe_int(matchups.get("count") or matchups.get("0", {}).get("count")) or 0

    out: List[Dict[str, Any]] = []
    for i in range(count):
        entry   = matchups.get(str(i)) or {}
        matchup = entry.get("matchup") or {}
        teams   = matchup.get("teams") or {}
        m_id    = i + 1

        for j in range(2):
            tm_entry = teams.get(str(j)) or {}
            tm       = tm_entry.get("team") or []
            tm_meta  = tm[0] if tm else {}
            if isinstance(tm_meta, list):
                tm_meta = tm_meta[0] if tm_meta else {}

            roster_id = _safe_int(
                tm_meta.get("team_id")
                or (tm_meta.get("team_key") or "").split(".")[-1]
            )
            pts_block = (tm[1] if len(tm) > 1 else {}) if isinstance(tm, list) else {}
            points    = _safe_float(pts_block.get("team_points", {}).get("total") if isinstance(pts_block, dict) else 0)

            out.append({
                "points":          points,
                "players":         [],
                "roster_id":       roster_id,
                "custom_points":   None,
                "matchup_id":      m_id,
                "starters":        [],
                "starters_points": [],
                "players_points":  {},
            })
    return out


def get_transactions(season: int, league_id: str, week: int, access_token: str) -> List[Dict[str, Any]]:
    try:
        raw = _yahoo_get(
            access_token,
            f"league/{_league_key(league_id)}/transactions;types=add,drop,trade",
        )
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


def get_drafts(season: int, league_id: str, access_token: str) -> List[Dict[str, Any]]:
    from datetime import datetime as _dt
    start_ts_ms = int(_dt(int(season), 8, 1).timestamp() * 1000)
    return [{
        "draft_id":   f"yahoo_{league_id}_{season}",
        "league_id":  str(league_id),
        "season":     int(season),
        "season_type": "regular",
        "start_time": start_ts_ms,
        "status":     "complete",
        "type":       "snake",
    }]


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
        meta = lg[0] if lg else {}
        settings = meta.get("settings") or {}
    except Exception as exc:
        logger.warning("[yahoo] get_league_globals failed: %s", exc)
        return {}

    # PPR
    scoring_type = (meta.get("scoring_type") or "head").lower()
    ppr_map = {"head_one_win": 0.0, "head": 0.0, "headone": 0.0, "point": 1.0}
    is_ppr  = any(k in scoring_type for k in ("ppr", "point"))
    ppr     = 1.0 if is_ppr else 0.5 if "half" in scoring_type else 0.0

    scoring_settings: Dict[str, Any] = {
        "rec":      ppr,
        "pass_yd":  0.04,
        "pass_td":  4.0,
        "pass_int": -2.0,
        "rush_yd":  0.1,
        "rush_td":  6.0,
        "rec_yd":   0.1,
        "rec_td":   6.0,
        "fum_lost": -2.0,
        "2pt":      2.0,
    }

    # Roster positions
    roster_positions_raw = settings.get("roster_positions") or {}
    pos_count = roster_positions_raw.get("roster_position") or []
    if isinstance(pos_count, dict):
        pos_count = [pos_count]

    _YAHOO_SLOT = {
        "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
        "W/R/T": "FLEX", "W/R": "FLEX", "RB/WR/TE": "FLEX",
        "Q/W/R/T": "SUPER_FLEX", "OP": "SUPER_FLEX",
        "K": "K", "DEF": "DEF", "D": "DEF",
        "BN": "BN", "IR": "IR",
    }
    roster_positions: List[str] = []
    for slot in pos_count:
        abbr  = slot.get("position") or ""
        count = _safe_int(slot.get("count")) or 0
        norm  = _YAHOO_SLOT.get(abbr.upper(), abbr.upper())
        roster_positions.extend([norm] * count)

    num_teams = _safe_int(meta.get("num_teams")) or 0
    league_settings: Dict[str, Any] = {
        "playoff_teams": _safe_int(settings.get("num_playoff_teams")) or 4,
        "num_teams":     num_teams,
        "type":          0,
    }

    return {
        "scoring_settings": scoring_settings,
        "roster_positions": roster_positions,
        "league_settings":  league_settings,
        "total_rosters":    num_teams,
    }
