from __future__ import annotations

import requests
from typing import List

SLEEPER_BASE = "https://api.sleeper.app/v1"
SCORING_DEFAULTS = {
    # Passing
    "twoPointConversions": 2, "passYards": 0.04, "passAttempts": -0.5, "passTD": 4,
    "passCompletions": 1, "passInterceptions": -2,
    # Receiving
    "pointsPerReception": 1, "receivingYards": 0.1, "receivingTD": 6, "targets": 0.1,
    # Rushing
    "carries": 0.2, "rushYards": 0.1, "rushTD": 6, "fumbles": -2,
    # Kicking
    "fgMade": 3, "fgMissed": -1, "xpMade": 1, "xpMissed": -1,
}
TANK01_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
BASE = f"https://{TANK01_HOST}"


def _avatar_from_users(users: list[dict], owner_id: Optional[str]) -> Optional[str]:
    if not owner_id:
        return None
    u = next((u for u in users if u.get("user_id") == owner_id), None)
    if not u:
        return None
    meta = u.get("metadata") or {}
    avatar_meta = meta.get("avatar")
    profile_id = u.get("avatar")
    if avatar_meta:
        return avatar_meta
    if profile_id:
        return f"https://sleepercdn.com/avatars/{profile_id}"
    return None


def fetch_json(path: str, timeout: int = 25) -> dict:
    url = f"{SLEEPER_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_league_name(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}")


def get_users(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/users")


def get_rosters(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/rosters")


def get_matchups(league_id: str, week: int) -> List[dict]:
    return fetch_json(f"/league/{league_id}/matchups/{week}")


def get_nfl_state() -> dict:
    return fetch_json("/state/nfl") or {}


def get_transactions(league_id: str, week: int) -> List[dict]:
    return fetch_json(f"/league/{league_id}/transactions/{week}")


def _avatar_url(avatar_id: str) -> Union[str, None]:
    if not avatar_id:
        return None
    return f"{avatar_id}"


def _headers(rapidapi_key: str) -> Dict[str, str]:
    return {
        "x-rapidapi-host": TANK01_HOST,
        "x-rapidapi-key": str(rapidapi_key),
    }

def get_bracket(league_id: str, bracket: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/{bracket}_bracket")
