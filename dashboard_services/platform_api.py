from __future__ import annotations

from typing import Any, Dict, List

from dashboard_services.api import (
    get_league as sleeper_get_league,
    get_users as sleeper_get_users,
    get_rosters as sleeper_get_rosters,
    get_matchups as sleeper_get_matchups,
    get_traded_picks as sleeper_get_traded_picks,
    get_bracket as sleeper_get_bracket,
)
from dashboard_services.providers.espn_api import (
    get_league as espn_get_league,
    get_users as espn_get_users,
    get_rosters as espn_get_rosters,
    get_matchups as espn_get_matchups, espn_get_bracket_like,
)


def norm_platform(platform: str) -> str:
    return (platform or "sleeper").lower().strip()


def get_league(platform: str, league_id: str, season: int) -> Dict[str, Any]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_league(season, league_id)
    return sleeper_get_league(league_id)


def get_users(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_users(season, league_id)
    return sleeper_get_users(league_id)


def get_rosters(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_rosters(season, league_id)
    return sleeper_get_rosters(league_id)


def get_matchups(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_matchups(season, league_id, week)
    return sleeper_get_matchups(league_id, week)


def get_traded_picks(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return []  # ESPN trades later
    return sleeper_get_traded_picks(league_id)


def get_bracket(platform: str, league_id: str, kind: str, season: int):
    if platform == "espn":
        import os
        return espn_get_bracket_like(
            league_id=league_id,
            season=season,
            kind=kind,
        )
    return sleeper_get_bracket(league_id, kind)
