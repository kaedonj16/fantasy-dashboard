from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from dashboard_services.api import (
    get_league as sleeper_get_league,
    get_users as sleeper_get_users,
    get_rosters as sleeper_get_rosters,
    get_matchups as sleeper_get_matchups,
    get_traded_picks as sleeper_get_traded_picks,
    get_bracket as sleeper_get_bracket,
    get_drafts as sleeper_get_drafts,
    get_transactions as sleeper_get_transactions,
    set_league_globals,
)
from dashboard_services.providers.espn_api import (
    get_league as espn_get_league,
    get_users as espn_get_users,
    get_rosters as espn_get_rosters,
    get_matchups as espn_get_matchups,
    espn_get_bracket_like,
    get_drafts as espn_get_drafts,
    get_league_globals as espn_get_league_globals,
    get_transactions as espn_get_transactions,
)


def norm_platform(platform: str) -> str:
    return (platform or "sleeper").lower().strip()


def _yahoo_token() -> str:
    """Retrieve the Yahoo access token from the Flask request session."""
    try:
        from flask import session
        return session.get("yahoo_access_token") or ""
    except RuntimeError:
        return ""


def get_league(platform: str, league_id: str, season: int) -> Dict[str, Any]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_league(season, league_id)
    if platform == "yahoo":
        from dashboard_services.providers.yahoo_api import get_league as yahoo_get_league
        return yahoo_get_league(season, league_id, _yahoo_token())
    return sleeper_get_league(league_id)


def get_users(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_users(season, league_id)
    if platform == "yahoo":
        from dashboard_services.providers.yahoo_api import get_users as yahoo_get_users
        return yahoo_get_users(season, league_id, _yahoo_token())
    return sleeper_get_users(league_id)


def get_rosters(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_rosters(season, league_id)
    if platform == "yahoo":
        from dashboard_services.providers.yahoo_api import get_rosters as yahoo_get_rosters
        return yahoo_get_rosters(season, league_id, _yahoo_token())
    return sleeper_get_rosters(league_id)


def get_matchups(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform == "espn":
        return espn_get_matchups(season, league_id, week)
    if platform == "yahoo":
        from dashboard_services.providers.yahoo_api import get_matchups as yahoo_get_matchups
        return yahoo_get_matchups(season, league_id, week, _yahoo_token())
    return sleeper_get_matchups(league_id, week)


def get_traded_picks(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    platform = norm_platform(platform)
    if platform in ("espn", "yahoo"):
        return []
    return sleeper_get_traded_picks(league_id)


def get_bracket(platform: str, league_id: str, kind: str, season: int):
    p = norm_platform(platform)
    if p == "espn":
        return espn_get_bracket_like(league_id=league_id, season=season, kind=kind)
    if p == "yahoo":
        from dashboard_services.providers.yahoo_api import get_bracket_like as yahoo_bracket
        return yahoo_bracket(league_id, season, kind, _yahoo_token())
    return sleeper_get_bracket(league_id, kind)


def get_drafts(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    p = norm_platform(platform)
    if p == "espn":
        return espn_get_drafts(season, league_id)
    if p == "yahoo":
        from dashboard_services.providers.yahoo_api import get_drafts as yahoo_get_drafts
        return yahoo_get_drafts(season, league_id, _yahoo_token())
    return sleeper_get_drafts(league_id)


def get_transactions(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    """Platform-agnostic transaction fetch for a single week."""
    p = norm_platform(platform)
    if p == "espn":
        return espn_get_transactions(season, league_id, week)
    if p == "yahoo":
        from dashboard_services.providers.yahoo_api import get_transactions as yahoo_get_transactions
        return yahoo_get_transactions(season, league_id, week, _yahoo_token())
    return sleeper_get_transactions(league_id, week) or []


def sync_league_globals(platform: str, league_id: str, season: int) -> None:
    """
    Populate the api.py module globals (SCORING_SETTINGS, ROSTER_POSITIONS, etc.)
    for the given league. For Sleeper these are already populated by get_league().
    For ESPN/Yahoo we extract them explicitly here.
    """
    p = norm_platform(platform)
    if p not in ("espn", "yahoo"):
        return
    try:
        if p == "espn":
            data = espn_get_league_globals(season, league_id)
        else:
            from dashboard_services.providers.yahoo_api import get_league_globals as yahoo_globals
            data = yahoo_globals(season, league_id, _yahoo_token())
        if data:
            set_league_globals(
                scoring_settings=data.get("scoring_settings"),
                roster_positions=data.get("roster_positions"),
                league_settings=data.get("league_settings"),
                total_rosters=data.get("total_rosters"),
            )
    except Exception as e:
        logger.warning("[sync_league_globals] %s failed for league %s: %s", p, league_id, e)
