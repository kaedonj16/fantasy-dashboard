"""Backward-compatible fantasy platform facade backed by the provider registry."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from dashboard_services.providers.registry import (
    get_provider, get_provider_capabilities, normalize_platform,
)

logger = logging.getLogger(__name__)


def norm_platform(platform: str) -> str:
    """Retain the legacy blank-to-Sleeper behavior; explicit unknowns fail."""
    return normalize_platform(platform)


def _yahoo_token(league_id: str = "", season: int = 0) -> str:
    """Resolve Yahoo credentials from a request session or a stored owner token."""
    from dashboard_services.providers.yahoo_api import get_valid_access_token, get_league_token
    try:
        from flask import session
        guid = session.get("yahoo_guid") or ""
        if guid:
            token = get_valid_access_token(guid)
            if token:
                session["yahoo_access_token"] = token
                return token
            # Refresh failed — do not fall back to a stale session bearer.
            session.pop("yahoo_access_token", None)
        else:
            raw = session.get("yahoo_access_token") or ""
            if raw:
                return raw
    except RuntimeError:
        pass
    return (get_league_token(league_id, season or 0) or "") if league_id else ""


def get_league(platform: str, league_id: str, season: int) -> Dict[str, Any]:
    return get_provider(platform).get_league(league_id, season)


def get_users(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_users(league_id, season)


def get_rosters(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_rosters(league_id, season)


def get_matchups(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_matchups(league_id, season, week)


def get_traded_picks(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_traded_picks(league_id, season)


def get_bracket(platform: str, league_id: str, kind: str, season: int):
    return get_provider(platform).get_bracket(league_id, season, kind)


def get_drafts(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_drafts(league_id, season)


def get_transactions(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_transactions(league_id, season, week)


def sync_league_globals(platform: str, league_id: str, season: int) -> None:
    """Populate legacy globals until league settings become request-context data.

    TODO: replace process-global league configuration in a dedicated refactor.
    """
    provider = get_provider(platform)
    try:
        data = provider.get_league_globals(league_id, season)
        if data:
            from dashboard_services.api import set_league_globals
            set_league_globals(scoring_settings=data.get("scoring_settings"),
                               roster_positions=data.get("roster_positions"),
                               league_settings=data.get("league_settings"),
                               total_rosters=data.get("total_rosters"))
    except Exception as exc:
        logger.warning("[sync_league_globals] %s failed for league %s: %s",
                       provider.metadata.key, league_id, type(exc).__name__)
