"""Backward-compatible fantasy platform facade backed by the provider registry."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from dashboard_services.providers.base import BRACKET, UnsupportedCapabilityError
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
        # Never keep access tokens in the session cookie after DB storage.
        session.pop("yahoo_access_token", None)
        guid = session.get("yahoo_guid") or ""
        if guid:
            token = get_valid_access_token(guid)
            if token:
                return token
    except RuntimeError:
        pass
    return (get_league_token(league_id, season or 0) or "") if league_id else ""


def get_league(platform: str, league_id: str, season: int) -> Dict[str, Any]:
    return get_provider(platform).get_league(league_id, season)


def get_users(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_users(league_id, season)


def get_rosters(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_rosters(league_id, season)


def get_matchups(
    platform: str, league_id: str, week: int, season: int, *,
    cache_ttl: float | None = None,
) -> List[Dict[str, Any]]:
    """Return matchups, optionally bounding provider cache age for live views."""
    provider = get_provider(platform)
    if cache_ttl is None:
        return provider.get_matchups(league_id, season, week)
    return provider.get_matchups(league_id, season, week, cache_ttl=cache_ttl)


def get_traded_picks(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_traded_picks(league_id, season)


def get_bracket(platform: str, league_id: str, kind: str, season: int):
    """Return a playoff bracket, or [] when none can be derived.

    Sleeper/ESPN publish a native bracket. Fleaflicker and MFL derive one
    from playoff-week matchups (or project the first round from standings).
    Callers treat an empty list as "no Playoff Picture"; raising would 500
    pages that only need the bracket when it exists.
    """
    provider = get_provider(platform)
    if not provider.supports(BRACKET):
        return []
    try:
        return provider.get_bracket(league_id, season, kind) or []
    except (UnsupportedCapabilityError, Exception):
        logger.debug(
            "get_bracket unavailable platform=%s league=%s kind=%s",
            provider.metadata.key, league_id, kind, exc_info=True,
        )
        return []


def get_drafts(platform: str, league_id: str, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_drafts(league_id, season)


def get_transactions(platform: str, league_id: str, week: int, season: int) -> List[Dict[str, Any]]:
    return get_provider(platform).get_transactions(league_id, season, week)


def sync_league_globals(platform: str, league_id: str, season: int) -> None:
    """Replace the league-config context with this league's settings.

    Isolation guarantee: when this returns, the request/thread context holds
    exactly this league's config -- or nothing at all when the provider has no
    data (failed or empty response). It can never hold a *previous* league's
    config: the context is cleared before the provider's data is consulted, so
    a failed or partial sync cannot leave stale cross-league state behind for
    later readers. (This is the scoped fix for the old process-global TODO:
    the storage itself is request-scoped via ``flask.g`` inside the app, with
    a thread-local fallback outside it.)

    Residual risk / full refactor: the context is still ambient state rather
    than an explicit parameter, so a code path that reads the globals without
    syncing first sees empty defaults instead of failing loudly. The full
    refactor is to thread an explicit league-context object through the call
    chain instead of consulting ambient state; until then, out-of-request
    consumers that run several leagues on one thread (background jobs, crons)
    must re-sync -- or call
    ``dashboard_services.api.clear_league_globals`` -- between leagues.
    """
    provider = get_provider(platform)
    from dashboard_services.api import clear_league_globals, set_league_globals
    # Clear before the provider data is consulted: a failed/empty/partial
    # response must leave "no config", never the previous league's config.
    clear_league_globals()
    try:
        data = provider.get_league_globals(league_id, season)
        if data:
            set_league_globals(scoring_settings=data.get("scoring_settings"),
                               roster_positions=data.get("roster_positions"),
                               league_settings=data.get("league_settings"),
                               total_rosters=data.get("total_rosters"))
    except Exception as exc:
        logger.warning("[sync_league_globals] %s failed for league %s: %s",
                       provider.metadata.key, league_id, type(exc).__name__)
