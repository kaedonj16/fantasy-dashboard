"""Yahoo live-draft relay: thin wrapper over the shared snapshot store.

The Chrome/Edge extension observes the open Yahoo draft room and POSTs picks to
``/api/draft/yahoo-relay`` with the Draft Room session (same-origin). Observe-only
— never talks to Yahoo and never submits picks.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashboard_services.espn_draft_relay import (
    clear_relay_snapshot as _clear,
    get_relay_snapshot as _get,
    merge_live_with_relay,
    put_relay_snapshot as _put,
)

__all__ = [
    "clear_relay_snapshot",
    "get_relay_snapshot",
    "merge_live_with_relay",
    "put_relay_snapshot",
]


def put_relay_snapshot(
    league_id: str,
    season: int,
    snapshot: Mapping[str, Any],
    *,
    source: str = "relay",
) -> Dict[str, Any]:
    return _put(league_id, season, snapshot, source=source, platform="yahoo")


def get_relay_snapshot(league_id: str, season: int) -> Optional[Dict[str, Any]]:
    return _get(league_id, season, platform="yahoo")


def clear_relay_snapshot(league_id: str, season: int) -> None:
    _clear(league_id, season, platform="yahoo")
