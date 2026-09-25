"""Deploy-time warmup of league-independent shared caches.

Every deploy wipes the per-worker in-memory caches, so the first requests on
each gunicorn worker pay full cold rebuilds (league context builds stall
requests for 7-28s). Gunicorn's ``post_fork`` hook (see ``gunicorn_conf.py``)
runs :func:`warm_shared_caches` in a daemon thread inside each worker AFTER
the port bind, so warmup never blocks the bind and never stalls a request.

Only league-INDEPENDENT data is warmed here: per-league contexts depend on
which leagues users visit and must not be pre-built in every worker.
Warming is best-effort and fail-soft: each step is isolated, exceptions are
logged and swallowed, and the whole pass never raises.

Disable with ``STARTUP_WARMUP_ENABLED=0``.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("STARTUP_WARMUP_ENABLED", "1").lower() not in {
    "0", "false", "no", "off",
}

_warmup_thread: threading.Thread | None = None


def _step(name: str, fn) -> None:
    started = time.monotonic()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - warmup must never raise
        logger.warning("[startup-warmup] %s failed: %s", name, exc)
    else:
        logger.info(
            "[startup-warmup] %s warmed in %.1fs",
            name, time.monotonic() - started,
        )


def _current_season() -> int:
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state() or {}
        return int(state.get("season") or 0) or _fallback_season()
    except Exception:  # noqa: BLE001 - fall back to calendar year
        return _fallback_season()


def _fallback_season() -> int:
    from datetime import datetime
    return datetime.now().year


def _load_model_value_table() -> None:
    # Deferred import: app.py is fully loaded by the time post_fork fires.
    from app import get_model_value_table_cached
    get_model_value_table_cached()


def warm_shared_caches() -> None:
    """Warm league-independent caches in this worker. Never raises."""
    if not _ENABLED:
        logger.info("[startup-warmup] disabled via STARTUP_WARMUP_ENABLED=0")
        return
    season = _current_season()

    def _nfl_state():
        from dashboard_services.api import get_nfl_state
        get_nfl_state()

    def _players_index():
        from utils.utils import load_players_index
        load_players_index()

    def _nfl_players():
        from dashboard_services.api import get_nfl_players
        get_nfl_players()

    def _season_projections():
        from data_building.fetch_projections import (
            fetch_sleeper_season_ppg_variants,
        )
        fetch_sleeper_season_ppg_variants(season)

    def _usage_trends():
        from data_building.weekly_metrics import get_usage_trends
        get_usage_trends(season)

    for name, fn in (
        ("nfl_state", _nfl_state),
        ("players_index", _players_index),
        ("nfl_players", _nfl_players),
        ("season_projections", _season_projections),
        ("usage_trends", _usage_trends),
        ("model_value_table", _load_model_value_table),
    ):
        _step(name, fn)


def warm_shared_caches_async() -> threading.Thread | None:
    """Spawn the warmup in a daemon thread; returns the thread (or None)."""
    global _warmup_thread
    if not _ENABLED:
        return None
    thread = threading.Thread(
        target=warm_shared_caches,
        name="startup-warmup",
        daemon=True,
    )
    thread.start()
    _warmup_thread = thread
    return thread
