"""Small self-contained utility API endpoints.

Routes:
    /api/changelog
    /api/nfl-state
    /api/advanced-metrics/seasons
    /api/trade-count
    /risers-fallers            (legacy 301 -> /top-movers)

Extracted from app.py to reduce monolith size.
Dependencies: dashboard_services.* / data_building.* only - no app.py internals.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, redirect

from dashboard_services.changelog import CHANGELOG

logger = logging.getLogger(__name__)

misc_api_bp = Blueprint("misc_api", __name__)


@misc_api_bp.route("/api/changelog")
def api_changelog():
    """Return the changelog entries."""
    return jsonify(CHANGELOG)


@misc_api_bp.route("/api/nfl-state")
def api_nfl_state():
    """Get current NFL state from Sleeper API."""
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state()
        return jsonify(state or {})
    except Exception as e:
        logger.info(f"[nfl-state] Error: {e}")
        return jsonify({}), 500


@misc_api_bp.route("/api/advanced-metrics/seasons")
def api_advanced_metrics_seasons():
    """Return available seasons in player_advanced_metrics, newest first."""
    from data_building.advanced_metrics import get_available_seasons
    return jsonify({"seasons": get_available_seasons()})


@misc_api_bp.route("/api/trade-count")
def api_trade_count():
    """Get the count of trades from trade_intel_trades table."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS n FROM trade_intel_trades")
            count = cursor.fetchone()["n"]
        return jsonify({"count": count})
    except Exception:
        # Return fallback count if table doesn't exist or other error
        return jsonify({"count": 15000})


@misc_api_bp.route("/risers-fallers")
def risers_fallers_redirect():
    return redirect("/top-movers", 301)
