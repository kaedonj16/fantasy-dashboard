"""Small self-contained utility API endpoints.

Routes:
    /api/changelog
    /api/nfl-state
    /api/advanced-metrics/seasons

Extracted from app.py to reduce monolith size.
Dependencies: dashboard_services.* / data_building.* only - no app.py internals.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

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
