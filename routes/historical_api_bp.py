"""Historical deep-panel API (JSON lookup, no parquet)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from dashboard_services.historical.aggregates_store import load_profile_aggregates
from dashboard_services.historical.board import build_deep_panel

logger = logging.getLogger(__name__)

historical_api_bp = Blueprint("historical_api", __name__)


@historical_api_bp.route("/api/historical-player/<player_id>")
def api_historical_player(player_id: str):
    """Lazy deep panel: named comps and rates from precomputed JSON leaves."""
    aggs = load_profile_aggregates()
    if not aggs:
        return jsonify({"available": False, "player_id": str(player_id or "")})
    extra = {}
    adp = request.args.get("adp")
    if adp not in (None, ""):
        extra["adp"] = adp
    redraft = request.args.get("redraft_avg_pick")
    if redraft not in (None, ""):
        extra["redraft_avg_pick"] = redraft
    pos = request.args.get("position")
    if pos:
        extra["position"] = pos
    try:
        payload = build_deep_panel(player_id, aggs, extra=extra or None)
    except Exception:
        logger.exception("[historical-player] %s failed", player_id)
        return jsonify({"available": False, "player_id": str(player_id or "")})
    return jsonify(payload)
