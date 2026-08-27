"""Historical deep-panel API (JSON lookup, no parquet)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from dashboard_services.historical.aggregates_store import load_profile_aggregates
from dashboard_services.historical.board import build_deep_panel, build_historical_trends

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
    proj = request.args.get("proj_ppg")
    if proj not in (None, ""):
        extra["proj_ppg"] = proj
        extra["projected_ppg"] = proj
    proj_rk = request.args.get("proj_rk")
    if proj_rk not in (None, ""):
        extra["projected_positional_rank"] = proj_rk
        extra["proj_rk"] = proj_rk
    adp_rk = request.args.get("adp_rk")
    if adp_rk not in (None, ""):
        extra["adp_positional_rank"] = adp_rk
        extra["adp_rk"] = adp_rk
    try:
        payload = build_deep_panel(player_id, aggs, extra=extra or None)
    except Exception:
        logger.exception("[historical-player] %s failed", player_id)
        return jsonify({"available": False, "player_id": str(player_id or "")})
    return jsonify(payload)


@historical_api_bp.route("/api/historical-trends")
def api_historical_trends():
    """Position-level historical trend tables for the cheat-sheet Trends tab."""
    aggs = load_profile_aggregates()
    if not aggs:
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})
    try:
        payload = build_historical_trends(aggs)
    except Exception:
        logger.exception("[historical-trends] failed")
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})
    return jsonify(payload)
