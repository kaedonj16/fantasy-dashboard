"""Historical deep-panel API (JSON lookup, no parquet)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from dashboard_services.historical.aggregates_store import aggregates_version, load_profile_aggregates
from dashboard_services.historical.board import build_deep_panel, build_historical_trends
from dashboard_services.historical.cohorts import evaluate_cohort
from dashboard_services.historical.filters import scout_matching_players

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
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-player] %s failed", player_id)
        return jsonify({"available": False, "player_id": str(player_id or "")})


@historical_api_bp.route("/api/historical-trends")
def api_historical_trends():
    """Position-level historical trend tables for the cheat-sheet Trends tab."""
    aggs = load_profile_aggregates()
    if not aggs:
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})
    try:
        payload = build_historical_trends(aggs)
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-trends] failed")
        return jsonify({"available": False, "descriptive_only": True, "not_in_ranking": True})


@historical_api_bp.route("/api/historical-cohort", methods=["POST"])
def api_historical_cohort():
    """Combined historical hit rate for selected Trends buckets. JSON index only."""
    aggs = load_profile_aggregates()
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        body = {}
    pos = body.get("position")
    filters = body.get("filters") or []
    if not isinstance(filters, list):
        filters = []
    tier = body.get("tier") or "top_12"
    if not aggs:
        return jsonify({
            "available": False,
            "descriptive_only": True,
            "not_in_ranking": True,
            "not_in_pick_score": True,
            "position": pos,
            "unknown_reason": "aggregates_missing",
        })
    try:
        payload = evaluate_cohort(
            aggs,
            position=pos,
            filters=filters,
            tier=tier,
            data_version=aggregates_version(),
        )
        # Scout matches use the same Python predicates as the cohort index.
        # board_features is request-scoped and must not enter the cohort cache.
        board_features = body.get("board_features")
        if isinstance(board_features, dict) and filters:
            payload = dict(payload)
            payload["scout_matches"] = scout_matching_players(board_features, filters)
        return jsonify(payload)
    except Exception:
        logger.exception("[historical-cohort] failed")
        return jsonify({
            "available": False,
            "descriptive_only": True,
            "not_in_ranking": True,
            "not_in_pick_score": True,
            "position": pos,
            "unknown_reason": "error",
        })

