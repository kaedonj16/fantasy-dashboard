"""
Rookie prospect API blueprint.

Endpoints:
    GET /api/rookies/rankings?year=2026&pos=WR&league_type=1qb
    GET /api/rookies/player/<player_id>
    GET /api/rookies/active-class
    POST /api/rookies/refresh          (triggers pipeline re-run)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

rookie_bp = Blueprint("rookies", __name__, url_prefix="/api/rookies")

# In-memory cache so we don't re-run the pipeline on every page load.
# Invalidated on refresh or on first hit per process.
_cache: Dict[int, List[Dict[str, Any]]] = {}


def _get_rankings(draft_year: int) -> List[Dict[str, Any]]:
    if draft_year not in _cache:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db
        _cache[draft_year] = get_rookie_rankings_from_db(draft_year)
    return _cache[draft_year]


def _safe_float(v, default=None):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _row_to_dict(row: Dict) -> Dict:
    """Serialise a row dict to JSON-safe types."""
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


@rookie_bp.route("/active-class")
def active_class():
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class
    year = get_active_rookie_class()
    return jsonify({"draft_class_year": year})


@rookie_bp.route("/rankings")
def rankings():
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital

        year = request.args.get("year", type=int) or get_active_rookie_class()
        pos  = (request.args.get("pos") or "").upper() or None
        league_type = (request.args.get("league_type") or "1qb").lower()
        league_size = request.args.get("league_size", type=int) or 10

        rows = _get_rankings(year)

        # Position filter
        if pos:
            rows = [r for r in rows if (r.get("position") or "").upper() == pos]

        # Build response list with value field chosen by league settings
        result = []
        for r in rows:
            d = _row_to_dict(r)

            # Choose value based on settings
            if league_type == "sf":
                val_key = "rookie_sf_value" if league_size == 10 else f"rookie_sf_value_{league_size}"
                d["display_value"] = d.get(val_key) or d.get("rookie_sf_value")
            else:
                val_key = "rookie_value" if league_size == 10 else f"rookie_value_{league_size}"
                d["display_value"] = d.get(val_key) or d.get("rookie_value")

            # Draft capital label
            d["draft_capital_label"] = format_draft_capital(
                d.get("projected_round"),
                d.get("projected_pick"),
                d.get("projected_pick_low"),
                d.get("projected_pick_high"),
            )
            result.append(d)

        return jsonify({"draft_class_year": year, "count": len(result), "rankings": result})

    except Exception as exc:
        log.exception("[rookie_api] /rankings error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/player/<player_id>")
def player_detail(player_id: str):
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        year = request.args.get("year", type=int) or get_active_rookie_class()
        rows = _get_rankings(year)
        row  = next((r for r in rows if r["player_id"] == player_id), None)
        if not row:
            return jsonify({"error": "Player not found"}), 404
        return jsonify(_row_to_dict(row))
    except Exception as exc:
        log.exception("[rookie_api] /player error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/refresh", methods=["POST"])
def refresh():
    """Re-run the pipeline and bust the in-memory cache."""
    try:
        from data_building.rookie_pipeline.pipeline import (
            get_active_rookie_class, run_rookie_pipeline,
        )
        year = request.json.get("year") if request.json else None
        if year is None:
            year = get_active_rookie_class()
        year = int(year)

        _cache.pop(year, None)
        run_rookie_pipeline(year)
        _cache.pop(year, None)  # force fresh DB read on next request

        return jsonify({"status": "ok", "draft_class_year": year})
    except Exception as exc:
        log.exception("[rookie_api] /refresh error")
        return jsonify({"error": str(exc)}), 500


def register_rookie_routes(app):
    app.register_blueprint(rookie_bp)
    return app
