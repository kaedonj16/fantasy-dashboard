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


@misc_api_bp.route("/api/top-movers")
def api_top_movers():
    """Compact dynasty risers/fallers for the home page live-values ticker.

    Percentage change is ratio-invariant, so we skip the (heavier) displayed-value
    map and stay cheap; get_top_movers is cached by snapshot date. Fully
    defensive: any failure or an empty board just yields no items, and the
    client leaves the ticker hidden.
    """
    try:
        from data_building.player_value_history import get_top_movers
        movers = get_top_movers(days=7, limit=10, min_baseline_value=10)
    except Exception:
        return jsonify({"items": []})

    def _pct(mover):
        try:
            old_value = float(mover.get("old_value") or 0)
            delta = float(mover.get("delta") or 0)
            return (delta / old_value * 100.0) if old_value else 0.0
        except (TypeError, ValueError):
            return 0.0

    items, seen = [], set()
    for mover in list(movers.get("risers") or [])[:6] + list(movers.get("fallers") or [])[:6]:
        name = (mover.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        pct = _pct(mover)
        items.append({"name": name, "pct": round(abs(pct), 1), "up": pct >= 0})
    items.sort(key=lambda item: item["pct"], reverse=True)
    return jsonify({"items": items[:12]})


@misc_api_bp.route("/risers-fallers")
def risers_fallers_redirect():
    return redirect("/top-movers", 301)
