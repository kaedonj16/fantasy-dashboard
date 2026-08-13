"""Admin health/monitoring API endpoints.

Routes:
    /api/health/errors   - warning/error counts since process start
    /api/health/timing   - per-endpoint request timing since process start

Both require the X-Admin-Secret header. Extracted from app.py; depends only on
extensions.limiter + utils monitors, no app.py internals.
"""
from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from extensions import limiter

health_bp = Blueprint("health", __name__)


def _forbidden_unless_admin():
    """Return a 403 response tuple when the admin secret is missing/wrong, else None."""
    secret = request.headers.get("X-Admin-Secret", "")
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret or secret != admin_secret:
        return jsonify({"error": "Forbidden"}), 403
    return None


@health_bp.route("/api/health/errors")
@limiter.limit("30 per minute")
def api_health_errors():
    """Warning/error counts since process start, most frequent first.
    Requires X-Admin-Secret header. Pass ?reset=1 to clear the counters."""
    denied = _forbidden_unless_admin()
    if denied:
        return denied
    from utils import error_monitor
    if request.args.get("reset") == "1":
        error_monitor.reset()
        return jsonify({"ok": True, "reset": True})
    try:
        limit = int(request.args.get("limit") or 100)
    except ValueError:
        limit = 100
    return jsonify(error_monitor.snapshot(limit=limit))


@health_bp.route("/api/health/timing")
@limiter.limit("30 per minute")
def api_health_timing():
    """Per-endpoint request timing since process start, slowest first.
    Requires X-Admin-Secret. ?reset=1 clears; ?sort=avg|max|slow|total; ?limit=N."""
    denied = _forbidden_unless_admin()
    if denied:
        return denied
    from utils import perf_monitor
    if request.args.get("reset") == "1":
        perf_monitor.reset()
        return jsonify({"ok": True, "reset": True})
    try:
        limit = int(request.args.get("limit") or 100)
    except ValueError:
        limit = 100
    return jsonify(perf_monitor.snapshot(limit=limit, sort=request.args.get("sort", "total")))
