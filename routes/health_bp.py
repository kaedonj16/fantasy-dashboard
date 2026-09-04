"""Admin health/monitoring API endpoints.

Routes:
    /healthz/version      - deploy smoke: bundle hashes + process start time
    /api/health/errors    - warning/error counts since process start
    /api/health/timing    - per-endpoint request timing since process start
    /api/health/pipeline  - last cron-step timestamps from pipeline_health.json

Admin routes require the X-Admin-Secret header. ``/healthz/version`` is public
(deploy checks). Extracted from app.py; version hashes are lazy-imported from
app so this module stays free of circular imports at registration time.
"""
from __future__ import annotations

import json
import os

from flask import Blueprint, jsonify, request

from extensions import limiter

health_bp = Blueprint("health", __name__)


def _forbidden_unless_admin():
    """Return a 403 response tuple when the admin secret is missing/wrong, else None."""
    import hmac
    secret = request.headers.get("X-Admin-Secret", "") or ""
    admin_secret = os.environ.get("ADMIN_SECRET", "") or ""
    if not admin_secret or not secret or not hmac.compare_digest(secret, admin_secret):
        return jsonify({"error": "Forbidden"}), 403
    return None


@health_bp.route("/healthz/version")
def healthz_version():
    """Deploy smoke-check: content hashes of the bundles this process is
    actually serving, plus the git SHA when available. After a deploy, hit this
    and confirm the hashes changed / match the built files — so "is it live yet?"
    is a one-request answer instead of guessing at a stale cache. Cache-busting
    headers so an intermediary can never hand back a previous deploy's answer."""
    # Lazy import: app registers this blueprint at import time; version stamps
    # live as module-level constants on app and are safe to read per-request.
    import app as _app

    resp = jsonify({
        "app_js": _app._APP_JS_V,
        "public_js": _app._PUBLIC_JS_V,
        "rankings_js": _app._RANKINGS_JS_V,
        "teams_js": _app._TEAMS_JS_V,
        "redzone_js": _app._REDZONE_JS_V,
        "css": _app._CSS_V,
        "git_sha": os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_SHA") or "",
        "started_at": _app._PROCESS_STARTED_AT,
    })
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


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


@health_bp.route("/api/health/pipeline")
@limiter.limit("30 per minute")
def api_health_pipeline():
    """Last-success / last-status timestamps for each cron_daily step.

    Reads ``CACHE_DIR/pipeline_health.json`` written by
    ``cron_daily.record_pipeline_health``. Missing file → empty object.
    Requires X-Admin-Secret.
    """
    denied = _forbidden_unless_admin()
    if denied:
        return denied
    from utils.paths import CACHE_DIR
    dest = CACHE_DIR / "pipeline_health.json"
    if not dest.exists():
        return jsonify({})
    try:
        data = json.loads(dest.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return jsonify(data)
