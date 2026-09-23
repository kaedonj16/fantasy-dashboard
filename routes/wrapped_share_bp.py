"""Shareable public links for Wrapped decks.

Routes:
    POST /api/wrapped/share   -- mint a public link for a deck (rate-limited)
    GET  /wrapped/<token>     -- public, no-auth deck viewer

The client POSTs the pristine overlay HTML it already fetched plus the share
payload; the server stores the *rendered* deck so the link needs no auth and
survives league deletion. Payloads live in Postgres, not in process memory.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from extensions import limiter

logger = logging.getLogger(__name__)

wrapped_share_bp = Blueprint("wrapped_share", __name__)

#: Hard cap on the stored overlay HTML (a typical deck is ~10-30KB).
MAX_OVERLAY_BYTES = 200 * 1024
#: Hard cap on the share payload JSON.
MAX_SHARE_DATA_BYTES = 50 * 1024

_VALID_KINDS = ("season", "weekly")


def _share_label(kind: str, share_data: dict) -> str:
    league = str(share_data.get("league") or "League")
    if kind == "weekly" and share_data.get("week"):
        return f"{league} — Week {share_data['week']} Wrapped"
    season = str(share_data.get("season") or "").strip()
    return f"{league} — {season} Season Wrapped".replace("  ", " ").strip(" —")


@wrapped_share_bp.route("/api/wrapped/share", methods=["POST"])
@limiter.limit("20 per hour")
def api_wrapped_share_create():
    """Mint a public share link for a Wrapped deck.

    Body: {kind: "season"|"weekly", ns, overlay_html, share_data}.
    Returns {"url": "<public url>"}.
    """
    from dashboard_services.wrapped_shares import create_wrapped_share

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    kind = str(body.get("kind") or "").strip().lower()
    ns = str(body.get("ns") or "wrapped").strip() or "wrapped"
    overlay_html = body.get("overlay_html") or ""
    share_data = body.get("share_data") or {}

    if kind not in _VALID_KINDS:
        return jsonify({"error": "kind must be 'season' or 'weekly'"}), 400
    if not isinstance(overlay_html, str) or "wrapped-slide" not in overlay_html:
        return jsonify({"error": "overlay_html missing or invalid"}), 400
    if len(overlay_html.encode("utf-8")) > MAX_OVERLAY_BYTES:
        return jsonify({"error": "Deck too large"}), 413
    if not isinstance(share_data, dict):
        return jsonify({"error": "share_data must be an object"}), 400
    import json as _json
    if len(_json.dumps(share_data).encode("utf-8")) > MAX_SHARE_DATA_BYTES:
        return jsonify({"error": "Share data too large"}), 413
    if len(ns) > 64 or not ns.replace("-", "").replace("_", "").isalnum():
        return jsonify({"error": "Invalid ns"}), 400

    label = _share_label(kind, share_data)
    try:
        token = create_wrapped_share(
            kind=kind, ns=ns, overlay_html=overlay_html,
            share_data=share_data, label=label,
        )
    except Exception:
        logger.exception("[api_wrapped_share_create] store failed")
        return jsonify({"error": "Could not create link"}), 500

    url = f"{request.host_url.rstrip('/')}/wrapped/{token}"
    return jsonify({"url": url})


@wrapped_share_bp.route("/wrapped/<token>")
def wrapped_share_view(token: str):
    """Public no-auth viewer for a shared Wrapped deck. 404 when unknown/expired."""
    from dashboard_services.pages.history_page import render_wrapped_share_page
    from dashboard_services.wrapped_shares import get_wrapped_share

    share = get_wrapped_share(token)
    if not share:
        return (
            "<!DOCTYPE html><html><head><title>Link expired</title>"
            '<meta name="robots" content="noindex, nofollow"></head>'
            "<body style='background:#0b0b16;color:#fff;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100vh;"
            "margin:0'><p>This Wrapped link has expired or doesn't exist.</p>"
            "</body></html>",
            404,
        )

    share_data = share.get("share_data") or {}
    if isinstance(share_data, str):
        import json as _json
        try:
            share_data = _json.loads(share_data)
        except Exception:
            share_data = {}

    html = render_wrapped_share_page(
        overlay_html=share.get("overlay_html") or "",
        share_data=share_data,
        label=share.get("label") or "Fantasy Wrapped",
        ns=share.get("ns") or "wrapped",
        css_url="/static/dashboard.css",
        logo_url=f"{request.host_url.rstrip('/')}/static/BR_Logo_dark.png",
    )
    from flask import Response
    return Response(html, mimetype="text/html")
