"""Custom draft board override persistence API (pro).

GET  /api/draft-board/overrides?platform=&league_id=&board_key=
POST /api/draft-board/overrides   {platform, league_id, board_key, overrides}

Overrides are personal ranking tweaks (pin / mute / tier bump) layered on the
model board. Premium only; keyed by the viewer's account (or Sleeper id) so they
follow the user across devices. See docs/custom-draft-board.md.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

draft_board_bp = Blueprint("draft_board_bp", __name__)


def has_premium_for_viewer(*a, **k):
    from app import has_premium_for_viewer as _fn
    return _fn(*a, **k)


def _owner_key() -> str:
    """Stable per-user key: the logged-in account, else the Sleeper viewer id."""
    aid = session.get("account_id")
    if aid:
        return "acct:" + str(aid)
    vid = session.get("viewer_user_id")
    if vid:
        return "sleeper:" + str(vid)
    return ""


@draft_board_bp.route("/api/draft-board/overrides", methods=["GET"])
def get_board_overrides():
    platform = (request.args.get("platform") or "sleeper").strip()
    league_id = (request.args.get("league_id") or "").strip()
    board_key = (request.args.get("board_key") or "").strip()[:40]
    season = request.args.get("season")
    # Free viewers simply have no stored board; return empty rather than an error.
    if not has_premium_for_viewer(session.get("viewer_username"), session.get("viewer_user_id"),
                                  league_id or None, platform, season):
        return jsonify({"paywall": True, "overrides": {}})
    owner = _owner_key()
    if not owner:
        return jsonify({"overrides": {}})
    from dashboard_services.draft_board import get_overrides
    return jsonify({"overrides": get_overrides(owner, platform, league_id, board_key)})


@draft_board_bp.route("/api/draft-board/overrides", methods=["POST", "PUT"])
def put_board_overrides():
    body = request.get_json(silent=True) or {}
    platform = str(body.get("platform") or "sleeper").strip()
    league_id = str(body.get("league_id") or "").strip()
    board_key = str(body.get("board_key") or "").strip()[:40]
    season = body.get("season")
    overrides = body.get("overrides")
    if not isinstance(overrides, dict):
        return jsonify({"error": "overrides must be an object"}), 400
    if len(overrides) > 1000:
        return jsonify({"error": "too many overrides"}), 400
    if not has_premium_for_viewer(session.get("viewer_username"), session.get("viewer_user_id"),
                                  league_id or None, platform, season):
        return jsonify({"paywall": True}), 403
    owner = _owner_key()
    if not owner:
        return jsonify({"error": "no owner"}), 400
    from dashboard_services.draft_board import save_overrides
    ok = save_overrides(owner, platform, league_id, board_key, overrides)
    return jsonify({"ok": bool(ok)})
