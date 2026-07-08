"""Server-synced player watchlist (per signed-in user).

The watchlist is device-local (localStorage) by default; when the viewer is
signed in (session has a Sleeper user id) it also syncs to the account so it
follows them across devices. The client is local-first and write-through:
localStorage updates immediately, and these endpoints mirror the change.

Routes (all no-op with synced:false when not signed in, so the client falls
back to local-only cleanly):
    GET    /api/watchlist            - the account's watched players
    POST   /api/watchlist            - add/update one
    DELETE /api/watchlist/<pid>      - remove one
    POST   /api/watchlist/merge      - union the client's local list into the
                                       account and return the merged result

Depends on extensions.limiter + dashboard_services.db only - no app.py internals.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, session

from extensions import limiter

logger = logging.getLogger(__name__)

watchlist_bp = Blueprint("watchlist", __name__)

_WL_TABLE_INIT = False


def _init_wl_table():
    global _WL_TABLE_INIT
    if _WL_TABLE_INIT:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_watchlist (
                    user_key   TEXT NOT NULL,
                    player_id  TEXT NOT NULL,
                    name       TEXT,
                    position   TEXT,
                    team       TEXT,
                    added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, player_id)
                )
                """
            )
            conn.commit()
        _WL_TABLE_INIT = True
    except Exception as exc:
        logger.warning("[watchlist] table init failed: %s", exc)


def _user_key():
    """Stable per-account key, or None when not signed in."""
    return str(session.get("viewer_user_id") or "").strip() or None


def _rows_for(user_key):
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, name, position, team, added_at FROM user_watchlist "
            "WHERE user_key = %s ORDER BY added_at DESC",
            (user_key,),
        ).fetchall()
    return [
        {
            "player_id": str(r[0]),
            "name": r[1] or "",
            "position": r[2] or "",
            "team": r[3] or "",
            "added_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


def _upsert(conn, user_key, item):
    pid = str((item or {}).get("player_id") or "").strip()
    if not pid:
        return
    conn.execute(
        """
        INSERT INTO user_watchlist (user_key, player_id, name, position, team)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_key, player_id) DO UPDATE SET
            name     = COALESCE(EXCLUDED.name,     user_watchlist.name),
            position = COALESCE(EXCLUDED.position, user_watchlist.position),
            team     = COALESCE(EXCLUDED.team,     user_watchlist.team)
        """,
        (user_key, pid, item.get("name"), item.get("position"), item.get("team")),
    )


@watchlist_bp.route("/api/watchlist", methods=["GET"])
def api_watchlist_get():
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False, "items": []})
    _init_wl_table()
    try:
        return jsonify({"synced": True, "items": _rows_for(uk)})
    except Exception as exc:
        logger.warning("[watchlist] get failed: %s", exc)
        return jsonify({"synced": False, "items": []})


@watchlist_bp.route("/api/watchlist", methods=["POST"])
@limiter.limit("120 per minute")
def api_watchlist_add():
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False})
    data = request.get_json(force=True) or {}
    if not str(data.get("player_id") or "").strip():
        return jsonify({"error": "player_id required"}), 400
    _init_wl_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            _upsert(conn, uk, data)
            conn.commit()
        return jsonify({"synced": True, "ok": True})
    except Exception as exc:
        logger.warning("[watchlist] add failed: %s", exc)
        return jsonify({"synced": False})


@watchlist_bp.route("/api/watchlist/<player_id>", methods=["DELETE"])
@limiter.limit("120 per minute")
def api_watchlist_remove(player_id):
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False})
    _init_wl_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                "DELETE FROM user_watchlist WHERE user_key = %s AND player_id = %s",
                (uk, str(player_id)),
            )
            conn.commit()
        return jsonify({"synced": True, "ok": True})
    except Exception as exc:
        logger.warning("[watchlist] remove failed: %s", exc)
        return jsonify({"synced": False})


@watchlist_bp.route("/api/watchlist/merge", methods=["POST"])
@limiter.limit("60 per minute")
def api_watchlist_merge():
    """Union the client's local items into the account, return the merged list.
    Called on load/sign-in so neither the device's local list nor the account's
    synced list is lost."""
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False, "items": []})
    data = request.get_json(force=True) or {}
    items = data.get("items") or []
    _init_wl_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            for it in items[:500]:
                _upsert(conn, uk, it)
            conn.commit()
        return jsonify({"synced": True, "items": _rows_for(uk)})
    except Exception as exc:
        logger.warning("[watchlist] merge failed: %s", exc)
        return jsonify({"synced": False, "items": []})
