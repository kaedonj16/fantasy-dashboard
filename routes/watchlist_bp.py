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
                    note       TEXT,
                    added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, player_id)
                )
                """
            )
            # Add the note column to tables created before notes existed.
            conn.execute("ALTER TABLE user_watchlist ADD COLUMN IF NOT EXISTS note TEXT")
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
            "SELECT player_id, name, position, team, note, added_at FROM user_watchlist "
            "WHERE user_key = %s ORDER BY added_at DESC",
            (user_key,),
        ).fetchall()
    # get_conn() uses psycopg's dict_row factory, so each row is a dict keyed by
    # column name - indexing by position (r[0]) raises KeyError and silently
    # emptied every read, so nothing ever synced across devices.
    out = []
    for r in rows:
        added = r.get("added_at")
        out.append({
            "player_id": str(r.get("player_id")),
            "name": r.get("name") or "",
            "position": r.get("position") or "",
            "team": r.get("team") or "",
            "note": r.get("note") or "",
            "added_at": added.isoformat() if hasattr(added, "isoformat") else added,
        })
    return out


def _upsert(conn, user_key, item):
    pid = str((item or {}).get("player_id") or "").strip()
    if not pid:
        return
    conn.execute(
        """
        INSERT INTO user_watchlist (user_key, player_id, name, position, team, note)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_key, player_id) DO UPDATE SET
            name     = COALESCE(EXCLUDED.name,     user_watchlist.name),
            position = COALESCE(EXCLUDED.position, user_watchlist.position),
            team     = COALESCE(EXCLUDED.team,     user_watchlist.team),
            note     = COALESCE(EXCLUDED.note,     user_watchlist.note)
        """,
        (user_key, pid, item.get("name"), item.get("position"), item.get("team"), item.get("note")),
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


@watchlist_bp.route("/api/watchlist", methods=["DELETE"])
@limiter.limit("60 per minute")
def api_watchlist_clear():
    """Clear the entire watchlist for the signed-in account (all players, all
    devices). Distinct from the per-player DELETE below."""
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False})
    _init_wl_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute("DELETE FROM user_watchlist WHERE user_key = %s", (uk,))
            conn.commit()
        return jsonify({"synced": True, "ok": True})
    except Exception as exc:
        logger.warning("[watchlist] clear failed: %s", exc)
        return jsonify({"synced": False})


@watchlist_bp.route("/api/watchlist/note", methods=["POST"])
@limiter.limit("120 per minute")
def api_watchlist_note():
    """Set (or clear) the personal note on a watched player. Unlike the add
    upsert, this writes the note verbatim so an empty string clears it."""
    uk = _user_key()
    if not uk:
        return jsonify({"synced": False})
    data = request.get_json(force=True) or {}
    pid = str(data.get("player_id") or "").strip()
    if not pid:
        return jsonify({"error": "player_id required"}), 400
    note = data.get("note")
    note = ("" if note is None else str(note))[:500]
    _init_wl_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO user_watchlist (user_key, player_id, name, position, team, note)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_key, player_id) DO UPDATE SET note = EXCLUDED.note
                """,
                (uk, pid, data.get("name"), data.get("position"), data.get("team"), note),
            )
            conn.commit()
        return jsonify({"synced": True, "ok": True})
    except Exception as exc:
        logger.warning("[watchlist] note failed: %s", exc)
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
