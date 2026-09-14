"""Per-user, per-league waiver preferences — currently the "Do not drop" list.

Mirrors the account-scoped persistence pattern in routes/ui_prefs_bp.py (a small
JSONB table keyed by a stable user key), scoped additionally to the league so a
"never cut this player" preference applies to the right roster (#3). Reads no-op
cleanly for guests, so the waiver surfaces can always call them.

    GET  /api/waiver-do-not-drop?platform&league_id       -> {pids: [...]}
    POST /api/waiver-do-not-drop  body: {platform, league_id, player_id, action}
         action: "add" (default) | "remove"

The waiver-candidates route reads the same store via ``load_do_not_drop`` so a
protected player is never suggested as a drop.
"""
from __future__ import annotations

import json
import logging

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

waiver_prefs_bp = Blueprint("waiver_prefs", __name__)

_TABLE_READY = False


def _init_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS waiver_do_not_drop (
                    scope_key   TEXT PRIMARY KEY,
                    pids        JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.warning("[waiver-prefs] table init failed: %s", exc)


def _scope_key(account_id, platform: str, league_id: str) -> "str | None":
    if account_id in (None, ""):
        return None
    return f"acct:{str(account_id).strip()}:{platform}:{league_id}"


def _load(scope_key: str) -> list:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT pids FROM waiver_do_not_drop WHERE scope_key = %s",
            (scope_key,),
        ).fetchone()
    if not row:
        return []
    pids = row["pids"] if isinstance(row, dict) else row[0]
    if isinstance(pids, str):
        try:
            pids = json.loads(pids)
        except Exception:
            return []
    return [str(p) for p in pids] if isinstance(pids, list) else []


def _save(scope_key: str, pids: list) -> list:
    from dashboard_services.db import get_conn
    # De-dupe, keep insertion order, cap so a runaway client can't bloat the row.
    seen, clean = set(), []
    for p in pids:
        p = str(p)
        if p and p not in seen:
            seen.add(p)
            clean.append(p)
        if len(clean) >= 200:
            break
    payload = json.dumps(clean)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO waiver_do_not_drop (scope_key, pids, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (scope_key) DO UPDATE
              SET pids = EXCLUDED.pids, updated_at = NOW()
            """,
            (scope_key, payload),
        )
        conn.commit()
    return clean


def load_do_not_drop(account_id, platform: str, league_id: str) -> set:
    """Set of protected player ids for this account + league. Best-effort: any
    failure (guest, no table yet, DB down) returns an empty set so the caller
    degrades to unprotected rather than erroring."""
    try:
        key = _scope_key(account_id, str(platform or "sleeper"), str(league_id or ""))
        if not key:
            return set()
        _init_table()
        return set(_load(key))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return set()


@waiver_prefs_bp.route("/api/waiver-do-not-drop", methods=["GET", "POST"])
def api_waiver_do_not_drop():
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    account_id = session.get("account_id")
    key = _scope_key(account_id, platform, league_id)
    if not key:
        # Guests keep the list client-side; server no-ops cleanly.
        return jsonify({"ok": True, "synced": False, "pids": []})

    _init_table()
    try:
        if request.method == "GET":
            return jsonify({"ok": True, "synced": True, "pids": _load(key)})

        body = request.get_json(silent=True) or {}
        pid = str(body.get("player_id") or "").strip()
        action = str(body.get("action") or "add").strip().lower()
        if not pid:
            return jsonify({"ok": False, "error": "player_id required"}), 400
        pids = _load(key)
        if action == "remove":
            pids = [p for p in pids if p != pid]
        else:
            if pid not in pids:
                pids.append(pid)
        saved = _save(key, pids)
        return jsonify({"ok": True, "synced": True, "pids": saved})
    except Exception as exc:
        logger.warning("[waiver-prefs] error: %s", exc)
        return jsonify({"ok": False, "synced": False, "pids": []}), 500
