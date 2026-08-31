"""Account-scoped UI preferences + lightweight product events.

Prefs (local-first on the client; mirrored here when signed in):
    GET  /api/ui-prefs
    PUT  /api/ui-prefs   body: {prefs: {...}}  (shallow merge)

Events (fire-and-forget analytics; no PII expected):
    POST /api/events     body: {event: str, props?: object}

Both no-op cleanly for guests so the client can always call them.
"""
from __future__ import annotations

import json
import logging

from flask import Blueprint, jsonify, request, session

from extensions import limiter

logger = logging.getLogger(__name__)

ui_prefs_bp = Blueprint("ui_prefs", __name__)

_TABLE_READY = False
_ALLOWED_PREF_KEYS = frozenset({
    "site_tour_done",
    "sub_welcome_done",
})
_ALLOWED_EVENTS = frozenset({
    "site_tour_start",
    "site_tour_step",
    "site_tour_complete",
    "site_tour_skip",
    "site_tour_later",
    "sub_welcome_show",
    "sub_welcome_dismiss",
    "sub_welcome_cta",
    "sub_welcome_start_tour",
    "home_league_selected",
    "home_create_account_nudge",
})


def _user_key():
    account_id = session.get("account_id")
    if account_id not in (None, ""):
        return "acct:" + str(account_id).strip()
    return None


def _init_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_ui_prefs (
                    user_key    TEXT PRIMARY KEY,
                    prefs       JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.warning("[ui-prefs] table init failed: %s", exc)


def _load_prefs(user_key: str) -> dict:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT prefs FROM user_ui_prefs WHERE user_key = %s",
            (user_key,),
        ).fetchone()
    if not row:
        return {}
    prefs = row["prefs"] if isinstance(row, dict) else row[0]
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except Exception:
            return {}
    return prefs if isinstance(prefs, dict) else {}


def _save_prefs(user_key: str, prefs: dict) -> dict:
    from dashboard_services.db import get_conn
    payload = json.dumps(prefs)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_ui_prefs (user_key, prefs, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (user_key) DO UPDATE
              SET prefs = EXCLUDED.prefs, updated_at = NOW()
            """,
            (user_key, payload),
        )
        conn.commit()
    return prefs


def _sanitize_prefs(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if key not in _ALLOWED_PREF_KEYS:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif value in (0, 1, "0", "1", "true", "false"):
            out[key] = value in (1, "1", "true", True)
    return out


@ui_prefs_bp.route("/api/ui-prefs", methods=["GET", "PUT"])
@limiter.limit("60/minute")
def api_ui_prefs():
    _init_table()
    user_key = _user_key()
    if not user_key:
        return jsonify({"ok": True, "synced": False, "prefs": {}})

    try:
        if request.method == "GET":
            return jsonify({"ok": True, "synced": True, "prefs": _load_prefs(user_key)})

        body = request.get_json(silent=True) or {}
        incoming = _sanitize_prefs(body.get("prefs") if "prefs" in body else body)
        if not incoming:
            return jsonify({"ok": False, "error": "No valid prefs."}), 400
        merged = _load_prefs(user_key)
        merged.update(incoming)
        _save_prefs(user_key, merged)
        return jsonify({"ok": True, "synced": True, "prefs": merged})
    except Exception as exc:
        logger.warning("[ui-prefs] error: %s", exc)
        return jsonify({"ok": False, "synced": False, "prefs": {}}), 500


@ui_prefs_bp.route("/api/events", methods=["POST"])
@limiter.limit("120/minute")
def api_events():
    """Record a product analytics event (structured log only)."""
    body = request.get_json(silent=True) or {}
    event = str(body.get("event") or "").strip()
    if event not in _ALLOWED_EVENTS:
        return jsonify({"ok": False, "error": "Unknown event."}), 400
    props = body.get("props") if isinstance(body.get("props"), dict) else {}
    # Keep props small and non-PII: coerce values to short strings/numbers/bools.
    clean = {}
    for key, value in list(props.items())[:12]:
        k = str(key)[:40]
        if isinstance(value, bool) or value is None:
            clean[k] = value
        elif isinstance(value, (int, float)):
            clean[k] = value
        else:
            clean[k] = str(value)[:80]
    account = bool(session.get("account_id"))
    logger.info(
        "[event] name=%s account=%s props=%s",
        event, account, json.dumps(clean, separators=(",", ":")),
    )
    return jsonify({"ok": True})
