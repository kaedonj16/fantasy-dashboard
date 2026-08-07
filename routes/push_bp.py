"""Web push notification subsystem: VAPID keys, subscription CRUD, per-league
preferences, admin broadcast, and the cron notification hook.

Routes:
    /api/push/vapid-public-key
    /api/push/subscribe        (POST)
    /api/push/unsubscribe      (POST)
    /api/push/leagues
    /api/push/preferences      (GET, PUT)
    /api/push/broadcast        (POST, admin)
    /api/cron/notifications    (POST, admin)

Extracted from app.py. Depends on extensions.limiter + dashboard_services/utils
(DB, pywebpush, push_notifications) - no app.py internals. _push_broadcast is
imported by app.py's changelog startup notifier.
"""
from __future__ import annotations

import logging
import os
import threading

from flask import Blueprint, jsonify, request

from extensions import limiter

logger = logging.getLogger(__name__)

push_bp = Blueprint("push", __name__)


# ── Push notifications ─────────────────────────────────────────────────────────

_PUSH_TABLE_INIT = False
_VAPID_KEYS: dict | None = None


def _init_push_table():
    global _PUSH_TABLE_INIT
    if _PUSH_TABLE_INIT:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id         SERIAL PRIMARY KEY,
                    endpoint   TEXT UNIQUE NOT NULL,
                    p256dh     TEXT NOT NULL,
                    auth       TEXT NOT NULL,
                    league_id  TEXT,
                    platform   TEXT DEFAULT 'sleeper',
                    owner_id   TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Migrate existing tables that predate the league/owner/prefs columns
            for col, defn in [("league_id", "TEXT"), ("platform", "TEXT DEFAULT 'sleeper'"), ("owner_id", "TEXT"), ("prefs", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE push_subscriptions ADD COLUMN IF NOT EXISTS {col} {defn}")
                except Exception:
                    logger.debug("suppressed exception", exc_info=True)
            conn.commit()
            # Multi-league support: a device (endpoint) can subscribe to several
            # leagues — one row per (endpoint, league_id). Replace the old
            # endpoint-unique constraint with a composite unique index. All
            # league_ids are normalized to '' (never NULL) so the index is exact.
            try:
                conn.execute("UPDATE push_subscriptions SET league_id = '' WHERE league_id IS NULL")
                conn.execute("""
                    DELETE FROM push_subscriptions a USING push_subscriptions b
                    WHERE a.id < b.id AND a.endpoint = b.endpoint
                      AND COALESCE(a.league_id, '') = COALESCE(b.league_id, '')
                """)
                conn.execute("ALTER TABLE push_subscriptions DROP CONSTRAINT IF EXISTS push_subscriptions_endpoint_key")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS push_sub_endpoint_league_idx ON push_subscriptions (endpoint, league_id)")
                conn.commit()
            except Exception as _mig_exc:
                logger.warning("[push] composite-key migration skipped: %s", _mig_exc)
        _PUSH_TABLE_INIT = True
    except Exception as exc:
        logger.warning("[push] table init failed: %s", exc)


def _get_vapid_keys() -> dict | None:
    global _VAPID_KEYS
    if _VAPID_KEYS:
        return _VAPID_KEYS
    pub  = os.environ.get("VAPID_PUBLIC_KEY", "").strip()
    priv = os.environ.get("VAPID_PRIVATE_KEY", "").replace("\\n", "\n").strip()
    if pub and priv:
        from utils.push_notifications import _normalize_vapid_private_key
        _VAPID_KEYS = {"public": pub, "private": _normalize_vapid_private_key(priv)}
        return _VAPID_KEYS
    # Generate ephemeral keys for this session; set env vars to persist
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        import base64 as _b64
        priv_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        pub_raw  = priv_key.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        priv_pem = priv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ).decode()
        pub_b64  = _b64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()
        logger.info(
            "[vapid] Generated ephemeral VAPID keys. Add to Render env vars to persist:\n"
            "  VAPID_PUBLIC_KEY=%s\n  VAPID_PRIVATE_KEY=%s",
            pub_b64,
            priv_pem.replace("\n", "\\n"),
        )
        _VAPID_KEYS = {"public": pub_b64, "private": priv_pem}
        return _VAPID_KEYS
    except Exception as exc:
        logger.warning("[vapid] key generation failed: %s", exc)
        return None


@push_bp.route("/api/push/vapid-public-key")
def api_push_vapid_public_key():
    keys = _get_vapid_keys()
    if not keys:
        return jsonify({"error": "Push not configured"}), 503
    return jsonify({"publicKey": keys["public"]})


@push_bp.route("/api/push/subscribe", methods=["POST"])
@limiter.limit("30 per minute")
def api_push_subscribe():
    data      = request.get_json(force=True) or {}
    endpoint  = data.get("endpoint", "").strip()
    p256dh    = (data.get("keys") or {}).get("p256dh", "").strip()
    auth      = (data.get("keys") or {}).get("auth",   "").strip()
    platform  = (data.get("platform")  or "sleeper").strip()
    owner_id  = (data.get("owner_id")  or "").strip() or None
    # Accept either a single league_id or a league_ids[] array (register the
    # device for every league at once — the default-to-all subscribe path).
    raw_leagues = data.get("league_ids")
    if not isinstance(raw_leagues, list) or not raw_leagues:
        raw_leagues = [data.get("league_id")]
    leagues = list(dict.fromkeys((str(l).strip() if l else "") for l in raw_leagues))
    if not (endpoint and p256dh and auth):
        return jsonify({"error": "Invalid subscription"}), 400
    _init_push_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            for lid in leagues:
                conn.execute(
                    """
                    INSERT INTO push_subscriptions (endpoint, p256dh, auth, league_id, platform, owner_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (endpoint, league_id) DO UPDATE
                        SET p256dh    = EXCLUDED.p256dh,
                            auth      = EXCLUDED.auth,
                            platform  = COALESCE(EXCLUDED.platform,  push_subscriptions.platform),
                            owner_id  = COALESCE(EXCLUDED.owner_id,  push_subscriptions.owner_id)
                    """,
                    (endpoint, p256dh, auth, lid, platform, owner_id),
                )
            # Keep notification-type prefs consistent across all of this device's
            # league rows (prefs are a device-level choice, not per-league).
            conn.execute(
                """
                UPDATE push_subscriptions SET prefs = (
                    SELECT prefs FROM push_subscriptions p2
                    WHERE p2.endpoint = %s AND p2.prefs IS NOT NULL LIMIT 1
                ) WHERE endpoint = %s AND prefs IS NULL
                """,
                (endpoint, endpoint),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("[push] subscribe error: %s", exc)
        return jsonify({"error": "Could not save subscription"}), 500
    return jsonify({"ok": True})


@push_bp.route("/api/push/unsubscribe", methods=["POST"])
@limiter.limit("30 per minute")
def api_push_unsubscribe():
    data     = request.get_json(force=True) or {}
    endpoint = data.get("endpoint", "").strip()
    # When a league_id is supplied, only that league is toggled off for this
    # device; otherwise the whole device is unsubscribed (all leagues).
    league_id = (data.get("league_id") or "").strip()
    if not endpoint:
        return jsonify({"error": "Missing endpoint"}), 400
    _init_push_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            if league_id:
                conn.execute(
                    "DELETE FROM push_subscriptions WHERE endpoint = %s AND league_id = %s",
                    (endpoint, league_id),
                )
            else:
                conn.execute(
                    "DELETE FROM push_subscriptions WHERE endpoint = %s", (endpoint,)
                )
            conn.commit()
    except Exception as exc:
        logger.warning("[push] unsubscribe error: %s", exc)
    return jsonify({"ok": True})


@push_bp.route("/api/push/leagues")
@limiter.limit("60 per minute")
def api_push_leagues():
    """Return the league_ids this device (endpoint) is currently subscribed to,
    so the notification settings modal can show per-league toggle state."""
    endpoint = request.args.get("endpoint", "").strip()
    if not endpoint:
        return jsonify({"league_ids": []})
    _init_push_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT league_id FROM push_subscriptions "
                "WHERE endpoint = %s AND COALESCE(league_id, '') <> ''",
                (endpoint,),
            ).fetchall()
        return jsonify({"league_ids": [r["league_id"] for r in rows]})
    except Exception as exc:
        logger.warning("[push] leagues get error: %s", exc)
        return jsonify({"league_ids": []})


@push_bp.route("/api/push/preferences", methods=["GET", "PUT"])
@limiter.limit("60 per minute")
def api_push_preferences():
    import json as _json
    _init_push_table()
    if request.method == "GET":
        endpoint = request.args.get("endpoint", "").strip()
        if not endpoint:
            return jsonify({"error": "Missing endpoint"}), 400
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT prefs FROM push_subscriptions WHERE endpoint = %s", (endpoint,)
                ).fetchone()
            prefs_raw = (row["prefs"] if row else None) or "{}"
            return jsonify({"prefs": _json.loads(prefs_raw)})
        except Exception as exc:
            logger.warning("[push] preferences get error: %s", exc)
            return jsonify({"prefs": {}})
    else:  # PUT
        data = request.get_json(force=True) or {}
        endpoint = data.get("endpoint", "").strip()
        prefs = data.get("prefs") or {}
        if not endpoint:
            return jsonify({"error": "Missing endpoint"}), 400
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                conn.execute(
                    "UPDATE push_subscriptions SET prefs = %s WHERE endpoint = %s",
                    (_json.dumps(prefs), endpoint),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("[push] preferences put error: %s", exc)
            return jsonify({"error": str(exc)}), 500
        return jsonify({"ok": True})


@push_bp.route("/api/push/broadcast", methods=["POST"])
@limiter.limit("10 per minute")
def api_push_broadcast():
    """Send a push to all subscribers. Requires X-Admin-Secret header."""
    secret = request.headers.get("X-Admin-Secret", "")
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret or secret != admin_secret:
        return jsonify({"error": "Forbidden"}), 403
    data  = request.get_json(force=True) or {}
    title = data.get("title", "BR Fantasy Update")
    body  = data.get("body",  "Your weekly dynasty risers and fallers are ready!")
    url   = data.get("url",   "/top-movers")
    tag   = data.get("tag",   "weekly-update")
    # Sending is a per-device network round-trip, so a real audience takes tens of
    # seconds — longer than an HTTP client (or the gunicorn worker) will wait,
    # which surfaces as a timeout and can kill the worker mid-blast. Fail fast on
    # a misconfigured server, then run the send loop on a background thread and
    # acknowledge immediately; the loop keeps going after the response returns.
    if not _get_vapid_keys():
        return jsonify({"error": "Push not configured"}), 503
    logger.info("[push] broadcast queued: title=%r tag=%r url=%r", title, tag, url)
    threading.Thread(
        target=_push_broadcast,
        kwargs={"title": title, "body": body, "url": url, "tag": tag},
        daemon=True,
    ).start()
    return jsonify({"ok": True, "queued": True}), 202


@push_bp.route("/api/cron/notifications", methods=["POST"])
@limiter.limit("60 per minute")
def api_cron_notifications():
    """Cron hook for push notifications. Pass type='hourly' or type='daily'.
    Call hourly for lineup lock; call once at your preferred daytime hour for daily alerts."""
    secret       = request.headers.get("X-Admin-Secret", "")
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret or secret != admin_secret:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(force=True) or {}
    kind = data.get("type", "hourly")
    try:
        from utils.push_notifications import run_hourly, run_all_daily
        if kind == "daily":
            run_all_daily()
        else:
            run_hourly()
    except Exception as exc:
        logger.warning("[cron/notifications] failed: %s", exc)
    return jsonify({"ok": True})


def _push_broadcast(title: str, body: str, url: str = "/", tag: str = "update"):
    """Send a push to every subscribed device. Returns a (body_dict, status)
    tuple - deliberately NOT a Flask response, so it is safe to call outside a
    request/app context (e.g. the changelog startup notifier). The HTTP endpoint
    wraps the dict in jsonify."""
    import json as _json
    keys = _get_vapid_keys()
    if not keys:
        return {"error": "Push not configured"}, 503
    _init_push_table()
    try:
        from pywebpush import webpush, WebPushException
        from dashboard_services.db import get_conn
        with get_conn() as _pconn:
            # One send per person, not per subscription row. A user accumulates
            # several endpoints on the same phone (Safari vs installed PWA,
            # re-granting notifications, etc.) plus a row per league — all of
            # which would each receive a copy. Collapse by owner_id (the
            # signed-in user) so a global broadcast lands once; anonymous rows
            # with no owner still de-dupe per endpoint. Newest row (id DESC) wins,
            # so we send to the user's most recent subscription.
            rows = _pconn.execute(
                "SELECT DISTINCT ON (dedupe_key) endpoint, p256dh, auth FROM ("
                "  SELECT endpoint, p256dh, auth, id, "
                "         COALESCE(NULLIF(owner_id, ''), 'ep:' || endpoint) AS dedupe_key "
                "  FROM push_subscriptions"
                ") s ORDER BY dedupe_key, id DESC"
            ).fetchall()
    except Exception as exc:
        logger.warning("[push] broadcast query failed: %s", exc)
        return {"error": "DB error"}, 500

    payload       = _json.dumps({"title": title, "body": body, "url": url, "tag": tag})
    try:
        from utils.push_notifications import _make_vapid
        vapid_obj = _make_vapid(keys["private"])
    except Exception as exc:
        logger.warning("[push] Could not build Vapid object: %s", exc)
        return {"error": "VAPID key error"}, 500
    sent = failed = 0
    stale         = []

    for row in rows:
        ep, p256dh, auth = row["endpoint"], row["p256dh"], row["auth"]
        try:
            webpush(
                subscription_info={"endpoint": ep, "keys": {"p256dh": p256dh, "auth": auth}},
                data=payload,
                vapid_private_key=vapid_obj,
                vapid_claims={"sub": "mailto:admin@brfantasy.com"},
            )
            sent += 1
        except WebPushException as exc:
            # A 404/410 means the subscription is permanently dead and must be
            # pruned so we stop retrying it forever. Note: requests.Response is
            # falsy for non-2xx (its __bool__ returns .ok), so `if exc.response`
            # skips real 410s - check `is not None` and fall back to parsing the
            # status out of the message.
            resp = getattr(exc, "response", None)
            status = resp.status_code if resp is not None else None
            if status is None:
                import re as _re
                m = _re.search(r"\b([45]\d\d)\b", str(exc))
                status = int(m.group(1)) if m else 0
            if status in (404, 410):
                stale.append(ep)
            else:
                logger.warning("[push] send failed %s: %s", ep[:50], exc)
            failed += 1
        except Exception as exc:
            logger.warning("[push] unexpected: %s", exc)
            failed += 1

    if stale:
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                conn.execute(
                    "DELETE FROM push_subscriptions WHERE endpoint = ANY(%s)", (stale,)
                )
                conn.commit()
            logger.info("[push] pruned %d dead subscription(s)", len(stale))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

    # Log the outcome so a background (fire-and-forget) broadcast still leaves a
    # visible record in the app logs — the HTTP caller only gets a 202 and never
    # sees these totals.
    logger.info("[push] broadcast complete: sent=%d failed=%d pruned=%d", sent, failed, len(stale))
    return {"ok": True, "sent": sent, "failed": failed}, 200


