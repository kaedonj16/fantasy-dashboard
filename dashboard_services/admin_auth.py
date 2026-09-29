"""Admin authentication for data-management endpoints.

Some endpoints mutate shared data (rookie prospects, player values, pipeline
runs) and must not be callable anonymously. Admin status is granted when the
request carries the configured ``ADMIN_KEY`` - via the ``X-Admin-Key`` header,
an ``admin_key`` field in the JSON body / form, or an ``admin_key`` query
parameter - or when the server-side session was already marked admin by a prior
successful key OR password check (so an operator can authenticate once by
visiting a page with ``?admin_key=...`` or by logging in at ``/admin/login``
and then use the controls normally).

Password login uses the ``ADMIN_PASSWORD`` env var and is meant for browser
use; the key flow remains for scripts and API-style access. If neither
``ADMIN_KEY`` nor ``ADMIN_PASSWORD`` is configured the admin surface is
disabled (fail closed).
"""
from __future__ import annotations

import hmac
import logging
import os
from functools import wraps

from flask import jsonify, request, session

log = logging.getLogger(__name__)

#: Session key marking a browser session as admin-authenticated.
ADMIN_SESSION_KEY = "is_admin"


def _configured_key() -> str:
    return os.environ.get("ADMIN_KEY", "") or ""


def _configured_password() -> str:
    return os.environ.get("ADMIN_PASSWORD", "") or ""


def _provided_key() -> str:
    key = request.headers.get("X-Admin-Key")
    if not key and request.is_json:
        data = request.get_json(silent=True) or {}
        key = data.get("admin_key")
    if not key:
        key = request.values.get("admin_key")
    return key or ""


def mark_admin_session() -> None:
    """Mark the current browser session as admin-authenticated."""
    session.permanent = True
    session[ADMIN_SESSION_KEY] = True


def verify_admin_password(provided: str) -> bool:
    """Check a submitted password against ADMIN_PASSWORD (constant time).

    Returns False when no password is configured (fail closed).
    """
    configured = _configured_password()
    if not configured or not provided:
        return False
    return hmac.compare_digest(provided, configured)


def is_admin() -> bool:
    """Whether the current request is authenticated as an admin.

    A session flag set by a prior successful key or password check is honored
    on its own. Otherwise a valid ADMIN_KEY (header/body/query) marks the
    session admin and returns True. Fail closed when nothing is configured.
    """
    if session.get(ADMIN_SESSION_KEY):
        return True
    configured = _configured_key()
    if not configured:
        return False
    provided = _provided_key()
    if provided and hmac.compare_digest(provided, configured):
        mark_admin_session()
        return True
    return False


def admin_required(fn):
    """Restrict a Flask route to authenticated admins."""
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if is_admin():
            return fn(*args, **kwargs)
        if not _configured_key():
            log.warning("admin_required: ADMIN_KEY not configured; denying %s", request.path)
            return jsonify({"error": "Admin features are not configured"}), 503
        return jsonify({"error": "Admin access required"}), 403

    return _wrapper
