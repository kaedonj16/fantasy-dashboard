"""Admin authentication for data-management endpoints.

Some endpoints mutate shared data (rookie prospects, player values, pipeline
runs) and must not be callable anonymously. Admin status is granted when the
request carries the configured ``ADMIN_KEY`` — via the ``X-Admin-Key`` header,
an ``admin_key`` field in the JSON body / form, or an ``admin_key`` query
parameter — or when the server-side session was already marked admin by a prior
successful key check (so an operator can authenticate once by visiting a page
with ``?admin_key=...`` and then use the controls normally).

If ``ADMIN_KEY`` is not configured the admin surface is disabled (fail closed).
"""
from __future__ import annotations

import hmac
import logging
import os
from functools import wraps

from flask import jsonify, request, session

log = logging.getLogger(__name__)


def _configured_key() -> str:
    return os.environ.get("ADMIN_KEY", "") or ""


def _provided_key() -> str:
    key = request.headers.get("X-Admin-Key")
    if not key and request.is_json:
        data = request.get_json(silent=True) or {}
        key = data.get("admin_key")
    if not key:
        key = request.values.get("admin_key")
    return key or ""


def is_admin() -> bool:
    """Whether the current request is authenticated as an admin.

    A valid key (header/body/query) also marks the session admin so subsequent
    requests in the same browser session are recognized without re-sending it.
    """
    configured = _configured_key()
    if not configured:
        return False
    if session.get("is_admin"):
        return True
    provided = _provided_key()
    if provided and hmac.compare_digest(provided, configured):
        session.permanent = True
        session["is_admin"] = True
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
