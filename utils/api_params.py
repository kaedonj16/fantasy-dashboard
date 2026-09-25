"""Typed query-parameter parsing for /api/* routes.

Bad values (``?season=abc``) must surface as 400 with a JSON body, not an
unhandled ValueError that becomes a 500. The app's 400 error handler renders
the JSON envelope for /api/* paths and JSON-Accept callers.
"""
from __future__ import annotations

from flask import abort, request

_MISSING = object()


def api_int(name: str, default=_MISSING, *, minimum: int | None = None,
            maximum: int | None = None) -> int | None:
    """Read an integer query parameter.

    - Missing or blank -> ``default`` (or 400 when no default is given).
    - Present but not an integer, or outside [minimum, maximum] -> 400.
    """
    raw = request.args.get(name)
    if raw is None or not str(raw).strip():
        if default is _MISSING:
            abort(400, description=f"Missing required query parameter: {name}.")
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        abort(400, description=f"Invalid '{name}' parameter: expected an integer.")
    if minimum is not None and value < minimum:
        abort(400, description=f"Invalid '{name}' parameter: must be at least {minimum}.")
    if maximum is not None and value > maximum:
        abort(400, description=f"Invalid '{name}' parameter: must be at most {maximum}.")
    return value


def api_str(name: str, default: str = "", *, max_length: int = 200) -> str:
    """Read a string query parameter, stripped and length-capped."""
    raw = request.args.get(name, default)
    text = str(raw or "").strip()
    return text[:max_length]
