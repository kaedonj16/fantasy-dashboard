"""Same-origin redirect helpers.

Used by OAuth callbacks, Stripe return URLs, and other places that accept a
``next`` / ``return_to`` query param. Rejects protocol-relative and absolute
off-site URLs so open redirects cannot be chained through auth flows.
"""
from __future__ import annotations


def safe_local_url(value: str | None, fallback: str = "/", *, host_url: str | None = None) -> str:
    """Return ``value`` when it is a same-site path (or absolute same-host URL).

    Args:
        value: Candidate redirect target from user/query input.
        fallback: Used when ``value`` is empty or unsafe.
        host_url: Optional ``request.host_url`` (trailing slash ok). When set,
            absolute URLs matching that host are accepted; otherwise only
            root-relative paths are allowed.
    """
    value = str(value or "").strip()
    if not value:
        return fallback
    # Root-relative only — never protocol-relative ("//evil.com").
    if value.startswith("/") and not value.startswith("//"):
        return value
    if host_url:
        base = host_url.rstrip("/")
        if value.startswith(base + "/") or value == base:
            return value
    return fallback
