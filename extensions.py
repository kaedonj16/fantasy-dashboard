"""Shared Flask extensions, created without an app instance so both app.py and
the route blueprints can import them (the standard app-factory pattern). app.py
binds them to the application via ``limiter.init_app(app)``.

Before this, the rate limiter was constructed inside app.py bound to the app,
so any blueprint that wanted ``@limiter.limit(...)`` would have had to import
app.py and create a circular import. Centralizing it here lets ops/admin routes
move out of the monolith while keeping their rate limits.
"""
from __future__ import annotations

import os

_redis_url = os.environ.get("REDIS_URL", "")


def _limiter_storage_uri() -> str:
    """Storage URI for Flask-Limiter with bounded socket timeouts.

    redis-py blocks forever on a wedged connection by default
    (socket_timeout=None). The limiter runs before every rate-limited view,
    so a hung Redis would permanently wedge gunicorn request threads --
    exactly the failure mode that took the trade endpoints down. Fail fast
    instead: the storage call raises, the request 500s, and the thread is
    freed for the next request.
    """
    if not _redis_url:
        return "memory://"
    base = f"redis://{_redis_url.split('://')[-1]}"
    try:
        _sock_to = max(1, int(os.getenv("LIMITER_REDIS_SOCKET_TIMEOUT", "2")))
        _conn_to = max(1, int(os.getenv("LIMITER_REDIS_CONNECT_TIMEOUT", "2")))
    except (TypeError, ValueError):
        _sock_to, _conn_to = 2, 2
    sep = "&" if "?" in base else "?"
    return (
        f"{base}{sep}socket_timeout={_sock_to}"
        f"&socket_connect_timeout={_conn_to}"
    )


_limiter_storage = _limiter_storage_uri()

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        get_remote_address,
        default_limits=[],
        storage_uri=_limiter_storage,
    )
    LIMITER_BACKEND = "redis" if _redis_url else "memory (set REDIS_URL for multi-worker)"
except ImportError:  # flask_limiter missing → decorators become no-ops
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def decorator(f):
                return f
            return decorator

        def init_app(self, app):  # match Limiter's interface so app.py can bind unconditionally
            pass

    limiter = _NoopLimiter()
    LIMITER_BACKEND = None
