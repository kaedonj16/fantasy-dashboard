from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator

import psycopg

logger = logging.getLogger(__name__)
from psycopg.rows import dict_row
from psycopg.types.json import set_json_dumps

# Connection pooling. A fresh psycopg.connect() per call pays TCP+TLS+auth +
# an isolation-level round-trip to the remote Postgres on every query; with a
# pool we reuse warm connections. Optional import so the app still runs (falling
# back to direct connections) if psycopg_pool isn't installed.
try:
    from psycopg_pool import ConnectionPool, PoolTimeout
    _POOL_AVAILABLE = True
except Exception:  # pragma: no cover - dependency missing
    ConnectionPool = None  # type: ignore
    PoolTimeout = Exception  # type: ignore
    _POOL_AVAILABLE = False

_pool = None
_pool_pid = None
_pool_lock = threading.Lock()


def _json_default(obj):
    """
    Custom JSON serializer for types that stdlib json.dumps can't handle.

    psycopg returns PostgreSQL NUMERIC columns as decimal.Decimal, and
    DATE/TIMESTAMP columns as datetime objects.  Without this hook, any
    call to psycopg's Json() with DB-sourced data raises TypeError.

    Registered globally via set_json_dumps so it applies to ALL Json()
    calls throughout the application.
    """
    if isinstance(obj, Decimal):
        # Preserve integer precision where possible
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_json_dumps(obj) -> str:
    """Drop-in replacement for json.dumps used by psycopg's Json() adapter."""
    return json.dumps(obj, default=_json_default)


# Register globally so every Json() call in the application uses this encoder.
set_json_dumps(_safe_json_dumps)


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")

    bad_tokens = ("USER", "PASSWORD", "HOST")
    if any(token in url for token in bad_tokens):
        raise RuntimeError(
            "DATABASE_URL still contains placeholder values. "
            "Replace USER, PASSWORD, and HOST with real connection details."
        )

    return url


def is_connection_healthy(conn: psycopg.Connection) -> bool:
    """Check if the database connection is still alive and usable."""
    if not conn or conn.closed:
        return False
    try:
        # Simple health check - execute a lightweight query
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _configure_pooled_conn(conn: psycopg.Connection) -> None:
    """Run once per pooled connection: set the session's default isolation level
    so individual checkouts don't pay a per-request SET round-trip."""
    try:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED")
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            logger.debug("suppressed exception", exc_info=True)


def _get_pool():
    """Lazily build a process-local connection pool, fork-safe under gunicorn
    ``--preload``: a pool created in the master before fork must never be shared
    across worker processes, so we rebuild when the pid changes."""
    global _pool, _pool_pid
    pid = os.getpid()
    if _pool is not None and _pool_pid == pid:
        return _pool
    with _pool_lock:
        if _pool is not None and _pool_pid == pid:
            return _pool
        # New process: build a fresh pool bound to this pid. Abandon (do NOT
        # close) any inherited pool — closing would disturb the parent's sockets.
        max_size = int(os.getenv("DB_POOL_MAX", str(int(os.getenv("WEB_THREADS", "2")) + 2)))
        _pool = ConnectionPool(
            get_database_url(),
            min_size=1,
            max_size=max(2, max_size),
            kwargs={"row_factory": dict_row},
            configure=_configure_pooled_conn,
            timeout=30.0,
            max_idle=300.0,
            name=f"brfantasy-{pid}",
            open=True,
        )
        _pool_pid = pid
        return _pool


@contextmanager
def _get_conn_direct(autocommit: bool, retries: int) -> Iterator[psycopg.Connection]:
    """Fallback path (no psycopg_pool): a fresh connection per call."""
    url = get_database_url()
    last_err: Exception = RuntimeError("get_conn: no attempts made")
    conn = None
    for attempt in range(retries):
        try:
            conn = psycopg.connect(url, row_factory=dict_row)
            break
        except psycopg.OperationalError as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    if conn is None:
        raise last_err
    try:
        conn.autocommit = autocommit
        if not autocommit:
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            try:
                if is_connection_healthy(conn):
                    conn.rollback()
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
        raise
    finally:
        conn.close()


@contextmanager
def get_conn(autocommit: bool = False, retries: int = 3) -> Iterator[psycopg.Connection]:
    if not _POOL_AVAILABLE:
        with _get_conn_direct(autocommit, retries) as conn:
            yield conn
        return

    # Acquire a pooled connection (with retry/backoff on connect/timeout).
    last_err: Exception = RuntimeError("get_conn: no attempts made")
    cm = None
    conn = None
    for attempt in range(retries):
        try:
            cm = _get_pool().connection()
            conn = cm.__enter__()
            break
        except (psycopg.OperationalError, PoolTimeout) as e:  # type: ignore
            last_err = e
            cm = None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    if conn is None or cm is None:
        raise last_err

    # The pool default is autocommit=False; flip per-checkout when requested and
    # restore before returning the connection so the next borrower sees default.
    set_ac = bool(autocommit) and not conn.autocommit
    if set_ac:
        conn.autocommit = True
    try:
        yield conn
    except BaseException as e:
        if set_ac:
            try:
                conn.autocommit = False
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
        # pool.connection().__exit__ rolls back (non-autocommit) and returns it.
        cm.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        if set_ac:
            try:
                conn.autocommit = False
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
        # pool.connection().__exit__ commits (non-autocommit) and returns it.
        cm.__exit__(None, None, None)
