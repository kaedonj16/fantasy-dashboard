from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


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


# ---------------------------------------------------------------------------
# Connection pool — created once at import time and shared across threads.
#
# min_size=1 keeps at least one warm connection alive so the first request
# after a cold start doesn't pay the full TCP+auth handshake cost.
# max_size=10 caps total server-side connections (tune to your Render plan).
# open=False defers the actual TCP connections until first use so the module
# can be imported even in environments where the DB isn't reachable yet
# (e.g. local dev without DATABASE_URL set).
# ---------------------------------------------------------------------------
def _make_pool() -> ConnectionPool | None:
    try:
        url = get_database_url()
    except RuntimeError:
        return None

    return ConnectionPool(
        conninfo=url,
        kwargs={"row_factory": dict_row},
        min_size=1,
        max_size=10,
        open=False,          # lazy — opens on first connection() call
        timeout=30.0,        # max seconds to wait for a free connection
        reconnect_timeout=5.0,
    )


_pool: ConnectionPool | None = _make_pool()


def _ensure_pool_open() -> None:
    """Open the pool if it hasn't been opened yet (idempotent)."""
    if _pool is not None and _pool.closed:
        _pool.open()


@contextmanager
def get_conn(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """
    Context manager that yields a psycopg connection from the shared pool.

    Uses the pool when available; falls back to a direct connection when the
    pool was not initialised (e.g. missing DATABASE_URL in local dev).
    """
    if _pool is not None:
        _ensure_pool_open()
        with _pool.connection() as conn:
            try:
                conn.autocommit = autocommit
                yield conn
                if not autocommit:
                    try:
                        conn.commit()
                    except Exception as commit_error:
                        print(f"[db] COMMIT FAILED for {id(conn)}: {commit_error}")
                        raise
            except Exception as e:
                print(f"[db] Exception in connection {id(conn)}: {type(e).__name__}: {e}")
                if not autocommit:
                    conn.rollback()
                    print(f"[db] Rollback complete: {id(conn)}")
                raise
    else:
        # Pool could not be created (DB URL missing) — open a direct connection
        # so callers get a clear error rather than a confusing AttributeError.
        conn = psycopg.connect(get_database_url(), row_factory=dict_row)
        try:
            conn.autocommit = autocommit
            yield conn
            if not autocommit:
                try:
                    conn.commit()
                except Exception as commit_error:
                    print(f"[db] COMMIT FAILED for {id(conn)}: {commit_error}")
                    raise
        except Exception as e:
            print(f"[db] Exception in connection {id(conn)}: {type(e).__name__}: {e}")
            if not autocommit:
                conn.rollback()
            raise
        finally:
            conn.close()
