from __future__ import annotations

import datetime
import json
import logging
import os
import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator

import psycopg

logger = logging.getLogger(__name__)
from psycopg.rows import dict_row
from psycopg.types.json import set_json_dumps


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


@contextmanager
def get_conn(autocommit: bool = False, retries: int = 3) -> Iterator[psycopg.Connection]:
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
                wait = 2 ** attempt
                logger.warning("DB connection failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, retries, wait, e)
                time.sleep(wait)
    if conn is None:
        raise last_err
    try:
        conn.autocommit = autocommit
        # Set isolation level to READ COMMITTED to reduce deadlock likelihood
        # Only set when not in autocommit mode to avoid transaction conflicts
        if not autocommit:
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        yield conn
        if not autocommit:
            try:
                conn.commit()
            except Exception as commit_error:
                logger.error("COMMIT FAILED for conn %d: %s", id(conn), commit_error)
                raise
    except Exception as e:
        logger.error("Exception in conn %d: %s: %s", id(conn), type(e).__name__, e)
        if not autocommit:
            conn.rollback()
            logger.info("Rollback complete: conn %d", id(conn))
        raise
    finally:
        conn.close()
