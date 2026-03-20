from __future__ import annotations

import os
import psycopg
from contextlib import contextmanager
from psycopg.rows import dict_row
from typing import Iterator


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
def get_conn(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(get_database_url(), row_factory=dict_row)
    try:
        conn.autocommit = autocommit
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()
