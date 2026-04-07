from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row


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
    print(f"[db] Connection opened: {id(conn)}, autocommit={autocommit}, status={conn.info.transaction_status}")
    try:
        conn.autocommit = autocommit
        yield conn
        if not autocommit:
            print(f"[db] About to commit connection: {id(conn)}, status={conn.info.transaction_status}")
            try:
                conn.commit()
                print(f"[db] Commit complete: {id(conn)}, status={conn.info.transaction_status}")
            except Exception as commit_error:
                print(f"[db] COMMIT FAILED for {id(conn)}: {commit_error}")
                raise
    except Exception as e:
        print(f"[db] Exception in connection {id(conn)}: {type(e).__name__}: {e}")
        if not autocommit:
            print(f"[db] Rolling back connection: {id(conn)}")
            conn.rollback()
            print(f"[db] Rollback complete: {id(conn)}")
        raise
    finally:
        print(f"[db] Closing connection: {id(conn)}, status={conn.info.transaction_status}")
        conn.close()
        print(f"[db] Connection closed: {id(conn)}")
