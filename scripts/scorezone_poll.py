#!/usr/bin/env python3
"""Own the ScoreZone play-store poll from a Render cron (one poll per run).

Replaces the in-app poller thread that used to start at app import: under
gunicorn ``--preload`` that thread ran once in the master process, so deploys
silently orphaned it and ScoreZone data froze. This script runs a single
``poll_once()`` and exits; leadership is the same Postgres advisory lock the
in-app thread used, so overlapping runs (a slow poll vs the next minute's
cron) exit cleanly instead of double-polling.

Exit 0 on a successful poll or a clean lock-held skip. Exit 1 on unexpected
errors so Render marks the run failed and it shows up in logs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# `python scripts/scorezone_poll.py` (the Render cron) puts scripts/ on
# sys.path, not the repo root, so the project packages don't import. Add the
# repo root explicitly, matching the other cron scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scorezone-poll")


def main() -> int:
    import psycopg
    from psycopg.rows import dict_row

    from dashboard_services.db import get_database_url
    from utils.scorezone_store import _LEADER_LOCK_KEY, poll_once

    try:
        conn = psycopg.connect(get_database_url(), row_factory=dict_row)
    except Exception as exc:
        logger.warning("[scorezone-poll] db connect failed: %s", exc)
        return 1
    acquired = False
    try:
        conn.autocommit = True
        try:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (_LEADER_LOCK_KEY,)
            ).fetchone()
        except Exception as exc:
            logger.warning("[scorezone-poll] lock query failed: %s", exc)
            return 1
        if not bool(row["pg_try_advisory_lock"]):
            logger.info("[scorezone-poll] lock held elsewhere; skipping this run")
            return 0
        acquired = True
        try:
            stats = poll_once()
        except Exception as exc:
            logger.warning("[scorezone-poll] poll_once failed: %s", exc)
            return 1
        logger.info(
            "[scorezone-poll] done: games=%s plays=%s",
            stats.get("games"), stats.get("plays"),
        )
        return 0
    finally:
        if acquired:
            try:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_LEADER_LOCK_KEY,))
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
