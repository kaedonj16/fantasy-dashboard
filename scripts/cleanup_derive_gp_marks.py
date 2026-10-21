#!/usr/bin/env python3
"""Remove derive-written 'gp' week marks from rookie_stat_fetched_weeks.

Background: ingestion.fetch_cfbd_games_played is a week-incremental
system built for the live pipeline - rookie_stat_fetched_weeks records
which (season, week) pairs have been folded into the GP totals stored
in rookie_prospect_source_data, and a run fetches only unmarked weeks
and ADDS to the stored totals. The v4 derive (script v1.3) loaded its
historical classes through that tracked path but never persists GP
totals anywhere: the first v1.3 run marked every historical week
fetched, its process died, the in-memory totals evaporated, and every
later tracked load read the marks over an empty store and resolved
games played for 0 players ("Skipping <yr> - all candidate weeks
already fetched"). Derive v1.4 fetches GP stateless and never marks
again; this script removes the marks the v1.3 runs wrote, so the
tracker is honest about what has actually been folded into stored
totals.

SAFETY BOUND - read before running: only rows with kind='gp' AND
season <= 2022 are touched. Season 2023 is EXCLUDED on purpose: the
live pipeline's current-class (2026) GP tracking shares that season,
and its weeks' counts are already folded into stored totals there.
Deleting live marks would make the live accumulate logic re-fetch
those weeks and ADD their counts a second time - double-counting
real players' games played. Seasons after 2023 are likewise never
touched. Nothing in this script can delete a 'targets' row.

- Dry-run by default: prints the per-season row counts it WOULD
  delete and writes nothing.
- --write performs the delete in one statement, same bound.
- Fails closed: no DATABASE_URL -> exit 1, nothing touched.

Usage:
    python3 scripts/cleanup_derive_gp_marks.py           # dry run
    python3 scripts/cleanup_derive_gp_marks.py --write   # delete
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("cleanup_derive_gp_marks")

# The hard safety bound: derive-written marks exist only for the
# historical seasons the v1.3 runs loaded (2013-2023); 2023 is cut
# off because the live pipeline shares it (see module docstring).
MAX_SEASON = 2022
KIND = "gp"

# Same DDL the ingestion helpers run before touching the table, so a
# database where the tracker was never used reads as "nothing to do"
# instead of erroring.
_DDL = """
CREATE TABLE IF NOT EXISTS rookie_stat_fetched_weeks (
    kind TEXT NOT NULL,
    season INT NOT NULL,
    week INT NOT NULL,
    team TEXT NOT NULL DEFAULT '',
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (kind, season, week, team)
)
"""

_COUNT_SQL = """
SELECT season, COUNT(*)
FROM rookie_stat_fetched_weeks
WHERE kind = %s AND season <= %s
GROUP BY season
ORDER BY season
"""

_DELETE_SQL = """
DELETE FROM rookie_stat_fetched_weeks
WHERE kind = %s AND season <= %s
"""


def _connect():
    """Fresh psycopg connection from DATABASE_URL (isolated so
    tests can substitute a fake)."""
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="apply the deletes (default is a dry run)")
    args = ap.parse_args(argv)

    # Fail closed: no database, no run.
    if not os.getenv("DATABASE_URL", "").strip():
        log.error("[gp-marks] DATABASE_URL is not set - failing closed, "
                  "nothing touched.")
        return 1

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            cur.execute(_COUNT_SQL, (KIND, MAX_SEASON))
            counts = [(int(season), int(n)) for season, n in cur.fetchall()]
            total = sum(n for _, n in counts)
            for season, n in counts:
                log.info("[gp-marks] season %d: %d %r week mark(s)%s",
                         season, n, KIND,
                         "" if args.write else " (would delete)")
            if not args.write:
                log.info("[gp-marks] DRY RUN - %d row(s) would be deleted "
                         "(kind=%r, season <= %d). Seasons >= %d and all "
                         "other kinds are never touched. Re-run with "
                         "--write to apply.",
                         total, KIND, MAX_SEASON, MAX_SEASON + 1)
                return 0
            cur.execute(_DELETE_SQL, (KIND, MAX_SEASON))
            deleted = cur.rowcount or 0
        conn.commit()

    log.info("[gp-marks] DELETED %d row(s) (kind=%r, season <= %d). "
             "Seasons >= %d and all other kinds untouched.",
             deleted, KIND, MAX_SEASON, MAX_SEASON + 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
