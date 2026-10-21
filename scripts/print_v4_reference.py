#!/usr/bin/env python3
"""Print the stored v4 reference artifact from Postgres.

scripts/derive_v4_reference.py (v1.5+) upserts the finished
artifact into prospect_v4_reference_artifact in the same run that
writes data_building/rookie_pipeline/v4_reference.json, because
files written to the app tree from a Render shell do not survive
shell recycling - the artifact file itself vanished within
minutes, three separate times. This script is the way back out:
run it in any shell with DATABASE_URL and the artifact prints to
stdout, ready to save or paste wherever it needs to land.

    python3 scripts/print_v4_reference.py            # compact JSON
    python3 scripts/print_v4_reference.py --base64   # one base64 block

Fails closed (exit 1, clean error, nothing partial printed) when
DATABASE_URL is unset or no artifact has been stored yet.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SELECT_SQL = """
SELECT artifact FROM prospect_v4_reference_artifact WHERE id = 1
"""


def _connect():
    """Fresh psycopg connection from DATABASE_URL (isolated so
    tests can substitute a fake)."""
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"])


def _load_artifact():
    """The stored artifact dict, or None when no row exists."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_SELECT_SQL)
            row = cur.fetchone()
    if row is None:
        return None
    artifact = row[0]
    if isinstance(artifact, str):
        artifact = json.loads(artifact)
    return artifact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base64", action="store_true",
                    help="print the JSON as a single base64 block "
                         "(paste-safe transfer)")
    args = ap.parse_args(argv)

    # Fail closed: no database, no artifact.
    if not os.getenv("DATABASE_URL", "").strip():
        print("print_v4_reference: DATABASE_URL is not set - "
              "failing closed.", file=sys.stderr)
        return 1
    try:
        artifact = _load_artifact()
    except Exception as exc:
        print(f"print_v4_reference: could not read "
              f"prospect_v4_reference_artifact ({exc!r}); no "
              f"artifact printed.", file=sys.stderr)
        return 1
    if artifact is None:
        print("print_v4_reference: no v4 reference artifact is "
              "stored yet (run scripts/derive_v4_reference.py "
              "first).", file=sys.stderr)
        return 1

    text = json.dumps(artifact, separators=(",", ":"))
    if args.base64:
        print(base64.b64encode(text.encode("utf-8")).decode("ascii"))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
