#!/usr/bin/env python3
"""Repair stored broad_jump_in values corrupted by the
nflverse feet/inches concatenation (model v3.14).

Background (the defect, PR fixing the read path): the
nflverse combine feed's broad_jump column is a feet/inches
concatenation, not a decimal - 1002 means 10 ft 2 in (122
in), 911 means 9 ft 11 in. ingestion.fetch_nflverse_combine
stored the bare number, so 78 live 2026 rows carry
broad_jump_in values of 800-1100 "inches" (every carrier
graded as the best broad jump in the pool's history; the TE
saturation lab caught it pricing Tanner Koziol's
athleticism). The ingestion path now decodes at read time
(ingestion._nflverse_broad_jump_in); this script repairs
the rows already stored.

What it does: for every rookie_prospect_athleticism row
whose broad_jump_in is in the concatenated range (>= 200)
with a legal inches part (value % 100 <= 11), rewrite it as
feet * 12 + inches. The guard is the decode rule itself: a
genuine inches value cannot reach 200 (the world record is
~145), and an inches part above 11 cannot be a concatenation
of feet/inches, so no legitimate row can match.

Safety:
- Dry-run by default; nothing is written without --write.
- Fails closed: no DATABASE_URL -> exit 1, nothing touched.
- The UPDATE re-states the guard in its WHERE clause, so a
  row that changed between the census and the write is not
  touched unless it still matches the corruption pattern.

Run the repair together with the post-3.14 regrade: grades
stored before the repair priced the corrupted jumps.
"""
from __future__ import annotations

import os
import sys

_GUARD = "broad_jump_in >= 200 AND (broad_jump_in % 100) <= 11"
# psycopg3 parses % as a placeholder marker in parameterized
# statements, so the UPDATE (the only parameterized statement
# here) needs the guard's modulo escaped as %%. The census
# SELECT takes no parameters and uses _GUARD verbatim.
_GUARD_PARAMS = _GUARD.replace("%", "%%")


def _decode(v: float) -> float:
    return (int(v) // 100) * 12 + (int(v) % 100)


def main() -> int:
    write = "--write" in sys.argv[1:]
    if not os.getenv("DATABASE_URL", "").strip():
        print("[broad-jump repair] DATABASE_URL is not set - "
              "failing closed, nothing touched.", file=sys.stderr)
        return 1
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        raw = conn.execute(
            f"""SELECT a.player_id, p.name, p.draft_class_year, a.broad_jump_in
                FROM rookie_prospect_athleticism a
                LEFT JOIN rookie_prospects p ON p.player_id = a.player_id
                WHERE {_GUARD}
                ORDER BY a.broad_jump_in DESC""").fetchall()
        rows = [(r["player_id"], r["name"], r["draft_class_year"],
                 float(r["broad_jump_in"])) for r in raw]
        print(f"[broad-jump repair] {len(rows)} stored rows match "
              f"the corruption guard ({_GUARD})")
        for pid, name, year, v in rows[:10]:
            print(f"  {name} ({year}) player_id={pid}: "
                  f"{v:g} -> {_decode(v):g} in")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")
        if not write:
            print("[broad-jump repair] dry run - nothing written. "
                  "Re-run with --write to repair.")
            return 0
        fixed = 0
        for pid, _name, _year, v in rows:
            cur = conn.execute(
                f"""UPDATE rookie_prospect_athleticism
                    SET broad_jump_in = %s, updated_at = NOW()
                    WHERE player_id = %s AND {_GUARD_PARAMS}""",
                (_decode(v), pid))
            fixed += cur.rowcount or 0
        conn.commit()
        print(f"[broad-jump repair] wrote {fixed} repaired rows.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
