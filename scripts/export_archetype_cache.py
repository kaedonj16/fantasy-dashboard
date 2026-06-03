#!/usr/bin/env python3
"""
Export PFF-style archetype profiles from player_advanced_metrics to a JSON cache
the (DB-free) breakout rebuild reads for role-fit context labels.

Run on a machine with DB access:
    python scripts/export_archetype_cache.py 2025

Writes cache/archetype_{season}.json:
    {"season": 2025, "players": {"<sleeper_id>": {"adot": .., "slot_rate": ..,
      "wide_rate": .., "inline_rate": .., "yac_per_rec": ..}, ...}}

Only rows with a non-null avg_depth_of_target are included (a player without it
has no usable archetype). Role-fit labels are simply omitted for players absent
from this cache, so partial coverage degrades safely.
"""
import json
import sys
from pathlib import Path

from dashboard_services.db import get_conn


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/export_archetype_cache.py <season>")
        sys.exit(1)
    season = int(sys.argv[1])

    query = """
        SELECT DISTINCT ON (player_id)
            player_id,
            avg_depth_of_target,
            slot_rate,
            wide_rate,
            inline_rate,
            yards_after_catch_per_reception
        FROM player_advanced_metrics
        WHERE season = %s
          AND position IN ('WR','TE')
          AND avg_depth_of_target IS NOT NULL
        ORDER BY player_id, as_of_date DESC
    """

    players = {}
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, (season,))
        for row in cur.fetchall():
            r = dict(row)
            pid = str(r["player_id"])
            players[pid] = {
                "adot": _num(r["avg_depth_of_target"]),
                "slot_rate": _num(r["slot_rate"]),
                "wide_rate": _num(r["wide_rate"]),
                "inline_rate": _num(r["inline_rate"]),
                "yac_per_rec": _num(r["yards_after_catch_per_reception"]),
            }

    out_dir = Path("cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"archetype_{season}.json"
    out_path.write_text(json.dumps({"season": season, "players": players}, indent=2))
    print(f"Wrote {len(players)} WR/TE archetype profiles -> {out_path}")


def _num(v):
    return round(float(v), 3) if v is not None else None


if __name__ == "__main__":
    main()
