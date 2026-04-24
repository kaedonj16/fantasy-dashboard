#!/usr/bin/env python3
"""
Import Reception Perception CSVs into rookie_prospect_source_data.

Usage:
    python scripts/import_reception_perception.py
    python scripts/import_reception_perception.py --season 2025 --source pff_college
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn

DATA_DIR = Path(__file__).parent.parent / "data"

# Each file maps to (csv_player_col, {csv_col: db_col})
RP_FILES = {
    "Success Rate vs. Coverage - 2026 Draft Prospects.csv": {
        "player_col": "Player",
        "columns": {
            "Success Rate vs. Press": "success_rate_vs_press",
            "Success Rate vs. Man":  "success_rate_vs_man",
            "Success Rate vs. Zone": "success_rate_vs_zone",
        },
    },
    "Contested Catch - 2026 Draft Prospects.csv": {
        "player_col": "Player",
        "columns": {
            "Contested Catch Rate": "contested_catch_rate_rp",
        },
    },
    "Tackle Breaking Data - 2026 Draft Prospects.csv": {
        "player_col": "Player",
        "columns": {
            "1 Broken Tackle": "tackle_break_rate",
        },
    },
    "Target Data - 2026 Draft Prospects.csv": {
        "player_col": "Player",
        "columns": {
            "Route Target Rate": "route_target_rate",
        },
    },
}


def _player_id(name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]", "_", name.upper())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"ROOKIE_2026_{slug}"


def _to_float(raw: str) -> float | None:
    raw = raw.strip().rstrip("%")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def import_rp(season: int = 2024) -> None:
    total_updated = 0
    total_skipped = 0

    with get_conn() as conn:
        cur = conn.cursor()

        for filename, cfg in RP_FILES.items():
            path = DATA_DIR / filename
            if not path.exists():
                print(f"Missing: {path}")
                continue

            player_col = cfg["player_col"]
            col_map = cfg["columns"]
            updated = skipped = 0

            with open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    raw_name = row.get(player_col, "").strip()
                    if not raw_name:
                        continue

                    # Strip asterisk suffix used for rookies in RP data
                    player_name = raw_name.rstrip("*").strip()
                    player_id = _player_id(player_name)

                    cur.execute(
                        "SELECT player_id FROM rookie_prospects WHERE player_id = %s",
                        (player_id,),
                    )
                    if not cur.fetchone():
                        skipped += 1
                        continue

                    update_data = {}
                    for csv_col, db_col in col_map.items():
                        val = _to_float(row.get(csv_col, ""))
                        if val is not None:
                            update_data[db_col] = val

                    if not update_data:
                        skipped += 1
                        continue

                    set_clause = ", ".join(f"{c} = %s" for c in update_data)
                    cur.execute(
                        f"UPDATE rookie_prospect_source_data "
                        f"SET {set_clause} "
                        f"WHERE player_id = %s AND season = %s AND source = 'cfbd'",
                        [*update_data.values(), player_id, season],
                    )
                    if cur.rowcount == 0:
                        cols = ["player_id", "season", "source", *update_data]
                        vals = [player_id, season, "cfbd", *update_data.values()]
                        cur.execute(
                            f"INSERT INTO rookie_prospect_source_data "
                            f"({', '.join(cols)}) VALUES ({', '.join(['%s']*len(cols))})",
                            vals,
                        )

                    updated += 1

            print(f"{filename}: {updated} updated, {skipped} skipped")
            total_updated += updated
            total_skipped += skipped

    print(f"\nTotal: {total_updated} updated, {total_skipped} skipped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args()
    import_rp(season=args.season)
