#!/usr/bin/env python3
"""
Import PFF college stat CSVs into rookie_prospect_source_data.

Replaces update_passing_data.py, update_receiving_data.py, and update_rushing_data.py.

Usage:
    python scripts/import_pff_stats.py --type passing
    python scripts/import_pff_stats.py --type receiving
    python scripts/import_pff_stats.py --type rushing
    python scripts/import_pff_stats.py --type all
"""

import csv
import os
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.db import get_conn

# ---------------------------------------------------------------------------
# Config: CSV file paths and column mappings per stat type
# ---------------------------------------------------------------------------

CONFIGS = {
    'passing': {
        'csv': 'data/passing_summary.csv',
        'columns': {
            'grades_offense': 'grades_offense',
            'grades_pass': 'pff_passing_grade',
            'big_time_throws': 'big_time_throw_rate',
            'completion_percent': 'adjusted_completion_rate',
            'pressure_to_sack_rate': 'pressure_to_sack_rate',
            'qb_rating': 'nfl_passer_rating',
        },
    },
    'receiving': {
        'csv': 'data/receiving_summary.csv',
        'columns': {
            'yards_after_catch': 'yards_after_catch',
            'yards_after_catch_per_reception': 'yards_after_catch_per_reception',
            'avg_depth_of_target': 'avg_depth_of_target',
            'contested_catch_rate': 'contested_catch_rate',
            'avoided_tackles': 'avoided_tackles',
            'drop_rate': 'drop_rate',
            'slot_rate': 'slot_rate',
            'wide_rate': 'wide_rate',
            'inline_rate': 'inline_rate',
            'pass_block_rate': 'pass_block_rate',
            'grades_offense': 'grades_offense',
            'grades_pass_block': 'grades_pass_block',
            'grades_pass_route': 'grades_pass_route',
            'yprr': 'yprr',
            'explosive_runs_10_plus': 'explosive_runs_10_plus',
            'breakaway_percentage': 'breakaway_percentage',
            'elusive_rating': 'elusive_rating',
            'pff_rushing_grade': 'pff_rushing_grade',
        },
    },
    'rushing': {
        'csv': 'data/rushing_summary.csv',
        'columns': {
            'explosive': 'explosive_runs_10_plus',
            'breakaway_percent': 'breakaway_percentage',
            'elusive_rating': 'elusive_rating',
            'grades_offense': 'grades_offense',
            'grades_run': 'pff_rushing_grade',
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_player_id(player_name: str) -> str:
    """Format player name to ROOKIE_2025_NAME_SLUG format."""
    name_slug = re.sub(r'[^A-Z0-9]', '_', player_name.upper())
    name_slug = re.sub(r'_+', '_', name_slug).strip('_')
    return f"ROOKIE_2025_{name_slug}"


def _convert(value: str, db_col: str):
    """Convert a raw CSV string to an appropriate Python type."""
    if db_col in ('avoided_tackles', 'explosive_runs_10_plus'):
        return int(float(value))
    return float(value)


# ---------------------------------------------------------------------------
# Core import logic
# ---------------------------------------------------------------------------

def import_stats(stat_type: str, season: int = 2026, source: str = 'pff_college') -> bool:
    config = CONFIGS[stat_type]
    csv_file = config['csv']
    column_mapping = config['columns']

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return False

    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            with open(csv_file, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                updated_count = 0
                skipped_count = 0

                for row in reader:
                    try:
                        player_name = row['player']
                        player_id = format_player_id(player_name)

                        cursor.execute(
                            "SELECT player_id FROM rookie_prospects WHERE player_id = %s",
                            (player_id,),
                        )
                        if not cursor.fetchone():
                            print(f"Skipping {player_name} – not in rookie_prospects")
                            skipped_count += 1
                            continue

                        update_data = {}
                        for csv_col, db_col in column_mapping.items():
                            raw = row.get(csv_col, '').strip()
                            if raw:
                                try:
                                    update_data[db_col] = _convert(raw, db_col)
                                except (ValueError, TypeError):
                                    pass

                        if not update_data:
                            skipped_count += 1
                            continue

                        set_clauses = [f"{col} = %s" for col in update_data]
                        values = list(update_data.values()) + [player_id, season, source]

                        cursor.execute(
                            f"UPDATE rookie_prospect_source_data "
                            f"SET {', '.join(set_clauses)} "
                            f"WHERE player_id = %s AND season = %s AND source = %s",
                            values,
                        )

                        if cursor.rowcount == 0:
                            insert_cols = ['player_id', 'season', 'source'] + list(update_data)
                            insert_vals = [player_id, season, source] + list(update_data.values())
                            placeholders = ', '.join(['%s'] * len(insert_cols))
                            cursor.execute(
                                f"INSERT INTO rookie_prospect_source_data "
                                f"({', '.join(insert_cols)}) VALUES ({placeholders})",
                                insert_vals,
                            )

                        updated_count += 1
                        print(f"Updated {player_name} ({player_id})")

                    except Exception:
                        continue

            print(f"\nSummary ({stat_type}):")
            print(f"  Updated/Inserted: {updated_count}")
            print(f"  Skipped:          {skipped_count}")
            return True

    except Exception as e:
        print(f"Database error: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Import PFF college stats into rookie_prospect_source_data'
    )
    parser.add_argument(
        '--type',
        required=True,
        choices=['passing', 'receiving', 'rushing', 'all'],
        help='Stat type to import',
    )
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--source', default='pff_college')
    args = parser.parse_args()

    types = list(CONFIGS.keys()) if args.type == 'all' else [args.type]
    success = True
    for t in types:
        print(f"\nImporting {t} stats...")
        if not import_stats(t, season=args.season, source=args.source):
            success = False

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
