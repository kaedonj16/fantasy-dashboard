#!/usr/bin/env python3
"""
Run database migrations for the subscription system.

Usage:
    python scripts/run_migrations.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.db import get_conn


def run_migrations():
    """Run all SQL migrations in order."""
    migrations_dir = project_root / "migrations"

    if not migrations_dir.exists():
        print(f"Error: Migrations directory not found at {migrations_dir}")
        return False

    # Get all .sql files sorted by name
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        print("No migration files found.")
        return True

    print(f"Found {len(migration_files)} migration file(s)")

    for migration_file in migration_files:
        print(f"\nRunning migration: {migration_file.name}")

        try:
            with open(migration_file, 'r') as f:
                sql = f.read()

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)

            print(f"✓ Successfully ran {migration_file.name}")

        except Exception as e:
            print(f"✗ Error running {migration_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ All migrations completed successfully!")
    return True


if __name__ == "__main__":
    success = run_migrations()
    sys.exit(0 if success else 1)
