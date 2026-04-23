#!/usr/bin/env python3
"""
Migrate all data from a local Postgres database to the production database.
Each table is read from the local DB and upserted into the prod DB so that
local data wins on conflict. Tables that have foreign-key dependencies are
migrated in the correct order.
Usage:
    LOCAL_DATABASE_URL=postgresql://... PROD_DATABASE_URL=postgresql://... \\
        python scripts/migrate_local_to_prod.py
    # Migrate only rookie-related tables:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --tables rookie
    # Migrate specific tables:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --tables rookie_prospects rookie_rankings
    # Preview row counts without writing anything:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import datetime
import decimal
import json

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb, set_json_dumps


def _json_default(obj):
    """Handle types that stdlib json.dumps can't serialise (mirrors db.py)."""
    if isinstance(obj, decimal.Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)  # sort for determinism
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# Register globally so every Jsonb() call in this script uses the same encoder.
set_json_dumps(lambda v: json.dumps(v, default=_json_default))


# ---------------------------------------------------------------------------
# Table configuration
# Each entry is a dict with:
#   table        : str   — table name
#   conflict_cols: list  — columns for ON CONFLICT (...) — use UNIQUE constraint cols
#                          for tables whose PK is a SERIAL id
#   conflict_expr: str   — literal SQL expression for the conflict target when
#                          an expression index is involved (overrides conflict_cols)
#   skip_cols    : list  — columns to exclude from INSERT (e.g. SERIAL "id")
#
# Order matters: parent tables before their FK-dependent children.
# ---------------------------------------------------------------------------
TABLE_CONFIG = [
    # Subscription tables (no FK deps)
    {
        "table": "league_subscriptions",
        "conflict_cols": ["league_id"],
        "skip_cols": ["id"],
    },
    {
        "table": "user_subscriptions",
        "conflict_cols": ["user_id", "platform"],
        "skip_cols": ["id"],
    },
    # Player values (no FK deps)
    {
        "table": "player_values",
        "conflict_cols": ["player_id"],
        "skip_cols": ["value_8", "value_10", "value_12", "value_14", "sf_value_8", "sf_value_10", "sf_value_12", "sf_value_14", "sf_pos_rank", "sf_pos_rank_label"],
    },
    {
        "table": "player_value_history",
        "conflict_cols": ["as_of_date", "player_id", "source"],
        "skip_cols": [],
    },
    # League analysis (no FK deps)
    {
        "table": "playoff_odds",
        "conflict_cols": ["league_id", "season", "week", "roster_id"],
        "skip_cols": [],
    },
    {
        "table": "luck_index",
        "conflict_cols": ["league_id", "season", "roster_id"],
        "skip_cols": [],
    },
    # Breakout engine (no FK deps)
    {
        "table": "roster_changes",
        "conflict_cols": ["player_id", "old_team", "new_team", "season"],
        "skip_cols": ["id"],
    },
    {
        "table": "vacated_opportunity",
        "conflict_cols": ["team", "position", "season"],
        "skip_cols": ["id"],
    },
    {
        "table": "projected_opportunity",
        "conflict_cols": ["player_id", "season"],
        "skip_cols": ["id"],
    },
    {
        "table": "breakout_opportunity_scores",
        "conflict_cols": ["player_id", "season", "as_of_date"],
        "skip_cols": ["id"],
    },
    # Advanced metrics (no FK deps)
    {
        "table": "player_advanced_metrics",
        "conflict_cols": ["player_id", "as_of_date"],
        "skip_cols": ["id"],
    },
    # Trade Intelligence (migrate in FK order)
    {
        "table": "trade_intel_leagues",
        "conflict_cols": ["league_id"],
        "skip_cols": [],
    },
    {
        "table": "trade_intel_trades",
        "conflict_cols": ["transaction_id"],
        "skip_cols": ["id"],
    },
    {
        "table": "trade_intel_assets",
        "conflict_cols": ["trade_id", "side", "asset_type"],
        "skip_cols": ["id"],
    },
    {
        "table": "trade_intel_player_stats",
        "conflict_cols": ["player_id", "season"],
        "skip_cols": [],
    },
    {
        "table": "trade_intel_packages",
        "conflict_cols": ["anchor_player_id", "package_key", "season"],
        "skip_cols": ["id"],
    },
    # Rookie tables — migrate in FK order (active_class and prospects first)
    {
        "table": "rookie_active_class",
        "conflict_cols": ["draft_class_year"],
        "skip_cols": [],
    },
    {
        "table": "rookie_prospects",
        "conflict_cols": ["player_id"],
        "skip_cols": [],
    },
    {
        "table": "rookie_prospect_source_data",
        "conflict_cols": ["player_id", "season", "source"],
        "skip_cols": ["id"],
    },
    {
        "table": "rookie_prospect_athleticism",
        "conflict_cols": ["player_id"],
        "skip_cols": [],
    },
    # rookie_mock_draft_entries uses an expression index with COALESCE(analyst_name, '')
    # so the conflict target must include the expression, not just the column name.
    {
        "table": "rookie_mock_draft_entries",
        "conflict_expr": "player_id, source_name, mock_date, (COALESCE(analyst_name, ''))",
        "skip_cols": ["id"],
    },
    {
        "table": "rookie_mock_draft_consensus",
        "conflict_cols": ["player_id"],
        "skip_cols": [],
    },
    {
        "table": "rookie_rankings",
        "conflict_cols": ["player_id", "draft_class_year"],
        "skip_cols": [],
    },
    {
        "table": "rookie_value_history",
        "conflict_cols": ["player_id", "snapshot_date"],
        "skip_cols": [],
    },
]

ROOKIE_TABLE_NAMES = {
    "rookie_active_class",
    "rookie_prospects",
    "rookie_prospect_source_data",
    "rookie_prospect_athleticism",
    "rookie_mock_draft_entries",
    "rookie_mock_draft_consensus",
    "rookie_rankings",
    "rookie_value_history",
}


def _adapt(v: Any) -> Any:
    """Wrap dict/list/set values as Jsonb so psycopg3 can serialise them for JSONB columns."""
    if isinstance(v, (dict, list, set)):
        return Jsonb(v)
    return v


def _get_url(env_var: str) -> str:
    url = os.getenv(env_var, "").strip()
    if not url:
        raise RuntimeError(f"{env_var} environment variable is not set.")
    bad_tokens = ("USER", "PASSWORD", "HOST")
    if any(t in url for t in bad_tokens):
        raise RuntimeError(f"{env_var} still contains placeholder values.")
    return url


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = %s AND table_schema = 'public'",
        (table,),
    ).fetchone()
    return row is not None


def _migrate_table(
    local_conn: psycopg.Connection,
    prod_conn: psycopg.Connection,
    cfg: dict[str, Any],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Upsert all rows from `cfg["table"]` in local_conn into prod_conn.
    Returns (rows_read, rows_upserted).
    """
    table = cfg["table"]

    if not _table_exists(local_conn, table):
        return 0, 0

    rows = local_conn.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        return 0, 0

    if dry_run:
        return len(rows), 0

    skip_cols: list[str] = cfg.get("skip_cols", [])
    all_cols = list(rows[0].keys())
    insert_cols = [c for c in all_cols if c not in skip_cols]

    col_names = ", ".join(insert_cols)
    col_placeholders = ", ".join(["%s"] * len(insert_cols))

    # Build the ON CONFLICT target
    if "conflict_expr" in cfg:
        conflict_target = cfg["conflict_expr"]
    else:
        conflict_target = ", ".join(cfg["conflict_cols"])

    # Columns to update (everything except the conflict key columns)
    conflict_key_set = set(cfg.get("conflict_cols", []))
    update_cols = [c for c in insert_cols if c not in conflict_key_set]

    if update_cols:
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        upsert_sql = (
            f"INSERT INTO {table} ({col_names}) VALUES ({col_placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}"
        )
    else:
        upsert_sql = (
            f"INSERT INTO {table} ({col_names}) VALUES ({col_placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO NOTHING"
        )

    upserted = 0
    with prod_conn.cursor() as cur:
        for row in rows:
            values = [_adapt(row[c]) for c in insert_cols]
            cur.execute(upsert_sql, values)
            upserted += cur.rowcount

    prod_conn.commit()
    return len(rows), upserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate local DB data to prod DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        metavar="TABLE",
        help=(
            "Tables to migrate. Use 'rookie' as a shorthand for all rookie tables. "
            "Omit to migrate all tables."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print row counts from local DB without writing to prod.",
    )
    args = parser.parse_args()

    local_url = _get_url("LOCAL_DATABASE_URL")
    prod_url = _get_url("PROD_DATABASE_URL")

    # Resolve which tables to migrate
    if args.tables is None:
        tables_to_run = TABLE_CONFIG
    else:
        expanded: set[str] = set()
        for t in args.tables:
            if t == "rookie":
                expanded |= ROOKIE_TABLE_NAMES
            else:
                expanded.add(t)
        tables_to_run = [cfg for cfg in TABLE_CONFIG if cfg["table"] in expanded]
        if not tables_to_run:
            print(f"No matching tables found for: {args.tables}")
            return 1

    print("Local DB -> Prod DB Migration")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN — no writes to prod DB")
    else:
        print(f"Migrating {len(tables_to_run)} table(s) to prod DB.")
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0
    print()

    total_read = total_upserted = 0
    errors: list[tuple[str, Exception]] = []

    with (
        psycopg.connect(local_url, row_factory=dict_row) as local_conn,
        psycopg.connect(prod_url, row_factory=dict_row) as prod_conn,
    ):
        for cfg in tables_to_run:
            table = cfg["table"]
            try:
                n_read, n_upserted = _migrate_table(
                    local_conn, prod_conn, cfg, dry_run=args.dry_run
                )
                if n_read == 0:
                    print(f"  {table:<42} (skipped — table empty or not found)")
                elif args.dry_run:
                    print(f"  {table:<42} {n_read} rows")
                else:
                    print(f"  {table:<42} {n_read} read, {n_upserted} upserted")
                total_read += n_read
                total_upserted += n_upserted
            except Exception as exc:
                print(f"  {table:<42} ERROR: {exc}")
                errors.append((table, exc))
                try:
                    prod_conn.rollback()
                except Exception:
                    pass

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"Dry run complete — {total_read} rows across {len(tables_to_run)} table(s).")
    else:
        print(f"Done — {total_read} rows read, {total_upserted} upserted.")

    if errors:
        print(f"\n{len(errors)} table(s) had errors:")
        for table, exc in errors:
            print(f"  {table}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())