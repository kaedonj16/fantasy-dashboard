#!/usr/bin/env python3
"""
Sync data from a local Postgres database to the production database.

Only ADDS rows that prod doesn't have yet - never overwrites existing prod data.
Tables with foreign-key dependencies are synced in the correct order.

Usage:
    LOCAL_DATABASE_URL=postgresql://... PROD_DATABASE_URL=postgresql://... \\
        python scripts/migrate_local_to_prod.py

    # Sync only ADP tables:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --tables adp

    # Sync only rookie-related tables:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --tables rookie

    # Sync specific tables:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --tables draft_adp rookie_rankings

    # Preview row counts without writing anything:
    LOCAL_DATABASE_URL=... PROD_DATABASE_URL=... \\
        python scripts/migrate_local_to_prod.py --dry-run
"""

import logging
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
#   table        : str   - table name
#   conflict_cols: list  - columns for ON CONFLICT (...) - use UNIQUE constraint cols
#                          for tables whose PK is a SERIAL id
#   conflict_expr: str   - literal SQL expression for the conflict target when
#                          an expression index is involved (overrides conflict_cols)
#   skip_cols    : list  - columns to exclude from INSERT (e.g. SERIAL "id")
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
        "table": "trade_intel_users",
        "conflict_cols": ["user_id"],
        "skip_cols": [],
    },
    {
        "table": "trade_intel_trades",
        "conflict_cols": ["transaction_id"],
        "skip_cols": ["id"],
    },
    # trade_intel_assets is handled separately via _migrate_assets() because
    # its trade_id FK references the BIGSERIAL id of trade_intel_trades, which
    # auto-increments independently on local and prod.  We remap via transaction_id.
    {
        "table": "trade_intel_assets",
        "special": "assets",  # triggers _migrate_assets() instead of _migrate_table()
        "skip_cols": [],
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
    # Draft ADP (migrate in FK order: leagues → drafts → picks → aggregated)
    {
        "table": "draft_adp_drafts",
        "conflict_cols": ["draft_id"],
        "skip_cols": [],
    },
    {
        "table": "draft_adp_picks",
        "conflict_cols": ["draft_id", "pick_no"],
        "skip_cols": ["id"],
    },
    {
        "table": "draft_adp",
        "conflict_cols": ["player_id", "draft_type", "season", "is_superflex", "num_teams"],
        "skip_cols": [],
    },
    # Rookie tables - migrate in FK order (active_class and prospects first)
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

TABLE_GROUPS: dict[str, set[str]] = {
    "rookie": {
        "rookie_active_class",
        "rookie_prospects",
        "rookie_prospect_source_data",
        "rookie_prospect_athleticism",
        "rookie_mock_draft_entries",
        "rookie_mock_draft_consensus",
        "rookie_rankings",
        "rookie_value_history",
    },
    "adp": {
        "draft_adp_drafts",
        "draft_adp_picks",
        "draft_adp",
    },
    "trade": {
        "trade_intel_leagues",
        "trade_intel_users",
        "trade_intel_trades",
        "trade_intel_assets",
        "trade_intel_player_stats",
        "trade_intel_packages",
    },
}

# Keep for backwards compatibility
ROOKIE_TABLE_NAMES = TABLE_GROUPS["rookie"]


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


def _migrate_assets(
    local_conn: psycopg.Connection,
    prod_conn: psycopg.Connection,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Sync trade_intel_assets with trade_id remapping.

    trade_intel_trades.id is a BIGSERIAL that auto-increments independently on
    local and prod, so we can't copy trade_id values directly.  Instead we:
      1. Build a map  transaction_id → prod trade id  (after trades are synced).
      2. Find prod trade ids that already have assets (skip those - assumed complete).
      3. For each local asset whose trade is new to prod, insert it using the
         prod-assigned trade id.
    """
    if not _table_exists(local_conn, "trade_intel_assets"):
        return 0, 0

    local_assets = local_conn.execute("SELECT * FROM trade_intel_assets").fetchall()
    if not local_assets:
        return 0, 0

    if dry_run:
        return len(local_assets), 0

    # local trade_id → transaction_id
    local_txn_by_id: dict[int, str] = {
        row["id"]: row["transaction_id"]
        for row in local_conn.execute("SELECT id, transaction_id FROM trade_intel_trades").fetchall()
    }

    # transaction_id → prod trade id  (only trades that exist in prod)
    prod_id_by_txn: dict[str, int] = {
        row["transaction_id"]: row["id"]
        for row in prod_conn.execute("SELECT id, transaction_id FROM trade_intel_trades").fetchall()
    }

    # prod trade ids that already have assets - skip these entirely
    prod_trades_with_assets: set[int] = {
        row["trade_id"]
        for row in prod_conn.execute("SELECT DISTINCT trade_id FROM trade_intel_assets").fetchall()
    }

    insert_sql = """
        INSERT INTO trade_intel_assets
            (trade_id, side, asset_type, player_id, pick_season, pick_round, pick_order)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    inserted = 0
    with prod_conn.cursor() as cur:
        for asset in local_assets:
            txn_id = local_txn_by_id.get(asset["trade_id"])
            if not txn_id:
                continue
            prod_trade_id = prod_id_by_txn.get(txn_id)
            if not prod_trade_id:
                continue
            if prod_trade_id in prod_trades_with_assets:
                continue
            cur.execute(insert_sql, (
                prod_trade_id,
                asset["side"],
                asset["asset_type"],
                asset["player_id"],
                asset["pick_season"],
                asset["pick_round"],
                asset["pick_order"],
            ))
            inserted += cur.rowcount

    prod_conn.commit()
    return len(local_assets), inserted


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
            "Tables to sync. Use a group shorthand (rookie, adp, trade) or individual "
            "table names. Omit to sync all tables."
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
            if t in TABLE_GROUPS:
                expanded |= TABLE_GROUPS[t]
            else:
                expanded.add(t)
        tables_to_run = [cfg for cfg in TABLE_CONFIG if cfg["table"] in expanded]
        if not tables_to_run:
            print(f"No matching tables found for: {args.tables}")
            print(f"Groups available: {', '.join(TABLE_GROUPS)}")
            return 1

    print("Local DB -> Prod DB Sync  (add-only, never overwrites prod data)")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN - no writes to prod DB")
    else:
        print(f"Syncing {len(tables_to_run)} table(s) to prod DB.")
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
                if cfg.get("special") == "assets":
                    n_read, n_upserted = _migrate_assets(
                        local_conn, prod_conn, dry_run=args.dry_run
                    )
                else:
                    n_read, n_upserted = _migrate_table(
                        local_conn, prod_conn, cfg, dry_run=args.dry_run
                    )
                if n_read == 0:
                    print(f"  {table:<42} (skipped - table empty or not found)")
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
                    logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"Dry run complete - {total_read} rows across {len(tables_to_run)} table(s).")
    else:
        print(f"Done - {total_read} rows read, {total_upserted} inserted (skipped existing).")

    if errors:
        print(f"\n{len(errors)} table(s) had errors:")
        for table, exc in errors:
            print(f"  {table}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())