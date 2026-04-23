#!/usr/bin/env python3
"""
Backfill script to update pick_order with actual numbers from Sleeper API.

This script:
1. Finds trades with draft picks where pick_order is null/old categorical
2. Re-fetches the original trade data from Sleeper
3. Updates pick_order with actual pick numbers
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Union

import requests
from dashboard_services.api import get_transactions
from dashboard_services.db import get_conn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLEEPER_BASE = "https://api.sleeper.app/v1"


def get_trade_transaction(league_id: str, transaction_id: str) -> Union[Dict[str, Any], None]:
    """Get original transaction data from Sleeper API."""
    try:
        url = f"{SLEEPER_BASE}/league/{league_id}/transaction/{transaction_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error(f"Failed to fetch transaction {transaction_id}: {exc}")
        return None


def extract_pick_order(pick: Dict[str, Any]) -> Union[str, None]:
    """Extract actual pick number from Sleeper pick data."""
    order = pick.get("order")
    return str(order) if order is not None else None


def backfill_league_trades(league_id: str, season: int, dry_run: bool = True) -> int:
    """Backfill pick orders for a specific league and season."""
    updated_count = 0
    
    with get_conn() as conn:
        # Find trades with draft picks that need updating
        trades_to_update = conn.execute("""
            SELECT DISTINCT t.id, t.transaction_id, t.league_id
            FROM trade_intel_trades t
            JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE t.league_id = %s
              AND t.season = %s
              AND a.asset_type = 'pick'
              AND (a.pick_order IS NULL OR a.pick_order IN ('early', 'mid', 'late'))
        """, (league_id, season)).fetchall()
        
        logger.info(f"Found {len(trades_to_update)} trades to update for league {league_id}")
        
        for trade in trades_to_update:
            trade_id = trade["id"]
            transaction_id = trade["transaction_id"]
            
            # Get original transaction data
            txn = get_trade_transaction(league_id, transaction_id)
            if not txn:
                logger.warning(f"Could not fetch transaction {transaction_id}")
                continue
            
            # Extract draft picks from original data
            draft_picks = txn.get("draft_picks", [])
            if not draft_picks:
                continue
            
            # Build map of picks by their identifying info
            pick_updates = {}
            for pick in draft_picks:
                season = pick.get("season")
                round_num = pick.get("round")
                order = extract_pick_order(pick)
                
                if season and round_num and order:
                    key = (season, round_num)
                    pick_updates[key] = order
            
            # Update assets in database
            assets_updated = 0
            for (season, round_num), order in pick_updates.items():
                if dry_run:
                    logger.info(f"DRY RUN: Would update pick_order to {order} for season {season}, round {round_num}")
                    assets_updated += 1
                else:
                    result = conn.execute("""
                        UPDATE trade_intel_assets 
                        SET pick_order = %s 
                        WHERE trade_id = %s 
                          AND asset_type = 'pick'
                          AND pick_season = %s 
                          AND pick_round = %s
                          AND (pick_order IS NULL OR pick_order IN ('early', 'mid', 'late'))
                    """, (order, trade_id, season, round_num))
                    
                    if result.rowcount > 0:
                        assets_updated += result.rowcount
                        logger.info(f"Updated {result.rowcount} picks for trade {trade_id}")
            
            if assets_updated > 0:
                updated_count += 1
                logger.info(f"Trade {trade_id}: {assets_updated} pick orders updated")
    
    logger.info(f"Total trades updated: {updated_count}")
    return updated_count


def backfill_all_trades(dry_run: bool = True) -> Dict[str, int]:
    """Backfill pick orders for all leagues and seasons in the database."""
    results = {}
    
    with get_conn() as conn:
        # Get all leagues and seasons that have trades with picks needing updates
        leagues_seasons = conn.execute("""
            SELECT t.league_id, t.season, COUNT(*) as trades_to_update
            FROM trade_intel_trades t
            JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE a.asset_type = 'pick'
              AND (a.pick_order IS NULL OR a.pick_order IN ('early', 'mid', 'late'))
            GROUP BY t.league_id, t.season
            ORDER BY t.season DESC, trades_to_update DESC
        """).fetchall()
        
        logger.info(f"Found {len(leagues_seasons)} league/season combinations to backfill")
        
        total_updated = 0
        for league_season in leagues_seasons:
            league_id = league_season["league_id"]
            season = league_season["season"]
            trades_count = league_season["trades_to_update"]
            
            logger.info(f"Processing league {league_id}, season {season} ({trades_count} trades)")
            
            try:
                updated = backfill_league_trades(league_id, season, dry_run=dry_run)
                results[f"{league_id}_{season}"] = updated
                total_updated += updated
                
                if not dry_run:
                    logger.info(f"Completed league {league_id}, season {season}: {updated} trades updated")
                else:
                    logger.info(f"Dry run for league {league_id}, season {season}: {updated} trades would be updated")
                    
            except Exception as exc:
                logger.error(f"Error processing league {league_id}, season {season}: {exc}")
                results[f"{league_id}_{season}"] = -1
        
        logger.info(f"Total trades processed: {total_updated}")
        results["total"] = total_updated
        
        return results


def backfill_by_season(season: int, dry_run: bool = True) -> Dict[str, int]:
    """Backfill pick orders for all leagues in a specific season."""
    results = {}
    
    with get_conn() as conn:
        # Get all leagues for the specified season
        leagues = conn.execute("""
            SELECT t.league_id, COUNT(*) as trades_to_update
            FROM trade_intel_trades t
            JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE t.season = %s
              AND a.asset_type = 'pick'
              AND (a.pick_order IS NULL OR a.pick_order IN ('early', 'mid', 'late'))
            GROUP BY t.league_id
            ORDER BY trades_to_update DESC
        """, (season,)).fetchall()
        
        logger.info(f"Found {len(leagues)} leagues for season {season}")
        
        total_updated = 0
        for league in leagues:
            league_id = league["league_id"]
            trades_count = league["trades_to_update"]
            
            logger.info(f"Processing league {league_id} ({trades_count} trades)")
            
            try:
                updated = backfill_league_trades(league_id, season, dry_run=dry_run)
                results[league_id] = updated
                total_updated += updated
                
                if not dry_run:
                    logger.info(f"Completed league {league_id}: {updated} trades updated")
                else:
                    logger.info(f"Dry run for league {league_id}: {updated} trades would be updated")
                    
            except Exception as exc:
                logger.error(f"Error processing league {league_id}: {exc}")
                results[league_id] = -1
        
        logger.info(f"Total trades for season {season}: {total_updated}")
        results["total"] = total_updated
        
        return results


def main():
    """Main backfill function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill pick orders with actual numbers")
    parser.add_argument("--league-id", help="Specific league ID to backfill")
    parser.add_argument("--season", type=int, help="Specific season to backfill")
    parser.add_argument("--all", action="store_true", help="Backfill all leagues and seasons")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be updated without making changes")
    parser.add_argument("--execute", action="store_true", help="Actually apply changes (overrides --dry-run)")
    
    args = parser.parse_args()
    
    # Set dry_run based on arguments
    dry_run = args.dry_run and not args.execute
    
    if args.execute:
        dry_run = False
        logger.info("EXECUTE MODE: Changes will be applied to database")
    else:
        logger.info(f"DRY RUN MODE: No changes will be made (use --execute to apply changes)")
    
    try:
        if args.all:
            # Backfill all trades
            logger.info("Backfilling ALL trades with draft picks...")
            results = backfill_all_trades(dry_run=dry_run)
            
        elif args.season:
            # Backfill specific season across all leagues
            logger.info(f"Backfilling season {args.season} across all leagues...")
            results = backfill_by_season(args.season, dry_run=dry_run)
            
        elif args.league_id and args.season:
            # Backfill specific league and season
            logger.info(f"Backfilling league {args.league_id}, season {args.season}...")
            updated = backfill_league_trades(args.league_id, args.season, dry_run=dry_run)
            results = {"total": updated}
            
        else:
            parser.error("Must specify --all, --season, or both --league-id and --season")
            return
        
        # Summary
        if dry_run:
            logger.info("Dry run completed. Use --execute to apply changes.")
        else:
            logger.info("Backfill completed successfully.")
        
        logger.info(f"Summary: {results}")
        
    except Exception as exc:
        logger.error(f"Backfill failed: {exc}")
        raise


if __name__ == "__main__":
    main()
