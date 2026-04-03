#!/usr/bin/env python3
"""
Cleanup stale breakout opportunity data to keep database fresh and performant.

This script removes old breakout scores and projections while preserving
recent data and historical trends.
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.db import get_conn


def cleanup_stale_breakout_scores(days_to_keep=30):
    """
    Remove breakout scores older than specified days.
    
    Args:
        days_to_keep: Number of days to retain (default: 30)
    """
    cutoff_date = date.today() - timedelta(days=days_to_keep)
    
    with get_conn() as conn:
        # Delete old breakout scores
        deleted_scores = conn.execute("""
            DELETE FROM breakout_opportunity_scores 
            WHERE as_of_date < %s
        """, (cutoff_date,)).rowcount
        
        # Delete old projections (keep longer since they're offseason-focused)
        proj_cutoff_date = date.today() - timedelta(days=90)
        deleted_projections = conn.execute("""
            DELETE FROM projected_opportunity 
            WHERE calculated_at::date < %s
        """, (proj_cutoff_date,)).rowcount
        
        # Delete old vacated opportunity (keep for reference)
        vacated_cutoff_date = date.today() - timedelta(days=180)
        deleted_vacated = conn.execute("""
            DELETE FROM vacated_opportunity 
            WHERE calculated_at::date < %s
        """, (vacated_cutoff_date,)).rowcount
        
        # Keep roster changes longer (historical reference)
        # Only delete very old ones (> 2 years)
        roster_cutoff_date = date.today() - timedelta(days=730)
        deleted_roster = conn.execute("""
            DELETE FROM roster_changes 
            WHERE created_at::date < %s
        """, (roster_cutoff_date,)).rowcount
        
        conn.commit()
        
        print(f"🧹 Cleanup completed:")
        print(f"   - Deleted {deleted_scores} old breakout scores (older than {days_to_keep} days)")
        print(f"   - Deleted {deleted_projections} old projections (older than 90 days)")
        print(f"   - Deleted {deleted_vacated} old vacated opportunities (older than 180 days)")
        print(f"   - Deleted {deleted_roster} old roster changes (older than 730 days)")


def optimize_breakout_tables():
    """Optimize tables after cleanup for better performance."""
    with get_conn() as conn:
        # Update table statistics
        conn.execute("ANALYZE breakout_opportunity_scores;")
        conn.execute("ANALYZE projected_opportunity;")
        conn.execute("ANALYZE vacated_opportunity;")
        conn.execute("ANALYZE roster_changes;")
        conn.commit()
        
        print("🔧 Table optimization completed")


def get_data_retention_stats():
    """Show current data retention statistics."""
    with get_conn() as conn:
        # Count records by age
        stats = {}
        
        # Breakout scores
        result = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN as_of_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as last_7_days,
                COUNT(CASE WHEN as_of_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as last_30_days,
                COUNT(CASE WHEN as_of_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as last_90_days
            FROM breakout_opportunity_scores
        """).fetchone()
        
        stats['breakout_scores'] = result
        
        # Projections
        result = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN calculated_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as last_30_days,
                COUNT(CASE WHEN calculated_at >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as last_90_days
            FROM projected_opportunity
        """).fetchone()
        
        stats['projections'] = result
        
        print("📊 Current Data Retention Stats:")
        print(f"   Breakout Scores: {stats['breakout_scores']['total']} total")
        print(f"     - Last 7 days: {stats['breakout_scores']['last_7_days']}")
        print(f"     - Last 30 days: {stats['breakout_scores']['last_30_days']}")
        print(f"     - Last 90 days: {stats['breakout_scores']['last_90_days']}")
        print(f"   Projections: {stats['projections']['total']} total")
        print(f"     - Last 30 days: {stats['projections']['last_30_days']}")
        print(f"     - Last 90 days: {stats['projections']['last_90_days']}")


def main():
    """Main cleanup function."""
    print("🧹 Starting breakout data cleanup...")
    print("=" * 50)
    
    try:
        # Show current stats
        get_data_retention_stats()
        print()
        
        # Clean up stale data
        cleanup_stale_breakout_scores(days_to_keep=30)
        
        # Optimize tables
        optimize_breakout_tables()
        
        print("=" * 50)
        print("✅ Breakout data cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
