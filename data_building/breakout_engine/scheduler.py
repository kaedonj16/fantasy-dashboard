#!/usr/bin/env python3
"""
Breakout Detection Scheduler

Automated job scheduler for running breakout detection at key points:
- Daily during offseason (post-FA, post-draft phases)
- Weekly during season (in-season phase with role trajectory)
- On-demand for specific dates

Can be run as:
1. Cron job: python3 -m data_building.breakout_engine.scheduler --cron
2. One-time: python3 -m data_building.breakout_engine.scheduler --run-now
3. Daemon: python3 -m data_building.breakout_engine.scheduler --daemon
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

from dashboard_services.api import get_nfl_state
from data_building.breakout_engine.calculate_breakouts_with_real_data import main as calculate_breakouts


# Ensure DATABASE_URL is set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = f"postgresql://{os.environ.get('USER')}@localhost:5432/brfantasy"


# =============================================================================
# SCHEDULING LOGIC
# =============================================================================

def should_run_today() -> bool:
    """
    Determine if breakout scoring should run today based on NFL calendar.

    Schedule:
    - Offseason (post-FA): Daily (March 15 - April 30)
    - Post-draft: Daily (May 1 - July 15)
    - Training camp: Weekly (July 16 - Aug 31)
    - In-season: Weekly on Tuesdays (Sept 1 - Dec 31)
    - Playoffs/offseason: Skip (Jan 1 - March 14)

    Returns:
        True if scoring should run today
    """
    today = date.today()
    weekday = today.weekday()  # 0=Monday, 1=Tuesday, ...

    month = today.month
    day = today.day

    # Post-FA phase: Daily (March 15 - April 30)
    if (month == 3 and day >= 15) or month == 4:
        return True

    # Post-draft phase: Daily (May 1 - July 15)
    if month == 5 or month == 6 or (month == 7 and day <= 15):
        return True

    # Training camp: Weekly on Mondays (July 16 - Aug 31)
    if (month == 7 and day >= 16) or month == 8:
        return weekday == 0  # Monday

    # In-season: Weekly on Tuesdays (Sept 1 - Dec 31)
    if month >= 9 and month <= 12:
        return weekday == 1  # Tuesday

    # Playoffs/offseason: Skip (Jan 1 - March 14)
    return False


def get_next_run_time() -> datetime:
    """
    Calculate the next scheduled run time.

    Returns:
        datetime for next run
    """
    now = datetime.now()
    target_hour = 3  # 3 AM local time

    # Start with tomorrow at 3 AM
    next_run = datetime(now.year, now.month, now.day, target_hour, 0, 0) + timedelta(days=1)

    # Keep advancing until we find a day that should run
    while not should_run_today_for_date(next_run.date()):
        next_run += timedelta(days=1)

    return next_run


def should_run_today_for_date(check_date: date) -> bool:
    """Helper to check if a specific date should run."""
    weekday = check_date.weekday()
    month = check_date.month
    day = check_date.day

    # Post-FA phase: Daily (March 15 - April 30)
    if (month == 3 and day >= 15) or month == 4:
        return True

    # Post-draft phase: Daily (May 1 - July 15)
    if month == 5 or month == 6 or (month == 7 and day <= 15):
        return True

    # Training camp: Weekly on Mondays (July 16 - Aug 31)
    if (month == 7 and day >= 16) or month == 8:
        return weekday == 0

    # In-season: Weekly on Tuesdays (Sept 1 - Dec 31)
    if month >= 9 and month <= 12:
        return weekday == 1

    return False


# =============================================================================
# EXECUTION
# =============================================================================

def run_breakout_scoring(dry_run: bool = False) -> dict:
    """
    Execute breakout scoring job.

    Args:
        dry_run: If True, don't actually run scoring, just report what would happen

    Returns:
        Job execution result dict
    """
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get('season', 2026))
    week = int(nfl_state.get('week', 0))
    season_type = str(nfl_state.get('season_type', 'off'))

    print(f"\n{'='*80}")
    print(f"Breakout Detection Scheduled Job")
    print(f"{'='*80}")
    print(f"Run time: {datetime.now()}")
    print(f"Season: {season}, Week: {week}, Type: {season_type}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")

    if dry_run:
        print("[DRY RUN] Would execute breakout scoring now")
        return {
            'status': 'dry_run',
            'season': season,
            'week': week,
            'season_type': season_type,
            'timestamp': datetime.now().isoformat()
        }

    try:
        # Run the scoring
        result = calculate_breakouts()

        print(f"\n✓ Breakout scoring completed successfully")
        print(f"  - Candidates analyzed: {result.get('raw_candidates', 0)}")
        print(f"  - Scores saved: {result.get('saved_count', 0)}")

        return {
            'status': 'success',
            'season': season,
            'week': week,
            'season_type': season_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"\n✗ Breakout scoring failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def run_cron_job(dry_run: bool = False):
    """
    Run as a cron job (check if today should run, then execute).

    Use in crontab:
        0 3 * * * cd /path/to/fantasy-dashboard && python3 -m data_building.breakout_engine.scheduler --cron
    """
    print(f"[cron] Checking if breakout scoring should run today...")

    if not should_run_today():
        print(f"[cron] Skipping - not a scheduled run day")
        return

    print(f"[cron] Scheduled run day detected - executing scoring")
    run_breakout_scoring(dry_run=dry_run)


def run_daemon(check_interval: int = 3600, dry_run: bool = False):
    """
    Run as a daemon that checks every N seconds if it should execute.

    Args:
        check_interval: Seconds between checks (default: 3600 = 1 hour)
        dry_run: If True, don't actually run scoring
    """
    print(f"[daemon] Starting breakout scoring daemon")
    print(f"[daemon] Check interval: {check_interval} seconds")

    last_run_date = None

    while True:
        today = date.today()

        # Only run once per day, even if daemon runs 24/7
        if today != last_run_date and should_run_today():
            print(f"\n[daemon] Executing scheduled run for {today}")
            run_breakout_scoring(dry_run=dry_run)
            last_run_date = today
        else:
            next_run = get_next_run_time()
            print(f"[daemon] {datetime.now()} - Next scheduled run: {next_run}")

        time.sleep(check_interval)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Breakout Detection Scheduler"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--cron',
        action='store_true',
        help='Run as cron job (check if today should run, then exit)'
    )
    group.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon (continuously check and run on schedule)'
    )
    group.add_argument(
        '--run-now',
        action='store_true',
        help='Run scoring immediately (ignore schedule)'
    )
    group.add_argument(
        '--next-run',
        action='store_true',
        help='Show when next scheduled run will occur'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (don\'t actually execute scoring)'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=3600,
        help='Daemon check interval in seconds (default: 3600)'
    )

    args = parser.parse_args()

    if args.cron:
        run_cron_job(dry_run=args.dry_run)

    elif args.daemon:
        run_daemon(check_interval=args.check_interval, dry_run=args.dry_run)

    elif args.run_now:
        run_breakout_scoring(dry_run=args.dry_run)

    elif args.next_run:
        next_run = get_next_run_time()
        print(f"Next scheduled breakout scoring run: {next_run}")
        print(f"  ({(next_run - datetime.now()).total_seconds() / 3600:.1f} hours from now)")


if __name__ == '__main__':
    main()
