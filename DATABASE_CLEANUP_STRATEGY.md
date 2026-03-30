# Database Cleanup Strategy

## Problem
Your value model formula has changed, so old values in `player_value_history` may be inaccurate or incompatible with new calculations.

## Strategy Options

### Option 1: Complete Reset (Recommended for Major Model Changes)
**When to use:** Formula fundamentally changed, old values are incompatible

```sql
-- Backup first (optional but recommended)
CREATE TABLE player_value_history_backup AS
SELECT * FROM player_value_history;

-- Clear all old data
TRUNCATE TABLE player_value_history;

-- Then run cron_daily.py to repopulate with new values
```

### Option 2: Keep Recent Data, Clear Old
**When to use:** Only recent values matter, want to save space

```sql
-- Keep last 90 days, delete everything older
DELETE FROM player_value_history
WHERE as_of_date < CURRENT_DATE - INTERVAL '90 days';

-- Vacuum to reclaim space
VACUUM FULL player_value_history;
```

### Option 3: Selective Source Cleanup
**When to use:** Only specific sources need updating

```sql
-- Clear only 'model' source, keep other sources intact
DELETE FROM player_value_history
WHERE source = 'model';

-- Keep vendor sources like FantasyCalc, DynastyProcess
-- DELETE FROM player_value_history
-- WHERE source IN ('fantasycalc', 'dynastyprocess');
```

### Option 4: Gradual Migration (Zero Downtime)
**When to use:** Need to keep site running while transitioning

```python
# In cron_daily.py, add a migration flag
def migrate_old_values():
    """
    One-time migration to recalculate all values with new formula.
    Run this manually once, then remove it.
    """
    from datetime import date, timedelta
    from data_building.value_model_training import rewrite_value_table_with_model
    from data_building.player_value_history import record_model_value_snapshot

    # Get dates you want to backfill (e.g., last 30 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    current = start_date
    while current <= end_date:
        print(f"Recalculating values for {current}")

        # Delete old values for this date
        with get_conn() as conn:
            conn.execute(
                "DELETE FROM player_value_history WHERE as_of_date = %s AND source = 'model'",
                (current,)
            )

        # Recalculate with new formula
        # (You'd need to modify rewrite_value_table_with_model to accept a date parameter)
        model_value_table = load_model_value_table() or []
        record_model_value_snapshot(model_value_table, as_of=current)

        current += timedelta(days=1)

# Run once:
# migrate_old_values()
```

## Python Script for Quick Cleanup

Create `cleanup_db.py`:

```python
#!/usr/bin/env python3
"""
Database cleanup script for player value history.
Usage: python cleanup_db.py [--all|--days=N|--source=NAME]
"""

import argparse
from dashboard_services.db import get_conn
from datetime import date, timedelta

def cleanup_all():
    """Remove all value history (use with caution!)"""
    with get_conn() as conn:
        result = conn.execute("SELECT COUNT(*) FROM player_value_history")
        count = result.fetchone()[0]

        confirm = input(f"Delete all {count} records? (yes/no): ")
        if confirm.lower() == 'yes':
            conn.execute("TRUNCATE TABLE player_value_history")
            print(f"✓ Deleted {count} records")
        else:
            print("Cancelled")

def cleanup_older_than(days: int):
    """Remove history older than N days"""
    cutoff = date.today() - timedelta(days=days)

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM player_value_history WHERE as_of_date < %s",
            (cutoff,)
        )
        print(f"✓ Deleted {result.rowcount} records older than {cutoff}")

def cleanup_source(source: str):
    """Remove all records from a specific source"""
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM player_value_history WHERE source = %s",
            (source,)
        )
        print(f"✓ Deleted {result.rowcount} records from source '{source}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up player value history")
    parser.add_argument("--all", action="store_true", help="Delete all records")
    parser.add_argument("--days", type=int, help="Delete records older than N days")
    parser.add_argument("--source", type=str, help="Delete records from specific source")

    args = parser.parse_args()

    if args.all:
        cleanup_all()
    elif args.days:
        cleanup_older_than(args.days)
    elif args.source:
        cleanup_source(args.source)
    else:
        parser.print_help()
```

Run with:
```bash
python cleanup_db.py --all  # Delete everything
python cleanup_db.py --days=60  # Keep last 60 days
python cleanup_db.py --source=model  # Delete only 'model' source
```

## Recommended Approach

1. **Backup** (always safe):
   ```bash
   # If using PostgreSQL
   pg_dump your_db > backup_$(date +%Y%m%d).sql
   ```

2. **Clear old model values**:
   ```sql
   DELETE FROM player_value_history WHERE source = 'model';
   ```

3. **Run cron job** to populate fresh data:
   ```bash
   python cron_daily.py
   ```

4. **Verify** new values are being saved:
   ```sql
   SELECT COUNT(*), MIN(as_of_date), MAX(as_of_date)
   FROM player_value_history
   WHERE source = 'model';
   ```

## Database Maintenance Tips

- **Regular vacuuming**: Run `VACUUM ANALYZE player_value_history;` weekly
- **Retention policy**: Keep 90-180 days of history (enough for trends, not too much storage)
- **Archive old data**: Export to CSV before deleting for historical analysis:
  ```sql
  COPY (SELECT * FROM player_value_history WHERE as_of_date < '2024-01-01')
  TO '/tmp/old_values.csv' CSV HEADER;
  ```

## Monitoring

After cleanup, monitor:
- Disk space: `SELECT pg_size_pretty(pg_total_relation_size('player_value_history'));`
- Row count: `SELECT COUNT(*) FROM player_value_history;`
- Date range: `SELECT MIN(as_of_date), MAX(as_of_date) FROM player_value_history;`
