# Auto-Calculation Feature for Breakout Data

## Overview

The `BreakoutDataManager` now automatically calculates missing data when it detects that tables are empty, preventing the KeyError issues and ensuring data is always available.

## How It Works

### Auto-Calculation Triggers

The system automatically triggers calculations when:

1. **Breakout Scores Missing**: `score_count == 0` AND `latest_score_date is None`
2. **Opportunity Projections Missing**: `proj_count == 0` AND `latest_proj_date is None`

### Auto-Calculation Process

1. **Detection**: During `get_data_freshness_report()`, the system checks if data exists
2. **Trigger**: If no data is found, it automatically runs the appropriate calculation functions
3. **Refresh**: After calculation, it refreshes the counts to verify data was created
4. **Reporting**: Results are logged and included in the freshness report

### Manual Force Refresh

A new method `force_refresh_all_data()` is available to manually trigger all calculations:

```python
from data_building.breakout_data_manager import BreakoutDataManager

manager = BreakoutDataManager()
results = manager.force_refresh_all_data()
print(results)
```

## Integration with Cron Daily

The `cron_daily.py` now:

1. **Auto-Calculates**: When `get_data_freshness_report()` is called, missing data is automatically generated
2. **Reports**: Shows any auto-calculations performed in the logs
3. **Fallback**: If the main workflow fails, it attempts a force refresh as backup

## Benefits

- **No More KeyError**: Eliminates crashes when database tables are empty
- **Self-Healing**: System automatically populates missing data
- **Better Reliability**: Reduces manual intervention needed
- **Graceful Degradation**: Failed calculations are logged but don't crash the system

## Error Handling

- Auto-calculations are wrapped in try-catch blocks
- Failed attempts are logged but don't stop the process
- Results (success/failure) are included in the freshness report under `auto_calculations`

## Example Output

```
[cron] Auto-calculations performed:
  - Generated 150 breakout scores
  - Generated opportunity projections
```

Or on failure:

```
[cron] Auto-calculations performed:
  - Failed to calculate breakout scores: Database connection error
  - Generated opportunity projections
```
