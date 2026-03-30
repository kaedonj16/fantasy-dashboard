# Testing Advanced Metrics Integration

## Overview

This guide walks through testing the advanced metrics system and its integration with the value model.

## Prerequisites

Before testing, ensure metrics are populated:

```bash
# Run the daily cron job to populate metrics
python cron_daily.py
```

This will:
1. Generate usage table (`usage_table_YYYY-MM-DD.json`)
2. Calculate advanced metrics for all players
3. Store metrics in `player_advanced_metrics` table

## Step 1: Verify Metrics Database

Check that metrics were calculated successfully:

```python
from dashboard_services.db import get_conn

with get_conn() as conn:
    # Check if table exists and has data
    result = conn.execute("""
        SELECT COUNT(*) as count,
               MIN(as_of_date) as earliest,
               MAX(as_of_date) as latest
        FROM player_advanced_metrics
    """).fetchone()

    print(f"Total metrics: {result['count']}")
    print(f"Date range: {result['earliest']} to {result['latest']}")

    # Check sample player metrics
    sample = conn.execute("""
        SELECT player_id, position, role_score, yards_per_target, snap_share
        FROM player_advanced_metrics
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM player_advanced_metrics)
        ORDER BY role_score DESC NULLS LAST
        LIMIT 10
    """).fetchall()

    print("\nTop 10 players by role score:")
    for row in sample:
        print(f"  {row['player_id']} ({row['position']}): "
              f"Role Score {row['role_score']:.1f}, "
              f"YPT {row['yards_per_target'] or 0:.1f}, "
              f"Snap% {(row['snap_share'] or 0) * 100:.0f}%")
```

**Expected Output:**
```
Total metrics: 600
Date range: 2025-01-15 to 2025-01-15

Top 10 players by role score:
  4046 (RB): Role Score 87.3, YPT 8.5, Snap% 85%
  7564 (WR): Role Score 84.2, YPT 9.2, Snap% 78%
  ...
```

## Step 2: Test Breakout Detection

Test the multi-factor breakout algorithm:

```python
from data_building.advanced_metrics import detect_breakout_candidates

# Get breakout candidates
candidates = detect_breakout_candidates(lookback_days=14, min_games=2)

print(f"Found {len(candidates)} breakout candidates\n")

for candidate in candidates[:5]:  # Show top 5
    print(f"{candidate['name']} ({candidate['position']})")
    print(f"  Breakout Score: {candidate['breakout_score']}")
    print(f"  Components: {candidate['score_components']}")
    print(f"  Current Role Score: {candidate['current_role_score']:.1f}")
    print(f"  Value Delta: +{candidate['value_delta']:.0f}")
    print()
```

**Expected Output:**
```
Found 12 breakout candidates

Puka Nacua (WR)
  Breakout Score: 67.3
  Components: {'snap_increase': 15.2, 'opportunity_increase': 22.5, ...}
  Current Role Score: 72.5
  Value Delta: +125

Tank Dell (WR)
  Breakout Score: 58.7
  Components: {'opportunity_increase': 28.3, 'efficiency_gains': 12.1, ...}
  Current Role Score: 68.3
  Value Delta: +98
...
```

## Step 3: Test Value Model Integration

Verify metrics are loaded into training data:

```python
from data_building.value_model_training import load_advanced_metrics_df, build_training_dataframe

# Test metrics loader
metrics_df = load_advanced_metrics_df()
print(f"Loaded {len(metrics_df)} players with metrics")
print(f"Columns: {list(metrics_df.columns)}")

# Test training dataframe
train_df = build_training_dataframe()
print(f"\nTraining dataframe: {len(train_df)} rows")

# Check if advanced metrics columns are present
metric_cols = [
    'yards_per_target', 'catch_rate', 'role_score',
    'usage_trend', 'efficiency_trend'
]

for col in metric_cols:
    if col in train_df.columns:
        non_null = train_df[col].notna().sum()
        print(f"  {col}: {non_null} non-null values")
```

**Expected Output:**
```
Loaded 600 players with metrics
Columns: ['sleeper_id', 'yards_per_target', 'catch_rate', ...]

Training dataframe: 450 rows
  yards_per_target: 312 non-null values
  catch_rate: 312 non-null values
  role_score: 598 non-null values
  usage_trend: 487 non-null values
  efficiency_trend: 487 non-null values
```

## Step 4: Retrain Value Model with Metrics

**IMPORTANT:** Only retrain after metrics are populated!

```bash
python -c "from data_building.value_model_training import train_trade_value_model; train_trade_value_model()"
```

Monitor output for:
```
[value_model] Loaded 600 players with advanced metrics
[value_model] Training dataframe: 450 rows
...
Model trained with MAE: 45.3
```

## Step 5: Compare Values Before/After

Compare player values before and after metrics integration:

```python
import pandas as pd
from pathlib import Path
from utils.paths import DATA_DIR
from datetime import date

# Load old and new value tables
old_values_path = Path(DATA_DIR) / "model_values_2025-01-14.json"  # Yesterday
new_values_path = Path(DATA_DIR) / f"model_values_{date.today().isoformat()}.json"  # Today

import json

with old_values_path.open() as f:
    old_values = {p['id']: p['value'] for p in json.load(f)}

with new_values_path.open() as f:
    new_values = {p['id']: p['value'] for p in json.load(f)}

# Compare top movers
changes = []
for pid in set(old_values.keys()) & set(new_values.keys()):
    delta = new_values[pid] - old_values[pid]
    if abs(delta) > 5:  # Significant changes only
        changes.append({
            'player_id': pid,
            'old_value': old_values[pid],
            'new_value': new_values[pid],
            'delta': delta
        })

changes.sort(key=lambda x: abs(x['delta']), reverse=True)

print("Top 10 value changes after metrics integration:\n")
for change in changes[:10]:
    direction = "↑" if change['delta'] > 0 else "↓"
    print(f"Player {change['player_id']}: "
          f"{change['old_value']:.1f} → {change['new_value']:.1f} "
          f"({direction}{abs(change['delta']):.1f})")
```

**Expected Output:**
```
Top 10 value changes after metrics integration:

Player 8136: 245.3 → 267.8 (↑22.5)  # High-efficiency WR with growing role
Player 7564: 412.1 → 395.4 (↓16.7)  # Declining efficiency trend detected
Player 4046: 678.2 → 697.1 (↑18.9)  # Elite role score + high YPC
...
```

## Step 6: Test API Endpoints

Test new API endpoints work correctly:

```bash
# Get player metrics
curl http://localhost:5000/api/player-advanced-metrics/8136 | jq

# Get top role players
curl "http://localhost:5000/api/advanced-metrics/top-role-players?position=WR&limit=10" | jq

# Get breakout candidates
curl "http://localhost:5000/api/advanced-metrics/breakout-candidates?lookback_days=14" | jq
```

**Expected Response** (player metrics):
```json
{
  "player_id": "8136",
  "position": "WR",
  "metrics": {
    "yards_per_target": 9.2,
    "catch_rate": 0.78,
    "yards_per_reception": 11.8,
    "role_score": 72.5,
    "snap_share": 0.85,
    "opportunity_share": 9.2,
    "usage_trend": 22.5,
    "efficiency_trend": 12.1
  },
  "as_of_date": "2025-01-15"
}
```

## Step 7: Frontend Integration Check

Test that breakout badges appear correctly:

1. Open trade calculator: `http://localhost:5000/trade`
2. Search for a breakout candidate (e.g., "Puka Nacua")
3. Verify orange "🔥 BREAKOUT" badge appears
4. Check player value list - breakouts should have badges
5. Check top movers panel - badges should appear there too

## Troubleshooting

### No Metrics Found

**Problem:** `[value_model] No advanced metrics available yet`

**Solution:**
```bash
# Run cron to populate metrics
python cron_daily.py

# Verify table exists
python -c "from data_building.advanced_metrics import init_advanced_metrics_db; init_advanced_metrics_db()"
```

### Model Performance Degraded

**Problem:** Model MAE increased after adding metrics

**Possible Causes:**
1. Not enough data (need 2+ weeks of metrics for trends)
2. Metrics have too many nulls (check database completeness)
3. Need to tune hyperparameters for larger feature set

**Solution:**
```python
# Check feature importance after training
from data_building.value_model_training import load_trained_bundle
import numpy as np

bundle = load_trained_bundle()
gbr = bundle.pipeline.named_steps['regressor']

# Get feature names (after preprocessing)
feature_names = bundle.feature_columns

print("Top 10 Most Important Features:")
importances = gbr.feature_importances_
indices = np.argsort(importances)[::-1][:10]

for i in indices:
    if i < len(feature_names):
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
```

### Breakout Detection Too Sensitive

**Problem:** Too many breakout candidates (50+)

**Solution:** Increase threshold in `app.py`:
```python
# In detect_breakout_candidates()
if total_score >= 40:  # Increase from 30 to 40
    breakouts.append(...)
```

### API Returns Empty Arrays

**Problem:** `/api/advanced-metrics/breakout-candidates` returns `[]`

**Check:**
1. Metrics table has data for 2+ dates (needed for trends)
2. Lookback period has data
3. Check server logs for errors

## Success Criteria

✅ **Metrics populated** - `player_advanced_metrics` table has 500+ rows
✅ **Breakouts detected** - 10-20 breakout candidates found
✅ **Model integration** - Training dataframe includes metric columns
✅ **Value changes** - Retraining produces different values for some players
✅ **API works** - All 3 new endpoints return valid JSON
✅ **Frontend badges** - Breakout badges appear in UI

## Next Steps

After successful testing:

1. **Monitor model performance** - Track MAE over time
2. **Tune breakout thresholds** - Adjust scoring weights based on accuracy
3. **Add UI features** - Display efficiency metrics on player cards
4. **Create dashboards** - Build efficiency leaderboards, trend charts
5. **Backfill history** - Run metrics calculation for past dates to enable historical analysis

## Performance Benchmarks

Expected timings on production data:

- `cron_daily.py` full run: ~5-8 minutes
- Advanced metrics calculation: ~30-45 seconds (600 players)
- Breakout detection: ~2-3 seconds
- Value model training: ~20-30 seconds
- API endpoint response: <200ms

If any operation takes significantly longer, check database indexes and data volume.
