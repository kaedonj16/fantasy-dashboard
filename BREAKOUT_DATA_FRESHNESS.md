# Breakout Opportunity Data Freshness Strategy

## 🎯 Overview

This document outlines the comprehensive strategy to prevent stale breakout opportunity data and ensure users always see relevant, up-to-date insights.

## 📊 Current Schema Advantages

The breakout data schema already includes excellent foundations for freshness management:

### ✅ Built-in Freshness Features
- **`as_of_date`** field in `breakout_opportunity_scores` - tracks calculation date
- **`calculated_at`** timestamps in all tables - automatic creation tracking
- **UNIQUE constraints** with date components - prevents duplicate daily data
- **Daily cleanup** already implemented in `cron_daily.py`

## 🚀 Multi-Layer Freshness Strategy

### Layer 1: Smart Data Refresh Logic

**File**: `data_building/breakout_data_manager.py`

#### 🧠 Intelligent Refresh Detection
```python
# Season-aware refresh intervals
if season_type == "regular" and days_old > 2:  # Refresh every 2 days during season
elif season_type in ("off", "pre") and days_old > 7:  # Refresh weekly in offseason
```

#### 🔄 Change-Driven Refreshes
- **High-impact changes**: Starting QB, WR1, RB1 trades/signings trigger immediate refresh
- **Low-impact changes**: Depth roster moves wait for scheduled refresh
- **Recent activity monitoring**: Last 3 days of roster changes analyzed

#### 📈 Freshness Reporting
- Real-time data age tracking
- Change-driven refresh recommendations
- Comprehensive freshness dashboard

### Layer 2: Automated Cleanup System

**File**: `scripts/cleanup_stale_breakout_data.py`

#### 🧹 Tiered Retention Policy
| Data Type | Retention Period | Rationale |
|-----------|------------------|-----------|
| **Breakout Scores** | 30 days | Daily calculations, most time-sensitive |
| **Projections** | 90 days | Offseason-focused, slower to change |
| **Vacated Opportunity** | 180 days | Reference data, medium volatility |
| **Roster Changes** | 730 days (2 years) | Historical reference, long-term value |

#### 🔧 Performance Optimization
- Automatic table statistics updates after cleanup
- Index maintenance for optimal query performance
- Storage optimization for large datasets

### Layer 3: Enhanced Cron Integration

**File**: `cron_daily.py` (updated)

#### 🤖 Smart Decision Making
```python
# Before running calculations
needs_refresh = data_manager.needs_refresh()
should_refresh_for_changes, refresh_reason = data_manager.should_refresh_for_changes()

if not needs_refresh and not should_refresh_for_changes:
    print("[cron] Breakout data fresh, skipping refresh")
    return
```

#### 📊 Real-time Freshness Reporting
- Data age displayed after each run
- Change-driven refresh notifications
- Performance metrics tracking

## 📋 Implementation Guide

### Step 1: Deploy Data Manager
```bash
# The breakout_data_manager.py is ready to use
# No additional setup required
```

### Step 2: Schedule Cleanup
```bash
# Add to crontab for weekly cleanup
0 2 * * 0 /usr/bin/python3 /path/to/scripts/cleanup_stale_breakout_data.py
```

### Step 3: Monitor Freshness
```bash
# Check current data freshness
python3 data_building/breakout_data_manager.py
```

## 🎛️ Configuration Options

### Refresh Intervals
```python
# In breakout_data_manager.py
REFRESH_INTERVALS = {
    "regular_season_scores": 2,  # days
    "offseason_scores": 7,       # days
    "regular_season_projections": 14,  # days
    "offseason_projections": 21,       # days
}
```

### Cleanup Retention
```python
# In cleanup_stale_breakout_data.py
RETENTION_PERIODS = {
    "breakout_scores": 30,    # days
    "projections": 90,        # days
    "vacated_opportunity": 180, # days
    "roster_changes": 730,    # days (2 years)
}
```

## 🚨 Monitoring & Alerts

### Freshness Metrics to Monitor
1. **Data Age**: How old is the newest breakout data?
2. **Change Detection**: Are we detecting significant roster changes?
3. **Cleanup Effectiveness**: Is stale data being removed properly?
4. **Performance**: Are queries running efficiently after cleanup?

### Alert Thresholds
- ⚠️ **Warning**: Breakout scores > 3 days old (regular season)
- 🚨 **Critical**: Breakout scores > 7 days old (regular season)
- ⚠️ **Warning**: No refresh in 14 days (offseason)
- 🚨 **Critical**: No refresh in 30 days (offseason)

## 🔄 Data Lifecycle

### Daily (During Season)
```
1. Check data freshness → Skip if fresh
2. Detect recent roster changes → Trigger if high-impact
3. Run modular workflow → Calculate fresh scores
4. Clean up today's data → Remove duplicates
5. Report freshness status → Log metrics
```

### Weekly (Maintenance)
```
1. Run cleanup script → Remove old data
2. Optimize tables → Update statistics
3. Generate freshness report → Monitor health
4. Check retention policies → Adjust if needed
```

### Seasonal (Strategy Review)
```
1. Analyze refresh patterns → Optimize intervals
2. Review retention periods → Adjust based on usage
3. Update change detection → Refine impact rules
4. Performance tuning → Optimize queries
```

## 📈 Benefits

### ✅ Immediate Benefits
- **No Stale Data**: Always fresh breakout insights
- **Efficient Resource Usage**: Skip unnecessary calculations
- **Change Responsiveness**: Immediate updates for major moves
- **Performance**: Optimized database size and query speed

### ✅ Long-term Benefits
- **Historical Accuracy**: Clean, timestamped data series
- **Storage Optimization**: Automatic cleanup prevents bloat
- **User Trust**: Consistently fresh, relevant insights
- **Scalability**: Efficient system handles growing data

## 🎯 Best Practices

### Development
- Always use `BreakoutDataManager` for data operations
- Check freshness before running expensive calculations
- Log refresh reasons for debugging

### Operations
- Schedule regular cleanup (weekly recommended)
- Monitor freshness metrics daily
- Adjust retention periods based on actual usage patterns

### Monitoring
- Set up alerts for stale data thresholds
- Track refresh frequency vs. roster change volume
- Monitor database performance after cleanup

## 🚀 Next Steps

1. **Deploy**: The enhanced cron_daily.py is ready
2. **Schedule**: Add weekly cleanup to crontab
3. **Monitor**: Check freshness reports regularly
4. **Optimize**: Adjust intervals based on usage patterns

This comprehensive strategy ensures your breakout opportunity data stays fresh, relevant, and performant! 🎯
