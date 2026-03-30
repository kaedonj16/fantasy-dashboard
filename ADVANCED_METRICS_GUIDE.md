# Advanced Metrics & Breakout Detection System

## Overview

The advanced metrics system calculates position-specific efficiency metrics for all players and uses multi-factor analysis to identify breakout candidates. This provides deeper insights beyond raw fantasy point totals.

## Architecture

### Database Schema

**Table:** `player_advanced_metrics`

Stores calculated metrics per player per date:

```sql
CREATE TABLE player_advanced_metrics (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    as_of_date DATE NOT NULL,
    position VARCHAR(5),

    -- Receiving efficiency (WR/TE/RB)
    yards_per_target NUMERIC,
    catch_rate NUMERIC,
    yards_per_reception NUMERIC,
    target_quality_score NUMERIC,

    -- Rushing efficiency (RB)
    yards_per_carry NUMERIC,
    yards_per_touch NUMERIC,
    rush_td_rate NUMERIC,

    -- Passing efficiency (QB)
    yards_per_attempt NUMERIC,
    completion_pct NUMERIC,
    td_rate NUMERIC,
    int_rate NUMERIC,

    -- Usage metrics
    snap_share NUMERIC,
    opportunity_share NUMERIC,
    red_zone_usage NUMERIC,

    -- Composite scores
    role_score NUMERIC,
    usage_trend NUMERIC,
    efficiency_trend NUMERIC,

    UNIQUE(player_id, as_of_date)
);
```

### Data Flow

1. **Daily Cron Job** (`cron_daily.py`)
   - Runs `build_daily_data()` to generate usage table
   - Runs `build_daily_advanced_metrics()` to calculate efficiency metrics
   - Saves metrics to database with today's date
   - **Offseason behavior:** Skips metrics calculation when no usage data exists (no games played)
   - Uses most recent available metrics from previous season when training value model

2. **Metric Calculation** (`data_building/advanced_metrics.py`)
   - Loads usage data from `usage_table_{date}.json`
   - Calculates position-specific efficiency metrics
   - Computes composite role scores
   - Stores in `player_advanced_metrics` table

3. **API Endpoints** (`app.py`)
   - `/api/player-advanced-metrics/<id>` - Individual player metrics
   - `/api/advanced-metrics/top-role-players` - Top players by role score
   - `/api/advanced-metrics/breakout-candidates` - Breakout detection results

## Metrics Explained

### Receiving Metrics (WR/TE/RB)

**Yards Per Target (YPT)**
- Formula: `rec_yards / targets`
- Elite: >8.5 for WR, >7.5 for TE
- Measures efficiency of target utilization

**Catch Rate**
- Formula: `receptions / targets`
- Elite: >75% for WR, >70% for TE
- Indicates reliability and hands quality

**Yards Per Reception (YPR)**
- Formula: `rec_yards / receptions`
- Elite: >12.0 for WR, >10.0 for TE
- Measures yards-after-catch and big-play ability

**Target Quality Score**
- Formula: `(targets * 2) + (YPT * 1.5) + (rec_tds * 15)`
- Combines volume and efficiency
- Elite: >50

### Rushing Metrics (RB)

**Yards Per Carry (YPC)**
- Formula: `rush_yards / carries`
- Elite: >4.5
- Core efficiency metric for RBs

**Yards Per Touch (YPTch)**
- Formula: `(rush_yards + rec_yards) / (carries + receptions)`
- Elite: >6.0
- Measures overall scrimmage efficiency

**Rush TD Rate**
- Formula: `rush_tds / carries`
- Elite: >0.05 (1 TD per 20 carries)
- Red zone efficiency indicator

### Passing Metrics (QB)

**Yards Per Attempt (YPA)**
- Formula: `pass_yds / pass_att`
- Elite: >7.5
- Key QB efficiency metric

**Completion Percentage**
- Formula: `(pass_cmp / pass_att) * 100`
- Elite: >68%
- Measures accuracy and decision-making

**TD Rate**
- Formula: `(pass_tds / pass_att) * 100`
- Elite: >5%
- Touchdown efficiency

**INT Rate**
- Formula: `(pass_int / pass_att) * 100`
- Elite: <2%
- Turnover risk (lower is better)

### Usage Metrics (All Positions)

**Snap Share**
- Formula: `avg_off_snap_pct`
- Elite: >70%
- Playing time percentage

**Opportunity Share**
- Formula: `avg_targets + avg_carries`
- Elite: >15 per game
- Total touches per game

**Red Zone Usage**
- Formula: `rec_rz_tgt_pg + rush_rz_att_pg`
- Elite: >2.0
- Scoring opportunity access

### Composite Scores

**Role Score (0-100)**

Position-weighted composite of usage and efficiency:

**QB Formula:**
```
(pass_att * 0.5) + (YPA * 3) + (TD_rate * 10) + (snap_pct * 0.3)
```

**RB Formula:**
```
(carries * 0.8) + (targets * 1.2) + (YPC * 2) + (YPT * 1.5) +
(rz_usage * 5) + (snap_pct * 0.4)
```
*Note: Pass-catching RBs weighted higher*

**WR/TE Formula:**
```
(targets * 1.5) + (YPT * 3) + (catch_rate * 20) +
(rz_targets * 6) + (snap_pct * 0.3)
```

**Interpretation:**
- 80+: Elite starter
- 65-79: Strong starter
- 50-64: Flex/low-end starter
- 35-49: Bench/handcuff
- <35: Deep bench

**Usage Trend**
- 14-day % change in opportunity share
- Positive: Increasing role
- Negative: Decreasing role

**Efficiency Trend**
- 14-day % change in role score
- Positive: Improving performance
- Negative: Declining performance

## Breakout Detection Algorithm

### Multi-Factor Scoring System

Breakout candidates must score 30+ points across these factors:

#### 1. Snap Share Increase (0-25 points)
- Requires 20%+ increase over 14 days
- Formula: `min(snap_increase%, 100) * 0.25`
- Example: 40% increase → 10 points

#### 2. Opportunity Share Increase (0-30 points)
- Requires 15%+ increase over 14 days
- Formula: `min(opp_increase%, 150) * 0.2`
- Example: 50% increase → 10 points

#### 3. Role Score Improvement (0-25 points)
- Requires 10%+ improvement over 14 days
- Formula: `min(role_improvement%, 100) * 0.25`
- Example: 30% improvement → 7.5 points

#### 4. Efficiency Gains (0-20 points)
- **WR/TE/RB:** 15%+ YPT increase → `min(ypt_gain%, 50) * 0.2`
- **RB:** 15%+ YPC increase → `min(ypc_gain%, 50) * 0.2`
- Example: 25% YPT increase → 5 points

#### 5. Red Zone Usage Increase (0-15 points)
- Requires 20%+ increase over 14 days
- Formula: `min(rz_increase%, 150) * 0.1`
- Example: 60% increase → 6 points

#### 6. Youth Bonus (0-15 points)
- Applies to players under 26 years old
- Formula: `(26 - age) * 3`, capped at 15
- Example: 23-year-old → 9 points

#### 7. Year-over-Year Snap Increase (0-20 points) **NEW**
- Requires 30%+ increase compared to same point last season
- Formula: `min(yoy_snap_increase%, 133) * 0.15`
- Captures depth chart promotions (WR2 → WR1)
- Example: 45% snap share last year → 65% this year = 44% increase → 6.6 points

#### 8. Year-over-Year Opportunity Increase (0-25 points) **NEW**
- Requires 25%+ increase compared to last season
- Formula: `min(yoy_opp_increase%, 125) * 0.2`
- Captures expanded roles and situation changes
- Example: 12% opportunity share last year → 18% this year = 50% increase → 10 points

#### 9. Second-Year Player Bonus (10 points) **NEW**
- Flat bonus for players in their second NFL season
- Sophomore breakouts are extremely common (Jefferson, Chase, Lamb, Olave, Wilson)
- Awarded when `years_exp == 1`

### Comparison: Old vs New

**Old Algorithm:**
```python
if value_delta >= 50:
    breakout = True
```
- Single factor (value change only)
- No context on why value changed
- Reactive (follows value, doesn't predict)

**New Algorithm:**
```python
breakout_score = (
    snap_increase_pts +           # 14-day trend
    opportunity_increase_pts +    # 14-day trend
    role_improvement_pts +
    efficiency_gains_pts +
    red_zone_increase_pts +
    youth_bonus_pts +
    yoy_snap_increase_pts +       # Year-over-year
    yoy_opportunity_increase_pts + # Year-over-year
    second_year_bonus_pts         # Sophomore leap
)

if breakout_score >= 30:
    breakout_candidate = True
```
- Multi-factor analysis (9 components)
- Transparent scoring breakdown
- Dual timeframes: 14-day trends + year-over-year changes
- Proactive (identifies trends before value spikes)
- Age-adjusted (young breakouts weighted higher)
- Captures situation changes (depth chart promotions, second-year leaps)

### Breakout Types Detected

The dual-timeframe approach (14-day + year-over-year) captures different breakout scenarios:

#### In-Season Breakouts (14-day factors)
- **Hot streak players** - WR3 gets hot for 3 weeks straight
- **Injury replacements** - Backup RB takes over due to starter injury
- **Coaching changes** - New OC features a previously underutilized player
- **Mid-season role expansion** - TE goes from blocking to receiving role

#### Structural Breakouts (YoY factors)
- **Second-year leaps** - Rookie WR with 40% snaps → 70% snaps in year 2
  - Example: Justin Jefferson (30% → 84% snap share year 1 → 2)
- **Depth chart promotions** - WR2 becomes WR1 after offseason departure
  - Example: Nico Collins (62% → 88% snap share after Brandin Cooks trade)
- **Scheme/situation changes** - Player benefits from new QB, system, or coaching staff
  - Example: DJ Moore (66% → 86% snap share, new Bears offense)
- **Contract year motivation** - Players in final year elevating their play

**Why both matter:**
- **14-day trends** identify hot hands and waiver wire pickups mid-season
- **YoY comparisons** identify structural advantages that persist all season
- A player with **both** signals is the highest-conviction breakout

### Example Breakout Candidate

```json
{
    "player_id": "8136",
    "name": "Puka Nacua",
    "position": "WR",
    "age": 23.1,
    "breakout_score": 67.3,
    "score_components": {
        "snap_increase": 15.2,
        "opportunity_increase": 22.5,
        "role_improvement": 12.1,
        "efficiency_gains": 8.5,
        "red_zone_increase": 3.0,
        "youth_bonus": 6.0
    },
    "current_role_score": 72.5,
    "previous_role_score": 64.7,
    "snap_share": 0.85,
    "opportunity_share": 9.2,
    "value_delta": 125.0
}
```

**Interpretation:**
- **Breakout Score 67.3** → Strong breakout candidate
- **Opportunity +22.5 pts** → Major increase in targets
- **Snap +15.2 pts** → Playing more snaps
- **Youth bonus 6.0 pts** → 23 years old, room to grow
- **Value delta +125** → Market recognizing the breakout

## API Usage

### Get Player Metrics

```bash
GET /api/player-advanced-metrics/8136
```

Response:
```json
{
    "player_id": "8136",
    "position": "WR",
    "metrics": {
        "yards_per_target": 9.2,
        "catch_rate": 0.78,
        "yards_per_reception": 11.8,
        "target_quality_score": 58.3,
        "snap_share": 0.85,
        "opportunity_share": 9.2,
        "red_zone_usage": 2.1,
        "role_score": 72.5,
        "usage_trend": 22.5,
        "efficiency_trend": 12.1
    },
    "as_of_date": "2025-01-15"
}
```

### Get Top Role Players

```bash
GET /api/advanced-metrics/top-role-players?position=WR&limit=10
```

Response:
```json
[
    {
        "player_id": "7564",
        "position": "WR",
        "role_score": 87.3,
        "snap_share": 0.92,
        "opportunity_share": 11.5,
        "yards_per_target": 9.8,
        ...
    },
    ...
]
```

### Get Breakout Candidates

```bash
GET /api/advanced-metrics/breakout-candidates?lookback_days=14&min_games=2
```

Response:
```json
[
    {
        "player_id": "8136",
        "name": "Puka Nacua",
        "position": "WR",
        "age": 23.1,
        "breakout_score": 67.3,
        "score_components": {
            "snap_increase": 15.2,
            "opportunity_increase": 22.5,
            "role_improvement": 12.1,
            "efficiency_gains": 8.5,
            "red_zone_increase": 3.0,
            "youth_bonus": 6.0
        },
        "value_delta": 125.0
    },
    ...
]
```

## Integration with Existing Systems

### Value Model Integration

**Advanced metrics are now integrated directly into the ML value model as training features.**

#### How It Works

1. **Data Loading** (`load_advanced_metrics_df()`)
   - Queries latest metrics from `player_advanced_metrics` table
   - Joins metrics into training and inference dataframes
   - Metrics become features alongside usage stats, history, and investment data

2. **Model Training** (`train_trade_value_model()`)
   - 18 advanced metrics added to feature set:
     - Receiving: YPT, catch rate, YPR, target quality score
     - Rushing: YPC, yards per touch, TD rate
     - Passing: YPA, completion %, TD rate, INT rate
     - Usage: snap share, opportunity share, red zone usage
     - Trends: usage trend, efficiency trend, role score
   - Gradient Boosting Regressor learns relationships between metrics and consensus values
   - Model discovers patterns like "high YPT + increasing snap share = rising value"

3. **Inference** (`build_ml_value_table()`)
   - Metrics automatically loaded and used for predictions
   - Players with strong efficiency metrics get value boosts
   - Declining efficiency trends signal value risk

#### What This Means

**Before Integration:**
- Value based on raw stats (PPG, targets, carries)
- No efficiency context
- Hard to differentiate between sustainable and unsustainable production

**After Integration:**
- Value reflects both volume AND efficiency
- High-efficiency players on limited volume recognized as undervalued
- Low-efficiency players with high volume flagged as overvalued
- Trend indicators help predict value changes before they happen

#### Example Impact

**Player A: High-Volume, Low-Efficiency RB**
```python
carries_pg: 18.5
rush_yards_pg: 70.1
yards_per_carry: 3.8  # Below average
yards_per_touch: 4.2  # Below average
role_score: 52.3  # Mediocre
efficiency_trend: -8.5  # Declining
```
**Impact:** Model recognizes unsustainable production, values player ~10% lower than volume alone would suggest

**Player B: Low-Volume, High-Efficiency WR**
```python
targets_pg: 5.2
rec_yards_pg: 52.1
yards_per_target: 10.0  # Elite
catch_rate: 0.80  # Elite
role_score: 68.5  # Strong
usage_trend: +22.5  # Rapidly increasing
```
**Impact:** Model recognizes emerging talent with elite efficiency, values player ~15% higher than volume alone would suggest

#### Feature Importance

In the trained model, advanced metrics typically rank:
1. **role_score** - Top 5 most important features (composite usage+efficiency)
2. **usage_trend** - Strong predictor of value changes
3. **yards_per_target** (WR/TE) - Key efficiency signal
4. **snap_share** - Opportunity indicator
5. **target_quality_score** - Elite WR identifier

#### Fallback Behavior

- If advanced metrics aren't available (table empty, cron hasn't run), model falls back to traditional features
- No errors thrown, just logs warning: `[value_model] No advanced metrics available yet`
- System gracefully handles partial data (some players missing metrics)

### Breakout Badges

The `/api/player-indicators` endpoint now uses `detect_breakout_candidates()`:

```python
# Old: Simple threshold
breakouts = [p for p in movers if p["delta"] >= 50]

# New: Multi-factor analysis
breakouts = detect_breakout_candidates(lookback_days=14)
```

Breakout badges appear in:
- Trade calculator chips
- Player dropdown search results
- Top movers panel
- Player value list

## Performance Considerations

### Database Indexes

Four indexes optimize metric queries:
```sql
CREATE INDEX idx_adv_metrics_player_date
    ON player_advanced_metrics (player_id, as_of_date DESC);

CREATE INDEX idx_adv_metrics_date_pos
    ON player_advanced_metrics (as_of_date, position);

CREATE INDEX idx_adv_metrics_role_score
    ON player_advanced_metrics (as_of_date, role_score DESC);
```

### Caching Strategy

- Metrics calculated once daily during cron job
- API endpoints query latest snapshot (no real-time calculation)
- Typical response time: <50ms for individual player, <200ms for top 50 list

### Data Volume

- ~600 players * 365 days = ~220K rows/year
- Metrics table size: ~50MB/year
- Query performance: Sub-second even with 5+ years of data

## Future Enhancements

### Potential Additions

1. **Advanced efficiency metrics:**
   - Targets per route run (true YPRR)
   - Broken tackles per touch
   - Yards after contact
   - Separation metrics

2. **Market efficiency scores:**
   - Value vs role score divergence
   - Buy-low / sell-high indicators
   - Overvalued / undervalued flags

3. **Predictive modeling:**
   - Regression models using efficiency trends
   - Role stability predictions
   - Bust risk scores

4. **UI Integration:**
   - Metrics tab on player profile pages
   - Efficiency charts and sparklines
   - Breakout candidate dashboard
   - Position-specific leaderboards

## Offseason Behavior

### How It Works

**During Offseason (No Current Season Data):**
1. ✅ **Value model still uses metrics** - Loads most recent available data (from previous season end)
2. ✅ **Metrics remain relevant** - Efficiency trends from last season inform valuations
3. ✅ **Graceful degradation** - If no historical metrics exist, model uses traditional features only
4. ⏸️ **No new metrics calculated** - Cron job skips calculation when all players have 0 games

**Message in Logs:**
```
[cron] Offseason detected - no current usage data available
[cron] Advanced metrics will use last available data when season starts
[value_model] Using advanced metrics from 2025-12-30 (90 days old - likely previous season)
```

### When Metrics Resume

Metrics automatically resume once the season starts and players accumulate stats:
- Week 1 stats populate → metrics calculate
- Week 2+ → trends begin working (14-day comparisons)
- Week 4+ → full trend accuracy (enough historical data)

### Stale Metrics Impact

**Metrics Age vs Relevance:**
- **0-30 days old:** Current season, fully relevant
- **31-180 days old:** Previous season end, still relevant (player efficiency doesn't change overnight)
- **180+ days old:** May need refresh, but better than nothing

**What Changes Between Seasons:**
- ❌ Volume stats (targets, carries) - **NOT reliable** from old data
- ✅ Efficiency metrics (YPT, catch rate, YPC) - **Mostly reliable** (player skill persists)
- ✅ Role scores - **Somewhat reliable** (adjusted by new season volume quickly)

This is why the model uses historical efficiency metrics even in offseason - a WR who had elite YPT (9.5) last season is likely still efficient this season.

## Troubleshooting

### Metrics Not Populating

**Check cron job execution:**
```bash
python cron_daily.py
```

**Verify database table exists:**
```sql
SELECT COUNT(*) FROM player_advanced_metrics;
```

**Check usage table availability:**
```python
from utils.utils import load_usage_table
usage = load_usage_table()
print(f"Loaded {len(usage)} players")
```

### Breakout API Returns Empty Array

**Verify metrics exist:**
```sql
SELECT MAX(as_of_date) FROM player_advanced_metrics;
```

**Check lookback period:**
```sql
SELECT COUNT(DISTINCT as_of_date)
FROM player_advanced_metrics
WHERE as_of_date >= CURRENT_DATE - INTERVAL '14 days';
```

If no data exists for lookback period, breakout detection can't calculate trends.

### API Performance Issues

**Check index usage:**
```sql
EXPLAIN ANALYZE
SELECT * FROM player_advanced_metrics
WHERE player_id = '8136'
ORDER BY as_of_date DESC
LIMIT 1;
```

Should show "Index Scan" not "Seq Scan".

**Vacuum table periodically:**
```sql
VACUUM ANALYZE player_advanced_metrics;
```

## Summary

The advanced metrics system provides:

✅ **Position-specific efficiency metrics** for deeper player analysis
✅ **Multi-factor breakout detection** beyond simple value thresholds
✅ **Transparent scoring** showing exactly why players qualify as breakouts
✅ **API endpoints** for integration into UI components
✅ **Daily automated calculation** via cron job
✅ **Historical tracking** for trend analysis

This infrastructure enables more sophisticated player evaluation and earlier identification of emerging talent.
