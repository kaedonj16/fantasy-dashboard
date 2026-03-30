# Offseason Breakout Detection System

## Overview

The offseason breakout detection system identifies dynasty breakout candidates **before the season starts** by analyzing roster changes and projecting how vacated opportunity will redistribute among remaining players.

Unlike in-season breakout detection (which analyzes performance trends), this system tracks **structural changes** like free agent departures, trades, and retirements to predict which players will see expanded roles.

## Key Scenarios Detected

### 1. **WR2 → WR1 Promotion**
**Example:** Mike Evans leaves Tampa Bay → Emeka Egbuka becomes primary target

- **Vacated:** 140 targets, 82% snap share, 24% team target share
- **Beneficiary:** Egbuka projects to gain +75 targets, +0.33 snap share
- **Breakout Score:** 65.5 (high confidence)

### 2. **Second-Year Leap**
**Example:** Rookie WR with limited year-1 role → Feature role in year 2

- **Previous:** 45 targets, 42% snap share (WR3 behind veterans)
- **Projected:** 110 targets, 75% snap share (veterans departed)
- **Bonus:** +15 points for sophomore status

### 3. **Backup RB → Lead Back**
**Example:** Team's RB1 leaves in free agency → Backup takes over

- **Vacated:** 200 carries, 75% snap share
- **Beneficiary:** Backup RB projects +120 carries
- **Breakout Score:** 58.2

### 4. **Scheme/QB Change Benefits**
**Example:** New QB/OC favors a position previously underutilized

- Tracked via year-over-year opportunity shifts
- Combined with roster change analysis

## System Architecture

### Database Schema

#### `roster_changes` Table
Tracks player movement between teams:

```sql
CREATE TABLE roster_changes (
    player_id VARCHAR(50),
    player_name VARCHAR(255),
    position VARCHAR(5),
    old_team VARCHAR(10),        -- Team departing from
    new_team VARCHAR(10),         -- Team joining (NULL for retirement)
    change_type VARCHAR(20),      -- 'free_agent', 'trade', 'retirement', 'cut'
    change_date DATE,
    season INT,

    -- Usage stats from previous season (what's being vacated)
    last_season_targets INT,
    last_season_carries INT,
    last_season_snap_share NUMERIC,
    last_season_opportunity_share NUMERIC,
    last_season_team_target_pct NUMERIC
);
```

#### `vacated_opportunity` Table
Aggregates opportunity left behind per team/position:

```sql
CREATE TABLE vacated_opportunity (
    team VARCHAR(10),
    position VARCHAR(5),
    season INT,

    total_targets_vacated INT,
    total_carries_vacated INT,
    total_snap_share_vacated NUMERIC,
    total_opportunity_share_vacated NUMERIC,

    departed_players JSONB         -- List of players who left
);
```

**Example Entry:**
```json
{
    "team": "TB",
    "position": "WR",
    "season": 2025,
    "total_targets_vacated": 140,
    "total_snap_share_vacated": 0.82,
    "departed_players": [
        {"name": "Mike Evans", "targets": 140, "change_type": "free_agent"}
    ]
}
```

#### `projected_opportunity` Table
Projects opportunity redistribution to remaining players:

```sql
CREATE TABLE projected_opportunity (
    player_id VARCHAR(50),
    season INT,
    team VARCHAR(10),
    position VARCHAR(5),

    -- Previous season baseline
    prev_season_targets INT,
    prev_season_snap_share NUMERIC,

    -- Projected for upcoming season
    projected_targets INT,
    projected_snap_share NUMERIC,

    -- Increase amounts
    target_increase INT,
    snap_share_increase NUMERIC,

    -- Offseason breakout score (0-100)
    breakout_score NUMERIC,

    -- Factors contributing to projection
    projection_factors JSONB
);
```

## Offseason Breakout Scoring

Players are scored 0-100 across five factors:

### 1. **Absolute Opportunity Increase** (0-30 points)
Raw number increase in targets/carries:

- 50+ targets increase → up to 30 points
- 50+ carries increase → up to 30 points
- Formula: `min(increase / 3, 30)`

**Example:** 45 → 120 targets = +75 targets → **25 points**

### 2. **Relative Opportunity Increase** (0-25 points)
Percentage increase from previous season:

- Requires 50%+ increase
- Formula: `min(pct_increase / 8, 25)`

**Example:** 45 → 120 targets = 167% increase → **20.9 points**

### 3. **Team Vacancy Size** (0-20 points)
How big is the hole left behind?

- Larger vacancies = more opportunity available
- Formula: `min(vacated_targets / 10, 20)`

**Example:** 140 targets vacated on team → **14 points**

### 4. **Youth/Experience Bonus** (0-15 points)
Younger players get higher scores:

- **Second-year player (years_exp == 1):** 15 points
- **Third-year player (years_exp == 2):** 10 points
- **Age < 26:** `(26 - age) * 2` points

**Example:** 23-year-old second-year player → **15 points**

### 5. **Established Role Bonus** (0-10 points)
Players already in rotation get priority:

- **40+ targets/carries last year:** 10 points (was WR2/RB2)
- **20+ targets/carries last year:** 5 points (was backup)
- **< 20:** 0 points (deep reserve)

**Rationale:** Teams promote from within. The WR2 becomes WR1, not the WR4.

### Scoring Threshold

**Minimum 30 points required** to qualify as offseason breakout candidate.

## Example Calculation

**Player:** Emeka Egbuka (TB WR)
**Scenario:** Mike Evans departs to Dallas

### Previous Season Stats
- Targets: 45
- Snap share: 42%
- Role: WR2 behind Evans

### Mike Evans Vacated
- Targets: 140
- Snap share: 82%

### Projected Stats
- Targets: 120 (45 + 75)
- Snap share: 75% (0.42 + 0.33)

### Breakout Score Breakdown

| Factor | Calculation | Points |
|--------|-------------|--------|
| Absolute increase | 75 targets / 3 | **25.0** |
| Relative increase | 167% / 8 | **20.9** |
| Team vacancy | 140 targets / 10 | **14.0** |
| Youth bonus | Second-year player | **15.0** |
| Established role | 45 targets (WR2) | **10.0** |
| **TOTAL** | | **84.9** |

**Result:** High-conviction offseason breakout (threshold: 30)

## Data Pipeline

### Step 1: Detect Roster Changes
```python
from data_building.populate_roster_changes import populate_offseason_data

# Automatically detect roster changes by comparing seasons
populate_offseason_data(season=2025)
```

**Process:**
1. Compares current players_index to previous season usage_table
2. Identifies players who changed teams
3. Enriches with previous season usage stats
4. Saves to `roster_changes` table

### Step 2: Calculate Vacated Opportunity
```python
from data_building.offseason_opportunity import calculate_vacated_opportunity

calculate_vacated_opportunity(season=2025)
```

**Process:**
1. Aggregates departed players by team/position
2. Sums targets, carries, snap share vacated
3. Saves to `vacated_opportunity` table

**Example Output:**
```
TB WR: 140 targets, 0 carries vacated (Mike Evans)
DAL RB: 235 carries, 85 targets vacated (Zeke Elliott)
```

### Step 3: Project Redistribution
```python
from data_building.offseason_opportunity import project_opportunity_redistribution

project_opportunity_redistribution(season=2025)
```

**Process:**
1. For each team with significant vacated opportunity:
   - Identifies remaining players at that position
   - Calculates their previous usage
   - Projects proportional share of vacated opportunity
   - Assigns offseason breakout scores
2. Saves projections to `projected_opportunity` table

**Example Output:**
```
Projecting TB WR (vacated: 140 tgts, 0 cars)
  ✓ Emeka Egbuka: 45→120 tgts (+75), score: 84.9
  ✓ Jalen McMillan: 58→98 tgts (+40), score: 42.3
```

### Step 4: Query Candidates
```python
from data_building.offseason_opportunity import get_offseason_breakout_candidates

candidates = get_offseason_breakout_candidates(season=2025, min_score=30)
```

Returns ranked list of breakout candidates.

## API Usage

### Get Offseason Breakout Candidates

**Endpoint:** `GET /api/offseason-breakout-candidates`

**Query Parameters:**
- `season` - Season year (default: current)
- `min_score` - Minimum breakout score (default: 30)
- `position` - Filter by position: QB/RB/WR/TE

**Example Request:**
```bash
GET /api/offseason-breakout-candidates?season=2025&position=WR&min_score=40
```

**Example Response:**
```json
[
    {
        "player_id": "9876",
        "name": "Emeka Egbuka",
        "team": "TB",
        "position": "WR",
        "age": 23,
        "years_exp": 1,
        "breakout_score": 84.9,
        "projection_factors": {
            "absolute_opportunity_increase": 25.0,
            "relative_opportunity_increase": 20.9,
            "team_vacancy_size": 14.0,
            "youth_experience_bonus": 15.0,
            "established_role_bonus": 10.0
        },
        "previous_season": {
            "targets": 45,
            "carries": 0,
            "snap_share": 0.42
        },
        "projected": {
            "targets": 120,
            "carries": 0,
            "snap_share": 0.75
        },
        "increases": {
            "targets": 75,
            "carries": 0,
            "snap_share": 0.33
        },
        "departed_players": ["Mike Evans"],
        "context": "Benefits from Mike Evans departure"
    }
]
```

### Player Indicators (Badges)

The `/api/player-indicators` endpoint automatically switches between in-season and offseason breakout detection:

**During Offseason:**
```json
{
    "rookies": ["1234", "5678"],
    "breakouts": ["9876"]  // From offseason opportunity tracking
}
```

**During Season:**
```json
{
    "rookies": ["1234", "5678"],
    "breakouts": ["4321"]  // From in-season performance trends
}
```

## Manual Roster Change Entry

For high-profile moves not auto-detected:

```python
from data_building.populate_roster_changes import manual_add_roster_change
from datetime import date

manual_add_roster_change(
    player_name="Mike Evans",
    old_team="TB",
    new_team="DAL",
    change_type="free_agent",
    season=2025,
    change_date=date(2025, 3, 15)
)
```

This will:
1. Find player in players_index
2. Load their previous season usage stats
3. Save to roster_changes table
4. Recalculate vacated opportunity
5. Update projections

## Maintenance

### Annual Offseason Update

Run once after free agency settles (March/April):

```bash
python data_building/populate_roster_changes.py 2025
```

This will:
1. Initialize database tables (if first time)
2. Detect all roster changes since last season
3. Calculate vacated opportunity
4. Project redistribution
5. Generate breakout candidates

### Mid-Offseason Updates

If a major trade happens after initial run:

```python
from data_building.populate_roster_changes import manual_add_roster_change
from data_building.offseason_opportunity import (
    calculate_vacated_opportunity,
    project_opportunity_redistribution
)

# Add the trade
manual_add_roster_change(...)

# Recalculate
calculate_vacated_opportunity(2025)
project_opportunity_redistribution(2025)
```

## Limitations & Future Enhancements

### Current Limitations

1. **No real-time tracking** - Relies on comparing player teams between seasons
2. **Equal distribution assumption** - Vacated opportunity split proportionally by previous usage
3. **No QB/scheme impact modeling** - Doesn't account for new OC bringing different philosophy
4. **No draft pick analysis** - Rookies must be manually added if drafted to team with vacancy

### Potential Enhancements

1. **Team-specific models** - Some teams promote WR3, others sign external FAs
2. **Historical promotion rates** - "Ravens RBs promoted from practice squad 60% of time"
3. **Draft capital integration** - 1st round WRs get higher share than undrafted FAs
4. **Scheme compatibility scores** - New OC run-heavy = RBs benefit more
5. **Injury history weighting** - Fragile players less likely to absorb full vacancy

## Comparison: In-Season vs Offseason Breakouts

| Aspect | In-Season Detection | Offseason Detection |
|--------|---------------------|---------------------|
| **Timing** | During season (weeks 1-17) | Offseason (March-August) |
| **Data Source** | Performance metrics | Roster changes |
| **Signal** | Usage/efficiency trends | Vacated opportunity |
| **Factors** | Snap %, YPT, role score, value delta | Target increase, team vacancy, youth |
| **Lookback** | 14 days + year-over-year | Previous season → projection |
| **Confidence** | Reactive (follows performance) | Predictive (structural change) |
| **Examples** | Mid-season hot streak, injury replacement | WR2→WR1, sophomore leap |
| **Threshold** | 30 points (9 factors) | 30 points (5 factors) |

**Best Case:** Player qualifies for **both** (high in-season score + high offseason projection)

## Recommended Usage

### Dynasty League Owners

**March-April (Post-Free Agency):**
1. Run offseason breakout detection
2. Identify high-score candidates (>60)
3. Compare to current dynasty values
4. Target undervalued players in trades before draft

**Example:**
- Egbuka has breakout score 84.9 (Mike Evans departure)
- Current dynasty value: WR35
- Projected post-breakout: WR18-22
- **Action:** Acquire before value spike

**August (Pre-Season):**
1. Refresh projections after training camp reports
2. Validate assumptions (did backup actually win job?)
3. Adjust dynasty rankings accordingly

**In-Season:**
1. Switch to in-season breakout detection
2. Monitor if projections materializing
3. Sell high if breakout confirmed

### Content Creators

Generate offseason content:
- "Top 10 Offseason Breakout Candidates"
- "Dynasty Buys Before Rookie Draft"
- "Roster Changes That Create Opportunity"

### Dynasty Value Model

Integrate offseason projections into value model:
- Players with high offseason scores get +5-15% boost
- Especially impactful for 2nd/3rd year players
- Helps prevent undervaluation of opportunity-driven breakouts

## Conclusion

The offseason breakout detection system fills a critical gap: identifying breakout candidates **before** the season starts based on structural changes rather than waiting for performance data.

By tracking roster changes and projecting opportunity redistribution, it enables proactive dynasty decisions and more accurate preseason valuations.
