# Unified Breakout Opportunity Scoring Engine

A year-round fantasy football breakout detection system that adapts scoring based on the NFL calendar phase.

## Overview

This engine replaces the previous dual-system approach (separate offseason and in-season formulas) with a unified
scoring system that uses **7 modular component scores** and **phase-based dynamic weighting**.

### Key Principles

- **Forward-looking**: Scores future opportunity, not past draft capital
- **Transaction-driven**: Roster changes (departures, signings, trades, draft picks) are primary signals
- **Explainable**: Every score includes text explanations and transaction summaries
- **Adaptive**: Weights change based on time of year

---

## Quick Start

```python
from data_building.breakout_engine import BreakoutEngine
from datetime import date

# Initialize engine for current season
engine = BreakoutEngine(season=2026)

# Build player list (simplified example)
player_list = [
    {
        'player_id': '11625',
        'player_name': 'Jalen McMillan',
        'team': 'TB',
        'position': 'WR',
        'age': 23,
        'years_exp': 1
    }
]

# Calculate breakout scores
candidates = engine.calculate_breakout_scores(player_list, min_score=30)

# View results
for candidate in candidates:
    print(f"{candidate.player_name}: {candidate.breakout_opportunity_score}")
    print(f"  {candidate.key_reasons}")
    print(f"  Role: {candidate.projected_role_tag}")
```

---

## Architecture

### Module Structure

```
data_building/breakout_engine/
├── __init__.py              # Exports BreakoutEngine
├── core.py                  # Main BreakoutEngine class (orchestration)
├── components.py            # 7 component score calculators
├── phases.py                # Phase detection & weight configs
├── transactions.py          # Transaction impact analyzer
├── explainability.py        # Text generation (key_reasons, summaries)
├── role_classifier.py       # Projected role tag generator
├── config.py                # Constants, thresholds, PHASE_WEIGHTS
└── db_helpers.py            # Database query functions
```

### Component Scores (All 0-100 scale)

1. **`opportunity_opened_score`** - Total opportunity vacated from team/position (targets, carries, snaps)
2. **`competition_removed_score`** - Specific high-value competitors who departed
3. **`competition_added_penalty`** - New competition from draft picks/signings (0 to -50)
4. **`team_environment_score`** - Offensive pace + QB quality
5. **`player_readiness_score`** - Age/efficiency/draft capital/usage history
6. **`role_trajectory_score`** - Recent usage trends (in-season only, neutral 50 in offseason)
7. **`confidence_score`** - Projection certainty (sample size, data completeness, phase)

### Phase-Based Weighting

The engine automatically detects the current NFL calendar phase and adjusts component weights:

| Phase                | Dates         | Key Characteristics                                             |
|----------------------|---------------|-----------------------------------------------------------------|
| **Offseason**        | Jan-Feb       | Focus on opportunity_opened (25%) and competition_removed (20%) |
| **Post-Free Agency** | Mar-Apr       | Competition_added_penalty increases (15%)                       |
| **Post-Draft**       | May-Jul       | Highest competition_added_penalty weight (20%)                  |
| **Preseason**        | Aug-early Sep | Role_trajectory begins to matter (20%)                          |
| **In-Season**        | Sep-Jan       | Role_trajectory dominates (40%)                                 |

---

## Component Score Details

### 1. Opportunity Opened Score

**What it measures**: Total opportunity vacated from team/position

**Calculation**:

- **WR/TE**: Score based on vacated targets (150 targets = max score 100)
- **RB**: Weighted sum of carries (primary, 70 pts max) + targets (secondary, 30 pts max)
- **QB**: Binary - starter left or not (70%+ snap share = 100)
- **Bonus**: Snap share vacated (up to +20 points)

**Example**: Mike Evans retires from TB, vacating 140 targets → WR2 on team scores ~93/100

### 2. Competition Removed Score

**What it measures**: Specific high-value competitors who departed

**Calculation**:

- Identifies departed players who were ahead on depth chart
- **High threat** (1.5x current player's usage): 40 pts max
- **Medium threat** (1.0x current player's usage): 25 pts max
- **Low threat** (0.5x current player's usage): 10 pts max

**Example**: Backup WR who had 45 targets benefits when WR1 (140 targets) leaves → 40 pts

### 3. Competition Added Penalty

**What it measures**: New competition from signings/draft (negative score)

**Calculation**:

- **Draft picks**:
    - Round 1: -30 pts
    - Round 2: -20 pts
    - Round 3: -10 pts
    - Round 4+: -5 pts
- **Free agent signings**: Based on previous season usage
    - 80+ targets last season: -25 pts
    - 50-80 targets: -15 pts
- **Cap**: Maximum penalty of -50

**Dual Impact**: Draft picks hurt existing players AND boost the rookie's own player_readiness_score

### 4. Team Environment Score

**What it measures**: Quality of offensive environment

**Calculation** (4 sub-components):

1. **Pace** (0-30 pts): Total plays per game (95+ plays = max)
2. **Pass rate** (0-30 pts): Position-dependent (WR/TE prefer high, RB balanced)
3. **Offensive ranking** (0-25 pts): Total yards per game (400+ = elite)
4. **QB quality** (0-15 pts): Pass TD/game (WR/TE only)

**Data source**: `teams_index.json` (populated by `team_enrichment.py`)

### 5. Player Readiness Score

**What it measures**: Player's ability to capitalize on opportunity

**Calculation** (4 sub-components):

1. **Age/experience** (0-30 pts): Year 2 = 30 (prime breakout window), Year 3 = 25
2. **Efficiency** (0-35 pts): YPT/YPC/catch rate from previous season
3. **Draft capital boost** (0-35 pts): For rookies - Round 1 = 35, Round 2 = 25
4. **Usage baseline** (0-20 pts): Previous season touches (shows trust)

**Note**: Rookies use draft capital instead of usage baseline

### 6. Role Trajectory Score

**What it measures**: Recent usage trends (in-season only)

**Calculation** (14-day lookback):

1. **Snap share trend** (0-30 pts): 30%+ increase = max
2. **Opportunity share trend** (0-35 pts): 25%+ increase = max
3. **Red zone usage trend** (0-20 pts): 30%+ increase = max
4. **Role score improvement** (0-15 pts): Composite role metric improvement

**Offseason behavior**: Returns neutral score (50) when no in-season data available

### 7. Confidence Score

**What it measures**: How certain is this projection

**Calculation**:

1. **Sample size** (0-40 pts): Games played × touches
2. **Data completeness** (0-25 pts): Have efficiency data + advanced metrics?
3. **Usage consistency** (0-20 pts): Low variance = more predictable
4. **Phase certainty** (0-15 pts): In-season (15) > Offseason (5)

---

## Output Format

### BreakoutCandidate Object

```python
{
    "player_id": "11625",
    "player_name": "Jalen McMillan",
    "team": "TB",
    "position": "WR",
    "season": 2026,
    "as_of_date": "2026-03-15",
    "phase": "post_free_agency",

    # Aggregate score (0-100)
    "breakout_opportunity_score": 74.3,

    # Component scores
    "opportunity_opened_score": 92.5,
    "competition_removed_score": 78.0,
    "competition_added_penalty": -15.0,
    "team_environment_score": 68.0,
    "player_readiness_score": 85.0,
    "role_trajectory_score": 50.0,
    "confidence_score": 65.0,

    # Explainability
    "directional_trend": "rising",  # 'rising' | 'falling' | 'stable'
    "key_reasons": "• Mike Evans retired (140 targets vacated)\n• Second-year WR in high-volume offense\n• High draft capital (Round 3)",
    "recent_transactions_affecting_player": "Mike Evans retired (140 targets vacated)",
    "vacated_usage_summary": "140 targets, 82% snap share from Mike Evans (retirement)",
    "added_competition_summary": "None",
    "projected_role_tag": "WR2 + Red Zone Target",  # Hybrid role classification

    # Component details (JSONB)
    "component_details": { ... }
}
```

### Projected Role Tags

Hybrid format combining depth chart position with specializations:

**WR/TE Examples**:

- "WR1"
- "WR2 + Red Zone Target"
- "WR3 + Slot"
- "TE1 + Goal Line"

**RB Examples**:

- "RB1 (Bellcow)"
- "RB2 + Passing Down"
- "RB2 + 3-Down Back"
- "Committee Back"

**QB Examples**:

- "QB1 (Locked Starter)"
- "QB1"
- "Backup QB"

---

## Database Schema

### breakout_opportunity_scores Table

```sql
CREATE TABLE breakout_opportunity_scores (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    season INT NOT NULL,
    as_of_date DATE NOT NULL,

    -- Context
    team VARCHAR(10),
    position VARCHAR(5),

    -- Component scores
    opportunity_opened_score NUMERIC,
    competition_removed_score NUMERIC,
    competition_added_penalty NUMERIC,
    team_environment_score NUMERIC,
    player_readiness_score NUMERIC,
    role_trajectory_score NUMERIC,
    confidence_score NUMERIC,

    -- Aggregate
    breakout_opportunity_score NUMERIC,

    -- Metadata
    phase VARCHAR(20),
    directional_trend VARCHAR(10),

    -- Explainability
    key_reasons TEXT,
    recent_transactions_affecting_player TEXT,
    vacated_usage_summary TEXT,
    added_competition_summary TEXT,
    projected_role_tag VARCHAR(100),

    -- Details (JSONB)
    component_details JSONB,

    calculated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(player_id, season, as_of_date)
);
```

### Supporting Tables

- `roster_changes` - Player movements (departures, signings, trades, draft picks)
    - Enhanced with `draft_metadata` JSONB column
- `vacated_opportunity` - Aggregated opportunity per team/position
- `player_advanced_metrics` - Daily efficiency/usage snapshots

---

## API Endpoints

### GET /api/breakout-candidates

Get breakout candidates using unified engine.

**Query Parameters**:

- `season` (int): Season year (default: current year)
- `min_score` (float): Minimum breakout score (default: 30)
- `position` (string): Filter by position (QB/RB/WR/TE)
- `as_of_date` (string): Date to calculate as of (YYYY-MM-DD, default: today)

**Response**: Array of BreakoutCandidate objects

### POST /api/calculate-breakout-scores

Calculate and save breakout scores for all players (admin endpoint).

**Query Parameters**:

- `season` (int): Season year
- `min_score` (float): Minimum score to save (default: 30)

**Response**:

```json
{
    "success": true,
    "candidates_calculated": 150,
    "candidates_saved": 150,
    "phase": "post_free_agency",
    "season": 2026
}
```

### GET /api/offseason-breakout-candidates (Legacy)

Backwards-compatible endpoint. Now uses unified engine by default.

**Query Parameters**:

- `season` (int): Season year
- `min_score` (float): Minimum breakout score (default: 30)
- `position` (string): Filter by position

---

## Integration with Existing Code

### Refactored Functions

Both legacy functions now use the unified engine by default with automatic fallback:

#### offseason_opportunity.py

```python
def get_offseason_breakout_candidates(
    season: int,
    min_score: float = 30,
    use_unified_engine: bool = True  # NEW: defaults to unified engine
) -> List[Dict]:
    """Uses unified engine by default, falls back to legacy if error."""
```

#### advanced_metrics.py

```python
def detect_breakout_candidates(
    lookback_days: int = 14,
    use_unified_engine: bool = True  # NEW: defaults to unified engine
) -> List[Dict]:
    """Uses unified engine by default, falls back to legacy if error."""
```

### Migration Path

1. **Current State**: Both endpoints use unified engine with automatic fallback
2. **Test Period**: Monitor for issues, compare unified vs legacy results
3. **Future**: Remove legacy implementations once validated

---

## Draft Pick Integration

### populate_draft_picks() Function

New function in `populate_roster_changes.py`:

```python
from data_building.populate_roster_changes import populate_draft_picks

draft_data = [
    {
        'player_id': '11625',
        'player_name': 'Jalen McMillan',
        'position': 'WR',
        'team': 'TB',
        'round': 3,
        'pick': 89,
        'college': 'Washington'
    }
]

populate_draft_picks(season=2025, draft_data=draft_data)
```

### Dual Impact of Draft Picks

1. **Existing Players**: Get `competition_added_penalty`
    - Round 1 pick at WR: Existing WRs lose 30 points
    - Round 2 pick: -20 points
    - Round 3 pick: -10 points

2. **Drafted Rookie**: Gets boosted `player_readiness_score`
    - Round 1: +35 points
    - Round 2: +25 points
    - Round 3: +15 points

---

## Testing

### Test Suite

Run the test suite:

```bash
python test_breakout_engine.py
```

Tests:

- ✓ Phase detection (5 test cases)
- ✓ Component score calculations
- ✓ Engine initialization
- ✓ Sample player scoring (requires DATABASE_URL)

### Manual Testing

```python
# Test phase detection
from data_building.breakout_engine.phases import PhaseDetector
from datetime import date

phase = PhaseDetector.detect_phase(date(2026, 3, 15))
print(phase)  # 'post_free_agency'

# Test component calculation
from data_building.breakout_engine.components import calculate_player_readiness_score

score, details = calculate_player_readiness_score(
    player_id="test",
    position="WR",
    season=2026,
    player_metadata={'age': 23, 'years_exp': 1},
    prev_usage={'targets': 45, 'yards_per_target': 8.5, 'catch_rate': 0.68, 'games': 12}
)

print(f"Score: {score}/100")  # ~70/100 (second-year + good efficiency)
```

---

## Configuration

### Tuning Component Weights

Edit `config.py` → `PHASE_WEIGHTS`:

```python
PHASE_WEIGHTS = {
    'in_season': {
        'opportunity_opened': 0.10,      # Lower in-season
        'role_trajectory': 0.40,         # Highest in-season
        'team_environment': 0.12,
        # ... other components
    }
}
```

### Adjusting Thresholds

Edit `config.py`:

```python
# Opportunity thresholds
MAX_VACATED_TARGETS_WR_TE = 150  # 150 targets = max score

# Draft penalties
DRAFT_PENALTY_ROUND_1 = -30
DRAFT_PENALTY_ROUND_2 = -20

# Efficiency thresholds
WR_ELITE_YARDS_PER_TARGET = 9.0
```

---

## Future Enhancements

1. **Team Stats Data Source** - Currently uses `teams_index.json`. Could integrate PFF data or NFL advanced stats.
2. **QB Quality Ratings** - Currently uses pass TD/game. Could add PFF grades or QB rating.
3. **Injury Integration** - Add `injury_status` as modifier to scores.
4. **Historical Validation** - Backtest against known breakouts (2023-2024 seasons).
5. **Machine Learning** - Train weights using historical breakout data.
6. **Alerts** - Notify users when a player's breakout score jumps 20+ points.

---

## Troubleshooting

### Database Connection Errors

```
RuntimeError: DATABASE_URL is not set.
```

**Solution**: Set DATABASE_URL environment variable:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/database"
```

### Missing Team Stats

If team_environment_score returns low scores:

```bash
# Enrich teams_index with current season stats
python -c "from data_building.external_data.team_enrichment import enrich_teams_index_with_team_offense; enrich_teams_index_with_team_offense(2026)"
```

### No Roster Changes Data

If opportunity_opened_score is always 0:

```bash
# Populate roster changes for a season
python data_building/populate_roster_changes.py 2026
```

---

## License

Internal use only - part of fantasy-dashboard project.

## Contributors

- Unified engine architecture and implementation
- Component score design
- Phase-based weighting system
- Explainability features
