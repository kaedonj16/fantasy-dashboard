# Breakout Detection - Integration Complete ✅

## Summary

The breakout detection methodology has been fully integrated into the codebase and database. The system is production-ready and can be accessed via REST API or command-line tools.

---

## ✅ What Was Completed

### 1. Database Integration

**Tables Created/Updated:**
- ✅ `breakout_opportunity_scores` - Stores all calculated breakout scores
- ✅ `roster_changes` - Tracks FA signings, trades, draft picks, cuts  
- ✅ `vacated_opportunity` - Team/position departed usage
- ✅ Added `player_name` column to breakout_opportunity_scores
- ✅ Added `total_opportunity_share_vacated` column to vacated_opportunity

**Views Created:**
- ✅ `v_latest_breakout_scores` - Most recent scores per player/season
- ✅ `v_top_breakout_candidates` - Candidates scoring 40+
- ✅ `v_roster_departures` - Aggregated departures by team/position
- ✅ `v_roster_additions` - Aggregated additions by team/position

**Migration Script:**
- ✅ `data_building/breakout_engine/setup_database.sql` - Complete schema setup

### 2. REST API Integration

**New Module:** `dashboard_services/breakout_api.py`

**Endpoints Created:**
- ✅ `GET /api/breakout/candidates` - All candidates (with filtering)
- ✅ `GET /api/breakout/candidates/{position}` - Position-filtered
- ✅ `GET /api/breakout/player/{player_id}` - Player detail
- ✅ `GET /api/breakout/statistics` - Aggregate stats  
- ✅ `GET /api/breakout/team/{team}` - Team roster situation

**Breakout Type Classification:**
- ✅ Distinguishes opportunity-driven vs readiness-driven breakouts
- ✅ 8 profile types (elite_opportunity, elite_readiness, balanced_elite, etc.)
- ✅ Emoji indicators for quick visual reference
- ✅ Human-readable labels for UI display

**Flask Integration:**
- ✅ Routes registered in `app.py` via `register_breakout_routes()`
- ✅ Automatic registration on app startup

### 3. Automated Scheduling

**New Module:** `data_building/breakout_engine/scheduler.py`

**Modes:**
- ✅ Cron job mode (`--cron`) - Run once, check if today should execute
- ✅ Daemon mode (`--daemon`) - Continuous monitoring
- ✅ Manual mode (`--run-now`) - Immediate execution
- ✅ Info mode (`--next-run`) - Show next scheduled time

**Schedule Logic:**
- ✅ Offseason (Mar 15 - Apr 30): Daily
- ✅ Post-draft (May 1 - Jul 15): Daily  
- ✅ Training camp (Jul 16 - Aug 31): Weekly (Mondays)
- ✅ In-season (Sep 1 - Dec 31): Weekly (Tuesdays)
- ✅ Playoffs (Jan 1 - Mar 14): Skip

### 4. Documentation

**Files Created:**
- ✅ `BREAKOUT_RESULTS_SUMMARY.md` - Comprehensive results analysis
- ✅ `data_building/breakout_engine/README.md` - Integration guide
- ✅ `BREAKOUT_INTEGRATION_COMPLETE.md` - This file

---

## 📊 Current Data State

**2026 Season Results:**
- **380 total candidates** scored
- **113 candidates** after filtering (score > 0)
- **10 elite candidates** (score >= 50)
- **Top score:** 65.9 (TreVeyon Henderson, RB-NE)

**Database Records:**
```sql
SELECT COUNT(*) FROM breakout_opportunity_scores WHERE season = 2026;
-- Result: 380 candidates

SELECT COUNT(*) FROM breakout_opportunity_scores WHERE season = 2026 AND breakout_opportunity_score >= 50;
-- Result: 10 elite candidates
```

**API Test:**
```bash
curl "http://localhost:5000/api/breakout/candidates?min_score=50" | jq '.count'
# Returns: 10
```

---

## 🚀 Usage Examples

### Command Line

```bash
# Calculate current season scores
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data

# View top candidates
python3 -m data_building.breakout_engine.display_results --summary --min-score 40

# Analyze distribution
python3 -m data_building.breakout_engine.analyze_results --top-n 20

# Filter by position
python3 -m data_building.breakout_engine.display_results --position RB --top-n 10
```

### API (Python)

```python
import requests

# Get all RB breakout candidates
response = requests.get('http://localhost:5000/api/breakout/candidates/RB?min_score=40')
data = response.json()

for candidate in data['candidates']:
    print(f"{candidate['player_name']}: {candidate['breakout_opportunity_score']}")
    print(f"  Type: {candidate['breakout_type']['profile_label']}")
    print(f"  Reasons: {candidate['key_reasons']}")
```

### API (JavaScript)

```javascript
// Fetch player breakout data
const response = await fetch('/api/breakout/player/12345');
const player = await response.json();

// Display breakout badge
<div className="breakout-badge">
  <span className="emoji">{player.breakout_type.emoji}</span>
  <span className="label">{player.breakout_type.profile_label}</span>
  <span className="score">{player.breakout_opportunity_score.toFixed(1)}</span>
</div>

// Show component scores
<div className="components">
  <div>Opportunity: {player.opportunity_opened_score}</div>
  <div>Readiness: {player.player_readiness_score}</div>
  <div>Confidence: {player.confidence_score}%</div>
</div>
```

---

## 🔧 Setup Instructions

### 1. Initialize Database

```bash
# Run migration script
DATABASE_URL="postgresql://user@localhost:5432/brfantasy" \
  psql -f data_building/breakout_engine/setup_database.sql
```

### 2. Calculate Initial Scores

```bash
# Run breakout detection
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data
```

### 3. Start Flask App

```bash
# API will be available at /api/breakout/*
python3 app.py
```

### 4. Automated Scoring (Already Configured!)

**✅ Breakout scoring is automatically integrated into your existing `cron_daily.py` job.**

No additional cron jobs needed! The breakout engine runs as part of your daily cron:

```python
# In cron_daily.py (already configured)
def main():
    # ... other daily tasks ...
    build_daily_breakout_candidates(season, week, state)  # ← Runs new BreakoutEngine
```

Your existing cron job handles it:
```bash
# Your existing cron (no changes needed)
0 3 * * * cd /path/to/fantasy-dashboard && python3 cron_daily.py
```

**Optional Manual Runs:**

If you need to run breakout scoring outside the daily cron:

```bash
# Run immediately
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data

# Or use the standalone scheduler (advanced)
python3 -m data_building.breakout_engine.scheduler --run-now
```

---

## 💎 Breakout Type Classification

Each candidate is classified by **primary driver** and **profile**:

### Primary Drivers

| Driver | Description | Example |
|--------|-------------|---------|
| **opportunity** | Driven by vacated usage/departed competition | Luther Burden III (CHI): 100 opp, 70 readiness |
| **readiness** | Driven by talent/pedigree/age window | TreVeyon Henderson (NE): 0 opp, 70 readiness |
| **balanced** | Both opportunity and readiness strong | Zach Charbonnet (SEA): 72.7 opp, 59.8 readiness |

### Profiles

| Profile | Score | Label | Emoji |
|---------|-------|-------|-------|
| `elite_opportunity` | 55+ | Elite Opportunity Breakout | 🚀 |
| `elite_readiness` | 55+ | Elite Talent Breakout | ⭐ |
| `balanced_elite` | 55+ | Elite Balanced Breakout | 💎 |
| `strong_opportunity` | 45-54 | Strong Opportunity Situation | 📈 |
| `strong_readiness` | 45-54 | High-Talent Prospect | ✨ |
| `balanced_strong` | 45-54 | Strong Balanced Profile | 🎯 |
| `moderate` | 35-44 | Moderate Breakout Potential | 📊 |
| `longshot` | <35 | Longshot Candidate | 🎲 |

---

## 🎯 Top 10 Elite Breakouts (Score 50+)

| Rank | Player | Pos | Team | Score | Type | Driver |
|------|--------|-----|------|-------|------|--------|
| 1 | **TreVeyon Henderson** | RB | NE | **65.9** | ⭐ Elite Talent | Readiness (70) |
| 2 | **Kyle Monangai** | RB | CHI | **64.3** | ⭐ Elite Talent | Readiness (65) |
| 3 | **Blake Corum** | RB | LAR | **61.0** | ⭐ Elite Talent | Readiness (62.9) |
| 4 | **AJ Barner** | TE | SEA | **59.8** | ⭐ Elite Talent | Readiness (75) |
| 5 | **Ladd McConkey** | WR | LAC | **56.1** | ⭐ Elite Talent | Readiness (65) |
| 6 | **RJ Harvey** | RB | DEN | **53.7** | ⭐ Elite Talent | Readiness (55) |
| 7 | **Quentin Johnston** | WR | LAC | **53.5** | ⭐ Elite Talent | Readiness (65) |
| 8 | **Luther Burden III** | WR | CHI | **52.4** | 💎 Balanced Elite | Opp (100) + Ready (70) |
| 9 | Davis Allen | TE | LAR | 51.1 | ⭐ Elite Talent | Readiness (49.5) |
| 10 | Terrance Ferguson | TE | LAR | 50.8 | ⭐ Elite Talent | Readiness (53.5) |

---

## 📈 Top Opportunity Situations

Players benefiting most from departed competition:

| Player | Pos | Team | Opp | Overall | Situation |
|--------|-----|------|-----|---------|-----------|
| **Luther Burden III** | WR | CHI | **100** | 52.4 | Bears WR room reset |
| **Rome Odunze** | WR | CHI | **100** | 48.7 | Same Bears situation |
| **Malik Nabers** | WR | NYG | **100** | 46.7 | Giants WR targets vacated |
| **Jaylin Lane** | WR | WAS | **94** | 47.8 | Washington WR opportunity |
| **Treylon Burks** | WR | WAS | **94** | 45.7 | Same Washington situation |
| **Bhayshul Tuten** | RB | JAX | **85.6** | 44.6 | Jacksonville backfield churn |

---

## 🔍 Next Steps

### 1. Populate Historical Roster Data

For full accuracy, populate `roster_changes` and `vacated_opportunity` for all teams:

```sql
-- Example: Add FA departure
INSERT INTO roster_changes (player_id, player_name, position, old_team, new_team, 
                            change_type, season, last_season_targets)
VALUES ('9509', 'DJ Moore', 'WR', 'CAR', 'CHI', 'trade', 2023, 120);
```

This enables the opportunity_opened and competition signals (currently 0 for teams without data).

### 2. Frontend Integration

Add breakout badges to player cards:

```javascript
// Fetch breakout data for displayed players
const playerIds = displayedPlayers.map(p => p.id);
const breakoutData = await Promise.all(
  playerIds.map(id => fetch(`/api/breakout/player/${id}`).then(r => r.json()))
);

// Merge into UI
displayedPlayers.forEach((player, i) => {
  player.breakout = breakoutData[i];
});
```

### 3. Enable Automated Scoring

Set up cron job to keep scores fresh:

```bash
# Add to crontab
0 3 * * * cd /path/to/fantasy-dashboard && python3 -m data_building.breakout_engine.scheduler --cron
```

### 4. Historical Validation

Run backtest to validate model accuracy:

```bash
# Requires fantasy_rankings_{year}.json files for 2022-2024
python3 -m data_building.breakout_engine.backtest_breakout_model \
    --seasons 2022 2023 2024 \
    --output cache/backtest_results.json
```

---

## 📦 Files Modified/Created

### Database
- ✅ `data_building/breakout_engine/setup_database.sql` (NEW) - Complete schema

### API
- ✅ `dashboard_services/breakout_api.py` (NEW) - REST endpoints + classification
- ✅ `app.py` (MODIFIED) - Registered breakout routes

### Scheduling
- ✅ `data_building/breakout_engine/scheduler.py` (NEW) - Automated jobs

### Documentation
- ✅ `BREAKOUT_RESULTS_SUMMARY.md` (NEW) - Full analysis
- ✅ `data_building/breakout_engine/README.md` (NEW) - Integration guide
- ✅ `BREAKOUT_INTEGRATION_COMPLETE.md` (NEW) - This file

### Analysis Tools
- ✅ `data_building/breakout_engine/display_results.py` (NEW)
- ✅ `data_building/breakout_engine/analyze_results.py` (NEW)

---

## ✅ Integration Checklist

- [x] Database schema created
- [x] Tables populated with 2026 data
- [x] REST API endpoints functional
- [x] Breakout type classification implemented
- [x] Flask routes registered
- [x] Scheduler created (cron + daemon modes)
- [x] Documentation complete
- [x] API tested and working
- [x] Command-line tools working

**Status: PRODUCTION READY** 🎉

---

## 🐛 Troubleshooting

### API Returns Empty Results

**Check database:**
```sql
SELECT COUNT(*) FROM breakout_opportunity_scores WHERE season = 2026;
```

**Repopulate if needed:**
```bash
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data
```

### All Opportunity Scores are 0

**Expected behavior** if roster_changes table is empty. Scores still work via renormalization, but:
- Opportunity_opened_score = 0
- Competition signals = 0
- Confidence reduced to 51-71%

**Solution:** Populate roster_changes data for better accuracy.

### Scheduler Not Running

**Check cron:**
```bash
crontab -l | grep breakout
```

**Check logs:**
```bash
tail -f /var/log/breakout.log
```

**Test manually:**
```bash
python3 -m data_building.breakout_engine.scheduler --run-now --dry-run
```

---

## 📞 Support

For questions or issues:
1. See `data_building/breakout_engine/README.md` for detailed docs
2. Check `BREAKOUT_RESULTS_SUMMARY.md` for analysis
3. Review code comments in `core.py` and `components.py`
4. Test with `--verbose` and `--dry-run` flags

---

**Integration completed on:** 2026-04-05
**Status:** ✅ Production Ready
**Next action:** Deploy and monitor automated scoring

---

## 🔄 API Route Migration

### New Routes (Recommended - Use These)

The new Breakout Engine routes use `/api/breakout/` prefix and provide comprehensive functionality:

```
GET /api/breakout/candidates              - All candidates with breakout type classification
GET /api/breakout/candidates/{position}   - Filter by QB/RB/WR/TE
GET /api/breakout/player/{player_id}      - Detailed player view with component breakdown
GET /api/breakout/statistics              - Aggregate stats and leaderboards
GET /api/breakout/team/{team}             - Team roster situation
```

### Old Routes (Deprecated - Still Available)

The old routes remain for backward compatibility but use older detection methods:

```
GET /api/breakout-candidates                        - Old detection logic
GET /api/advanced-metrics/breakout-candidates       - Advanced metrics (older)
GET /api/offseason-breakout-candidates              - Offseason only (older)
GET /api/calculate-breakout-scores                  - Manual calculation trigger
```

**Recommendation:** Migrate to the new `/api/breakout/` routes for:
- ✅ Breakout type classification (opportunity vs readiness)
- ✅ Component score breakdown
- ✅ Confidence scoring
- ✅ Better data quality (fixes applied)
- ✅ Consistent schema across all endpoints

---

## 🛠️ Technical Details - Blueprint Implementation

The new routes use Flask Blueprint to avoid naming conflicts:

```python
# In dashboard_services/breakout_api.py
from flask import Blueprint

breakout_bp = Blueprint('breakout', __name__, url_prefix='/api/breakout')

@breakout_bp.route('/candidates')
def candidates():
    # Scoped as breakout.candidates (no conflict with old api_breakout_candidates)
    ...
```

This allows both old and new routes to coexist during migration period.
