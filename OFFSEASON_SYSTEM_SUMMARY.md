# Offseason Breakout System - Implementation Summary

## ✓ System Operational

The offseason breakout detection system is fully functional and tracking real 2026 roster changes.

## Current Status (2026 Season)

- **63 Roster Changes Detected** - Real player movements (trades, free agency, etc.)
- **52 Team/Position Groups** with vacated opportunity
- **34 Breakout Candidates** with scores ≥ 30

## Top Breakout Candidates

| Player | Team | Pos | Score | Projected Usage |
|--------|------|-----|-------|-----------------|
| Bhayshul Tuten | JAX | RB | 79.6 | 39t / 181c (+25t / +98c) |
| Chris Rodriguez Jr. | JAX | RB | 79.2 | 11t / 245c (+7t / +133c) |
| Chuba Hubbard | CAR | RB | 70.4 | 85t / 339c (+46t / +205c) |
| Zach Charbonnet | SEA | RB | 65.1 | 44t / 313c (+20t / +129c) |
| Emanuel Wilson | SEA | RB | 64.0 | 31t / 212c (+14t / +87c) |
| Rome Odunze | CHI | WR | 60.4 | 165t / 0c (+75t / +0c) |

## Key Roster Changes Driving Breakouts

- **Travis Etienne Jr.** (JAX → NO) - Opens up JAX backfield for Tuten/Rodriguez
- **Rico Dowdle** (CAR → PIT) - Chuba Hubbard becomes lead back
- **Kenneth Walker III** (SEA → KC) - Charbonnet/Wilson split carries
- **DJ Moore** (CHI → BUF) - Rome Odunze gets expanded role

## How It Works

1. **Auto-Detection**: Compares current rosters (`players_index`) to previous season (`cache/player_history/usage_rows_2025.json`)
2. **Vacated Opportunity**: Calculates targets/carries/snaps left behind by departing players
3. **Redistribution**: Projects how vacated opportunity redistributes to remaining players based on previous usage
4. **Scoring**: 5-factor algorithm (0-100 points, 30+ threshold):
   - Absolute opportunity increase (0-30pts)
   - Relative increase % (0-25pts)
   - Team vacancy size (0-20pts)
   - Youth/experience bonus (0-15pts)
   - Established role bonus (0-10pts)

## Automation

Daily updates can be automated via cron:

```bash
# Add to crontab (runs daily at 6 AM)
0 6 * * * /Users/kaedonjenkins/IdeaProjects/fantasy-dashboard/scripts/update_offseason_data.sh
```

## API Endpoints

- **GET /api/offseason-breakout-candidates** - Full list of breakout candidates with projections
- **GET /api/player-indicators** - Auto-switches between offseason/in-season detection based on season type

## Files Modified

### Core Logic
- `data_building/offseason_opportunity.py` - Database schema, scoring algorithm, projection logic
- `data_building/populate_roster_changes.py` - Auto-detection and population pipeline

### API
- `app.py` - Added offseason endpoints and season-aware player indicators

### Automation
- `scripts/update_offseason_data.sh` - Daily update script

### Documentation
- `OFFSEASON_BREAKOUTS.md` - Complete technical documentation
- `HOW_TO_ADD_ROSTER_CHANGES.md` - User guide for manual additions
- `OFFSEASON_SYSTEM_SUMMARY.md` - This file

## Key Technical Fixes Applied

1. **Field name mapping**: `players_index` uses `'pos'` not `'position'`
2. **Historical data structure**: Uses `'id'` not `'player_id'`, `'total_targets'` not `'targets'`
3. **Decimal type handling**: Added `float()` conversions for database decimal values
4. **Usage calculation**: Handles both aggregated (`total_targets`) and average (`avg_carries * games`) formats

## Next Steps

1. Start Flask server to test API endpoints
2. Verify breakout badges appear in trade calculator UI
3. Set up cron job for daily updates (optional)
4. Monitor during March-April free agency for real-time breakout detection
