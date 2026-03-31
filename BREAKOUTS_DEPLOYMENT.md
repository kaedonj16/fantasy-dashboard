# Deploying Breakout Candidates Feature

## Quick Fix for Production

If you're seeing "No breakout candidates found" on your deployed website, run this on your production server:

```bash
# Set your production database URL
export DATABASE_URL="postgresql://user@host:5432/dbname"

# Navigate to project directory
cd /path/to/fantasy-dashboard

# Run the population script
python3 scripts/populate_offseason_breakouts.py
```

**Expected output:**
```
============================================================
POPULATING OFFSEASON BREAKOUT DATA
============================================================

Step 1: Initializing database tables...
✓ Tables created/verified

Step 2: Populating data for 2026 season...
[populate_roster_changes] Found 63 roster changes
[offseason] Calculated vacated opportunity for 52 team/position groups
[offseason] Saved 34 opportunity projections

============================================================
✓ DEPLOYMENT COMPLETE
============================================================

Results:
  - Roster Changes: 63
  - Breakout Candidates: 34

The breakouts tab should now show 34 candidates!
```

## What This Does

1. **Creates database tables** (if they don't exist):
   - `roster_changes` - Tracks player movements
   - `vacated_opportunity` - Calculates targets/carries left behind
   - `projected_opportunity` - Projects breakout candidates

2. **Populates 2026 data** by:
   - Detecting roster changes (comparing current rosters to 2025 historical data)
   - Calculating vacated opportunity per team/position
   - Projecting opportunity redistribution to remaining players
   - Scoring breakout potential (0-100, 30+ shown)
   - Filtering candidates by:
     * Age ≤ 25 years old
     * Dynasty value < 2000 (not already a star)
     * Not already elite (position rank > 5)
     * Position-specific rank limits: QB ≤ 32, RB ≤ 45, WR ≤ 60, TE ≤ 20

3. **Enables the Breakouts tab** in the trade calculator

## Requirements

### Data Files Needed

The script requires these files in your deployment:
- `cache/player_history/usage_rows_2025.json` - Previous season usage stats
- `cache/players_index.json` - Current player roster data

If these are missing, the script will detect 0 roster changes and create 0 breakouts.

### Database Requirements

Your PostgreSQL database needs to be accessible via `DATABASE_URL`. The script will create the necessary tables automatically.

## Add to Your Deployment Process

### Option 1: Manual Run (Recommended for First Deploy)

```bash
python3 scripts/populate_offseason_breakouts.py
```

### Option 2: Add to Deployment Script

```bash
#!/bin/bash
# deploy.sh

export DATABASE_URL="your_production_db_url"
cd /path/to/fantasy-dashboard

# ... your other deployment steps ...

# Populate offseason breakouts
echo "Populating breakout candidates..."
python3 scripts/populate_offseason_breakouts.py || echo "Warning: Breakouts population failed"

# Start server
gunicorn app:app
```

### Option 3: Cron Job for Daily Updates

During offseason (March-August), roster changes happen frequently. Set up a daily update:

```bash
crontab -e

# Add this line (runs daily at 6 AM):
0 6 * * * cd /path/to/fantasy-dashboard && /usr/bin/python3 scripts/populate_offseason_breakouts.py >> /var/log/breakouts-update.log 2>&1
```

## Verification

### Check API Endpoint

```bash
curl https://your-domain.com/api/offseason-breakout-candidates?limit=5
```

You should see JSON with breakout candidates:
```json
[
  {
    "player_id": "11625",
    "name": "Bhayshul Tuten",
    "team": "JAX",
    "position": "RB",
    "age": 23.1,
    "breakout_score": 79.6,
    ...
  },
  ...
]
```

### Check Frontend

1. Open trade calculator
2. Click "Player Insights" panel in the right sidebar
3. Click "Breakouts" tab
4. Should see young players with expanded role opportunities

## Troubleshooting

### "No breakout candidates found"

**Cause**: Database tables are empty

**Fix**: Run the population script:
```bash
python3 scripts/populate_offseason_breakouts.py
```

### "ModuleNotFoundError: No module named 'data_building'"

**Cause**: Script not run from project root

**Fix**: Ensure you're in the project directory:
```bash
cd /path/to/fantasy-dashboard
python3 scripts/populate_offseason_breakouts.py
```

### "DATABASE_URL environment variable not set"

**Cause**: Database connection string not configured

**Fix**: Export your database URL:
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

### "0 roster changes detected"

**Cause**: Missing historical data files

**Fix**: Ensure these files exist:
- `cache/player_history/usage_rows_2025.json`
- `cache/players_index.json`

If missing, copy from your development environment to production.

### Frontend shows loading spinner forever

**Cause**: API endpoint error or network issue

**Fix**:
1. Check browser console for errors
2. Check Flask logs for API errors
3. Verify API endpoint is accessible:
   ```bash
   curl http://localhost:5000/api/offseason-breakout-candidates
   ```

## Data Updates

The breakout candidates are based on:
- **Current rosters**: From `players_index.json` (updated by daily Sleeper sync)
- **Historical stats**: From `cache/player_history/usage_rows_2025.json` (static, from 2025 season)

To update for new roster changes, re-run the population script. It will:
- Detect new roster changes
- Recalculate vacated opportunity
- Update breakout projections

## Season Transition

When the season starts (Week 1), the breakouts tab automatically switches to in-season breakout detection based on performance metrics instead of roster changes.

No action needed - the system detects season type automatically via NFL state API.
