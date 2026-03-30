# How to Add Real Roster Changes for Offseason Breakout Detection

The offseason breakout system automatically detects roster changes and calculates breakout projections.

## Automatic Daily Updates

The system can run automatically via cron to check for roster changes daily:

```bash
# Add to your crontab (runs daily at 6 AM)
crontab -e

# Add this line:
0 6 * * * /Users/kaedonjenkins/IdeaProjects/fantasy-dashboard/scripts/update_offseason_data.sh
```

This will:
1. Detect new roster changes by comparing current rosters to previous season
2. Calculate vacated opportunity for each team/position
3. Project opportunity redistribution to remaining players
4. Update breakout scores for all affected players

Logs are saved to: `logs/offseason_updates.log`

## When to Update

The automatic detection runs during these periods:
- **March-April:** Free agency period (majority of moves)
- **May-July:** Post-draft adjustments
- **August:** Final cuts and signings before season

## Method 1: Find Player IDs in Your Database

First, find the correct player IDs from your database:

```bash
export DATABASE_URL="postgresql://$USER@localhost:5432/brfantasy"

# Search for a player by name
python3 -c "
import sys
sys.path.insert(0, '.')
from utils.utils import load_players_index

players = load_players_index()

# Search for a player
search = 'Saquon Barkley'
matches = [(pid, p) for pid, p in players.items() if search.lower() in p.get('name', '').lower()]

for pid, player in matches:
    print(f'ID: {pid:6} | {player.get(\"name\"):25} | {player.get(\"team\"):3} | {player.get(\"position\"):2} | Age: {player.get(\"age\")}')
"
```

## Method 2: Add a Specific Roster Change

Once you have the player ID, add the roster change:

```python
python3 -c "
import sys
sys.path.insert(0, '.')
from datetime import date
from data_building.populate_roster_changes import manual_add_roster_change

# Example: Saquon Barkley leaves NYG for PHI
manual_add_roster_change(
    player_name='Saquon Barkley',
    old_team='NYG',
    new_team='PHI',
    change_type='free_agent',  # or 'trade', 'retirement', 'cut'
    season=2026
)
"
```

This will:
1. Find Saquon in players_index
2. Load his 2025 usage stats (targets, carries, snap share)
3. Save to `roster_changes` table

## Method 3: Batch Update After Free Agency

After free agency concludes, run the automatic detection:

```bash
python data_building/populate_roster_changes.py 2026
```

This compares player teams between 2025 and 2026 and detects all changes automatically.

## Real-World Example Workflow

### Scenario: Mike Evans signs with Cowboys

**Step 1: Find Mike Evans' player ID**
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from utils.utils import load_players_index

players = load_players_index()
matches = [(pid, p) for pid, p in players.items() if 'mike evans' in p.get('name', '').lower()]

for pid, player in matches:
    print(f'{pid}: {player.get(\"name\")} - {player.get(\"team\")} {player.get(\"position\")}')
"
```

Output: `4040: Mike Evans - TB WR`

**Step 2: Add the roster change**
```python
python3 -c "
import sys
sys.path.insert(0, '.')
from datetime import date
from data_building.populate_roster_changes import manual_add_roster_change

manual_add_roster_change(
    player_name='Mike Evans',
    old_team='TB',
    new_team='DAL',
    change_type='free_agent',
    season=2026,
    change_date=date(2026, 3, 15)
)
"
```

**Step 3: Calculate vacated opportunity**
```python
python3 -c "
import sys
sys.path.insert(0, '.')
from data_building.offseason_opportunity import (
    calculate_vacated_opportunity,
    project_opportunity_redistribution
)

# Recalculate for 2026
calculate_vacated_opportunity(2026)
project_opportunity_redistribution(2026, top_n_players=600)
"
```

**Step 4: Check results**
```bash
curl http://localhost:5000/api/offseason-breakout-candidates
```

You should now see TB WRs (like Jalen McMillan) as breakout candidates!

## Common Change Types

### Free Agent Signing
```python
change_type='free_agent'
old_team='TB'
new_team='DAL'
```

### Trade
```python
change_type='trade'
old_team='ATL'
new_team='PHI'
```

### Retirement
```python
change_type='retirement'
old_team='KC'
new_team=None  # No new team for retirement
```

### Cut/Released
```python
change_type='cut'
old_team='WAS'
new_team='FA'  # Free agent pool
```

## Quick Reference Script

Save this as `add_roster_change.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from datetime import date
from data_building.populate_roster_changes import manual_add_roster_change
from data_building.offseason_opportunity import (
    calculate_vacated_opportunity,
    project_opportunity_redistribution
)

# Add roster change
manual_add_roster_change(
    player_name='PLAYER_NAME',  # ← Change this
    old_team='OLD',             # ← Change this
    new_team='NEW',             # ← Change this (or None for retirement)
    change_type='free_agent',   # ← free_agent, trade, retirement, cut
    season=2026
)

# Recalculate projections
calculate_vacated_opportunity(2026)
project_opportunity_redistribution(2026, top_n_players=600)

print("\n✓ Roster change added and projections updated!")
print("Check: http://localhost:5000/api/offseason-breakout-candidates")
```

## Verification

After adding roster changes, verify they're working:

```bash
# Check player indicators endpoint (should show breakouts)
curl http://localhost:5000/api/player-indicators

# Check full details
curl http://localhost:5000/api/offseason-breakout-candidates | python3 -m json.tool

# Check database directly
python3 -c "
import sys
sys.path.insert(0, '.')
from dashboard_services.db import get_conn

with get_conn() as conn:
    changes = conn.execute('SELECT player_name, old_team, new_team, change_type FROM roster_changes').fetchall()
    for c in changes:
        print(f'{c[\"player_name\"]:25} {c[\"old_team\"]} → {c[\"new_team\"] or \"N/A\":3} ({c[\"change_type\"]})')
"
```

## Notes

- **Top 600 limit**: Only tracks opportunity for top 600 players by dynasty value
- **Previous season stats**: Automatically loaded from last season's usage_table
- **Recalculation**: Run vacated opportunity + redistribution after each batch of changes
- **Season timing**: Update in March/April for maximum value (before draft, before values spike)
