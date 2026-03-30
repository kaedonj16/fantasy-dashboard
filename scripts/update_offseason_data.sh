#!/bin/bash
#
# Daily script to update offseason breakout data
# Detects roster changes and recalculates breakout projections
#
# Add to crontab with:
# 0 6 * * * /Users/kaedonjenkins/IdeaProjects/fantasy-dashboard/scripts/update_offseason_data.sh

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Set database URL
export DATABASE_URL="postgresql://$USER@localhost:5432/brfantasy"

# Log file
LOG_FILE="$PROJECT_DIR/logs/offseason_updates.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Get current season (assuming we're always updating for next year during offseason)
CURRENT_YEAR=$(date +%Y)
NEXT_SEASON=$((CURRENT_YEAR + 1))

echo "[$(date)] Starting offseason data update for season $NEXT_SEASON" >> "$LOG_FILE"

# Run the update
python3 -c "
import sys
sys.path.insert(0, '.')
from data_building.populate_roster_changes import populate_offseason_data

try:
    populate_offseason_data($NEXT_SEASON)
    print('✓ Offseason data update completed successfully')
except Exception as e:
    print(f'✗ Error updating offseason data: {e}')
    sys.exit(1)
" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ Update completed successfully" >> "$LOG_FILE"
else
    echo "[$(date)] ✗ Update failed with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"

exit $EXIT_CODE
