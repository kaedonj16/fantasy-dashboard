-- Migration 008: Update projected_opportunity table schema
--
-- This migration updates the projected_opportunity table to match
-- the columns expected by breakout_workflow.py, adding:
-- - player_name field
-- - prev_season_* fields (renaming from baseline_*)
-- - *_increase fields for deltas
-- - prev_season_opportunity_share field

-- Step 1: Add new columns
ALTER TABLE projected_opportunity
ADD COLUMN IF NOT EXISTS player_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS prev_season_targets INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS prev_season_carries INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS prev_season_snap_share NUMERIC DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS prev_season_opportunity_share NUMERIC DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS projected_opportunity_share NUMERIC DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS target_increase INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS carry_increase INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS snap_share_increase NUMERIC DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS opportunity_share_increase NUMERIC DEFAULT 0.0;

-- Step 2: Migrate data from old columns to new columns (if any data exists)
UPDATE projected_opportunity
SET prev_season_targets = baseline_targets,
    prev_season_carries = baseline_carries,
    prev_season_snap_share = baseline_snap_share
WHERE prev_season_targets = 0
  AND prev_season_carries = 0
  AND prev_season_snap_share = 0
  AND baseline_targets IS NOT NULL;

-- Step 3: Keep old columns for backward compatibility (don't drop them)
-- The columns baseline_targets, baseline_carries, baseline_snap_share,
-- projected_targets, projected_carries, projected_snap_share remain

-- Step 4: Add comment for documentation
COMMENT ON TABLE projected_opportunity IS
'Player opportunity projections based on roster changes and breakout scores.
Contains both baseline (prev_season_*) and projected values with deltas (*_increase).';

COMMENT ON COLUMN projected_opportunity.prev_season_snap_share IS
'Player snap share from previous season (0-1 decimal).
Estimated from usage when real snap data unavailable.';

COMMENT ON COLUMN projected_opportunity.prev_season_opportunity_share IS
'Player opportunity share from previous season (0-1 decimal).
Calculated from targets and carries relative to team totals.';
