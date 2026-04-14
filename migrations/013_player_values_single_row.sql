-- Migration 013: Convert player_values to one-row-per-player
-- Previously (player_id, date) was the PK, accumulating a new row each day.
-- Now player_id is the sole PK; last_updated records when the value was refreshed.

-- Step 1: Keep only the latest snapshot per player (drop all older rows)
DELETE FROM player_values pv1
USING (
    SELECT player_id, MAX(date) AS max_date
    FROM player_values
    GROUP BY player_id
) latest
WHERE pv1.player_id = latest.player_id
  AND pv1.date < latest.max_date;

-- Step 2: Drop the old compound primary key
ALTER TABLE player_values DROP CONSTRAINT player_values_pkey;

-- Step 3: Rename date → last_updated to reflect the new semantics
ALTER TABLE player_values RENAME COLUMN date TO last_updated;

-- Step 4: New primary key on player_id only
ALTER TABLE player_values ADD PRIMARY KEY (player_id);

-- Step 5: Update the date index to reflect the new column name
DROP INDEX IF EXISTS idx_player_values_date;
CREATE INDEX IF NOT EXISTS idx_player_values_last_updated ON player_values(last_updated);
