-- Migration 010: Fix rookie_mock_draft_entries unique constraint
--
-- Problem: UNIQUE(player_id, source_name, mock_date) collapses all analyst picks
-- for the same player on the same day to a single row. When CBS Sports scrapes 6
-- analyst mocks in one run, each ON CONFLICT DO UPDATE overwrites the previous
-- analyst's pick, leaving only the last analyst's data in the table.
--
-- Fix: Include analyst_name in the constraint so each analyst's pick for a player
-- is stored as a separate row. COALESCE to '' handles any legacy NULL values.

-- Step 1: Drop the old constraint
ALTER TABLE rookie_mock_draft_entries
    DROP CONSTRAINT IF EXISTS rookie_mock_draft_entries_player_id_source_name_mock_date_key;

-- Step 2: Add the new constraint that includes analyst_name
-- Use a partial unique index with COALESCE so NULLs don't create duplicates
CREATE UNIQUE INDEX IF NOT EXISTS uq_mock_entry_per_analyst
    ON rookie_mock_draft_entries
    (player_id, source_name, mock_date, COALESCE(analyst_name, ''));
