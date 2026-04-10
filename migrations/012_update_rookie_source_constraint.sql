-- Migration 012: Update rookie_prospect_source_data unique constraint
-- Changes unique constraint from (player_id, season, source) to (player_id, season)
-- This allows updating existing records regardless of source instead of creating duplicates

-- Drop existing unique constraint
ALTER TABLE rookie_prospect_source_data DROP CONSTRAINT IF EXISTS rookie_prospect_source_data_player_id_season_source_key;

-- Add new unique constraint without source
ALTER TABLE rookie_prospect_source_data ADD CONSTRAINT rookie_prospect_source_data_player_id_season_key UNIQUE(player_id, season);
