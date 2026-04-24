-- Add route_target_rate from Reception Perception Target Data.
-- Measures how often a player is targeted per route run (RP's version of tprr).
-- Stored as a percentage (e.g. 28.5 = 28.5% target rate per route).
ALTER TABLE rookie_prospect_source_data
    ADD COLUMN IF NOT EXISTS route_target_rate DECIMAL(5,1);
