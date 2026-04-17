-- Migration 016: Add height and weight to athleticism table
-- These fields are processed from combine data but need to be stored for display

-- Add height and weight columns to rookie_prospect_athleticism table
ALTER TABLE rookie_prospect_athleticism 
ADD COLUMN IF NOT EXISTS height_inches DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS weight_lbs INTEGER;
