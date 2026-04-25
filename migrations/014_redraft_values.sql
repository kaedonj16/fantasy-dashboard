-- Add redraft value columns to player_values.
-- Populated daily from FantasyCalc API (redraftValue field).

ALTER TABLE player_values ADD COLUMN IF NOT EXISTS redraft_value_1qb DECIMAL(10,2);
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS redraft_value_sf  DECIMAL(10,2);

CREATE INDEX IF NOT EXISTS idx_pv_redraft ON player_values(redraft_value_1qb DESC NULLS LAST);
