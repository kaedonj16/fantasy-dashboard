-- Migration 002: Create player_values table for daily value tracking
-- This table stores daily snapshots of all player values for historical tracking

CREATE TABLE IF NOT EXISTS player_values (
    player_id TEXT NOT NULL,
    date DATE NOT NULL,
    value_1qb DECIMAL(10,2),
    value_sf DECIMAL(10,2),
    position TEXT,
    pos_rank INTEGER,
    pos_rank_label TEXT,
    age DECIMAL(5,2),
    team TEXT,
    years_exp INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (player_id, date)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_player_values_date ON player_values(date);
CREATE INDEX IF NOT EXISTS idx_player_values_player ON player_values(player_id);
CREATE INDEX IF NOT EXISTS idx_player_values_position ON player_values(position, date);

-- Comments
COMMENT ON TABLE player_values IS 'Daily snapshots of dynasty player values for historical tracking and trend analysis';
COMMENT ON COLUMN player_values.player_id IS 'Sleeper player ID';
COMMENT ON COLUMN player_values.date IS 'Date of this value snapshot';
COMMENT ON COLUMN player_values.value_1qb IS '1QB league dynasty value (0-999.9)';
COMMENT ON COLUMN player_values.value_sf IS 'Superflex league dynasty value (0-999.9)';
