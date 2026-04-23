-- Migration 005: Create luck_index table for luck vs skill analysis
-- This table stores season-long luck metrics for each team

CREATE TABLE IF NOT EXISTS luck_index (
    league_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    roster_id INTEGER NOT NULL,
    team_name TEXT,

    -- Schedule luck metrics
    avg_opponent_score DECIMAL(10,2),
    league_avg_opponent_score DECIMAL(10,2),
    schedule_luck_score DECIMAL(5,2), -- -100 to +100 (negative = unlucky)

    -- Close game metrics
    close_game_wins INTEGER,
    close_game_losses INTEGER,
    close_game_luck_score DECIMAL(5,2), -- -100 to +100

    -- Optimal lineup metrics
    actual_points DECIMAL(10,2),
    optimal_points DECIMAL(10,2),
    lineup_efficiency DECIMAL(5,2), -- 0-100 percentage

    -- Overall composite score
    overall_luck_score DECIMAL(5,2), -- 0-100 (50 = average luck)
    luck_tier TEXT, -- 'Very Lucky', 'Lucky', 'Average', 'Unlucky', 'Very Unlucky'

    -- Metadata
    weeks_analyzed INTEGER,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (league_id, season, roster_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_luck_index_league ON luck_index(league_id, season);
CREATE INDEX IF NOT EXISTS idx_luck_index_tier ON luck_index(luck_tier);

-- Comments
COMMENT ON TABLE luck_index IS 'Season-long luck vs skill analysis for fantasy teams';
COMMENT ON COLUMN luck_index.schedule_luck_score IS 'Did you face tough/easy opponents? -100 (unlucky) to +100 (lucky)';
COMMENT ON COLUMN luck_index.close_game_luck_score IS 'Win% in games decided by <10 points, -100 (unlucky) to +100 (lucky)';
COMMENT ON COLUMN luck_index.lineup_efficiency IS 'Percentage of optimal points scored (actual/optimal * 100)';
COMMENT ON COLUMN luck_index.overall_luck_score IS 'Composite luck score: 0 (very unlucky) to 100 (very lucky), 50 = average';
