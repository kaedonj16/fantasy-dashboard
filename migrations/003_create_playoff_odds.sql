-- Migration 003: Create playoff_odds table for Monte Carlo simulation results
-- This table stores weekly playoff probability calculations for each team

CREATE TABLE IF NOT EXISTS playoff_odds (
    league_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    roster_id INTEGER NOT NULL,
    team_name TEXT,

    -- Current standings
    current_wins INTEGER,
    current_losses INTEGER,
    current_ties INTEGER DEFAULT 0,

    -- Playoff probabilities (0-100%)
    playoff_probability DECIMAL(5,2),
    first_seed_probability DECIMAL(5,2),
    bye_probability DECIMAL(5,2),
    miss_playoffs_probability DECIMAL(5,2),

    -- Projected final records
    avg_final_wins DECIMAL(5,2),
    avg_final_losses DECIMAL(5,2),

    -- Metadata
    num_simulations INTEGER DEFAULT 10000,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (league_id, season, week, roster_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_playoff_odds_league_season ON playoff_odds(league_id, season, week);
CREATE INDEX IF NOT EXISTS idx_playoff_odds_team ON playoff_odds(roster_id);

-- Comments
COMMENT ON TABLE playoff_odds IS 'Weekly playoff probability calculations using Monte Carlo simulation';
COMMENT ON COLUMN playoff_odds.playoff_probability IS 'Probability of making playoffs (0-100%)';
COMMENT ON COLUMN playoff_odds.first_seed_probability IS 'Probability of earning #1 seed (0-100%)';
COMMENT ON COLUMN playoff_odds.num_simulations IS 'Number of Monte Carlo simulations run (default 10,000)';
