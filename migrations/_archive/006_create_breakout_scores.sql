-- Migration 006: Create breakout opportunity scores table and add draft metadata to roster changes
--
-- This migration creates the unified breakout scoring system with:
-- 1. roster_changes table (if not exists) for tracking player movements
-- 2. vacated_opportunity table (if not exists) for tracking departed usage
-- 3. New breakout_opportunity_scores table for storing all component scores
-- 4. Add draft_metadata column to roster_changes table for tracking draft picks

-- Step 1: Create roster_changes table (prerequisite)
CREATE TABLE IF NOT EXISTS roster_changes (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(255),
    position VARCHAR(5),
    old_team VARCHAR(10),
    new_team VARCHAR(10),
    change_type VARCHAR(20),  -- 'free_agent', 'trade', 'retirement', 'cut', 'draft'
    change_date DATE,
    season INT,

    -- Usage stats from previous season (what's being vacated)
    last_season_targets INT,
    last_season_carries INT,
    last_season_snap_share NUMERIC,
    last_season_opportunity_share NUMERIC,
    last_season_team_target_pct NUMERIC,
    last_season_team_carry_pct NUMERIC,

    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, old_team, new_team, season)
);

-- Step 2: Create vacated_opportunity table (prerequisite)
CREATE TABLE IF NOT EXISTS vacated_opportunity (
    id SERIAL PRIMARY KEY,
    team VARCHAR(10) NOT NULL,
    position VARCHAR(5) NOT NULL,
    season INT NOT NULL,

    total_targets_vacated INT DEFAULT 0,
    total_carries_vacated INT DEFAULT 0,
    total_snap_share_vacated NUMERIC DEFAULT 0.0,

    -- Store list of departed players as JSONB
    departed_players JSONB,

    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team, position, season)
);

-- Step 3: Create projected_opportunity table (optional - legacy)
CREATE TABLE IF NOT EXISTS projected_opportunity (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    season INT NOT NULL,
    team VARCHAR(10),
    position VARCHAR(5),

    baseline_targets INT DEFAULT 0,
    projected_targets INT DEFAULT 0,
    baseline_carries INT DEFAULT 0,
    projected_carries INT DEFAULT 0,
    baseline_snap_share NUMERIC DEFAULT 0.0,
    projected_snap_share NUMERIC DEFAULT 0.0,

    breakout_score NUMERIC DEFAULT 0.0,
    projection_factors JSONB,

    calculated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, season)
);

-- Step 4: Create breakout_opportunity_scores table
CREATE TABLE IF NOT EXISTS breakout_opportunity_scores (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    season INT NOT NULL,
    as_of_date DATE NOT NULL,

    -- Position and team context
    team VARCHAR(10),
    position VARCHAR(5),

    -- Component scores (0-100 each, except competition_added_penalty can be negative)
    opportunity_opened_score NUMERIC,
    competition_removed_score NUMERIC,
    competition_added_penalty NUMERIC,  -- Can be negative (0 to -50)
    team_environment_score NUMERIC,
    player_readiness_score NUMERIC,
    role_trajectory_score NUMERIC,
    confidence_score NUMERIC,

    -- Aggregate score (weighted sum based on phase, 0-100)
    breakout_opportunity_score NUMERIC,

    -- Metadata
    phase VARCHAR(20),  -- 'offseason', 'post_free_agency', 'post_draft', 'preseason', 'in_season'
    directional_trend VARCHAR(10),  -- 'rising', 'falling', 'stable'

    -- Explainability fields
    key_reasons TEXT,
    recent_transactions_affecting_player TEXT,
    vacated_usage_summary TEXT,
    added_competition_summary TEXT,
    projected_role_tag VARCHAR(100),

    -- Component details stored as JSONB for each component's breakdown
    component_details JSONB,

    -- Timestamps
    calculated_at TIMESTAMP DEFAULT NOW(),

    -- Ensure one score per player per season per date
    UNIQUE(player_id, season, as_of_date)
);

-- Step 2: Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_breakout_scores_player
    ON breakout_opportunity_scores(player_id);

CREATE INDEX IF NOT EXISTS idx_breakout_scores_season
    ON breakout_opportunity_scores(season);

CREATE INDEX IF NOT EXISTS idx_breakout_scores_date
    ON breakout_opportunity_scores(as_of_date);

CREATE INDEX IF NOT EXISTS idx_breakout_scores_score
    ON breakout_opportunity_scores(breakout_opportunity_score DESC);

CREATE INDEX IF NOT EXISTS idx_breakout_scores_position
    ON breakout_opportunity_scores(position);

-- Step 3: Add draft_metadata column to roster_changes table
ALTER TABLE roster_changes
ADD COLUMN IF NOT EXISTS draft_metadata JSONB;

-- draft_metadata structure:
-- {
--   "round": 1,
--   "pick": 15,
--   "overall_pick": 15,
--   "college": "Ohio State"
-- }

-- Step 4: Add comment for documentation
COMMENT ON TABLE breakout_opportunity_scores IS
'Unified breakout opportunity scoring system. Stores 7 component scores that adapt based on NFL calendar phase (offseason, post-draft, in-season). Includes explainability fields for user-facing text.';

COMMENT ON COLUMN breakout_opportunity_scores.opportunity_opened_score IS
'Score (0-100) based on total opportunity vacated from team/position (targets, carries, snaps)';

COMMENT ON COLUMN breakout_opportunity_scores.competition_removed_score IS
'Score (0-100) based on specific high-value competitors who departed';

COMMENT ON COLUMN breakout_opportunity_scores.competition_added_penalty IS
'Negative score (0 to -50) for new competition from draft picks, signings, trades';

COMMENT ON COLUMN breakout_opportunity_scores.team_environment_score IS
'Score (0-100) based on offensive environment quality (pace, pass rate, QB quality)';

COMMENT ON COLUMN breakout_opportunity_scores.player_readiness_score IS
'Score (0-100) based on player ability to capitalize (age, efficiency, draft capital, usage history)';

COMMENT ON COLUMN breakout_opportunity_scores.role_trajectory_score IS
'Score (0-100) based on recent usage trends (in-season only, neutral 50 in offseason)';

COMMENT ON COLUMN breakout_opportunity_scores.confidence_score IS
'Score (0-100) indicating projection certainty (sample size, data completeness, phase)';

COMMENT ON COLUMN breakout_opportunity_scores.phase IS
'NFL calendar phase: offseason, post_free_agency, post_draft, preseason, in_season';

COMMENT ON COLUMN breakout_opportunity_scores.component_details IS
'JSONB containing detailed breakdowns for each component score';

COMMENT ON COLUMN roster_changes.draft_metadata IS
'JSONB containing draft information for drafted players: round, pick, overall_pick, college';
