-- ============================================================================
-- Breakout Detection Engine - Database Schema Setup
-- ============================================================================
--
-- This script creates all necessary tables for the breakout detection engine:
-- 1. breakout_opportunity_scores - Stores calculated breakout scores
-- 2. vacated_opportunity - Tracks departed player usage by team/position
-- 3. roster_changes - Records all roster movements (FA, trades, draft, cuts)
--
-- Safe to run multiple times (uses CREATE TABLE IF NOT EXISTS)
-- ============================================================================

-- ============================================================================
-- 1. BREAKOUT OPPORTUNITY SCORES TABLE
-- ============================================================================
-- Stores breakout scores for all candidates by season and date
-- Primary key: (player_id, season, as_of_date) allows tracking score changes over time

CREATE TABLE IF NOT EXISTS breakout_opportunity_scores (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(255),
    season INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    team VARCHAR(10),
    position VARCHAR(5),

    -- Component scores (0-100 scale)
    opportunity_opened_score NUMERIC,
    competition_removed_score NUMERIC,
    competition_added_penalty NUMERIC,
    team_environment_score NUMERIC,
    player_readiness_score NUMERIC,
    role_trajectory_score NUMERIC,
    confidence_score NUMERIC,
    breakout_opportunity_score NUMERIC,

    -- Metadata
    phase VARCHAR(20),
    directional_trend VARCHAR(10),

    -- Explainability fields
    key_reasons TEXT,
    recent_transactions_affecting_player TEXT,
    vacated_usage_summary TEXT,
    added_competition_summary TEXT,
    projected_role_tag VARCHAR(100),
    component_details JSONB,

    calculated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(player_id, season, as_of_date)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_breakout_scores_player ON breakout_opportunity_scores(player_id);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_season ON breakout_opportunity_scores(season);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_date ON breakout_opportunity_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_position ON breakout_opportunity_scores(position);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_score ON breakout_opportunity_scores(breakout_opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_team ON breakout_opportunity_scores(team);


-- ============================================================================
-- 2. VACATED OPPORTUNITY TABLE
-- ============================================================================
-- Tracks departed player usage by team/position/season
-- Used to calculate opportunity_opened_score

CREATE TABLE IF NOT EXISTS vacated_opportunity (
    id SERIAL PRIMARY KEY,
    team VARCHAR(10) NOT NULL,
    position VARCHAR(5) NOT NULL,
    season INTEGER NOT NULL,

    total_targets_vacated INTEGER DEFAULT 0,
    total_carries_vacated INTEGER DEFAULT 0,
    total_snap_share_vacated NUMERIC DEFAULT 0.0,
    total_opportunity_share_vacated NUMERIC DEFAULT 0.0,

    departed_players JSONB,  -- Array of {player_id, name, targets, carries, snap_share}

    calculated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(team, position, season)
);

CREATE INDEX IF NOT EXISTS idx_vacated_opp_team_pos ON vacated_opportunity(team, position);
CREATE INDEX IF NOT EXISTS idx_vacated_opp_season ON vacated_opportunity(season);


-- ============================================================================
-- 3. ROSTER CHANGES TABLE
-- ============================================================================
-- Records all player movements: free agency, trades, draft picks, cuts
-- Used to calculate competition_removed and competition_added scores

CREATE TABLE IF NOT EXISTS roster_changes (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(255),
    position VARCHAR(5),

    old_team VARCHAR(10),
    new_team VARCHAR(10),
    change_type VARCHAR(20),  -- 'free_agent', 'trade', 'draft', 'cut', 'retirement'
    change_date DATE,
    season INTEGER,

    -- Previous season usage metrics (for evaluating impact)
    last_season_targets INTEGER,
    last_season_carries INTEGER,
    last_season_snap_share NUMERIC,
    last_season_opportunity_share NUMERIC,
    last_season_team_target_pct NUMERIC,
    last_season_team_carry_pct NUMERIC,

    -- Draft-specific metadata
    draft_metadata JSONB,  -- {round, pick, overall, college}

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(player_id, old_team, new_team, season)
);

CREATE INDEX IF NOT EXISTS idx_roster_changes_player ON roster_changes(player_id);
CREATE INDEX IF NOT EXISTS idx_roster_changes_old_team ON roster_changes(old_team);
CREATE INDEX IF NOT EXISTS idx_roster_changes_new_team ON roster_changes(new_team);
CREATE INDEX IF NOT EXISTS idx_roster_changes_season ON roster_changes(season);
CREATE INDEX IF NOT EXISTS idx_roster_changes_type ON roster_changes(change_type);
CREATE INDEX IF NOT EXISTS idx_roster_changes_position ON roster_changes(position);


-- ============================================================================
-- 4. HELPER VIEWS
-- ============================================================================

-- View: Latest breakout scores for current season
CREATE OR REPLACE VIEW v_latest_breakout_scores AS
SELECT DISTINCT ON (player_id, season)
    player_id,
    player_name,
    season,
    team,
    position,
    breakout_opportunity_score,
    opportunity_opened_score,
    competition_removed_score,
    competition_added_penalty,
    team_environment_score,
    player_readiness_score,
    role_trajectory_score,
    confidence_score,
    phase,
    directional_trend,
    key_reasons,
    projected_role_tag,
    as_of_date,
    calculated_at
FROM breakout_opportunity_scores
ORDER BY player_id, season, as_of_date DESC;

-- View: Top breakout candidates (score 40+)
CREATE OR REPLACE VIEW v_top_breakout_candidates AS
SELECT
    player_name,
    position,
    team,
    season,
    breakout_opportunity_score,
    opportunity_opened_score,
    player_readiness_score,
    confidence_score,
    key_reasons,
    projected_role_tag,
    as_of_date
FROM v_latest_breakout_scores
WHERE breakout_opportunity_score >= 40
ORDER BY breakout_opportunity_score DESC;

-- View: Roster departures by team/position
CREATE OR REPLACE VIEW v_roster_departures AS
SELECT
    old_team as team,
    position,
    season,
    COUNT(*) as total_departures,
    SUM(last_season_targets) as total_targets_lost,
    SUM(last_season_carries) as total_carries_lost,
    SUM(last_season_snap_share) as total_snap_share_lost,
    ARRAY_AGG(
        jsonb_build_object(
            'player_name', player_name,
            'change_type', change_type,
            'targets', last_season_targets,
            'carries', last_season_carries
        )
        ORDER BY COALESCE(last_season_targets, 0) + COALESCE(last_season_carries, 0) DESC
    ) as departed_players
FROM roster_changes
WHERE change_type IN ('free_agent', 'trade', 'cut', 'retirement')
  AND old_team IS NOT NULL
  AND old_team != ''
GROUP BY old_team, position, season;

-- View: Roster additions by team/position
CREATE OR REPLACE VIEW v_roster_additions AS
SELECT
    new_team as team,
    position,
    season,
    COUNT(*) as total_additions,
    COUNT(*) FILTER (WHERE change_type = 'draft') as draft_picks,
    COUNT(*) FILTER (WHERE change_type = 'free_agent') as fa_signings,
    COUNT(*) FILTER (WHERE change_type = 'trade') as trades_in,
    ARRAY_AGG(
        jsonb_build_object(
            'player_name', player_name,
            'change_type', change_type,
            'targets', last_season_targets,
            'carries', last_season_carries,
            'draft_round', draft_metadata->>'round'
        )
        ORDER BY
            CASE change_type
                WHEN 'draft' THEN (draft_metadata->>'round')::int
                ELSE 999
            END,
            COALESCE(last_season_targets, 0) DESC
    ) as added_players
FROM roster_changes
WHERE change_type IN ('free_agent', 'trade', 'draft')
  AND new_team IS NOT NULL
  AND new_team != ''
GROUP BY new_team, position, season;


-- ============================================================================
-- 5. SAMPLE DATA QUERY EXAMPLES
-- ============================================================================

-- Get top 20 breakout candidates for current season
-- SELECT * FROM v_top_breakout_candidates LIMIT 20;

-- Get all WR breakout candidates
-- SELECT * FROM v_latest_breakout_scores WHERE position = 'WR' AND breakout_opportunity_score >= 35 ORDER BY breakout_opportunity_score DESC;

-- Get breakout score history for a specific player
-- SELECT
--     player_name,
--     season,
--     as_of_date,
--     breakout_opportunity_score,
--     phase,
--     directional_trend,
--     key_reasons
-- FROM breakout_opportunity_scores
-- WHERE player_id = '12345'
-- ORDER BY season, as_of_date;

-- Get teams with most departed opportunity at WR
-- SELECT * FROM v_roster_departures WHERE position = 'WR' ORDER BY total_targets_lost DESC;

-- Get teams adding most competition at RB
-- SELECT * FROM v_roster_additions WHERE position = 'RB' ORDER BY total_additions DESC;


-- ============================================================================
-- 6. GRANTS (if using specific application user)
-- ============================================================================

-- Uncomment and modify if you have a specific application database user
-- GRANT SELECT, INSERT, UPDATE ON breakout_opportunity_scores TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE ON vacated_opportunity TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE ON roster_changes TO your_app_user;
-- GRANT SELECT ON v_latest_breakout_scores TO your_app_user;
-- GRANT SELECT ON v_top_breakout_candidates TO your_app_user;


-- ============================================================================
-- SETUP COMPLETE
-- ============================================================================
--
-- Next steps:
-- 1. Populate roster_changes table with historical FA/draft/trade data
-- 2. Populate vacated_opportunity table (or let engine calculate it)
-- 3. Run breakout scoring engine: python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data
-- 4. Query results: SELECT * FROM v_top_breakout_candidates;
--
-- ============================================================================
