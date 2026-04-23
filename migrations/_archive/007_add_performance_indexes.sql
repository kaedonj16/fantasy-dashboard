-- Add performance indexes for breakout opportunity queries
-- These indexes will significantly speed up the UI and API responses

-- Indexes for projected_opportunity table (most frequently queried)
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_score ON projected_opportunity(season, breakout_score DESC);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_position ON projected_opportunity(season, position);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_team_position_season ON projected_opportunity(team, position, season);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_player_season ON projected_opportunity(player_id, season);

-- Indexes for breakout_opportunity_scores table (unified engine)
CREATE INDEX IF NOT EXISTS idx_breakout_scores_season_score ON breakout_opportunity_scores(season, breakout_opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_position_score ON breakout_opportunity_scores(position, breakout_opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_team_position_season ON breakout_opportunity_scores(team, position, season);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_player_season ON breakout_opportunity_scores(player_id, season);

-- Indexes for roster_changes table (used in opportunity calculations)
CREATE INDEX IF NOT EXISTS idx_roster_changes_season_team_position ON roster_changes(season, new_team, position);
CREATE INDEX IF NOT EXISTS idx_roster_changes_player_season ON roster_changes(player_id, season);
CREATE INDEX IF NOT EXISTS idx_roster_changes_change_type_season ON roster_changes(change_type, season);

-- Indexes for vacated_opportunity table
CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_team_position_season ON vacated_opportunity(team, position, season);
CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_season ON vacated_opportunity(season);

-- Composite index for common UI query pattern (season + position + score)
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_ui_query ON projected_opportunity(season, position, breakout_score DESC) WHERE breakout_score >= 30;
CREATE INDEX IF NOT EXISTS idx_breakout_scores_ui_query ON breakout_opportunity_scores(season, position, breakout_opportunity_score DESC) WHERE breakout_opportunity_score >= 40;

-- Add comments for documentation
COMMENT ON INDEX idx_projected_opportunity_season_score IS 'For getting top candidates by season and score';
COMMENT ON INDEX idx_breakout_scores_season_score IS 'For unified engine top candidates by season and score';
COMMENT ON INDEX idx_projected_opportunity_ui_query IS 'Optimized index for common UI queries with minimum score filter';
