-- Migration 010: Add advanced metrics to rookie_prospect_source_data table
-- Adds PFF College and advanced receiving metrics for prospect evaluation

-- Add new columns to rookie_prospect_source_data table
ALTER TABLE rookie_prospect_source_data 
ADD COLUMN IF NOT EXISTS yards_after_catch DECIMAL(8,1),
ADD COLUMN IF NOT EXISTS yards_after_catch_per_reception DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS avg_depth_of_target DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS contested_catch_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS avoided_tackles INTEGER,
ADD COLUMN IF NOT EXISTS drop_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS slot_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS wide_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS inline_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS pass_block_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS grades_offense DECIMAL(4,1),
ADD COLUMN IF NOT EXISTS grades_pass_block DECIMAL(4,1),
ADD COLUMN IF NOT EXISTS explosive_runs_10_plus INTEGER,
ADD COLUMN IF NOT EXISTS breakaway_percentage DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS elusive_rating DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS pff_rushing_grade DECIMAL(4,1),
ADD COLUMN IF NOT EXISTS pff_passing_grade DECIMAL(4,1),
ADD COLUMN IF NOT EXISTS big_time_throw_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS adjusted_completion_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS pressure_to_sack_rate DECIMAL(5,3),
ADD COLUMN IF NOT EXISTS nfl_passer_rating DECIMAL(4,1);

-- Add comments for documentation
COMMENT ON COLUMN rookie_prospect_source_data.yards_after_catch IS 'Total yards after catch for the season';
COMMENT ON COLUMN rookie_prospect_source_data.yards_after_catch_per_reception IS 'Average yards after catch per reception';
COMMENT ON COLUMN rookie_prospect_source_data.avg_depth_of_target IS 'Average depth of target (aDOT) in yards';
COMMENT ON COLUMN rookie_prospect_source_data.contested_catch_rate IS 'Rate of catches on contested targets (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.avoided_tackles IS 'Total avoided tackles after the catch';
COMMENT ON COLUMN rookie_prospect_source_data.drop_rate IS 'Drop rate (drops / targets, 0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.slot_rate IS 'Percentage of snaps from slot position (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.wide_rate IS 'Percentage of snaps from wide receiver position (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.inline_rate IS 'Percentage of snaps from inline/tight end position (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.pass_block_rate IS 'Pass block rate for eligible positions (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.grades_offense IS 'PFF overall offensive grade (0-100 scale)';
COMMENT ON COLUMN rookie_prospect_source_data.grades_pass_block IS 'PFF pass blocking grade (0-100 scale)';
COMMENT ON COLUMN rookie_prospect_source_data.explosive_runs_10_plus IS 'Number of runs of 10+ yards';
COMMENT ON COLUMN rookie_prospect_source_data.breakaway_percentage IS 'Percentage of runs that are breakaways (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.elusive_rating IS 'PFF elusive rating for rushers';
COMMENT ON COLUMN rookie_prospect_source_data.pff_rushing_grade IS 'PFF rushing grade (0-100 scale)';
COMMENT ON COLUMN rookie_prospect_source_data.pff_passing_grade IS 'PFF passing grade (0-100 scale)';
COMMENT ON COLUMN rookie_prospect_source_data.big_time_throw_rate IS 'Rate of big time throws (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.adjusted_completion_rate IS 'Adjusted completion rate accounting for drops (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.pressure_to_sack_rate IS 'Rate of pressures converted to sacks (0-1)';
COMMENT ON COLUMN rookie_prospect_source_data.nfl_passer_rating IS 'NFL passer rating (0-158.3 scale)';
