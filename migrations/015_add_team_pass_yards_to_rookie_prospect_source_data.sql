-- Add team_pass_yards column to rookie_prospect_source_data table
-- This column stores team passing yards from CFBD team stats
-- Used in conjunction with sagarin team ratings for rookie evaluation

ALTER TABLE rookie_prospect_source_data 
ADD COLUMN team_pass_yards INTEGER;
