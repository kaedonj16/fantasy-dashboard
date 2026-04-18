-- Migration 015: Add team_pass_yards and sagarin_team_rating to rookie_prospect_source_data
--
-- team_pass_yards: team's net passing yards for the season (CFBD netPassingYards).
--   Used as the denominator for the WR/TE pass-share dominator metric in
--   prospect_model.py, replacing total team offense (pass + rush) which
--   penalised receivers on run-heavy teams.
--
-- sagarin_team_rating: Jeff Sagarin CFB predictor rating for the player's college
--   team in that season. Applied as a multiplicative adjustment to the pass-share
--   metric: +6.47% cap (Alabama-tier), -9.3% floor (non-D1/unrated).

ALTER TABLE rookie_prospect_source_data
    ADD COLUMN IF NOT EXISTS team_pass_yards    INTEGER,
    ADD COLUMN IF NOT EXISTS sagarin_team_rating DECIMAL(6,2);
