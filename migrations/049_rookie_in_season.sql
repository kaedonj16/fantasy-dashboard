-- 049: Prospect model v3.4 two-mode grading flag on rookie_rankings.
--   in_season: TRUE when the row was graded in IN-SEASON mode (the
--     player's latest season was the in-progress season at 1-11 stored
--     games on the grading run; see prospect_model.py v3.4). The board
--     and the prospect modal show an "In-Season" tag for flagged rows,
--     and a row's flag flips on the first run after the player's season
--     completes and he grades in COMPLETE mode. Written on every
--     pipeline run by upsert_rankings; rows graded before v3.4 read
--     FALSE (they were all complete-mode grades).
-- Run BEFORE redeploying code that selects this column.

ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS in_season BOOLEAN NOT NULL DEFAULT FALSE;
