-- Add PFF and Reception Perception fields to rookie_prospect_source_data.
-- yprr and grades_pass_route exist in PFF CSVs but were never imported.
-- RP fields are new from Reception Perception dataset.
ALTER TABLE rookie_prospect_source_data
    ADD COLUMN IF NOT EXISTS grades_pass_route     DECIMAL(5,1),
    ADD COLUMN IF NOT EXISTS success_rate_vs_press DECIMAL(5,1),
    ADD COLUMN IF NOT EXISTS success_rate_vs_man   DECIMAL(5,1),
    ADD COLUMN IF NOT EXISTS success_rate_vs_zone  DECIMAL(5,1),
    ADD COLUMN IF NOT EXISTS contested_catch_rate_rp DECIMAL(5,1),
    ADD COLUMN IF NOT EXISTS tackle_break_rate     DECIMAL(5,1);
