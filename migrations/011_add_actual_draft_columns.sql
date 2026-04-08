-- Migration 011: Add actual NFL draft result columns to rookie_prospects
--
-- After the NFL Draft (late April each year), these columns are populated by
-- fetch_nflverse_draft_picks() / upsert_actual_draft_picks() in pipeline.py.
-- The pipeline then uses actual picks instead of mock draft projections for
-- all scoring and consensus calculations.

ALTER TABLE rookie_prospects
    ADD COLUMN IF NOT EXISTS actual_pick     INTEGER,
    ADD COLUMN IF NOT EXISTS actual_round    INTEGER,
    ADD COLUMN IF NOT EXISTS actual_nfl_team TEXT,
    ADD COLUMN IF NOT EXISTS draft_confirmed BOOLEAN DEFAULT FALSE;

-- Index so post-draft queries (WHERE draft_confirmed = TRUE) are fast
CREATE INDEX IF NOT EXISTS idx_rp_draft_confirmed
    ON rookie_prospects (draft_class_year, draft_confirmed)
    WHERE draft_confirmed = TRUE;
