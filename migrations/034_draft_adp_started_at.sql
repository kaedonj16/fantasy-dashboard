-- Persist when a crawled draft actually started (Sleeper start_time / last_picked /
-- created). Used by BR Fantasy Live ADP (past-N-days window). crawled_at alone is
-- a poor proxy: a historical backfill stamps many old drafts with "today".
ALTER TABLE draft_adp_drafts
    ADD COLUMN IF NOT EXISTS draft_started_at TIMESTAMP NULL;

CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_started
    ON draft_adp_drafts (draft_started_at)
    WHERE draft_started_at IS NOT NULL;
