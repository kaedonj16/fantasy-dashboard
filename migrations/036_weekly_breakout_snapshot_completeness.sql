-- Atomic, versioned publication metadata for weekly breakout snapshots.
ALTER TABLE weekly_breakout_scores
    ADD COLUMN IF NOT EXISTS run_id BIGINT;

ALTER TABLE weekly_breakout_runs
    ADD COLUMN IF NOT EXISTS expected_row_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS inserted_row_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP;

ALTER TABLE weekly_breakout_scores
    DROP CONSTRAINT IF EXISTS weekly_breakout_scores_player_id_season_as_of_week_key;
ALTER TABLE weekly_breakout_runs
    DROP CONSTRAINT IF EXISTS weekly_breakout_runs_season_as_of_week_status_key;

-- Attempt logs are not snapshots and intentionally carry NULL version so they
-- cannot conflict with the one published run for a versioned target.
UPDATE weekly_breakout_runs
SET scoring_version = NULL
WHERE status NOT IN ('success', 'completed');

CREATE UNIQUE INDEX IF NOT EXISTS uq_wbs_snapshot_player_version
    ON weekly_breakout_scores (player_id, season, as_of_week, scoring_version);
CREATE UNIQUE INDEX IF NOT EXISTS uq_wbr_snapshot_version
    ON weekly_breakout_runs (season, as_of_week, scoring_version);

-- Existing successful snapshots were transactionally written by the legacy
-- writer. Mark them completed so they remain available during rollout.
UPDATE weekly_breakout_scores AS scores
SET run_id = runs.id
FROM weekly_breakout_runs AS runs
WHERE scores.run_id IS NULL
  AND runs.status = 'success'
  AND runs.records_saved > 0
  AND scores.season = runs.season
  AND scores.as_of_week = runs.as_of_week
  AND scores.scoring_version = runs.scoring_version;

UPDATE weekly_breakout_runs
SET status = 'completed',
    expected_row_count = records_saved,
    inserted_row_count = records_saved,
    completed_at = COALESCE(completed_at, calculated_at)
WHERE status = 'success' AND records_saved > 0;
