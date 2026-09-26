-- Perf: composite indexes for hot query paths (see perf audit 2026-09-25).
-- All IF NOT EXISTS so re-runs are safe.

-- player_weekly_metrics: usage-trends rebuild and waiver discovery filter by
-- season and sort by (player_id, week); the existing (season, week) index
-- does not cover the sort.
CREATE INDEX IF NOT EXISTS idx_pwm_season_player_week
    ON player_weekly_metrics (season, player_id, week);
