-- Shared, deploy-independent NFL team pace snapshot built by the daily cron.
CREATE TABLE IF NOT EXISTS team_play_volume (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    plays_faced_pg DOUBLE PRECISION,
    plays_faced_l4_pg DOUBLE PRECISION,
    off_plays_pg DOUBLE PRECISION,
    games INTEGER,
    nfl_avg_plays_faced_pg DOUBLE PRECISION,
    generated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, team)
);

CREATE INDEX IF NOT EXISTS idx_team_play_volume_generated_at
    ON team_play_volume (season, generated_at DESC);
