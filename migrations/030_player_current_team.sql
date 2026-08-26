-- Current NFL team affiliation per player.
-- Written daily by cron from Sleeper; read by the web service to overlay
-- stale git-committed players_index.json (cron and web do not share a disk).

CREATE TABLE IF NOT EXISTS player_current_team (
    player_id  TEXT        PRIMARY KEY,
    team       TEXT        NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_player_current_team_updated
    ON player_current_team (updated_at DESC);
