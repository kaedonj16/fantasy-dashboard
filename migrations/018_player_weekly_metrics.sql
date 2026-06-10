-- Per-player per-week usage metrics (snap share, target share, touches).
-- Feeds waivers usage risers, leaderboard/modal weekly trends, start/sit
-- usage factor, and the breakout engine's role trajectory.
CREATE TABLE IF NOT EXISTS player_weekly_metrics (
    player_id    TEXT    NOT NULL,
    season       INTEGER NOT NULL,
    week         INTEGER NOT NULL,
    position     TEXT,
    snap_pct     NUMERIC,
    snaps        INTEGER,
    team_snaps   INTEGER,
    targets      INTEGER,
    receptions   INTEGER,
    rec_yards    NUMERIC,
    carries      INTEGER,
    rush_yards   NUMERIC,
    touches      INTEGER,
    target_share NUMERIC,
    ppr_pts      NUMERIC,
    PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_pwm_season_week ON player_weekly_metrics (season, week);
