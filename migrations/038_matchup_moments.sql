-- Sunday drama: win-probability snapshots for live lead-change moments and
-- archived turning points. Rows are small and throttled (first snapshot, then
-- a favorite flip or a meaningful move at most every 10 minutes per matchup),
-- so a full Sunday of 60s polls stays to a few dozen rows per matchup.
-- Render has no persistent disk, so this lives in the managed Postgres.
CREATE TABLE IF NOT EXISTS matchup_moments (
    id BIGSERIAL PRIMARY KEY,
    league_id TEXT NOT NULL,
    season TEXT NOT NULL,
    week INTEGER NOT NULL,
    -- Stable per-week key: "mid:<provider matchup id>" or
    -- "pair:<rosterA>-<rosterB>" for providers without one.
    matchup_key TEXT NOT NULL,
    left_roster_id TEXT NOT NULL,
    right_roster_id TEXT NOT NULL,
    left_name TEXT,
    right_name TEXT,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Left team's win probability (0.0-1.0) from the existing model at poll time.
    left_win_prob DOUBLE PRECISION NOT NULL,
    -- Live-projected totals at poll time, for the turning-point game state.
    left_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    right_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    games_live INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_matchup_moments_lookup
    ON matchup_moments (league_id, season, week, matchup_key, observed_at);

COMMENT ON TABLE matchup_moments IS
    'Win-probability snapshots behind Sunday drama: live lead-change detection and archived turning points.';
COMMENT ON COLUMN matchup_moments.left_win_prob IS
    'Observed output of the existing win-probability model; the model itself is not changed by this table.';
