-- Durable, normalized game-market context.  Unlike provider response caches,
-- this survives restarts and can safely bridge a temporary provider outage.
CREATE TABLE IF NOT EXISTS market_team_environments (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    implied_points NUMERIC NOT NULL,
    league_average NUMERIC NOT NULL,
    environment_score NUMERIC NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, team)
);
CREATE INDEX IF NOT EXISTS idx_market_team_environment_freshness
    ON market_team_environments(season, observed_at DESC);
