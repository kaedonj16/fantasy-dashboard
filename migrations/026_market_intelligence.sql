CREATE TABLE IF NOT EXISTS player_external_ids (
    provider TEXT NOT NULL,
    provider_player_id TEXT NOT NULL,
    canonical_player_id TEXT NOT NULL,
    match_confidence NUMERIC(5,4) NOT NULL,
    match_method TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, provider_player_id)
);
CREATE INDEX IF NOT EXISTS idx_external_ids_canonical ON player_external_ids(canonical_player_id, provider);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    provider_player_id TEXT NOT NULL,
    canonical_player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER,
    context TEXT NOT NULL CHECK (context IN ('weekly', 'season')),
    stat_type TEXT NOT NULL,
    market_type TEXT NOT NULL,
    period TEXT NOT NULL,
    sportsbook TEXT NOT NULL,
    line NUMERIC NOT NULL,
    over_price NUMERIC,
    under_price NUMERIC,
    event_start_time TIMESTAMPTZ NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    source_updated_at TIMESTAMPTZ,
    UNIQUE(provider, provider_event_id, provider_player_id, stat_type, sportsbook, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_read ON market_snapshots(season, week, canonical_player_id, stat_type, observed_at DESC);

CREATE TABLE IF NOT EXISTS market_consensus (
    canonical_player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER,
    context TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    consensus_line NUMERIC NOT NULL,
    fair_over_probability NUMERIC,
    book_count INTEGER NOT NULL,
    dispersion NUMERIC NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL,
    UNIQUE NULLS NOT DISTINCT (canonical_player_id, season, week, context, stat_type)
);
CREATE INDEX IF NOT EXISTS idx_market_consensus_context ON market_consensus(season, week, context, calculated_at DESC);

CREATE TABLE IF NOT EXISTS market_projections (
    canonical_player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER,
    context TEXT NOT NULL,
    fantasy_points NUMERIC NOT NULL,
    coverage NUMERIC(5,4) NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    components JSONB NOT NULL DEFAULT '{}'::jsonb,
    calculated_at TIMESTAMPTZ NOT NULL,
    UNIQUE NULLS NOT DISTINCT (canonical_player_id, season, week, context)
);
CREATE INDEX IF NOT EXISTS idx_market_projections_read ON market_projections(season, week, context, canonical_player_id);
