-- Bulk history lookup used to build rolling weekly market rates. The existing
-- read index begins with (season, week); this order serves player/stat histories
-- without N+1 queries as the season grows.
CREATE INDEX IF NOT EXISTS idx_market_consensus_rolling
    ON market_consensus(season, context, canonical_player_id, stat_type, week, calculated_at DESC);
