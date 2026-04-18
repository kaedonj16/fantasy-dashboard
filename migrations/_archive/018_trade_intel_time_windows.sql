-- Migration 018: Add time-windowed market value columns to trade_intel_player_stats
-- Replaces the single flat market_value with:
--   weighted_market_value_*  — time-decay weighted median (primary signal)
--   market_value_*_14d/30d/90d — unweighted window medians (for trend math)
--   market_trend_*           — 14d median minus 90d median (directional signal)
--   trade_count_14d          — how fresh the data actually is

ALTER TABLE trade_intel_player_stats
    ADD COLUMN IF NOT EXISTS weighted_market_value_1qb  DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS weighted_market_value_sf   DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_1qb_14d       DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_sf_14d        DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_1qb_30d       DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_sf_30d        DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_1qb_90d       DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_value_sf_90d        DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_trend_1qb           DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS market_trend_sf            DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS trade_count_14d            INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_tips_weighted ON trade_intel_player_stats(weighted_market_value_1qb DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_tips_trend    ON trade_intel_player_stats(market_trend_1qb DESC NULLS LAST);
