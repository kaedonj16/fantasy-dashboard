-- Per-format analytics buckets (feat/trade-intel-expansion item 1).
--
-- trade_intel_player_stats was a single all-formats aggregate keyed by
-- (player_id, season). Keeper-league inclusion needs a real third bucket, so
-- rows are now keyed by (player_id, season, league_format). Existing rows and
-- all current readers use the 'all' bucket, preserving behavior.

ALTER TABLE trade_intel_player_stats
    ADD COLUMN IF NOT EXISTS league_format TEXT NOT NULL DEFAULT 'all';

-- ADD COLUMN ... DEFAULT backfills existing rows with 'all' already; this is
-- belt-and-braces for rows that predate the default.
UPDATE trade_intel_player_stats
SET league_format = 'all'
WHERE league_format IS NULL OR league_format = '';

ALTER TABLE trade_intel_player_stats
    DROP CONSTRAINT IF EXISTS trade_intel_player_stats_pkey;

ALTER TABLE trade_intel_player_stats
    ADD PRIMARY KEY (player_id, season, league_format);
