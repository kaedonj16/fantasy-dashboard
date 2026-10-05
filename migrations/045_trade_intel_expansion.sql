-- Trade-intel expansion (feat/trade-intel-expansion).
--
-- 1. Keeper-league inclusion: no schema change (league_type=1 rows were simply
--    never inserted). Queries now accept 0/1/2.
-- 2. previous_league_id chain walking: link each season's league row to its
--    predecessor so the crawler can backfill multi-season history.
-- 4. Velocity-prioritized recrawls: per-league trade counters drive crawl order.
-- 6. Opt-in multi-platform trade contribution: per-user consent flag.
-- 7. Trade-context snapshots: per-trade JSONB with each side's record at trade time.

ALTER TABLE trade_intel_leagues
    ADD COLUMN IF NOT EXISTS previous_league_id TEXT,
    ADD COLUMN IF NOT EXISTS total_trades INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_trade_at TIMESTAMPTZ;

ALTER TABLE trade_intel_trades
    ADD COLUMN IF NOT EXISTS trade_context JSONB;

ALTER TABLE trade_intel_users
    ADD COLUMN IF NOT EXISTS contrib_opt_in BOOLEAN NOT NULL DEFAULT FALSE;

-- Velocity ordering needs this; NULLS LAST keeps never-traded leagues crawlable.
CREATE INDEX IF NOT EXISTS idx_til_last_trade_at
    ON trade_intel_leagues (last_trade_at DESC NULLS LAST);

-- Backfill counters from already-crawled trades so velocity ordering is
-- meaningful from day one instead of waiting for the next full recrawl cycle.
UPDATE trade_intel_leagues l
SET total_trades = COALESCE(s.n, 0),
    last_trade_at = s.last_at
FROM (
    SELECT league_id, COUNT(*) AS n, MAX(created_at) AS last_at
    FROM trade_intel_trades
    GROUP BY league_id
) s
WHERE l.league_id = s.league_id;
