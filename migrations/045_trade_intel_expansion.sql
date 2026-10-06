-- Trade-intel expansion (feat/trade-intel-expansion).
--
-- 1. Keeper-league inclusion: no schema change (league_type=1 rows were simply
--    never inserted). Queries now accept 0/1/2.
-- 2. previous_league_id chain walking: link each season's league row to its
--    predecessor so the crawler can backfill multi-season history.
-- 4. Velocity-prioritized recrawls: per-league trade counters drive crawl order.
-- 6. Opt-in multi-platform trade contribution: per-user consent flag.
-- 7. Trade-context snapshots: per-trade JSONB with each side's record at trade time.
--
-- This migration carries its own statement timeout: the counter backfill
-- below aggregates trade_intel_trades, which is large on prod, and the
-- Render default statement_timeout killed it (blocking every later
-- migration, since the runner stops at the first failure).
-- run_migrations executes each file inside one transaction, so SET LOCAL
-- is scoped to this migration only.
SET LOCAL statement_timeout = '30min';
-- The backfill below is aggregate-heavy over a large table; the default
-- work_mem/maintenance_work_mem spill those sorts to disk. Keep them in
-- memory for this migration only.
SET LOCAL work_mem = '256MB';
SET LOCAL maintenance_work_mem = '256MB';

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
-- The EXISTS guard skips the aggregate entirely when every league already
-- has counters (the common case on re-run); it short-circuits on the first
-- league still missing them. The UPDATE itself only touches leagues whose
-- counters are still NULL, so a re-run after a successful backfill is a
-- cheap no-op instead of re-aggregating the whole trades table.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM trade_intel_leagues l
    WHERE l.last_trade_at IS NULL
      AND EXISTS (
        SELECT 1 FROM trade_intel_trades t WHERE t.league_id = l.league_id
      )
  ) THEN
    UPDATE trade_intel_leagues l
    SET total_trades = s.n,
        last_trade_at = s.last_at
    FROM (
        SELECT t.league_id, COUNT(*) AS n, MAX(t.created_at) AS last_at
        FROM trade_intel_trades t
        JOIN trade_intel_leagues l2
          ON l2.league_id = t.league_id
         AND l2.last_trade_at IS NULL
        GROUP BY t.league_id
    ) s
    WHERE l.league_id = s.league_id;
  END IF;
END
$$;
