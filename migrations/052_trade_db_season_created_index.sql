-- Trade Database load speed: /api/trade-database sorts a season's trades
-- by created_at on every request. The existing indexes cannot serve that
-- order: idx_tit_season_week is (season, week) and idx_tit_created is
-- (created_at) alone, so the planner scans and sorts every matching
-- trade (~545k for the current season; 11-40s live, first paint timed
-- out at 30s on 2026-11-04) before the LIMIT applies.
--
-- (season, created_at DESC NULLS LAST, id DESC) matches the default
-- date_desc sort exactly (NULLS LAST is spelled out because plain DESC
-- defaults to NULLS FIRST, which would not match the query's order),
-- so the planner reads it forward and stops after LIMIT + 1 rows.
-- When a league filter (superflex / format) is active the planner
-- walks this index and filters through the 1:1 leagues join, still
-- bounded by how many rows it must skip to fill the page.
-- IF NOT EXISTS so re-runs are safe.
CREATE INDEX IF NOT EXISTS idx_tit_season_created
    ON trade_intel_trades (season, created_at DESC NULLS LAST, id DESC);
