-- All rows in trade_intel_leagues were inserted by the discovery pipeline,
-- which only stores dynasty leagues (settings.type == 2).  Rows with
-- league_type IS NULL predate the column being written on insert/conflict;
-- backfill them so the crawler's WHERE league_type = 2 filter matches them.
UPDATE trade_intel_leagues
SET    league_type = 2
WHERE  league_type IS NULL;
