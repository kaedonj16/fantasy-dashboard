-- RedZone play store: stamp the NFL week on every play so week-scoped
-- readers (e.g. /api/redzone/moments) can ask for one week's plays instead
-- of "whatever happened in the last N days".
--
-- The collector stamps week at write time going forward. Rows already in
-- the table predate the column; backfill them from the game_id date
-- (``YYYYMMDD_AWAY@HOME``). The 2026 regular season opened Thursday
-- 2026-09-10, so week = ((game_date - 2026-09-10) / 7) + 1.
-- Only season 2026 is backfilled: the table was created for the 2026
-- season (041_redzone_plays.sql) and holds no earlier seasons.

ALTER TABLE redzone_plays ADD COLUMN IF NOT EXISTS week INTEGER;

UPDATE redzone_plays
SET week = ((to_date(substring(game_id from 1 for 8), 'YYYYMMDD') - DATE '2026-09-10') / 7) + 1
WHERE season = 2026
  AND week IS NULL
  AND substring(game_id from 1 for 8) ~ '^[0-9]{8}$'
  AND ((to_date(substring(game_id from 1 for 8), 'YYYYMMDD') - DATE '2026-09-10') / 7) + 1 >= 1;

CREATE INDEX IF NOT EXISTS idx_redzone_plays_week
    ON redzone_plays (season, week);
