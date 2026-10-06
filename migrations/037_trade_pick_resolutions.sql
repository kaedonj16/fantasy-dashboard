-- Verified relationship between a traded fantasy pick and its eventual draft
-- selection.  The transaction fields remain immutable; these columns are a
-- refreshable cache of authoritative provider draft results.
--
-- This migration carries its own statement timeout: the dedup DELETE and the
-- index builds below scan trade_intel_assets, which is large on prod, and the
-- Render default statement_timeout killed earlier versions of this file
-- (blocking every later migration, since the runner stops at the first
-- failure). run_migrations executes each file inside one transaction, so
-- SET LOCAL is scoped to this migration only.
SET LOCAL statement_timeout = '30min';
-- The statements below are sort-heavy (dedup window + two index builds) over
-- a large table; the default work_mem/maintenance_work_mem spill those sorts
-- to disk. Keep them in memory for this migration only.
SET LOCAL work_mem = '256MB';
SET LOCAL maintenance_work_mem = '256MB';

ALTER TABLE trade_intel_assets
    ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'sleeper',
    ADD COLUMN IF NOT EXISTS resolved_draft_id TEXT,
    ADD COLUMN IF NOT EXISTS resolved_player_id TEXT,
    ADD COLUMN IF NOT EXISTS resolved_pick_no INTEGER,
    ADD COLUMN IF NOT EXISTS resolved_round_slot INTEGER,
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_tia_resolved_player
    ON trade_intel_assets (resolved_player_id)
    WHERE resolved_player_id IS NOT NULL;

-- Deduplicate pick assets before the unique index: the same pick
-- (provider / trade / season / round / original roster) was occasionally
-- recorded twice for one trade (e.g. Sleeper listing the same draft pick
-- twice in a transaction). Keep the resolved row when one exists, else the
-- latest row. The NOT NULL filters mirror the unique index exactly: rows
-- with a NULL key column can never conflict, so they are left alone.
-- Window-function form (single scan + sort): the earlier NOT IN +
-- DISTINCT ON subquery timed out on large tables (statement timeout),
-- blocking every later migration. The EXISTS guard skips the sort entirely
-- when there is nothing to dedupe (the common case); it short-circuits on
-- the first duplicate found.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM trade_intel_assets
    WHERE asset_type = 'pick'
      AND pick_roster_id IS NOT NULL
      AND provider IS NOT NULL
      AND pick_season IS NOT NULL
      AND pick_round IS NOT NULL
    GROUP BY provider, trade_id, pick_season, pick_round, pick_roster_id
    HAVING COUNT(*) > 1
  ) THEN
    DELETE FROM trade_intel_assets a
    USING (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY provider, trade_id, pick_season,
                                pick_round, pick_roster_id
                   ORDER BY (resolved_player_id IS NOT NULL) DESC,
                            id DESC
               ) AS rn
        FROM trade_intel_assets
        WHERE asset_type = 'pick'
          AND pick_roster_id IS NOT NULL
          AND provider IS NOT NULL
          AND pick_season IS NOT NULL
          AND pick_round IS NOT NULL
    ) ranked
    WHERE a.id = ranked.id
      AND ranked.rn > 1;
  END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_tia_verified_pick_resolution
    ON trade_intel_assets
       (provider, trade_id, pick_season, pick_round, pick_roster_id)
    WHERE asset_type = 'pick' AND pick_roster_id IS NOT NULL;

COMMENT ON COLUMN trade_intel_assets.resolved_player_id IS
    'Authoritative fantasy draft selection; never NFL draft/ADP/projection data.';
COMMENT ON COLUMN trade_intel_assets.resolved_round_slot IS
    'Actual within-round selection position derived from provider pick_no.';
