-- Verified relationship between a traded fantasy pick and its eventual draft
-- selection.  The transaction fields remain immutable; these columns are a
-- refreshable cache of authoritative provider draft results.
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
DELETE FROM trade_intel_assets a
WHERE a.asset_type = 'pick'
  AND a.pick_roster_id IS NOT NULL
  AND a.provider IS NOT NULL
  AND a.pick_season IS NOT NULL
  AND a.pick_round IS NOT NULL
  AND a.id NOT IN (
      SELECT DISTINCT ON (b.provider, b.trade_id, b.pick_season,
                          b.pick_round, b.pick_roster_id) b.id
      FROM trade_intel_assets b
      WHERE b.asset_type = 'pick'
        AND b.pick_roster_id IS NOT NULL
        AND b.provider IS NOT NULL
        AND b.pick_season IS NOT NULL
        AND b.pick_round IS NOT NULL
      ORDER BY b.provider, b.trade_id, b.pick_season, b.pick_round,
               b.pick_roster_id,
               (b.resolved_player_id IS NOT NULL) DESC,
               b.id DESC
  );

CREATE UNIQUE INDEX IF NOT EXISTS uq_tia_verified_pick_resolution
    ON trade_intel_assets
       (provider, trade_id, pick_season, pick_round, pick_roster_id)
    WHERE asset_type = 'pick' AND pick_roster_id IS NOT NULL;

COMMENT ON COLUMN trade_intel_assets.resolved_player_id IS
    'Authoritative fantasy draft selection; never NFL draft/ADP/projection data.';
COMMENT ON COLUMN trade_intel_assets.resolved_round_slot IS
    'Actual within-round selection position derived from provider pick_no.';
