-- Store the original pick owner so we can look up draft order later
-- and backfill pick_order (early / mid / late).

ALTER TABLE trade_intel_assets
    ADD COLUMN IF NOT EXISTS pick_roster_id TEXT;

CREATE INDEX IF NOT EXISTS idx_tia_roster
    ON trade_intel_assets(pick_roster_id)
    WHERE pick_roster_id IS NOT NULL;
