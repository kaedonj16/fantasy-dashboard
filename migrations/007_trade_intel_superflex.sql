-- Add is_superflex flag to trade_intel_leagues so the WLS model can
-- train separately on SF vs 1QB trade data (QB values differ 3-5x).
-- DEFAULT NULL = "not yet inspected" so the backfill can detect them.
ALTER TABLE trade_intel_leagues ADD COLUMN IF NOT EXISTS is_superflex BOOLEAN DEFAULT NULL;
CREATE INDEX IF NOT EXISTS idx_til_sf ON trade_intel_leagues(is_superflex);
