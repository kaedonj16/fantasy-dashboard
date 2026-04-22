-- Add exact pick slot (position within the round, e.g. 6 for "1.06")
-- sourced from Sleeper's /traded_picks endpoint.

ALTER TABLE trade_intel_assets
    ADD COLUMN IF NOT EXISTS pick_slot INTEGER;

COMMENT ON COLUMN trade_intel_assets.pick_slot IS
    '1-based slot within the round (e.g. 6 for pick 1.06 in a 12-team league)';
