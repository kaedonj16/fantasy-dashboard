-- Migration 017: Add market-calibrated value columns to player_values
-- These are separate from raw model values so we never lose the model signal.
-- calibrated_value_1qb/sf = blend of model + market; used by trade calculator.
-- calibration_weight = 0.0 (pure model) → 0.7 (heavily market-influenced).
-- For rookies the weight reflects position-tier anchoring, not direct trade data.

ALTER TABLE player_values
    ADD COLUMN IF NOT EXISTS calibrated_value_1qb DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS calibrated_value_sf  DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS calibration_weight   DECIMAL(4,3),
    ADD COLUMN IF NOT EXISTS calibration_source   TEXT;  -- 'direct' | 'tier_anchor' | 'model_only'

CREATE INDEX IF NOT EXISTS idx_pv_calibrated ON player_values(calibrated_value_1qb DESC NULLS LAST);
