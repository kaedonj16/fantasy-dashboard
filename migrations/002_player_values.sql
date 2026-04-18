-- Player values table — one row per player, updated daily by cron.
-- Consolidates migrations 002, 013, 017.

CREATE TABLE IF NOT EXISTS player_values (
    player_id            TEXT        PRIMARY KEY,
    last_updated         DATE,
    value_1qb            DECIMAL(10,2),
    value_sf             DECIMAL(10,2),
    -- Market-calibrated values (COALESCE over raw values in all reads)
    calibrated_value_1qb DECIMAL(10,2),
    calibrated_value_sf  DECIMAL(10,2),
    calibration_weight   DECIMAL(4,3),
    calibration_source   TEXT,        -- 'direct' | 'tier_anchor' | 'model_only'
    position             TEXT,
    pos_rank             INTEGER,
    pos_rank_label       TEXT,
    age                  DECIMAL(5,2),
    team                 TEXT,
    years_exp            INTEGER,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Idempotent column additions for existing databases
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS calibrated_value_1qb DECIMAL(10,2);
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS calibrated_value_sf  DECIMAL(10,2);
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS calibration_weight   DECIMAL(4,3);
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS calibration_source   TEXT;
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS years_exp            INTEGER;

-- Existing databases: convert (player_id, date) PK → player_id-only PK
DO $$
BEGIN
    -- Keep only the latest snapshot per player
    DELETE FROM player_values pv1
    USING (
        SELECT player_id, MAX(COALESCE(last_updated, created_at::date)) AS max_date
        FROM player_values
        GROUP BY player_id
    ) latest
    WHERE pv1.player_id = latest.player_id
      AND COALESCE(pv1.last_updated, pv1.created_at::date) < latest.max_date;

    -- Rename date → last_updated if the old column name still exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'player_values' AND column_name = 'date'
    ) THEN
        ALTER TABLE player_values RENAME COLUMN date TO last_updated;
    END IF;

    -- Replace compound PK with player_id-only PK if needed
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_name = 'player_values'
          AND constraint_type = 'PRIMARY KEY'
          AND constraint_name != 'player_values_pkey'
    ) THEN
        ALTER TABLE player_values DROP CONSTRAINT IF EXISTS player_values_pkey;
        ALTER TABLE player_values ADD PRIMARY KEY (player_id);
    END IF;
EXCEPTION WHEN OTHERS THEN
    NULL; -- already migrated; safe to ignore
END $$;

CREATE INDEX IF NOT EXISTS idx_player_values_last_updated ON player_values(last_updated);
CREATE INDEX IF NOT EXISTS idx_player_values_position     ON player_values(position);
CREATE INDEX IF NOT EXISTS idx_pv_calibrated              ON player_values(calibrated_value_1qb DESC NULLS LAST);
