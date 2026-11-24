-- Prospect model 3.8 (under-rating audit M1): data-quality flag on
-- the historical percentile pool.
--
-- scripts/build_reference_population.py screens every row it builds
-- (no-seasons profiles, stray-season identity truncations, and the
-- FCS/fragment hole where dominator and team totals are missing for
-- a fragment-sized career) and persists the verdict here: 'ok' or
-- 'truncated'. Rows stay stored and displayable either way; the
-- percentile builders in dashboard_services/rookie_api.py exclude
-- 'truncated' rows from every pool and hide a truncated row's own
-- percentiles, because a fragment's component scores and per-game
-- marks describe a data artifact, not a player.
--
-- Existing rows default to 'ok': the quarantine takes effect as
-- classes are re-graded and re-built under 3.8, which is also when
-- their component scores move to the current model.

ALTER TABLE prospect_reference_population
    ADD COLUMN IF NOT EXISTS data_quality text NOT NULL DEFAULT 'ok';

COMMENT ON COLUMN prospect_reference_population.data_quality IS
    'Build-time validity screen (model 3.8): ok | truncated. Truncated rows are quarantined out of percentile pools at read time.';
