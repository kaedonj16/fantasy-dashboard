-- 048: Prospect model v3.2 stored fields on rookie_rankings.
--   archetype / archetype_receipt / archetype_confidence: scored
--     NFL-outlook archetype from
--     data_building/rookie_pipeline/prospect_archetypes.py (v2 rule set:
--     fit scores, not first-match thresholds; label only, never feeds
--     the grade), plus the one-line receipt naming the driving inputs
--     and the High/Medium/Low confidence in the label. The label itself
--     encodes a secondary trait as "Primary · Secondary" when the
--     runner-up fit lands within 6 points.
--   talent_score: the same profile graded with draft capital excluded
--     (weights renormalized, translation pick-neutral). NULL when no
--     projected pick exists, in which case the grade already excludes
--     capital and talent_score equals prospect_score once written.
--   capital_delta: prospect_score - talent_score. NULL when no projected
--     pick exists (no capital information, not a zero effect).
-- Run BEFORE redeploying code that selects these columns.

ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS archetype         TEXT;
ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS archetype_receipt TEXT;
ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS archetype_confidence TEXT;
ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS talent_score      DECIMAL(6,2);
ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS capital_delta     DECIMAL(6,2);
