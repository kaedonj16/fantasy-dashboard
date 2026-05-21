-- Add multitask prediction columns to breakout_opportunity_scores.
-- hit_probability: P(top-12 fantasy finish at position)
-- cumulative_ppr: expected PPR points over next 2 seasons
-- peak_ppr: expected peak single-season PPR

ALTER TABLE breakout_opportunity_scores
    ADD COLUMN IF NOT EXISTS hit_probability NUMERIC,
    ADD COLUMN IF NOT EXISTS cumulative_ppr  NUMERIC,
    ADD COLUMN IF NOT EXISTS peak_ppr        NUMERIC;
