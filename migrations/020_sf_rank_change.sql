-- Superflex rank-movement columns.
--
-- rank_change_7d / pos_rank_change_7d are computed from 1QB values, so the
-- movement arrows are wrong in Superflex (SF) view (e.g. QBs move very
-- differently once they're valued for SF). These columns hold the SF-ordered
-- 7-day movement, computed from the SF values already stored historically in
-- player_value_history (so they backfill on the next daily run — no waiting).
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS sf_rank_change_7d     INTEGER;
ALTER TABLE player_values ADD COLUMN IF NOT EXISTS sf_pos_rank_change_7d INTEGER;
