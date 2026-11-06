-- Draft Database: extend the crawled draft corpus (draft_adp_drafts, from
-- migrations/015 + 034) with the league/draft metadata the Draft Database
-- page needs, plus a persisted grade summary per draft.
--
-- Every new column is nullable on purpose: rows crawled before this
-- migration only gain values when the backfill
-- (data_building/trade_intel/draft_db_backfill.py) can re-fetch the Sleeper
-- draft/league payloads. Anything the payloads do not honestly support
-- stays NULL and the API/UI degrade (league name falls back to a
-- "League <id>" label, format filters simply do not match NULL rows).
--
-- league_format mirrors Sleeper league settings.type exactly as the rest
-- of the trade-intel corpus stores it: 0 = redraft, 1 = keeper,
-- 2 = dynasty, stored here as the label. scoring_type mirrors
-- trade_intel_leagues.scoring_type ('ppr' / 'half' / 'std'); when a draft
-- payload carries metadata.scoring_type it is normalized to the same set,
-- and stays NULL when neither source names one.
--
-- grade_summary is one JSON document per draft (see
-- utils/draft_db_grades.py for the shape): the at-draft family
-- (utils/draft.py letter scale over ADP diff, same bands as
-- /api/draft-grades) and, when season performance data exists, the
-- performance family (utils/draft_recap.py verdict bands). Each
-- superlative in the payload names the family it came from.

ALTER TABLE draft_adp_drafts
    ADD COLUMN IF NOT EXISTS league_name       TEXT,
    ADD COLUMN IF NOT EXISTS scoring_type      TEXT,
    ADD COLUMN IF NOT EXISTS league_format     TEXT,
    ADD COLUMN IF NOT EXISTS slot_to_roster_id JSONB,
    ADD COLUMN IF NOT EXISTS draft_order_type  TEXT,
    ADD COLUMN IF NOT EXISTS roster_names      JSONB,
    ADD COLUMN IF NOT EXISTS roster_owners     JSONB,
    ADD COLUMN IF NOT EXISTS grade_summary     JSONB,
    ADD COLUMN IF NOT EXISTS grades_computed_at TIMESTAMP;

-- The list API filters on season + recency first, then segments.
CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_season_started
    ON draft_adp_drafts (season, draft_started_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_segment
    ON draft_adp_drafts (draft_type, is_superflex, num_teams);

CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_format
    ON draft_adp_drafts (league_format)
    WHERE league_format IS NOT NULL;

-- Player search on the list API probes picks by player before joining
-- drafts; (player_id, draft_id) serves that probe directly.
CREATE INDEX IF NOT EXISTS idx_draft_adp_picks_player_draft
    ON draft_adp_picks (player_id, draft_id);
