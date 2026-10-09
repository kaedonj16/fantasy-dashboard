-- 047: New CFBD data sources for the rookie prospect pipeline.
--   1. rookie_prospect_recruiting: 247 Composite recruiting pedigree per prospect
--      (stars, composite rating, national + position rank).
--   2. rookie_prospect_wepa: opponent-adjusted efficiency (WEPA) per player/season.
--      Raw payload stored as JSONB; normalized adj_efficiency_score is best-effort.
--   3. rookie_team_context: per-school/per-season SP+ and 247 Team Talent Composite
--      for the competition adjustment.
--   4. rookie_rankings.model_version + recruiting_score: Kaedon's rule - every
--      grading change is versioned, never silently tweaked. Model v2.0 introduces
--      the recruiting-pedigree component, WEPA efficiency blend, and SP+/talent
--      competition inputs.

CREATE TABLE IF NOT EXISTS rookie_prospect_recruiting (
    player_id          TEXT        PRIMARY KEY,
    draft_class_year   INTEGER     NOT NULL,
    stars              INTEGER,
    composite_rating   DECIMAL(6,4),
    national_rank      INTEGER,
    position_rank      INTEGER,
    recruit_class_year INTEGER,
    committed_school   TEXT,
    created_at         TIMESTAMP   DEFAULT NOW(),
    updated_at         TIMESTAMP   DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rookie_prospect_wepa (
    player_id            TEXT        NOT NULL,
    season               INTEGER     NOT NULL,
    wepa_type            TEXT        NOT NULL,   -- 'passing' | 'rushing'
    adj_efficiency_score DECIMAL(6,2),           -- normalized 0-100, best-effort
    metrics              JSONB,
    created_at           TIMESTAMP   DEFAULT NOW(),
    PRIMARY KEY (player_id, season, wepa_type)
);

CREATE TABLE IF NOT EXISTS rookie_team_context (
    school            TEXT        NOT NULL,
    season            INTEGER     NOT NULL,
    sp_rating         DECIMAL(6,2),
    sp_offense        DECIMAL(6,2),
    sp_defense        DECIMAL(6,2),
    sp_sos            DECIMAL(6,2),
    talent_composite  DECIMAL(8,2),
    created_at        TIMESTAMP   DEFAULT NOW(),
    PRIMARY KEY (school, season)
);

ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS model_version    TEXT;
ALTER TABLE rookie_rankings ADD COLUMN IF NOT EXISTS recruiting_score DECIMAL(6,2);

CREATE INDEX IF NOT EXISTS idx_rookie_recruiting_class
    ON rookie_prospect_recruiting (draft_class_year);
CREATE INDEX IF NOT EXISTS idx_rookie_wepa_player
    ON rookie_prospect_wepa (player_id);
CREATE INDEX IF NOT EXISTS idx_rookie_team_context_school
    ON rookie_team_context (school, season);
