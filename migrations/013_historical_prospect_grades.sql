-- Historical prospect grades for draft classes 2018-2025.
-- Stores simplified pre-draft grades for use as comparables in the prospect modal.

CREATE TABLE IF NOT EXISTS historical_prospect_grades (
    player_id           TEXT        PRIMARY KEY,  -- HIST_{YEAR}_{NAME_SLUG}
    sleeper_id          TEXT,
    name                TEXT        NOT NULL,
    position            TEXT        NOT NULL,
    draft_class_year    INTEGER     NOT NULL,
    school              TEXT,
    prospect_score      DECIMAL(6,2),
    tier                INTEGER,
    tier_label          TEXT,
    overall_rank        INTEGER,
    position_rank       INTEGER,
    actual_pick         INTEGER,
    actual_round        INTEGER,
    actual_nfl_team     TEXT,
    production_score    DECIMAL(6,2),
    athleticism_score   DECIMAL(6,2),
    draft_capital_score DECIMAL(6,2),
    headshot_url        TEXT,
    created_at          TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hpg_position_score ON historical_prospect_grades(position, prospect_score);
CREATE INDEX IF NOT EXISTS idx_hpg_year           ON historical_prospect_grades(draft_class_year);
CREATE INDEX IF NOT EXISTS idx_hpg_sleeper        ON historical_prospect_grades(sleeper_id) WHERE sleeper_id IS NOT NULL;