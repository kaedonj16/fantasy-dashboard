-- Rookie prospect evaluation system.
-- Consolidates migrations 009, 010, 011, 012, 014, 015.

CREATE TABLE IF NOT EXISTS rookie_prospects (
    player_id          TEXT        PRIMARY KEY,  -- ROOKIE_{YEAR}_{NAME_SLUG}
    sleeper_id         TEXT,
    name               TEXT        NOT NULL,
    position           TEXT        NOT NULL,
    school             TEXT,
    age                DECIMAL(4,1),
    height_inches      INTEGER,
    weight_lbs         INTEGER,
    hometown           TEXT,
    state              TEXT,
    draft_class_year   INTEGER     NOT NULL,
    early_declare      BOOLEAN     DEFAULT FALSE,
    transfer_history   TEXT,
    headshot_url       TEXT,
    -- Actual NFL draft results (populated post-draft)
    actual_pick        INTEGER,
    actual_round       INTEGER,
    actual_nfl_team    TEXT,
    draft_confirmed    BOOLEAN     DEFAULT FALSE,
    created_at         TIMESTAMP   DEFAULT NOW(),
    updated_at         TIMESTAMP   DEFAULT NOW()
);

-- Idempotent additions for existing databases
ALTER TABLE rookie_prospects ADD COLUMN IF NOT EXISTS actual_pick      INTEGER;
ALTER TABLE rookie_prospects ADD COLUMN IF NOT EXISTS actual_round     INTEGER;
ALTER TABLE rookie_prospects ADD COLUMN IF NOT EXISTS actual_nfl_team  TEXT;
ALTER TABLE rookie_prospects ADD COLUMN IF NOT EXISTS draft_confirmed  BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS rookie_prospect_source_data (
    id                          SERIAL      PRIMARY KEY,
    player_id                   TEXT        NOT NULL REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    season                      INTEGER     NOT NULL,
    games_played                INTEGER,
    -- Passing
    pass_yards                  INTEGER,
    pass_tds                    INTEGER,
    pass_attempts               INTEGER,
    completions                 INTEGER,
    interceptions               INTEGER,
    -- Rushing
    rush_attempts               INTEGER,
    rush_yards                  INTEGER,
    rush_tds                    INTEGER,
    -- Receiving
    receptions                  INTEGER,
    targets                     INTEGER,
    receiving_yards             INTEGER,
    receiving_tds               INTEGER,
    -- Derived / advanced
    dominator_rating            DECIMAL(6,3),
    market_share_yards          DECIMAL(6,3),
    market_share_tds            DECIMAL(6,3),
    yds_per_carry               DECIMAL(5,2),
    yds_per_reception           DECIMAL(5,2),
    yds_per_attempt             DECIMAL(5,2),
    completion_pct              DECIMAL(5,2),
    td_int_ratio                DECIMAL(6,2),
    -- Team context
    team                        TEXT,
    conference                  TEXT,
    team_pass_rate              DECIMAL(5,3),
    team_total_yards            INTEGER,
    team_total_tds              INTEGER,
    team_pass_yards             INTEGER,        -- net passing yards (for WR/TE dominator)
    sagarin_team_rating         DECIMAL(6,2),   -- strength-of-schedule adjustment
    -- PFF / advanced receiving
    yards_after_catch           DECIMAL(8,1),
    yards_after_catch_per_reception DECIMAL(5,2),
    avg_depth_of_target         DECIMAL(5,2),
    contested_catch_rate        DECIMAL(5,3),
    avoided_tackles             INTEGER,
    drop_rate                   DECIMAL(5,3),
    slot_rate                   DECIMAL(5,3),
    wide_rate                   DECIMAL(5,3),
    inline_rate                 DECIMAL(5,3),
    pass_block_rate             DECIMAL(5,3),
    grades_offense              DECIMAL(4,1),
    grades_pass_block           DECIMAL(4,1),
    explosive_runs_10_plus      INTEGER,
    breakaway_percentage        DECIMAL(5,3),
    elusive_rating              DECIMAL(5,2),
    pff_rushing_grade           DECIMAL(4,1),
    pff_passing_grade           DECIMAL(4,1),
    big_time_throw_rate         DECIMAL(5,3),
    adjusted_completion_rate    DECIMAL(5,3),
    pressure_to_sack_rate       DECIMAL(5,3),
    nfl_passer_rating           DECIMAL(4,1),
    source                      TEXT        DEFAULT 'cfbd',
    fetched_at                  TIMESTAMP   DEFAULT NOW(),
    UNIQUE(player_id, season, source)
);

-- Idempotent additions for existing databases
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS team_pass_yards      INTEGER;
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS sagarin_team_rating  DECIMAL(6,2);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS yards_after_catch    DECIMAL(8,1);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS yards_after_catch_per_reception DECIMAL(5,2);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS avg_depth_of_target  DECIMAL(5,2);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS contested_catch_rate DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS avoided_tackles      INTEGER;
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS drop_rate            DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS slot_rate            DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS wide_rate            DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS inline_rate          DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS pass_block_rate      DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS grades_offense       DECIMAL(4,1);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS grades_pass_block    DECIMAL(4,1);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS explosive_runs_10_plus INTEGER;
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS breakaway_percentage DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS elusive_rating       DECIMAL(5,2);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS pff_rushing_grade    DECIMAL(4,1);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS pff_passing_grade    DECIMAL(4,1);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS big_time_throw_rate  DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS adjusted_completion_rate DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS pressure_to_sack_rate DECIMAL(5,3);
ALTER TABLE rookie_prospect_source_data ADD COLUMN IF NOT EXISTS nfl_passer_rating    DECIMAL(4,1);

CREATE TABLE IF NOT EXISTS rookie_prospect_athleticism (
    player_id        TEXT        PRIMARY KEY REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    forty_yard       DECIMAL(4,2),
    vertical_inches  DECIMAL(4,1),
    broad_jump_in    INTEGER,
    three_cone       DECIMAL(4,2),
    short_shuttle    DECIMAL(4,2),
    bench_reps       INTEGER,
    speed_score      DECIMAL(6,2),
    ras_score        DECIMAL(4,2),
    source           TEXT        DEFAULT 'manual',
    updated_at       TIMESTAMP   DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rookie_mock_draft_entries (
    id               SERIAL      PRIMARY KEY,
    player_id        TEXT        NOT NULL REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    draft_class_year INTEGER     NOT NULL,
    source_name      TEXT        NOT NULL,
    source_url       TEXT,
    projected_round  INTEGER,
    projected_pick   INTEGER,
    mock_date        DATE,
    analyst_name     TEXT,
    ingested_at      TIMESTAMP   DEFAULT NOW()
);

-- Fixed unique constraint: include analyst_name so multiple analysts on same day are separate rows
DROP INDEX IF EXISTS rookie_mock_draft_entries_player_id_source_name_mock_date_key;
ALTER TABLE rookie_mock_draft_entries
    DROP CONSTRAINT IF EXISTS rookie_mock_draft_entries_player_id_source_name_mock_date_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_mock_entry_per_analyst
    ON rookie_mock_draft_entries (player_id, source_name, mock_date, COALESCE(analyst_name, ''));

CREATE TABLE IF NOT EXISTS rookie_mock_draft_consensus (
    player_id                     TEXT        PRIMARY KEY REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    draft_class_year              INTEGER     NOT NULL,
    projected_round               INTEGER,
    projected_pick                INTEGER,
    projected_pick_low            INTEGER,
    projected_pick_high           INTEGER,
    projected_draft_capital_score DECIMAL(6,2),
    num_mocks_used                INTEGER     DEFAULT 0,
    consensus_confidence          DECIMAL(6,2),
    mock_sources                  JSONB       DEFAULT '[]',
    calculated_at                 TIMESTAMP   DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rookie_rankings (
    player_id                     TEXT        NOT NULL,
    draft_class_year              INTEGER     NOT NULL,
    overall_rank                  INTEGER,
    position_rank                 INTEGER,
    production_score              DECIMAL(6,2),
    efficiency_score              DECIMAL(6,2),
    age_score                     DECIMAL(6,2),
    breakout_profile_score        DECIMAL(6,2),
    athleticism_score             DECIMAL(6,2),
    competition_score             DECIMAL(6,2),
    environment_adjustment        DECIMAL(6,2),
    durability_score              DECIMAL(6,2),
    projected_draft_capital_score DECIMAL(6,2),
    fantasy_translation_score     DECIMAL(6,2),
    confidence_score              DECIMAL(6,2),
    prospect_score                DECIMAL(6,2),
    rookie_value                  DECIMAL(8,2),
    rookie_sf_value               DECIMAL(8,2),
    rookie_value_8                DECIMAL(8,2),
    rookie_value_12               DECIMAL(8,2),
    rookie_value_14               DECIMAL(8,2),
    rookie_sf_value_8             DECIMAL(8,2),
    rookie_sf_value_12            DECIMAL(8,2),
    rookie_sf_value_14            DECIMAL(8,2),
    tier                          INTEGER,
    tier_label                    TEXT,
    key_reasons                   TEXT,
    calculated_at                 TIMESTAMP   DEFAULT NOW(),
    PRIMARY KEY (player_id, draft_class_year)
);

CREATE TABLE IF NOT EXISTS rookie_value_history (
    player_id        TEXT        NOT NULL,
    draft_class_year INTEGER     NOT NULL,
    snapshot_date    DATE        NOT NULL,
    overall_rank     INTEGER,
    position_rank    INTEGER,
    rookie_value     DECIMAL(8,2),
    rookie_sf_value  DECIMAL(8,2),
    prospect_score   DECIMAL(6,2),
    PRIMARY KEY (player_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS rookie_active_class (
    draft_class_year INTEGER     PRIMARY KEY,
    is_active        BOOLEAN     DEFAULT TRUE,
    offseason_start  DATE,
    draft_date       DATE,
    season_start     DATE,
    season_end       DATE,
    notes            TEXT,
    created_at       TIMESTAMP   DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_rp_class          ON rookie_prospects(draft_class_year);
CREATE INDEX IF NOT EXISTS idx_rp_position        ON rookie_prospects(position);
CREATE INDEX IF NOT EXISTS idx_rp_draft_confirmed ON rookie_prospects(draft_class_year, draft_confirmed) WHERE draft_confirmed = TRUE;
CREATE INDEX IF NOT EXISTS idx_rpsd_player        ON rookie_prospect_source_data(player_id);
CREATE INDEX IF NOT EXISTS idx_rpsd_season        ON rookie_prospect_source_data(season);
CREATE INDEX IF NOT EXISTS idx_rmde_player        ON rookie_mock_draft_entries(player_id, draft_class_year);
CREATE INDEX IF NOT EXISTS idx_rr_class_rank      ON rookie_rankings(draft_class_year, overall_rank);
CREATE INDEX IF NOT EXISTS idx_rvh_date           ON rookie_value_history(snapshot_date);

-- Seed active draft classes
INSERT INTO rookie_active_class (draft_class_year, is_active, offseason_start, draft_date, season_start, season_end, notes)
VALUES
    (2025, FALSE, '2025-02-01', '2025-04-24', '2025-09-04', '2026-01-11',
     '2025 class played their rookie NFL season Sep 2025 – Jan 2026'),
    (2026, TRUE,  '2026-02-01', '2026-04-23', '2026-09-10', '2027-01-10',
     '2026 class — upcoming draft, pre-draft evaluation window active')
ON CONFLICT (draft_class_year) DO NOTHING;

-- Data fix: remove duplicate prospect caused by period-in-name slug bug (migration 012)
DELETE FROM rookie_rankings      WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
DELETE FROM rookie_value_history WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
DELETE FROM rookie_prospects     WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
