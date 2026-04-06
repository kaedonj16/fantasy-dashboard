-- Migration 009: Rookie prospect evaluation system
-- Stores draft prospects, college stats, mock draft data, scores, and values.

-- ─────────────────────────────────────────────────────────────────────────────
-- Core prospect bio / identity
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_prospects (
    player_id        TEXT PRIMARY KEY,        -- internal ID: ROOKIE_{YEAR}_{NAME_SLUG}
    sleeper_id       TEXT,                    -- linked after NFL draft
    name             TEXT NOT NULL,
    position         TEXT NOT NULL,           -- QB / RB / WR / TE
    school           TEXT,
    age              DECIMAL(4,1),
    height_inches    INTEGER,
    weight_lbs       INTEGER,
    hometown         TEXT,
    state            TEXT,
    draft_class_year INTEGER NOT NULL,
    early_declare    BOOLEAN  DEFAULT FALSE,
    transfer_history TEXT,
    headshot_url     TEXT,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Season-level college production (one row per player per year per source)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_prospect_source_data (
    id                   SERIAL PRIMARY KEY,
    player_id            TEXT    NOT NULL REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    season               INTEGER NOT NULL,
    games_played         INTEGER,
    -- Passing
    pass_yards           INTEGER,
    pass_tds             INTEGER,
    pass_attempts        INTEGER,
    completions          INTEGER,
    interceptions        INTEGER,
    -- Rushing
    rush_attempts        INTEGER,
    rush_yards           INTEGER,
    rush_tds             INTEGER,
    -- Receiving
    receptions           INTEGER,
    targets              INTEGER,
    receiving_yards      INTEGER,
    receiving_tds        INTEGER,
    -- Derived / advanced (computed during ingestion)
    dominator_rating     DECIMAL(6,3),   -- pct of team yards+TDs
    market_share_yards   DECIMAL(6,3),
    market_share_tds     DECIMAL(6,3),
    yds_per_carry        DECIMAL(5,2),
    yds_per_reception    DECIMAL(5,2),
    yds_per_attempt      DECIMAL(5,2),
    completion_pct       DECIMAL(5,2),
    td_int_ratio         DECIMAL(6,2),
    -- Team context
    team                 TEXT,
    conference           TEXT,
    team_pass_rate       DECIMAL(5,3),   -- 0-1, pass plays / total plays
    team_total_yards     INTEGER,
    team_total_tds       INTEGER,
    -- Metadata
    source               TEXT DEFAULT 'cfbd',
    fetched_at           TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, season, source)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Athletic testing (combine / pro day)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_prospect_athleticism (
    player_id        TEXT PRIMARY KEY REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    forty_yard       DECIMAL(4,2),    -- 40-yard dash (seconds)
    vertical_inches  DECIMAL(4,1),   -- vertical jump
    broad_jump_in    INTEGER,         -- broad jump inches
    three_cone       DECIMAL(4,2),   -- 3-cone drill
    short_shuttle    DECIMAL(4,2),   -- shuttle run
    bench_reps       INTEGER,         -- bench press reps
    speed_score      DECIMAL(6,2),   -- weight-adjusted speed (40*weight^0.5)
    ras_score        DECIMAL(4,2),   -- relative athletic score (0-10)
    source           TEXT DEFAULT 'manual',
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Raw mock draft entries (one per prospect per mock source per date)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_mock_draft_entries (
    id               SERIAL PRIMARY KEY,
    player_id        TEXT    NOT NULL REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    draft_class_year INTEGER NOT NULL,
    source_name      TEXT    NOT NULL,   -- e.g. "ESPN_McShay", "PFF", "TheAthletic"
    source_url       TEXT,
    projected_round  INTEGER,
    projected_pick   INTEGER,            -- overall pick number
    mock_date        DATE,
    analyst_name     TEXT,
    ingested_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, source_name, mock_date)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Aggregated mock draft consensus (one row per prospect per class)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_mock_draft_consensus (
    player_id                    TEXT    PRIMARY KEY REFERENCES rookie_prospects(player_id) ON DELETE CASCADE,
    draft_class_year             INTEGER NOT NULL,
    projected_round              INTEGER,
    projected_pick               INTEGER,  -- consensus median pick
    projected_pick_low           INTEGER,  -- best-case (lowest pick number seen)
    projected_pick_high          INTEGER,  -- worst-case (highest pick number seen)
    projected_draft_capital_score DECIMAL(6,2),  -- 0-100 normalized score
    num_mocks_used               INTEGER DEFAULT 0,
    consensus_confidence         DECIMAL(6,2),   -- 0-100 confidence metric
    mock_sources                 JSONB   DEFAULT '[]',
    calculated_at                TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Scored prospect rankings (recalculated each pipeline run)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_rankings (
    player_id                    TEXT    NOT NULL,
    draft_class_year             INTEGER NOT NULL,
    overall_rank                 INTEGER,
    position_rank                INTEGER,
    -- Component scores (each 0–100)
    production_score             DECIMAL(6,2),
    efficiency_score             DECIMAL(6,2),
    age_score                    DECIMAL(6,2),
    breakout_profile_score       DECIMAL(6,2),
    athleticism_score            DECIMAL(6,2),
    competition_score            DECIMAL(6,2),
    environment_adjustment       DECIMAL(6,2),
    durability_score             DECIMAL(6,2),
    projected_draft_capital_score DECIMAL(6,2),
    fantasy_translation_score    DECIMAL(6,2),
    confidence_score             DECIMAL(6,2),
    -- Final output
    prospect_score               DECIMAL(6,2),
    rookie_value                 DECIMAL(8,2),  -- 1QB dynasty value (0-999 scale)
    rookie_sf_value              DECIMAL(8,2),  -- SF dynasty value
    tier                         INTEGER,        -- 1-6
    tier_label                   TEXT,
    key_reasons                  TEXT,           -- human-readable summary
    calculated_at                TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (player_id, draft_class_year)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Value history snapshots (for tracking change over time)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_value_history (
    player_id        TEXT    NOT NULL,
    draft_class_year INTEGER NOT NULL,
    snapshot_date    DATE    NOT NULL,
    overall_rank     INTEGER,
    position_rank    INTEGER,
    rookie_value     DECIMAL(8,2),
    rookie_sf_value  DECIMAL(8,2),
    prospect_score   DECIMAL(6,2),
    PRIMARY KEY (player_id, snapshot_date)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Active class tracker
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rookie_active_class (
    draft_class_year  INTEGER PRIMARY KEY,
    is_active         BOOLEAN  DEFAULT TRUE,
    offseason_start   DATE,    -- approx start of evaluation window (Feb 1)
    draft_date        DATE,    -- approximate draft date
    season_start      DATE,    -- approx NFL season kickoff
    season_end        DATE,    -- approx end of rookie season (wild-card weekend)
    notes             TEXT,
    created_at        TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_rp_class        ON rookie_prospects(draft_class_year);
CREATE INDEX IF NOT EXISTS idx_rp_position     ON rookie_prospects(position);
CREATE INDEX IF NOT EXISTS idx_rpsd_player     ON rookie_prospect_source_data(player_id);
CREATE INDEX IF NOT EXISTS idx_rpsd_season     ON rookie_prospect_source_data(season);
CREATE INDEX IF NOT EXISTS idx_rmde_player     ON rookie_mock_draft_entries(player_id, draft_class_year);
CREATE INDEX IF NOT EXISTS idx_rr_class_rank   ON rookie_rankings(draft_class_year, overall_rank);
CREATE INDEX IF NOT EXISTS idx_rvh_date        ON rookie_value_history(snapshot_date);

-- Seed active class records for 2025 and 2026
INSERT INTO rookie_active_class (draft_class_year, is_active, offseason_start, draft_date, season_start, season_end, notes)
VALUES
    (2025, FALSE, '2025-02-01', '2025-04-24', '2025-09-04', '2026-01-11',
     '2025 class played their rookie NFL season Sep 2025 – Jan 2026'),
    (2026, TRUE,  '2026-02-01', '2026-04-23', '2026-09-10', '2027-01-10',
     '2026 class — upcoming draft, pre-draft evaluation window active')
ON CONFLICT (draft_class_year) DO NOTHING;
