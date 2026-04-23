-- Breakout and opportunity scoring system.
-- Consolidates migrations 006, 007, 008.

CREATE TABLE IF NOT EXISTS roster_changes (
    id                           SERIAL      PRIMARY KEY,
    player_id                    VARCHAR(50) NOT NULL,
    player_name                  VARCHAR(255),
    position                     VARCHAR(5),
    old_team                     VARCHAR(10),
    new_team                     VARCHAR(10),
    change_type                  VARCHAR(20),  -- 'free_agent','trade','retirement','cut','draft'
    change_date                  DATE,
    season                       INT,
    last_season_targets          INT,
    last_season_carries          INT,
    last_season_snap_share       NUMERIC,
    last_season_opportunity_share NUMERIC,
    last_season_team_target_pct  NUMERIC,
    last_season_team_carry_pct   NUMERIC,
    draft_metadata               JSONB,        -- {round, pick, overall_pick, college}
    created_at                   TIMESTAMP   DEFAULT NOW(),
    UNIQUE(player_id, old_team, new_team, season)
);

CREATE TABLE IF NOT EXISTS vacated_opportunity (
    id                       SERIAL      PRIMARY KEY,
    team                     VARCHAR(10) NOT NULL,
    position                 VARCHAR(5)  NOT NULL,
    season                   INT         NOT NULL,
    total_targets_vacated    INT         DEFAULT 0,
    total_carries_vacated    INT         DEFAULT 0,
    total_snap_share_vacated NUMERIC     DEFAULT 0.0,
    departed_players         JSONB,
    calculated_at            TIMESTAMP   DEFAULT NOW(),
    UNIQUE(team, position, season)
);

CREATE TABLE IF NOT EXISTS projected_opportunity (
    id                          SERIAL      PRIMARY KEY,
    player_id                   VARCHAR(50) NOT NULL,
    season                      INT         NOT NULL,
    team                        VARCHAR(10),
    position                    VARCHAR(5),
    player_name                 VARCHAR(255),
    -- Previous season baselines
    prev_season_targets         INT         DEFAULT 0,
    prev_season_carries         INT         DEFAULT 0,
    prev_season_snap_share      NUMERIC     DEFAULT 0.0,
    prev_season_opportunity_share NUMERIC   DEFAULT 0.0,
    -- Projections
    projected_targets           INT         DEFAULT 0,
    projected_carries           INT         DEFAULT 0,
    projected_snap_share        NUMERIC     DEFAULT 0.0,
    projected_opportunity_share NUMERIC     DEFAULT 0.0,
    -- Deltas
    target_increase             INT         DEFAULT 0,
    carry_increase              INT         DEFAULT 0,
    snap_share_increase         NUMERIC     DEFAULT 0.0,
    opportunity_share_increase  NUMERIC     DEFAULT 0.0,
    -- Legacy baseline columns kept for backward compatibility
    baseline_targets            INT         DEFAULT 0,
    baseline_carries            INT         DEFAULT 0,
    baseline_snap_share         NUMERIC     DEFAULT 0.0,
    breakout_score              NUMERIC     DEFAULT 0.0,
    projection_factors          JSONB,
    calculated_at               TIMESTAMP   DEFAULT NOW(),
    UNIQUE(player_id, season)
);

-- Migrate existing data from legacy baseline_* columns where prev_season_* is empty
UPDATE projected_opportunity
SET prev_season_targets      = baseline_targets,
    prev_season_carries      = baseline_carries,
    prev_season_snap_share   = baseline_snap_share
WHERE prev_season_targets = 0
  AND prev_season_carries = 0
  AND prev_season_snap_share = 0
  AND baseline_targets IS NOT NULL;

CREATE TABLE IF NOT EXISTS breakout_opportunity_scores (
    id                              SERIAL      PRIMARY KEY,
    player_id                       VARCHAR(50) NOT NULL,
    season                          INT         NOT NULL,
    as_of_date                      DATE        NOT NULL,
    team                            VARCHAR(10),
    position                        VARCHAR(5),
    opportunity_opened_score        NUMERIC,
    competition_removed_score       NUMERIC,
    competition_added_penalty       NUMERIC,
    team_environment_score          NUMERIC,
    player_readiness_score          NUMERIC,
    role_trajectory_score           NUMERIC,
    confidence_score                NUMERIC,
    breakout_opportunity_score      NUMERIC,
    phase                           VARCHAR(20),
    directional_trend               VARCHAR(10),
    key_reasons                     TEXT,
    recent_transactions_affecting_player TEXT,
    vacated_usage_summary           TEXT,
    added_competition_summary       TEXT,
    projected_role_tag              VARCHAR(100),
    component_details               JSONB,
    calculated_at                   TIMESTAMP   DEFAULT NOW(),
    UNIQUE(player_id, season, as_of_date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_roster_changes_season_team_position    ON roster_changes(season, new_team, position);
CREATE INDEX IF NOT EXISTS idx_roster_changes_player_season           ON roster_changes(player_id, season);
CREATE INDEX IF NOT EXISTS idx_roster_changes_change_type_season      ON roster_changes(change_type, season);
CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_team_position_season ON vacated_opportunity(team, position, season);
CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_season             ON vacated_opportunity(season);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_score     ON projected_opportunity(season, breakout_score DESC);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_position  ON projected_opportunity(season, position);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_player_season    ON projected_opportunity(player_id, season);
CREATE INDEX IF NOT EXISTS idx_projected_opportunity_ui_query         ON projected_opportunity(season, position, breakout_score DESC) WHERE breakout_score >= 30;
CREATE INDEX IF NOT EXISTS idx_breakout_scores_player                 ON breakout_opportunity_scores(player_id);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_season                 ON breakout_opportunity_scores(season);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_date                   ON breakout_opportunity_scores(as_of_date);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_season_score           ON breakout_opportunity_scores(season, breakout_opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_position_score         ON breakout_opportunity_scores(position, breakout_opportunity_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakout_scores_ui_query               ON breakout_opportunity_scores(season, position, breakout_opportunity_score DESC) WHERE breakout_opportunity_score >= 40;
