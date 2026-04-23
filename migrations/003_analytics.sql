-- League analytics tables — playoff odds and luck index.
-- Consolidates migrations 003, 005.

CREATE TABLE IF NOT EXISTS playoff_odds (
    league_id                TEXT        NOT NULL,
    season                   INTEGER     NOT NULL,
    week                     INTEGER     NOT NULL,
    roster_id                INTEGER     NOT NULL,
    team_name                TEXT,
    current_wins             INTEGER,
    current_losses           INTEGER,
    current_ties             INTEGER     DEFAULT 0,
    playoff_probability      DECIMAL(5,2),
    first_seed_probability   DECIMAL(5,2),
    bye_probability          DECIMAL(5,2),
    miss_playoffs_probability DECIMAL(5,2),
    avg_final_wins           DECIMAL(5,2),
    avg_final_losses         DECIMAL(5,2),
    num_simulations          INTEGER     DEFAULT 10000,
    calculated_at            TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (league_id, season, week, roster_id)
);

CREATE TABLE IF NOT EXISTS luck_index (
    league_id                   TEXT        NOT NULL,
    season                      INTEGER     NOT NULL,
    roster_id                   INTEGER     NOT NULL,
    team_name                   TEXT,
    avg_opponent_score          DECIMAL(10,2),
    league_avg_opponent_score   DECIMAL(10,2),
    schedule_luck_score         DECIMAL(5,2),
    close_game_wins             INTEGER,
    close_game_losses           INTEGER,
    close_game_luck_score       DECIMAL(5,2),
    actual_points               DECIMAL(10,2),
    optimal_points              DECIMAL(10,2),
    lineup_efficiency           DECIMAL(5,2),
    overall_luck_score          DECIMAL(5,2),
    luck_tier                   TEXT,
    weeks_analyzed              INTEGER,
    calculated_at               TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (league_id, season, roster_id)
);

CREATE INDEX IF NOT EXISTS idx_playoff_odds_league_season ON playoff_odds(league_id, season, week);
CREATE INDEX IF NOT EXISTS idx_playoff_odds_team          ON playoff_odds(roster_id);
CREATE INDEX IF NOT EXISTS idx_luck_index_league          ON luck_index(league_id, season);
CREATE INDEX IF NOT EXISTS idx_luck_index_tier            ON luck_index(luck_tier);
