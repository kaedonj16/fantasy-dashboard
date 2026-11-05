-- Wave 1 (league-rules truth): one shared rules record per league/season.
--
-- Playoff format, waiver system, trade deadline and designated keepers,
-- resolved with precedence user_confirmed > feed > derived >
-- default_unverified and applied to league contexts as a serve-time
-- overlay (utils/league_rules.py). Until this table existed, consumers
-- re-read provider settings that some adapters silently default (the
-- Fleaflicker adapter serves playoff_teams=6 / playoff_week_start=15 for
-- values its API never publishes), so a league whose real field is 8
-- computed playoff odds, the standings picture and the projected bracket
-- on the wrong field, and trade surfaces could not know which players
-- were designated keepers (untradeable in keeper leagues that bar it).
--
-- League-level (not per-account): a league's rules are the same for
-- every viewer. designated_keepers is {roster_id: [player_id, ...]}.
-- IF NOT EXISTS so re-runs are safe; the code also creates the table
-- idempotently and fails soft when it is absent.
CREATE TABLE IF NOT EXISTS league_rules (
    platform           TEXT NOT NULL,
    league_id          TEXT NOT NULL,
    season             INTEGER NOT NULL,
    playoff_teams      INTEGER,
    playoff_week_start INTEGER,
    playoff_round_type INTEGER,
    playoff_reseed     BOOLEAN,
    waiver_type        TEXT,
    waiver_process_time TIME,
    waiver_process_days TEXT,
    waiver_process_tz  TEXT DEFAULT 'America/New_York',
    trade_deadline     DATE,
    max_keepers        INTEGER,
    designated_keepers JSONB NOT NULL DEFAULT '{}'::jsonb,
    keepers_known      BOOLEAN NOT NULL DEFAULT FALSE,
    source             TEXT NOT NULL,
    confirmed_by       INTEGER,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, league_id, season)
);
