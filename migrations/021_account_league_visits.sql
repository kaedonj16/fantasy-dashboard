-- Account-scoped state for the one-time "Since your last visit" digest.
CREATE TABLE IF NOT EXISTS account_league_visits (
    account_id      INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    platform        TEXT NOT NULL,
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    roster_id       TEXT,
    last_visit_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    roster_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (account_id, platform, league_id, season)
);
