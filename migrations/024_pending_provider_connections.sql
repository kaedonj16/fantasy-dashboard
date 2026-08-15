CREATE TABLE IF NOT EXISTS pending_provider_connections (
    token_hash TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    connection_method TEXT NOT NULL,
    league_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    league_name TEXT,
    team_id TEXT,
    encrypted_credentials TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pending_provider_connections_expires_idx
    ON pending_provider_connections (expires_at);
