CREATE TABLE IF NOT EXISTS fantasy_provider_connections (
    id                    SERIAL PRIMARY KEY,
    account_id            INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    provider              TEXT NOT NULL,
    connection_method     TEXT NOT NULL CHECK (connection_method IN ('public', 'private')),
    encrypted_credentials TEXT,
    status                TEXT NOT NULL DEFAULT 'connected',
    last_authenticated_at TIMESTAMPTZ,
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now(),
    UNIQUE (account_id, provider, connection_method)
);

ALTER TABLE user_leagues
    ADD COLUMN IF NOT EXISTS provider_connection_id INTEGER
    REFERENCES fantasy_provider_connections(id) ON DELETE SET NULL;
