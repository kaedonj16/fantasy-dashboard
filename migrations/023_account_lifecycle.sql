ALTER TABLE accounts ADD COLUMN IF NOT EXISTS first_name TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_platform TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_league_id TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS last_active_season INTEGER;

CREATE TABLE IF NOT EXISTS account_auth_identities (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    auth_provider TEXT NOT NULL,
    auth_provider_subject TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (auth_provider, auth_provider_subject)
);

INSERT INTO account_auth_identities (account_id, auth_provider, auth_provider_subject)
SELECT id, 'google', google_sub FROM accounts WHERE google_sub IS NOT NULL
ON CONFLICT (auth_provider, auth_provider_subject) DO NOTHING;

ALTER TABLE fantasy_provider_connections ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ;
ALTER TABLE fantasy_provider_connections ADD COLUMN IF NOT EXISTS last_successful_sync_at TIMESTAMPTZ;
ALTER TABLE fantasy_provider_connections ADD COLUMN IF NOT EXISTS last_error_code TEXT;
ALTER TABLE fantasy_provider_connections ADD COLUMN IF NOT EXISTS credential_expires_at TIMESTAMPTZ;

ALTER TABLE fantasy_provider_connections DROP CONSTRAINT IF EXISTS fantasy_provider_connections_status_check;
ALTER TABLE fantasy_provider_connections ADD CONSTRAINT fantasy_provider_connections_status_check
    CHECK (status IN ('connected', 'reauth_required', 'sync_error', 'disconnected'));
