ALTER TABLE pending_provider_connections
    ADD COLUMN IF NOT EXISTS team_id TEXT;
