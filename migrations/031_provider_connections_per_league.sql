-- Allow multiple private connections per provider (MFL league APIKEY and
-- Fleaflicker tokens are league-scoped). ESPN still reuses one private row
-- via application logic. Lookup always goes through user_leagues.provider_connection_id.
ALTER TABLE fantasy_provider_connections
    DROP CONSTRAINT IF EXISTS fantasy_provider_connections_account_id_provider_connection_method_key;

CREATE INDEX IF NOT EXISTS fantasy_provider_connections_account_provider_idx
    ON fantasy_provider_connections (account_id, provider, connection_method);
