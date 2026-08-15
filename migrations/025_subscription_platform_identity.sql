-- League IDs are only unique within a fantasy provider. ESPN and Sleeper can
-- legitimately issue the same text ID, so subscriptions must use both values.
ALTER TABLE league_subscriptions
    DROP CONSTRAINT IF EXISTS league_subscriptions_league_id_key;

CREATE UNIQUE INDEX IF NOT EXISTS league_subscriptions_platform_league_uidx
    ON league_subscriptions (platform, league_id);
