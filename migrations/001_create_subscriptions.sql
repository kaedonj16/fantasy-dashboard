-- League-based subscription system
-- Supports two subscription types:
--   1. League subscriptions: entire league gets premium access
--   2. User subscriptions: user gets premium access across all their leagues

CREATE TABLE IF NOT EXISTS league_subscriptions (
    id SERIAL PRIMARY KEY,
    league_id TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL DEFAULT 'sleeper',
    subscriber_user_id TEXT NOT NULL,  -- Who paid for the subscription
    subscription_status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'canceled', 'expired'
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_status CHECK (subscription_status IN ('active', 'canceled', 'expired'))
);

CREATE TABLE IF NOT EXISTS user_subscriptions (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,  -- Sleeper username or user ID
    platform TEXT NOT NULL DEFAULT 'sleeper',
    subscription_status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'canceled', 'expired'
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_user_status CHECK (subscription_status IN ('active', 'canceled', 'expired')),
    UNIQUE (user_id, platform)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_league_subs_league_id ON league_subscriptions(league_id);
CREATE INDEX IF NOT EXISTS idx_league_subs_expires_at ON league_subscriptions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_subs_user_id ON user_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_subs_expires_at ON user_subscriptions(expires_at);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_league_subscriptions_updated_at ON league_subscriptions;
CREATE TRIGGER update_league_subscriptions_updated_at BEFORE UPDATE ON league_subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_subscriptions_updated_at ON user_subscriptions;
CREATE TRIGGER update_user_subscriptions_updated_at BEFORE UPDATE ON user_subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
