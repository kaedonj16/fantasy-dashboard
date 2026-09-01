-- Personal single-league subscriptions: one user unlocks PRO for one selected
-- league only (buyer-only — not shared with co-managers). Distinct from:
--   league_subscriptions  = shared PRO for every manager in a league ($15)
--   user_subscriptions    = personal PRO across all of a user's leagues ($10)

CREATE TABLE IF NOT EXISTS user_league_subscriptions (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'sleeper',
    league_id TEXT NOT NULL,
    subscription_status TEXT NOT NULL DEFAULT 'active',
    stripe_subscription_id TEXT,
    stripe_customer_id TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_user_league_status CHECK (
        subscription_status IN ('active', 'canceled', 'expired')
    ),
    UNIQUE (user_id, platform, league_id)
);

CREATE INDEX IF NOT EXISTS idx_user_league_subs_lookup
    ON user_league_subscriptions(user_id, platform, league_id);
CREATE INDEX IF NOT EXISTS idx_user_league_subs_league
    ON user_league_subscriptions(platform, league_id);
CREATE INDEX IF NOT EXISTS idx_user_league_subs_expires
    ON user_league_subscriptions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_league_subs_stripe
    ON user_league_subscriptions(stripe_subscription_id);

DROP TRIGGER IF EXISTS update_user_league_subscriptions_updated_at
    ON user_league_subscriptions;
CREATE TRIGGER update_user_league_subscriptions_updated_at
    BEFORE UPDATE ON user_league_subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE user_league_subscriptions IS
    'Buyer-only PRO for a single selected league ($5/year)';
