-- Plan restructure: plan_key on user_subscriptions + PRO league slots.
--
-- user_subscriptions.plan_key records which catalog plan a personal row was
-- granted for: 'starter' | 'all_pro' | 'hall_of_fame' (new catalog) or the
-- grandfathered 'user'. NULL means "legacy row" and reads as unlimited
-- personal PRO (the old user-plan semantics).
--
-- pro_league_slots holds the selected leagues for the slot-capped plans
-- (starter: 1, all_pro: 5). One row per (user_id, platform, league_id);
-- rows only grant access while the owning subscription row is active.

ALTER TABLE user_subscriptions
    ADD COLUMN IF NOT EXISTS plan_key TEXT;

CREATE TABLE IF NOT EXISTS pro_league_slots (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'sleeper',
    league_id TEXT NOT NULL,
    stripe_subscription_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_pro_league_slot UNIQUE (user_id, platform, league_id)
);

CREATE INDEX IF NOT EXISTS idx_pro_league_slots_user
    ON pro_league_slots (user_id, platform);
CREATE INDEX IF NOT EXISTS idx_pro_league_slots_league
    ON pro_league_slots (platform, league_id);
