-- Monthly billing: record which interval a subscription renews on.
-- 'month' | 'year'. Existing rows (all annual) keep the 'year' default.
-- Grants written before this migration deployed never set the column,
-- so the default keeps reads correct for legacy rows.

ALTER TABLE league_subscriptions
    ADD COLUMN IF NOT EXISTS billing_interval TEXT NOT NULL DEFAULT 'year';
ALTER TABLE user_subscriptions
    ADD COLUMN IF NOT EXISTS billing_interval TEXT NOT NULL DEFAULT 'year';
ALTER TABLE user_league_subscriptions
    ADD COLUMN IF NOT EXISTS billing_interval TEXT NOT NULL DEFAULT 'year';
