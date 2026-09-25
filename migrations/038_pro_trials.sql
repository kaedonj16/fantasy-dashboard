-- PRO free-trial state. One row per user ever: UNIQUE(user_key) enforces the
-- one-trial-per-user rule; a second claim finds the row and is refused.
-- Trial rows are keyed by the Google account (acct:<account_id>) because that
-- is the only stable cross-session identity (same rule as paid checkout).
-- Re-claim via a brand-new Google account is out of scope.
CREATE TABLE IF NOT EXISTS pro_trials (
    id SERIAL PRIMARY KEY,
    user_key TEXT NOT NULL UNIQUE,
    account_id BIGINT,
    trial_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trial_ends_at TIMESTAMPTZ NOT NULL,
    subscription_status TEXT NOT NULL DEFAULT 'active',
    ended_notified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_pro_trial_status CHECK (
        subscription_status IN ('active', 'expired')
    )
);

CREATE INDEX IF NOT EXISTS idx_pro_trials_ends_at ON pro_trials(trial_ends_at);
CREATE INDEX IF NOT EXISTS idx_pro_trials_account_id ON pro_trials(account_id);
