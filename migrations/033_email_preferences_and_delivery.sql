-- Extensible email notification preferences, delivery observability, and bounce
-- suppression. Backwards compatible with accounts.email_opt_out:
--   * existing TRUE opt-outs continue to suppress weekly_digest when no
--     preference row exists
--   * existing users without a preference row keep receiving weekly_digest

ALTER TABLE accounts ADD COLUMN IF NOT EXISTS email_opt_out BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS account_notification_preferences (
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    channel TEXT NOT NULL DEFAULT 'email',
    notification_type TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, channel, notification_type)
);

CREATE INDEX IF NOT EXISTS account_notification_preferences_type_idx
    ON account_notification_preferences (notification_type, enabled);

CREATE TABLE IF NOT EXISTS email_delivery_events (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    email TEXT,
    email_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_message_id TEXT,
    platform TEXT,
    league_id TEXT,
    season INTEGER,
    iso_week TEXT,
    status TEXT NOT NULL,
    error_category TEXT,
    error_detail TEXT,
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ,
    bounced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS email_delivery_events_account_week_idx
    ON email_delivery_events (account_id, email_type, iso_week);
CREATE INDEX IF NOT EXISTS email_delivery_events_message_id_idx
    ON email_delivery_events (provider_message_id);
CREATE INDEX IF NOT EXISTS email_delivery_events_email_idx
    ON email_delivery_events (email);

CREATE TABLE IF NOT EXISTS email_suppressions (
    email TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    provider TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
