-- Yahoo OAuth 2.0 token storage.
-- Tokens are keyed by the Yahoo user GUID (returned in the token response as
-- xoauth_yahoo_guid).  A single user may have tokens refreshed in-place, so
-- the primary key is simply the guid.

CREATE TABLE IF NOT EXISTS yahoo_oauth_tokens (
    guid            TEXT        NOT NULL PRIMARY KEY,
    access_token    TEXT        NOT NULL,
    refresh_token   TEXT        NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_yahoo_oauth_tokens_expires
    ON yahoo_oauth_tokens (expires_at);
