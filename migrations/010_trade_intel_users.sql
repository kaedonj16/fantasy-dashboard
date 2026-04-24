-- Stores Sleeper user IDs discovered during BFS expansion or site logins.
-- Used to seed future discovery runs from known dynasty league owners.
CREATE TABLE IF NOT EXISTS trade_intel_users (
    user_id        TEXT        PRIMARY KEY,
    username       TEXT,
    source         TEXT,       -- 'bfs' | 'login'
    discovered_at  TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    last_seeded_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tiu_seeded ON trade_intel_users(last_seeded_at ASC NULLS FIRST);
