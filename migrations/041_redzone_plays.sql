-- Server-side RedZone play store. One poller (advisory-lock elected) fetches
-- play-by-play upstream every ~15s during live games and upserts here; page
-- loads and the TD notifier read from this table instead of hitting ESPN/
-- Tank01 per viewer. Plays are immutable once final, so upserts are idempotent.
CREATE TABLE IF NOT EXISTS redzone_plays (
    season      INTEGER NOT NULL,
    game_id     TEXT NOT NULL,
    play_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    is_td       BOOLEAN NOT NULL DEFAULT FALSE,
    payload     JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, game_id, play_id)
);
CREATE INDEX IF NOT EXISTS idx_redzone_plays_game_seq
    ON redzone_plays (season, game_id, seq);
CREATE INDEX IF NOT EXISTS idx_redzone_plays_td_seen
    ON redzone_plays (season, observed_at) WHERE is_td;
CREATE INDEX IF NOT EXISTS idx_redzone_plays_observed
    ON redzone_plays (observed_at);
