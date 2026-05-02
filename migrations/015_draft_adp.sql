-- Draft ADP: tracks actual picks from startup and rookie drafts across all
-- known leagues to produce community ADP data segmented by draft type,
-- superflex format, league size, and season.

-- Track which leagues have had their drafts indexed and when, so we don't
-- re-hit the Sleeper draft list endpoint for every league on every daily run.
ALTER TABLE trade_intel_leagues
    ADD COLUMN IF NOT EXISTS last_draft_adp_crawled_at TIMESTAMP DEFAULT NULL;

-- One row per completed draft we have pick data for.
CREATE TABLE IF NOT EXISTS draft_adp_drafts (
    draft_id        TEXT PRIMARY KEY,
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    draft_type      TEXT NOT NULL,      -- 'startup' or 'rookie'
    num_teams       INTEGER,
    is_superflex    BOOLEAN NOT NULL DEFAULT FALSE,
    rounds          INTEGER,
    status          TEXT,               -- 'complete', 'drafting', 'pre_draft'
    total_picks     INTEGER,
    crawled_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_league   ON draft_adp_drafts (league_id);
CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_season   ON draft_adp_drafts (season);
CREATE INDEX IF NOT EXISTS idx_draft_adp_drafts_type_sf  ON draft_adp_drafts (draft_type, is_superflex);

-- Raw pick data — one row per pick in each crawled draft.
CREATE TABLE IF NOT EXISTS draft_adp_picks (
    id              BIGSERIAL PRIMARY KEY,
    draft_id        TEXT NOT NULL REFERENCES draft_adp_drafts(draft_id) ON DELETE CASCADE,
    player_id       TEXT NOT NULL,
    pick_no         INTEGER NOT NULL,
    round           INTEGER,
    pick_in_round   INTEGER,
    roster_id       TEXT,
    UNIQUE (draft_id, pick_no)
);

CREATE INDEX IF NOT EXISTS idx_draft_adp_picks_player ON draft_adp_picks (player_id);
CREATE INDEX IF NOT EXISTS idx_draft_adp_picks_draft  ON draft_adp_picks (draft_id);

-- Aggregated ADP — recomputed after each crawl batch.
-- Segmented by draft_type / season / superflex / num_teams so callers can
-- filter for exactly the context they care about.
CREATE TABLE IF NOT EXISTS draft_adp (
    player_id       TEXT        NOT NULL,
    draft_type      TEXT        NOT NULL,   -- 'startup' or 'rookie'
    season          INTEGER     NOT NULL,
    is_superflex    BOOLEAN     NOT NULL,
    num_teams       INTEGER     NOT NULL,
    avg_pick        NUMERIC(7,2),
    std_pick        NUMERIC(7,2),
    avg_round       NUMERIC(5,2),
    sample_size     INTEGER,
    updated_at      TIMESTAMP,
    PRIMARY KEY (player_id, draft_type, season, is_superflex, num_teams)
);

CREATE INDEX IF NOT EXISTS idx_draft_adp_lookup ON draft_adp (draft_type, season, is_superflex, num_teams);
