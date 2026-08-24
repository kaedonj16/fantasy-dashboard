-- Normalized ADP snapshots from the tokenless global feeds (Yahoo / ESPN / MFL)
-- and, in future, any other normalized source. Disk snapshots under
-- data/adp_snapshots/ remain the source of truth for the request-path resolver;
-- this table is a durable, queryable mirror kept for historical ADP-movement
-- analysis. It is intentionally additive and never alters the existing
-- draft_adp tables, which stay the home of BR Fantasy's observed-draft data.
--
-- Every dimension a source does not resolve is stored NULL (unknown/mixed) rather
-- than guessed, so a row records exactly what its feed represents.

CREATE TABLE IF NOT EXISTS adp_snapshots (
    source        TEXT        NOT NULL,   -- 'yahoo' | 'espn' | 'mfl' | ...
    season        INTEGER     NOT NULL,
    player_id     TEXT        NOT NULL,   -- canonical (Sleeper) id
    adp           NUMERIC(7,2),           -- overall average draft pick
    draft_type    TEXT        NOT NULL,   -- 'redraft' | 'startup' | 'rookie'
    qb_format     TEXT        NOT NULL,   -- '1qb' | '2qb' | 'superflex' | 'mixed'
    ppr           NUMERIC(3,2),           -- 0 / 0.5 / 1.0; NULL = unknown/mixed
    te_premium    NUMERIC(4,2),           -- additional TE reception premium; NULL = unknown
    num_teams     INTEGER,                -- league size; NULL = unknown/mixed
    source_scope  TEXT,                   -- 'global' | 'observed' | 'platform'
    sample_size   INTEGER,
    min_pick      NUMERIC(7,2),
    max_pick      NUMERIC(7,2),
    draft_pct     NUMERIC(6,2),
    collected_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, season, player_id, draft_type, qb_format)
);

CREATE INDEX IF NOT EXISTS idx_adp_snapshots_lookup
    ON adp_snapshots (source, season, draft_type, qb_format);
CREATE INDEX IF NOT EXISTS idx_adp_snapshots_player
    ON adp_snapshots (player_id);
