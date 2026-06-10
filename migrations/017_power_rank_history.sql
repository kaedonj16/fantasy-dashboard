-- Weekly power-ranking snapshots so the standings page can show movement arrows.
CREATE TABLE IF NOT EXISTS power_rank_history (
    league_id  TEXT    NOT NULL,
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    owner_key  TEXT    NOT NULL,
    rank       INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (league_id, season, week, owner_key)
);
