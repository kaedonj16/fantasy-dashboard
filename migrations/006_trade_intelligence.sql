-- Trade Intelligence Engine tables.
-- Consolidates migrations 016, 017 (trade portions), 018.

CREATE TABLE IF NOT EXISTS trade_intel_leagues (
    league_id         TEXT        PRIMARY KEY,
    season            INTEGER     NOT NULL,
    num_teams         INTEGER,
    scoring_type      TEXT,        -- ppr, half, std
    league_type       INTEGER,     -- 2 = dynasty
    discovered_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    last_crawled_at   TIMESTAMP,
    last_crawled_week INTEGER,
    crawl_enabled     BOOLEAN     DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_til_season ON trade_intel_leagues(season);
CREATE INDEX IF NOT EXISTS idx_til_crawl  ON trade_intel_leagues(crawl_enabled, last_crawled_at);

CREATE TABLE IF NOT EXISTS trade_intel_trades (
    id              BIGSERIAL   PRIMARY KEY,
    league_id       TEXT        NOT NULL,
    transaction_id  TEXT        NOT NULL UNIQUE,
    season          INTEGER     NOT NULL,
    week            INTEGER     NOT NULL,
    status          TEXT,
    created_at      TIMESTAMP,
    ingested_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (league_id) REFERENCES trade_intel_leagues(league_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tit_league      ON trade_intel_trades(league_id);
CREATE INDEX IF NOT EXISTS idx_tit_season_week ON trade_intel_trades(season, week);
CREATE INDEX IF NOT EXISTS idx_tit_created     ON trade_intel_trades(created_at);

CREATE TABLE IF NOT EXISTS trade_intel_assets (
    id          BIGSERIAL   PRIMARY KEY,
    trade_id    BIGINT      NOT NULL,
    side        CHAR(1)     NOT NULL,  -- 'a' or 'b'
    asset_type  TEXT        NOT NULL,  -- 'player' or 'pick'
    player_id   TEXT,
    pick_season INTEGER,
    pick_round  INTEGER,
    pick_order  TEXT,                  -- early / mid / late
    FOREIGN KEY (trade_id) REFERENCES trade_intel_trades(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tia_trade  ON trade_intel_assets(trade_id);
CREATE INDEX IF NOT EXISTS idx_tia_player ON trade_intel_assets(player_id);
CREATE INDEX IF NOT EXISTS idx_tia_type   ON trade_intel_assets(asset_type);

CREATE TABLE IF NOT EXISTS trade_intel_player_stats (
    player_id                TEXT        NOT NULL,
    season                   INTEGER     NOT NULL,
    week_updated             INTEGER,
    -- Trade frequency
    trade_count              INTEGER     DEFAULT 0,
    trade_count_7d           INTEGER     DEFAULT 0,
    trade_count_14d          INTEGER     DEFAULT 0,
    trade_count_30d          INTEGER     DEFAULT 0,
    -- Flat market values (legacy; now equals weighted_market_value)
    market_value_1qb         DECIMAL(10,2),
    market_value_sf          DECIMAL(10,2),
    -- Time-decay weighted median (primary calibration signal)
    weighted_market_value_1qb DECIMAL(10,2),
    weighted_market_value_sf  DECIMAL(10,2),
    -- Window medians for trend math
    market_value_1qb_14d     DECIMAL(10,2),
    market_value_sf_14d      DECIMAL(10,2),
    market_value_1qb_30d     DECIMAL(10,2),
    market_value_sf_30d      DECIMAL(10,2),
    market_value_1qb_90d     DECIMAL(10,2),
    market_value_sf_90d      DECIMAL(10,2),
    -- Directional momentum: 14d median minus 90d median
    market_trend_1qb         DECIMAL(10,2),
    market_trend_sf          DECIMAL(10,2),
    -- Package stats
    avg_package_value        DECIMAL(10,2),
    avg_received_value       DECIMAL(10,2),
    avg_sent_value           DECIMAL(10,2),
    -- Buy/sell pressure
    buy_count                INTEGER     DEFAULT 0,
    sell_count               INTEGER     DEFAULT 0,
    buy_sell_ratio           DECIMAL(5,2),
    updated_at               TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, season)
);

CREATE INDEX IF NOT EXISTS idx_tips_season       ON trade_intel_player_stats(season);
CREATE INDEX IF NOT EXISTS idx_tips_trade_count  ON trade_intel_player_stats(trade_count_7d DESC);
CREATE INDEX IF NOT EXISTS idx_tips_weighted     ON trade_intel_player_stats(weighted_market_value_1qb DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_tips_trend        ON trade_intel_player_stats(market_trend_1qb DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS trade_intel_packages (
    id               BIGSERIAL   PRIMARY KEY,
    anchor_player_id TEXT        NOT NULL,
    package_key      TEXT        NOT NULL,  -- sorted companion player_ids joined by '|'
    season           INTEGER     NOT NULL,
    occurrence_count NUMERIC     DEFAULT 1, -- decay-weighted count
    avg_value_diff   DECIMAL(10,2),
    last_seen_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (anchor_player_id, package_key, season)
);

CREATE INDEX IF NOT EXISTS idx_tip_anchor ON trade_intel_packages(anchor_player_id, season);
