-- Migration 016: Trade Intelligence Engine
-- Stores crawled trades from thousands of Sleeper leagues for market analysis

CREATE TABLE IF NOT EXISTS trade_intel_leagues (
    league_id       TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    num_teams       INTEGER,
    scoring_type    TEXT,        -- ppr, half, std
    league_type     INTEGER,     -- 0=redraft, 2=dynasty
    discovered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_crawled_at TIMESTAMP,
    last_crawled_week INTEGER,
    crawl_enabled   BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_til_season ON trade_intel_leagues(season);
CREATE INDEX IF NOT EXISTS idx_til_crawl ON trade_intel_leagues(crawl_enabled, last_crawled_at);

-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS trade_intel_trades (
    id              BIGSERIAL PRIMARY KEY,
    league_id       TEXT NOT NULL,
    transaction_id  TEXT NOT NULL UNIQUE,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    status          TEXT,        -- complete, failed
    created_at      TIMESTAMP,   -- when the trade was made in Sleeper
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (league_id) REFERENCES trade_intel_leagues(league_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tit_league ON trade_intel_trades(league_id);
CREATE INDEX IF NOT EXISTS idx_tit_season_week ON trade_intel_trades(season, week);
CREATE INDEX IF NOT EXISTS idx_tit_created ON trade_intel_trades(created_at);

-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS trade_intel_assets (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        BIGINT NOT NULL,
    side            CHAR(1) NOT NULL,  -- 'a' or 'b' (which roster received this asset)
    asset_type      TEXT NOT NULL,     -- 'player' or 'pick'
    player_id       TEXT,              -- Sleeper player_id (null for picks)
    pick_season     INTEGER,           -- for picks
    pick_round      INTEGER,           -- for picks
    pick_order      TEXT,              -- early/mid/late or null

    FOREIGN KEY (trade_id) REFERENCES trade_intel_trades(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tia_trade ON trade_intel_assets(trade_id);
CREATE INDEX IF NOT EXISTS idx_tia_player ON trade_intel_assets(player_id);
CREATE INDEX IF NOT EXISTS idx_tia_type ON trade_intel_assets(asset_type);

-- ---------------------------------------------------------------
-- Materialized stats refreshed by the analytics job

CREATE TABLE IF NOT EXISTS trade_intel_player_stats (
    player_id           TEXT NOT NULL,
    season              INTEGER NOT NULL,
    week_updated        INTEGER,
    trade_count         INTEGER DEFAULT 0,
    trade_count_7d      INTEGER DEFAULT 0,
    trade_count_30d     INTEGER DEFAULT 0,
    avg_package_value   DECIMAL(10,2),  -- avg total value of packages that included this player
    avg_received_value  DECIMAL(10,2),  -- avg value of what was received for this player
    avg_sent_value      DECIMAL(10,2),  -- avg value of what was given up for this player
    market_value_1qb    DECIMAL(10,2),  -- implied market value 1QB
    market_value_sf     DECIMAL(10,2),  -- implied market value SF
    buy_count           INTEGER DEFAULT 0,  -- times acquired
    sell_count          INTEGER DEFAULT 0,  -- times sent away
    buy_sell_ratio      DECIMAL(5,2),
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (player_id, season)
);

CREATE INDEX IF NOT EXISTS idx_tips_season ON trade_intel_player_stats(season);
CREATE INDEX IF NOT EXISTS idx_tips_trade_count ON trade_intel_player_stats(trade_count_7d DESC);

-- ---------------------------------------------------------------
-- Common trade packages: which players/picks frequently move together

CREATE TABLE IF NOT EXISTS trade_intel_packages (
    id              BIGSERIAL PRIMARY KEY,
    anchor_player_id TEXT NOT NULL,    -- the player being "traded for"
    package_key     TEXT NOT NULL,     -- sorted player_ids joined (deterministic)
    season          INTEGER NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    avg_value_diff  DECIMAL(10,2),     -- avg model value delta of this package
    last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (anchor_player_id, package_key, season)
);

CREATE INDEX IF NOT EXISTS idx_tip_anchor ON trade_intel_packages(anchor_player_id, season);
