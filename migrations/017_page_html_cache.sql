-- Shared page-HTML cache used by background build workers.
-- Replaces the in-process DASHBOARD_CACHE["page_html"] dict so that all
-- gunicorn workers can see the built HTML regardless of which worker ran
-- the background build.

CREATE TABLE IF NOT EXISTS page_html_cache (
    platform   TEXT        NOT NULL,
    season     INT         NOT NULL,
    league_id  TEXT        NOT NULL,
    page       TEXT        NOT NULL,
    html       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (platform, season, league_id, page)
);

CREATE INDEX IF NOT EXISTS ix_page_html_cache_created
    ON page_html_cache (created_at);
