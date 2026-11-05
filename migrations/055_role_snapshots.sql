-- Frontier 1 (week-over-week role snapshots): one compact role map
-- per league/season/week, captured at serve time from the Wave 2
-- situation layer (utils/role_state.py). Wave 2 v1 deliberately had
-- no snapshot store, so no surface could say "role changed since
-- last week"; this table is that store (utils/role_snapshots.py).
--
-- League-level (not per-account): a depth chart is the same for
-- every viewer of the league. states is the compact map
-- {player_id: {state, team, pos, depth_order}}.
-- IF NOT EXISTS so re-runs are safe; the code also creates the table
-- idempotently and fails soft when it is absent.
CREATE TABLE IF NOT EXISTS role_snapshots (
    platform    TEXT NOT NULL,
    league_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    states      JSONB NOT NULL DEFAULT '{}'::jsonb,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, league_id, season, week)
);
