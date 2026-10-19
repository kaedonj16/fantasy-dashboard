-- 050: Graded reference population for prospect percentiles.
--
-- The board/modal percentiles (dashboard_services/rookie_api.py) are
-- computed within position against a reference population of graded
-- prospects. The stored classes alone are a thin pool (classes
-- 2025-2027, ~34 WRs), so the best player in the pool prints 100 on
-- every component. This table holds one graded row per DRAFTED
-- QB/RB/WR/TE from classes 2016-2024, scored by the production model
-- at backfill time, so the percentile pool spans the recorded drafted
-- population.
--
-- REFERENCE ONLY: these rows are percentile anchors, never prospects.
-- No board, values, or rankings query reads this table; the only
-- readers are the percentile population functions in rookie_api.py.
--
-- Populated by: python scripts/build_reference_population.py
-- (run once on Render; resumable per class year, upserts on the
-- unique key below). Rollback: drop table prospect_reference_population;

create table if not exists prospect_reference_population (
    id                      serial primary key,
    player_name             text not null,
    position                text not null,
    draft_class_year        integer not null,
    actual_pick             integer,
    production_score        double precision,
    efficiency_score        double precision,
    breakout_profile_score  double precision,
    competition_score       double precision,
    -- NULL when the player has no recorded testing: an untested
    -- player's model athleticism score is a floor, and floors never
    -- anchor the athleticism percentile scale.
    athleticism_score       double precision,
    tested                  boolean not null default false,
    -- Advanced-tab marks {label: value} from the same code path the
    -- live modal uses (_build_advanced_metrics), component rows
    -- excluded. A metric earns a population percentile only when the
    -- union pool holds >= 30 values for it at the position.
    advanced_marks          jsonb not null default '{}'::jsonb,
    -- Completed-season primary per-game rates (Production tab pool).
    season_rates            jsonb not null default '[]'::jsonb,
    -- Recorded combine/testing values {field: value} (Testing pool).
    testing_marks           jsonb not null default '{}'::jsonb,
    -- The production model version that graded the row at backfill
    -- time (the script imports the model from main; never pinned).
    model_version           text not null,
    created_at              timestamp not null default now(),
    updated_at              timestamp not null default now(),
    unique (position, draft_class_year, player_name)
);

create index if not exists idx_prp_position
    on prospect_reference_population(position);
create index if not exists idx_prp_year
    on prospect_reference_population(draft_class_year);
