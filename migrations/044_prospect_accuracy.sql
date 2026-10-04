-- Prospect accuracy tracking: NFL outcomes and accuracy reports.
--
-- Closes the feedback loop on rookie prospect grades by storing actual NFL
-- performance (Y+1 through Y+3) for each graded prospect, plus precomputed
-- accuracy aggregates by tier/position/draft class.
--
-- Populated by: python scripts/prospect_accuracy_report.py
-- Scheduled: annually in February via cron_daily.py (Step: prospect_accuracy_report)

-- ── NFL outcomes per graded prospect ──────────────────────────────────────
create table if not exists prospect_nfl_outcomes (
    player_id           text        primary key
        references historical_prospect_grades(player_id) on delete cascade,
    draft_class_year    integer     not null,
    position            text        not null,
    -- PPR fantasy points per NFL season (Y+1 = draft year)
    ppr_y1              numeric(8,2) not null default 0,
    ppr_y2              numeric(8,2) not null default 0,
    ppr_y3              numeric(8,2) not null default 0,
    ppr_peak            numeric(8,2) not null default 0,
    ppr_cumulative      numeric(9,2) not null default 0,
    games_y1            integer      not null default 0,
    games_y2            integer      not null default 0,
    games_y3            integer      not null default 0,
    seasons_with_data   integer      not null default 0,
    -- Hit definition: ppr_peak >= position threshold
    -- (QB 310, WR 220, RB 240, TE 175 — see scripts/prospect_accuracy_report.py)
    is_hit              boolean      not null default false,
    hit_season          integer,  -- 1, 2, or 3 (first season the threshold was met)
    updated_at          timestamp    default now()
);

create index if not exists idx_pno_year     on prospect_nfl_outcomes(draft_class_year);
create index if not exists idx_pno_position on prospect_nfl_outcomes(position);
create index if not exists idx_pno_hit      on prospect_nfl_outcomes(is_hit) where is_hit;

-- ── Precomputed accuracy aggregates ───────────────────────────────────────
create table if not exists prospect_accuracy_reports (
    id                  serial      primary key,
    report_year         integer     not null,  -- year the report was generated
    draft_class_year    integer     not null,
    position            text        not null,  -- QB/RB/WR/TE or 'ALL'
    tier                integer,               -- 1-6, NULL = all tiers
    n_players           integer     not null,
    n_with_nfl_data     integer     not null,
    n_hits              integer     not null,
    hit_rate            numeric(5,2),           -- percentage (0-100)
    avg_prospect_score  numeric(6,2),
    created_at          timestamp   default now(),
    unique (report_year, draft_class_year, position, tier)
);

create index if not exists idx_par_report on prospect_accuracy_reports(report_year);

-- ── Grade snapshot timestamp (for future pipeline runs) ───────────────────
-- Records when each historical grade was computed, so the accuracy loop can
-- distinguish pre-draft grades from post-draft re-grades.
alter table historical_prospect_grades
    add column if not exists graded_at timestamp;
