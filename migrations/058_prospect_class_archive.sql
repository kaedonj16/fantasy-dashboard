-- 058: Past-class prospect archive boards.
--
-- The Prospects page can browse one draft class at a time, but only
-- classes the live pipeline has stored in rookie_rankings (2025+)
-- have boards. This table holds one graded row per DRAFTED QB/RB/WR/TE
-- for the classes the reference-population backfill grades
-- (2016-2026), so those classes get boards of their own and a combined
-- all-classes board can rank every class together by grade.
--
-- ARCHIVE, NOT REFERENCE: this is a separate surface from
-- prospect_reference_population (migration 050), which stays
-- reference-only (percentile anchors, never read by a board query).
-- The archive carries the overall grade (prospect_score) the
-- reference table deliberately omits, plus school and the actual
-- draft pick as reader context. Grades are capital-free talent
-- grades; the pick never feeds the grade.
--
-- All rows in this table must come from ONE model version for the
-- cross-class board to mean anything: the backfill grades every
-- class with the model current at run time, and model_version is
-- stored per row so a mixed-version state is visible, never silent.
--
-- Populated by: python scripts/build_reference_population.py
-- (the same per-year run that writes the reference rows; resumable
-- per class year, upserts on the unique key below).
-- Rollback: drop table prospect_class_archive;

create table if not exists prospect_class_archive (
    id                      serial primary key,
    player_name             text not null,
    position                text not null,
    draft_class_year        integer not null,
    school                  text,
    actual_pick             integer,
    prospect_score          double precision,
    production_score        double precision,
    efficiency_score        double precision,
    breakout_profile_score  double precision,
    competition_score       double precision,
    model_version           text not null,
    created_at              timestamp not null default now(),
    updated_at              timestamp not null default now(),
    unique (position, draft_class_year, player_name)
);

create index if not exists idx_pca_year
    on prospect_class_archive(draft_class_year);
create index if not exists idx_pca_position
    on prospect_class_archive(position);
