from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json


def _db_available() -> bool:
    try:
        from dashboard_services.db import get_database_url

        _ = get_database_url()
        return True
    except Exception:
        return False


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively convert types that are not JSON-serializable.

    psycopg returns PostgreSQL NUMERIC columns as decimal.Decimal.
    Dates and datetimes also need to be stringified.  Everything else
    that isn't a basic JSON type is cast to str so serialisation never
    raises TypeError.
    """
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        # Preserve integer precision where possible
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    # bool must be checked before int (bool is subclass of int in Python)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str, type(None))):
        return obj
    # Fallback: cast to str so we never crash
    return str(obj)


def _metric_value(metrics: Dict[str, Any], name: str) -> Optional[float]:
    """Extract the scalar numeric value from a metric payload dict, or None."""
    entry = (metrics or {}).get(name)
    if not isinstance(entry, dict):
        return None
    val = entry.get("value")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _bool_metric_value(metrics: Dict[str, Any], name: str) -> Optional[bool]:
    """Extract the scalar boolean value from a metric payload dict, or None."""
    entry = (metrics or {}).get(name)
    if not isinstance(entry, dict):
        return None
    val = entry.get("value")
    if val is None:
        return None
    return bool(val)


def init_rookie_eval_tables(conn) -> None:
    """Ensure rookie evaluation storage exists on the existing rookie tables."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE rookie_prospect_source_data
            ADD COLUMN IF NOT EXISTS rookie_eval_metrics JSONB,
            ADD COLUMN IF NOT EXISTS rookie_eval_missing JSONB,
            ADD COLUMN IF NOT EXISTS rookie_eval_updated_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS eval_routes_run NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_yprr NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_tprr NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_yac_per_att NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_mtf_per_att NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_explosive_run_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_adjusted_comp_pct NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_twp_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_player_level_sos NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_perf_vs_top_def NUMERIC,
            ADD COLUMN IF NOT EXISTS eval_true_early_declare BOOLEAN,
            ADD COLUMN IF NOT EXISTS games_played NUMERIC;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rookie_profiles_snapshots (
                snapshot_date DATE NOT NULL,
                draft_class_year INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                profile_json JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (snapshot_date, draft_class_year, player_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rookie_evaluation_runs (
                snapshot_date DATE NOT NULL,
                draft_class_year INTEGER NOT NULL,
                run_metadata JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (snapshot_date, draft_class_year)
            );
            """
        )


def _raw_int(stats: Dict[str, Any], name: str) -> Optional[int]:
    val = stats.get(name)
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _raw_float(stats: Dict[str, Any], name: str) -> Optional[float]:
    val = stats.get(name)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def save_rookie_evaluation_to_db(
    as_of_date: str,
    draft_class_year: int,
    by_player_metrics: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    rookie_profiles: List[Dict[str, Any]],
    run_metadata: Dict[str, Any],
    raw_seasons_by_player: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
) -> Dict[str, int]:
    """Persist rookie advanced metrics and profiles snapshots to Postgres."""
    if not _db_available():
        return {"db_metrics_rows": 0, "db_profiles_rows": 0, "db_runs_rows": 0}

    from dashboard_services.db import get_conn

    metrics_rows = 0
    profile_rows = 0
    run_rows = 0
    snapshot_dt = datetime.date.fromisoformat(as_of_date)

    with get_conn() as conn:
        init_rookie_eval_tables(conn)
        missing_by_player = {
            p.get("player_id"): ((p.get("rookie_profile") or {}).get("missing") or {})
            for p in rookie_profiles
            if p.get("player_id")
        }

        with conn.cursor() as cur:
            # Union all player IDs so Sportradar-injected seasons with no resolved
            # eval metrics (empty metrics_by_season entry) still get DB rows.
            all_player_ids = set(by_player_metrics.keys()) | set((raw_seasons_by_player or {}).keys())
            for player_id in all_player_ids:
                metrics_by_season = by_player_metrics.get(player_id) or {}
                raw_by_season = (raw_seasons_by_player or {}).get(player_id) or {}
                all_seasons = set(metrics_by_season.keys()) | set(raw_by_season.keys())
                for season in all_seasons:
                    metrics = metrics_by_season.get(season) or {}
                    missing_metrics = missing_by_player.get(player_id) or {}
                    raw = raw_by_season.get(season) or {}
                    cur.execute(
                        """
                        INSERT INTO rookie_prospect_source_data
                            (player_id, season, source,
                             rookie_eval_metrics, rookie_eval_missing, rookie_eval_updated_at,
                             eval_routes_run, eval_yprr, eval_tprr,
                             eval_yac_per_att, eval_mtf_per_att, eval_explosive_run_rate,
                             eval_adjusted_comp_pct, eval_twp_rate,
                             eval_player_level_sos, eval_perf_vs_top_def,
                             eval_true_early_declare, games_played, targets,
                             receptions, receiving_yards, receiving_tds,
                             rush_attempts, rush_yards, rush_tds,
                             pass_attempts, pass_yards, pass_tds, completions, interceptions,
                             yds_per_carry, yds_per_reception, yds_per_attempt,
                             completion_pct, td_int_ratio,
                             yards_after_catch, yards_after_catch_per_reception,
                             avg_depth_of_target, contested_catch_rate, avoided_tackles,
                             drop_rate, slot_rate, wide_rate, inline_rate, pass_block_rate,
                             grades_offense, grades_pass_block,
                             explosive_runs_10_plus, breakaway_percentage, elusive_rating,
                             pff_rushing_grade, pff_passing_grade,
                             big_time_throw_rate, adjusted_completion_rate,
                             pressure_to_sack_rate, nfl_passer_rating)
                        VALUES
                            (%s, %s, %s, %s, %s, NOW(),
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                             %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, season, source)
                        DO UPDATE SET
                            rookie_eval_metrics     = EXCLUDED.rookie_eval_metrics,
                            rookie_eval_missing     = EXCLUDED.rookie_eval_missing,
                            rookie_eval_updated_at  = NOW(),
                            eval_routes_run         = EXCLUDED.eval_routes_run,
                            eval_yprr               = EXCLUDED.eval_yprr,
                            eval_tprr               = EXCLUDED.eval_tprr,
                            eval_yac_per_att        = EXCLUDED.eval_yac_per_att,
                            eval_mtf_per_att        = EXCLUDED.eval_mtf_per_att,
                            eval_explosive_run_rate = EXCLUDED.eval_explosive_run_rate,
                            eval_adjusted_comp_pct  = EXCLUDED.eval_adjusted_comp_pct,
                            eval_twp_rate           = EXCLUDED.eval_twp_rate,
                            eval_player_level_sos   = EXCLUDED.eval_player_level_sos,
                            eval_perf_vs_top_def    = EXCLUDED.eval_perf_vs_top_def,
                            eval_true_early_declare = EXCLUDED.eval_true_early_declare,
                            games_played            = EXCLUDED.games_played,
                            targets                 = EXCLUDED.targets,
                            receptions        = COALESCE(rookie_prospect_source_data.receptions,        EXCLUDED.receptions),
                            receiving_yards   = COALESCE(rookie_prospect_source_data.receiving_yards,   EXCLUDED.receiving_yards),
                            receiving_tds     = COALESCE(rookie_prospect_source_data.receiving_tds,     EXCLUDED.receiving_tds),
                            rush_attempts     = COALESCE(rookie_prospect_source_data.rush_attempts,     EXCLUDED.rush_attempts),
                            rush_yards        = COALESCE(rookie_prospect_source_data.rush_yards,        EXCLUDED.rush_yards),
                            rush_tds          = COALESCE(rookie_prospect_source_data.rush_tds,          EXCLUDED.rush_tds),
                            pass_attempts     = COALESCE(rookie_prospect_source_data.pass_attempts,     EXCLUDED.pass_attempts),
                            pass_yards        = COALESCE(rookie_prospect_source_data.pass_yards,        EXCLUDED.pass_yards),
                            pass_tds          = COALESCE(rookie_prospect_source_data.pass_tds,          EXCLUDED.pass_tds),
                            completions       = COALESCE(rookie_prospect_source_data.completions,       EXCLUDED.completions),
                            interceptions     = COALESCE(rookie_prospect_source_data.interceptions,     EXCLUDED.interceptions),
                            yds_per_carry     = COALESCE(rookie_prospect_source_data.yds_per_carry,     EXCLUDED.yds_per_carry),
                            yds_per_reception = COALESCE(rookie_prospect_source_data.yds_per_reception, EXCLUDED.yds_per_reception),
                            yds_per_attempt   = COALESCE(rookie_prospect_source_data.yds_per_attempt,   EXCLUDED.yds_per_attempt),
                            completion_pct    = COALESCE(rookie_prospect_source_data.completion_pct,    EXCLUDED.completion_pct),
                            td_int_ratio      = COALESCE(rookie_prospect_source_data.td_int_ratio,      EXCLUDED.td_int_ratio),
                            yards_after_catch = COALESCE(rookie_prospect_source_data.yards_after_catch, EXCLUDED.yards_after_catch),
                            yards_after_catch_per_reception = COALESCE(rookie_prospect_source_data.yards_after_catch_per_reception, EXCLUDED.yards_after_catch_per_reception),
                            avg_depth_of_target    = COALESCE(rookie_prospect_source_data.avg_depth_of_target,    EXCLUDED.avg_depth_of_target),
                            contested_catch_rate   = COALESCE(rookie_prospect_source_data.contested_catch_rate,   EXCLUDED.contested_catch_rate),
                            avoided_tackles        = COALESCE(rookie_prospect_source_data.avoided_tackles,        EXCLUDED.avoided_tackles),
                            drop_rate              = COALESCE(rookie_prospect_source_data.drop_rate,              EXCLUDED.drop_rate),
                            slot_rate              = COALESCE(rookie_prospect_source_data.slot_rate,              EXCLUDED.slot_rate),
                            wide_rate              = COALESCE(rookie_prospect_source_data.wide_rate,              EXCLUDED.wide_rate),
                            inline_rate            = COALESCE(rookie_prospect_source_data.inline_rate,            EXCLUDED.inline_rate),
                            pass_block_rate        = COALESCE(rookie_prospect_source_data.pass_block_rate,        EXCLUDED.pass_block_rate),
                            grades_offense         = COALESCE(rookie_prospect_source_data.grades_offense,         EXCLUDED.grades_offense),
                            grades_pass_block      = COALESCE(rookie_prospect_source_data.grades_pass_block,      EXCLUDED.grades_pass_block),
                            explosive_runs_10_plus = COALESCE(rookie_prospect_source_data.explosive_runs_10_plus, EXCLUDED.explosive_runs_10_plus),
                            breakaway_percentage   = COALESCE(rookie_prospect_source_data.breakaway_percentage,   EXCLUDED.breakaway_percentage),
                            elusive_rating         = COALESCE(rookie_prospect_source_data.elusive_rating,         EXCLUDED.elusive_rating),
                            pff_rushing_grade      = COALESCE(rookie_prospect_source_data.pff_rushing_grade,      EXCLUDED.pff_rushing_grade),
                            pff_passing_grade      = COALESCE(rookie_prospect_source_data.pff_passing_grade,      EXCLUDED.pff_passing_grade),
                            big_time_throw_rate    = COALESCE(rookie_prospect_source_data.big_time_throw_rate,    EXCLUDED.big_time_throw_rate),
                            adjusted_completion_rate = COALESCE(rookie_prospect_source_data.adjusted_completion_rate, EXCLUDED.adjusted_completion_rate),
                            pressure_to_sack_rate  = COALESCE(rookie_prospect_source_data.pressure_to_sack_rate,  EXCLUDED.pressure_to_sack_rate),
                            nfl_passer_rating      = COALESCE(rookie_prospect_source_data.nfl_passer_rating,      EXCLUDED.nfl_passer_rating)
                        """,
                        (
                            player_id,
                            int(season),
                            "cfbd",
                            Json(_to_json_safe(metrics)),
                            Json(_to_json_safe(missing_metrics)),
                            _metric_value(metrics, "routes_run"),
                            _metric_value(metrics, "yprr"),
                            _metric_value(metrics, "tprr"),
                            _metric_value(metrics, "yac_per_att"),
                            _metric_value(metrics, "mtf_per_att"),
                            _metric_value(metrics, "explosive_run_rate"),
                            _metric_value(metrics, "adjusted_comp_pct"),
                            _metric_value(metrics, "twp_rate"),
                            _metric_value(metrics, "player_level_sos"),
                            _metric_value(metrics, "performance_vs_top_defenses"),
                            _bool_metric_value(metrics, "true_early_declare"),
                            _metric_value(metrics, "games_played"),
                            _metric_value(metrics, "targets"),
                            # Raw production stats from Sportradar/source season record
                            _raw_int(raw, "receptions"),
                            _raw_int(raw, "receiving_yards"),
                            _raw_int(raw, "receiving_tds"),
                            _raw_int(raw, "rush_attempts"),
                            _raw_int(raw, "rush_yards"),
                            _raw_int(raw, "rush_tds"),
                            _raw_int(raw, "pass_attempts"),
                            _raw_int(raw, "pass_yards"),
                            _raw_int(raw, "pass_tds"),
                            _raw_int(raw, "completions"),
                            _raw_int(raw, "interceptions"),
                            _raw_float(raw, "yds_per_carry"),
                            _raw_float(raw, "yds_per_reception"),
                            _raw_float(raw, "yds_per_attempt"),
                            _raw_float(raw, "completion_pct"),
                            _raw_float(raw, "td_int_ratio"),
                            # Advanced metrics (migration 011)
                            # yards_after_catch comes from Sportradar for all seasons;
                            # the remaining 19 fields come from the CSV import (2025 only)
                            # and are preserved via COALESCE on conflict.
                            _raw_float(raw, "yards_after_catch"),
                            _raw_float(raw, "yards_after_catch_per_reception"),
                            _raw_float(raw, "avg_depth_of_target"),
                            _raw_float(raw, "contested_catch_rate"),
                            _raw_int(raw,   "avoided_tackles"),
                            _raw_float(raw, "drop_rate"),
                            _raw_float(raw, "slot_rate"),
                            _raw_float(raw, "wide_rate"),
                            _raw_float(raw, "inline_rate"),
                            _raw_float(raw, "pass_block_rate"),
                            _raw_float(raw, "grades_offense"),
                            _raw_float(raw, "grades_pass_block"),
                            _raw_int(raw,   "explosive_runs_10_plus"),
                            _raw_float(raw, "breakaway_percentage"),
                            _raw_float(raw, "elusive_rating"),
                            _raw_float(raw, "pff_rushing_grade"),
                            _raw_float(raw, "pff_passing_grade"),
                            _raw_float(raw, "big_time_throw_rate"),
                            _raw_float(raw, "adjusted_completion_rate"),
                            _raw_float(raw, "pressure_to_sack_rate"),
                            _raw_float(raw, "nfl_passer_rating"),
                        ),
                    )
                    metrics_rows += 1

            for profile in rookie_profiles:
                player_id = profile.get("player_id")
                if not player_id:
                    continue
                cur.execute(
                    """
                    INSERT INTO rookie_profiles_snapshots
                        (snapshot_date, draft_class_year, player_id, profile_json, updated_at)
                    VALUES
                        (%s, %s, %s, %s, NOW())
                    ON CONFLICT (snapshot_date, draft_class_year, player_id)
                    DO UPDATE SET
                        profile_json = EXCLUDED.profile_json,
                        updated_at = NOW()
                    """,
                    (snapshot_dt, draft_class_year, player_id, Json(_to_json_safe(profile))),
                )
                profile_rows += 1

            cur.execute(
                """
                INSERT INTO rookie_evaluation_runs
                    (snapshot_date, draft_class_year, run_metadata)
                VALUES
                    (%s, %s, %s)
                ON CONFLICT (snapshot_date, draft_class_year)
                DO UPDATE SET
                    run_metadata = EXCLUDED.run_metadata,
                    created_at = NOW()
                """,
                (snapshot_dt, draft_class_year, Json(_to_json_safe(run_metadata))),
            )
            run_rows = 1

    return {
        "db_metrics_rows": metrics_rows,
        "db_profiles_rows": profile_rows,
        "db_runs_rows": run_rows,
    }


def backfill_bio_from_sportradar(
    bio_updates: Dict[str, Dict[str, Any]],
) -> int:
    """
    Update height_inches / weight_lbs on rookie_prospects where currently NULL.

    bio_updates: {player_id: {"height_inches": int|None, "weight_lbs": int|None}}
    Uses COALESCE so existing values are never overwritten.
    Returns the number of rows updated.
    """
    if not bio_updates or not _db_available():
        return 0

    from dashboard_services.db import get_conn

    updated = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for player_id, bio in bio_updates.items():
                h = bio.get("height_inches")
                w = bio.get("weight_lbs")
                if h is None and w is None:
                    continue
                cur.execute(
                    """
                    UPDATE rookie_prospects
                    SET height_inches = COALESCE(height_inches, %s),
                        weight_lbs    = COALESCE(weight_lbs,    %s)
                    WHERE player_id = %s
                      AND (height_inches IS NULL OR weight_lbs IS NULL)
                    """,
                    (h, w, player_id),
                )
                updated += cur.rowcount
        conn.commit()
    return updated


