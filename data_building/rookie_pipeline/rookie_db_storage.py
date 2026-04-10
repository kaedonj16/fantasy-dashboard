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


def save_rookie_evaluation_to_db(
    as_of_date: str,
    draft_class_year: int,
    by_player_metrics: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    rookie_profiles: List[Dict[str, Any]],
    run_metadata: Dict[str, Any],
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
            for player_id, seasons in by_player_metrics.items():
                for season, metrics in (seasons or {}).items():
                    missing_metrics = missing_by_player.get(player_id) or {}
                    cur.execute(
                        """
                        INSERT INTO rookie_prospect_source_data
                            (player_id, season, source,
                             rookie_eval_metrics, rookie_eval_missing, rookie_eval_updated_at,
                             eval_routes_run, eval_yprr, eval_tprr,
                             eval_yac_per_att, eval_mtf_per_att, eval_explosive_run_rate,
                             eval_adjusted_comp_pct, eval_twp_rate,
                             eval_player_level_sos, eval_perf_vs_top_def,
                             eval_true_early_declare, games_played, targets)
                        VALUES
                            (%s, %s, %s, %s, %s, NOW(),
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            targets                 = EXCLUDED.targets
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


