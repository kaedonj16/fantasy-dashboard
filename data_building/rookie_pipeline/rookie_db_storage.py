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
            ADD COLUMN IF NOT EXISTS eval_snap_counts NUMERIC;
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
                             eval_true_early_declare, eval_snap_counts)
                        VALUES
                            (%s, %s, %s, %s, %s, NOW(),
                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            eval_snap_counts        = EXCLUDED.eval_snap_counts
                        """,
                        (
                            player_id,
                            int(season),
                            "rookie_eval",
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
                            _metric_value(metrics, "snap_counts"),
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

    # Bridge profiles into player_advanced_metrics so model training can use
    # rookie evaluation fields as features.
    bridge_result = bridge_to_advanced_metrics(as_of_date, draft_class_year, rookie_profiles)

    return {
        "db_metrics_rows": metrics_rows,
        "db_profiles_rows": profile_rows,
        "db_runs_rows": run_rows,
        "db_bridge_rows": bridge_result,
    }


def bridge_to_advanced_metrics(
    as_of_date: str,
    draft_class_year: int,
    profiles: List[Dict],
) -> Dict[str, int]:
    """
    Bridge rookie evaluation profiles into player_advanced_metrics.

    Calls merge_rookie_profiles_to_advanced_metrics from advanced_metrics.py
    so that rookie_eval_* columns in player_advanced_metrics are populated.
    These columns are then available as features in value_model_training.py.

    Args:
        as_of_date:        ISO date string (YYYY-MM-DD).
        draft_class_year:  Draft class year (e.g. 2026).
        profiles:          List of rookie profile dicts from evaluation pipeline.

    Returns:
        {"updated": n, "inserted": n, "skipped": n}
    """
    if not profiles or not _db_available():
        return {"updated": 0, "inserted": 0, "skipped": 0}

    try:
        from data_building.advanced_metrics import merge_rookie_profiles_to_advanced_metrics
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            result = merge_rookie_profiles_to_advanced_metrics(profiles, as_of_date, conn=conn)
        print(
            f"[rookie_db_storage] bridge_to_advanced_metrics class={draft_class_year} "
            f"updated={result.get('updated')} inserted={result.get('inserted')} "
            f"skipped={result.get('skipped')}"
        )
        return result
    except Exception as exc:
        print(f"[rookie_db_storage] bridge_to_advanced_metrics failed: {exc}")
        return {"updated": 0, "inserted": 0, "skipped": 0, "error": str(exc)}
