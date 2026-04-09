from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from psycopg.types.json import Json


def _db_available() -> bool:
    try:
        from dashboard_services.db import get_database_url

        _ = get_database_url()
        return True
    except Exception:
        return False


def init_rookie_eval_tables(conn) -> None:
    """Ensure rookie evaluation storage exists on the existing rookie tables."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE rookie_prospect_source_data
            ADD COLUMN IF NOT EXISTS rookie_eval_metrics JSONB,
            ADD COLUMN IF NOT EXISTS rookie_eval_missing JSONB,
            ADD COLUMN IF NOT EXISTS rookie_eval_updated_at TIMESTAMP;
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
    snapshot_dt = date.fromisoformat(as_of_date)

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
                            (player_id, season, source, rookie_eval_metrics, rookie_eval_missing, rookie_eval_updated_at)
                        VALUES
                            (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (player_id, season, source)
                        DO UPDATE SET
                            rookie_eval_metrics = EXCLUDED.rookie_eval_metrics,
                            rookie_eval_missing = EXCLUDED.rookie_eval_missing,
                            rookie_eval_updated_at = NOW()
                        """,
                        (
                            player_id,
                            int(season),
                            "rookie_eval",
                            Json(metrics),
                            Json(missing_metrics),
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
                    (snapshot_dt, draft_class_year, player_id, Json(profile)),
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
                (snapshot_dt, draft_class_year, Json(run_metadata)),
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
