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
    """Create rookie evaluation snapshot tables if they do not already exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rookie_advanced_metrics_snapshots (
                snapshot_date DATE NOT NULL,
                draft_class_year INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                metrics_json JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (snapshot_date, draft_class_year, player_id, season)
            );
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

        with conn.cursor() as cur:
            for player_id, seasons in by_player_metrics.items():
                for season, metrics in (seasons or {}).items():
                    cur.execute(
                        """
                        INSERT INTO rookie_advanced_metrics_snapshots
                            (snapshot_date, draft_class_year, player_id, season, metrics_json, updated_at)
                        VALUES
                            (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (snapshot_date, draft_class_year, player_id, season)
                        DO UPDATE SET
                            metrics_json = EXCLUDED.metrics_json,
                            updated_at = NOW()
                        """,
                        (snapshot_dt, draft_class_year, player_id, int(season), Json(metrics)),
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

    return {
        "db_metrics_rows": metrics_rows,
        "db_profiles_rows": profile_rows,
        "db_runs_rows": run_rows,
    }
