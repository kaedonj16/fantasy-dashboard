"""
Persistence for in-season weekly breakout results.

Deliberately a SEPARATE table from the offseason ``breakout_opportunity_scores``
board: weekly and offseason results are computed from different evidence, mean
different things, and must never overwrite or masquerade as one another. The
weekly table is keyed on (player_id, season, as_of_week) so each week's snapshot
is its own row and the history is queryable.

A companion ``weekly_breakout_runs`` table records one row per scoring run
(season, cutoff week, mode, coverage, counts, status). It lets the API expose
freshness ("scored through week 6, 1 week stale") and lets the runner PRESERVE
the last good snapshot when a refresh fails - it records a skipped/stale run
without deleting the previous week's scores.

All DB access is confined to this module; the scorer itself is pure.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from dashboard_services.db import get_conn

WEEKLY_SCORES_TABLE = "weekly_breakout_scores"
WEEKLY_RUNS_TABLE = "weekly_breakout_runs"

_INIT_DONE = False


def init_weekly_breakout_db() -> None:
    """Create the weekly tables and indexes if absent. Idempotent; safe to call
    on every run (acts as the migration for existing databases)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {WEEKLY_SCORES_TABLE} (
                id               SERIAL PRIMARY KEY,
                player_id        VARCHAR(50) NOT NULL,
                player_name      VARCHAR(255),
                season           INTEGER NOT NULL,
                as_of_week       INTEGER NOT NULL,
                as_of_date       DATE NOT NULL,
                team             VARCHAR(10),
                position         VARCHAR(5),
                scoring_version  VARCHAR(40) NOT NULL,
                classification   VARCHAR(30),
                breakout_score   NUMERIC,
                confidence       NUMERIC,
                provisional      BOOLEAN DEFAULT FALSE,
                baseline_source  VARCHAR(20),
                evaluated_weeks  INTEGER[],
                recent_weeks     INTEGER[],
                baseline_weeks   INTEGER[],
                recent_games     INTEGER,
                baseline_games   INTEGER,
                coverage_fraction NUMERIC,
                reasons          TEXT,
                risks            TEXT,
                evidence         JSONB,
                run_id           BIGINT,
                calculated_at    TIMESTAMP DEFAULT NOW(),
                UNIQUE (player_id, season, as_of_week, scoring_version)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbs_season_week "
            f"ON {WEEKLY_SCORES_TABLE} (season, as_of_week)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbs_score "
            f"ON {WEEKLY_SCORES_TABLE} (breakout_score DESC)"
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {WEEKLY_RUNS_TABLE} (
                id               SERIAL PRIMARY KEY,
                season           INTEGER NOT NULL,
                as_of_week       INTEGER NOT NULL,
                as_of_date       DATE NOT NULL,
                scoring_version  VARCHAR(40),
                mode             VARCHAR(20),
                status           VARCHAR(20),
                candidates_scored INTEGER DEFAULT 0,
                records_saved    INTEGER DEFAULT 0,
                expected_row_count INTEGER DEFAULT 0,
                inserted_row_count INTEGER DEFAULT 0,
                weeks_covered    INTEGER DEFAULT 0,
                detail           JSONB,
                completed_at     TIMESTAMP,
                calculated_at    TIMESTAMP DEFAULT NOW(),
                UNIQUE (season, as_of_week, scoring_version)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbr_season "
            f"ON {WEEKLY_RUNS_TABLE} (season, calculated_at DESC)"
        )
        # Online migration for pre-completeness deployments.  The score-table
        # uniqueness migration is deliberately done after adding scoring_version
        # to the key: two scorer versions are separate immutable snapshots.
        conn.execute(f"ALTER TABLE {WEEKLY_SCORES_TABLE} ADD COLUMN IF NOT EXISTS run_id BIGINT")
        conn.execute(f"ALTER TABLE {WEEKLY_RUNS_TABLE} ADD COLUMN IF NOT EXISTS expected_row_count INTEGER DEFAULT 0")
        conn.execute(f"ALTER TABLE {WEEKLY_RUNS_TABLE} ADD COLUMN IF NOT EXISTS inserted_row_count INTEGER DEFAULT 0")
        conn.execute(f"ALTER TABLE {WEEKLY_RUNS_TABLE} ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP")
        conn.execute(
            f"ALTER TABLE {WEEKLY_SCORES_TABLE} DROP CONSTRAINT IF EXISTS "
            f"weekly_breakout_scores_player_id_season_as_of_week_key"
        )
        conn.execute(
            f"ALTER TABLE {WEEKLY_RUNS_TABLE} DROP CONSTRAINT IF EXISTS "
            f"weekly_breakout_runs_season_as_of_week_status_key"
        )
        conn.execute(
            f"UPDATE {WEEKLY_RUNS_TABLE} SET scoring_version=NULL "
            f"WHERE status <> 'success' AND status <> 'completed'"
        )
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_wbs_snapshot_player_version "
            f"ON {WEEKLY_SCORES_TABLE} (player_id, season, as_of_week, scoring_version)"
        )
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uq_wbr_snapshot_version "
            f"ON {WEEKLY_RUNS_TABLE} (season, as_of_week, scoring_version)"
        )
        conn.execute(
            f"UPDATE {WEEKLY_SCORES_TABLE} s SET run_id=r.id "
            f"FROM {WEEKLY_RUNS_TABLE} r WHERE s.run_id IS NULL "
            f"AND r.status='success' AND r.records_saved > 0 "
            f"AND s.season=r.season AND s.as_of_week=r.as_of_week "
            f"AND s.scoring_version=r.scoring_version"
        )
        conn.execute(
            f"UPDATE {WEEKLY_RUNS_TABLE} SET status='completed', "
            f"expected_row_count=records_saved, inserted_row_count=records_saved, "
            f"completed_at=COALESCE(completed_at, calculated_at) "
            f"WHERE status='success' AND records_saved > 0"
        )
    _INIT_DONE = True


def _result_row(result: Dict[str, Any], season: int, as_of_week: int, as_of_date: date) -> Dict[str, Any]:
    evidence = {
        "signals": result.get("signals"),
        "sample": result.get("sample"),
        "coverage": result.get("coverage"),
        "confidence_detail": result.get("confidence_detail"),
        "fantasy": result.get("fantasy"),
        "score_basis": result.get("score_basis"),
        "previous_breakout_status": result.get("previous_breakout_status"),
        "established_role_penalty": result.get("established_role_penalty"),
        "established_player": result.get("established_player"),
        "established_role_score": result.get("established_role_score"),
        "role_novelty_score": result.get("role_novelty_score"),
        "role_novelty_reason": result.get("role_novelty_reason"),
        "early_watch": result.get("early_watch"),
        "main_board_eligible": result.get("main_board_eligible"),
        "main_board_rejection_reasons": result.get("main_board_rejection_reasons"),
        "baseline_method": result.get("baseline_method"),
        "baseline_games_used": result.get("baseline_games_used"),
        "partial_games_excluded": result.get("partial_games_excluded"),
        "baseline_quality": result.get("baseline_quality"),
        "subscores": {key: result.get(key) for key in (
            "role_change_score", "current_role_score", "sustainability_score",
            "breakout_novelty_score", "expectation_delta_score", "ranking_score")},
        "signal_diagnostics": {key: result.get(key) for key in (
            "supporting_signal_count", "supporting_signals", "conflicting_signals",
            "signal_agreement_score")},
        "opportunity": {key: result.get(key) for key in (
            "opportunity_source", "opportunity_source_confidence",
            "opportunity_source_reason")},
        "lifecycle": result.get("lifecycle"),
        "reasons": result.get("reasons"),
        "risks": result.get("risks"),
    }
    sample = result.get("sample") or {}
    return {
        "player_id": str(result.get("player_id") or ""),
        "player_name": result.get("player_name"),
        "season": int(season),
        "as_of_week": int(as_of_week),
        "as_of_date": as_of_date,
        "team": result.get("team"),
        "position": result.get("position"),
        "scoring_version": result.get("scoring_version"),
        "classification": result.get("classification"),
        "breakout_score": result.get("breakout_score"),
        "confidence": result.get("confidence"),
        "provisional": bool(result.get("provisional")),
        "baseline_source": result.get("baseline_source"),
        "evaluated_weeks": result.get("evaluated_weeks") or [],
        "recent_weeks": result.get("recent_weeks") or [],
        "baseline_weeks": result.get("baseline_weeks") or [],
        "recent_games": sample.get("recent_games"),
        "baseline_games": sample.get("baseline_games"),
        "coverage_fraction": (result.get("coverage") or {}).get("fraction"),
        "reasons": "\n".join(result.get("reasons") or []),
        "risks": "\n".join(result.get("risks") or []),
        "evidence": json.dumps(evidence),
    }


def save_weekly_scores(
    season: int,
    as_of_week: int,
    results: List[Dict[str, Any]],
    as_of_date: Optional[date] = None,
) -> int:
    """Backward-compatible entry point using the atomic publisher."""
    return publish_weekly_snapshot(
        season, as_of_week, results, mode="weekly", as_of_date=as_of_date,
        detail={"source": "legacy_save_weekly_scores"},
    )


def publish_weekly_snapshot(
    season: int,
    as_of_week: int,
    results: List[Dict[str, Any]],
    *,
    mode: str,
    weeks_covered: int = 0,
    detail: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[date] = None,
) -> int:
    """Atomically publish one complete, versioned weekly snapshot.

    Calculation happens before this function is called.  The run row, target
    snapshot replacement, inserts, count validation, and completed marker share
    one transaction.  Any exception therefore leaves the formerly completed
    snapshot visible and rolls back every byte of the attempted replacement.
    """
    if not results:
        raise ValueError("refusing to publish an empty weekly breakout snapshot")
    as_of_date = as_of_date or date.today()
    init_weekly_breakout_db()
    versions = {str(row.get("scoring_version") or "") for row in results}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError("snapshot rows must have one non-empty scoring_version")
    scoring_version = next(iter(versions))
    rows = [_result_row(row, season, as_of_week, as_of_date) for row in results]
    cols = [
        "player_id", "player_name", "season", "as_of_week", "as_of_date", "team",
        "position", "scoring_version", "classification", "breakout_score",
        "confidence", "provisional", "baseline_source", "evaluated_weeks",
        "recent_weeks", "baseline_weeks", "recent_games", "baseline_games",
        "coverage_fraction", "reasons", "risks", "evidence", "run_id",
    ]
    placeholders = ", ".join(
        f"%({column})s::jsonb" if column == "evidence" else f"%({column})s"
        for column in cols
    )
    expected = len(rows)
    with get_conn() as conn:
        run = conn.execute(
            f"""
            INSERT INTO {WEEKLY_RUNS_TABLE}
                (season, as_of_week, as_of_date, scoring_version, mode, status,
                 candidates_scored, records_saved, expected_row_count,
                 inserted_row_count, weeks_covered, detail, completed_at)
            VALUES (%s,%s,%s,%s,%s,'writing',%s,0,%s,0,%s,%s::jsonb,NULL)
            ON CONFLICT (season, as_of_week, scoring_version) DO UPDATE SET
                as_of_date=EXCLUDED.as_of_date, mode=EXCLUDED.mode,
                status='writing', candidates_scored=EXCLUDED.candidates_scored,
                records_saved=0, expected_row_count=EXCLUDED.expected_row_count,
                inserted_row_count=0, weeks_covered=EXCLUDED.weeks_covered,
                detail=EXCLUDED.detail, completed_at=NULL, calculated_at=NOW()
            RETURNING id
            """,
            (int(season), int(as_of_week), as_of_date, scoring_version, mode,
             expected, expected, int(weeks_covered), json.dumps(detail or {})),
        ).fetchone()
        run_id = int(run["id"])
        conn.execute(
            f"DELETE FROM {WEEKLY_SCORES_TABLE} WHERE season=%s AND as_of_week=%s "
            f"AND scoring_version=%s",
            (int(season), int(as_of_week), scoring_version),
        )
        for row in rows:
            row["run_id"] = run_id
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {WEEKLY_SCORES_TABLE} ({', '.join(cols)}) "
                f"VALUES ({placeholders})", rows,
            )
        inserted = conn.execute(
            f"SELECT COUNT(*) AS n FROM {WEEKLY_SCORES_TABLE} WHERE run_id=%s",
            (run_id,),
        ).fetchone()
        inserted_count = int(inserted["n"])
        if inserted_count != expected:
            raise RuntimeError(
                f"weekly snapshot incomplete: expected {expected}, inserted {inserted_count}"
            )
        completed_detail = dict(detail or {})
        completed_telemetry = dict(completed_detail.get("pipeline_telemetry") or {})
        completed_telemetry["inserted_rows"] = inserted_count
        completed_detail.update({
            "pipeline_telemetry": completed_telemetry,
            "expected_row_count": expected,
            "inserted_row_count": inserted_count,
        })
        conn.execute(
            f"UPDATE {WEEKLY_RUNS_TABLE} SET status='completed', records_saved=%s, "
            f"inserted_row_count=%s, completed_at=NOW(), detail=%s::jsonb WHERE id=%s",
            (inserted_count, inserted_count, json.dumps(completed_detail), run_id),
        )
    return expected


def record_run(
    season: int,
    as_of_week: int,
    *,
    mode: str,
    status: str,
    candidates_scored: int = 0,
    records_saved: int = 0,
    weeks_covered: int = 0,
    detail: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[date] = None,
) -> None:
    """Record a non-published attempt (normally ``skipped`` or ``stale``).

    Completed runs are created only by :func:`publish_weekly_snapshot`, so an
    error report can never overwrite the completion marker for a served run.
    """
    if as_of_date is None:
        as_of_date = date.today()
    init_weekly_breakout_db()
    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {WEEKLY_RUNS_TABLE}
                (season, as_of_week, as_of_date, scoring_version, mode, status,
                 candidates_scored, records_saved, weeks_covered, detail)
            VALUES (%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                int(season), int(as_of_week), as_of_date,
                mode, status, int(candidates_scored), int(records_saved), int(weeks_covered),
                json.dumps(detail or {}),
            ),
        )


def latest_scored_week(season: int) -> Optional[int]:
    """Most recent snapshot compatible with the current scorer.

    Old rows remain as history, but are never advertised as the current board.
    """
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT MAX(r.as_of_week) AS w FROM {WEEKLY_RUNS_TABLE} r "
            f"WHERE r.season = %s AND r.scoring_version = %s "
            f"AND r.status='completed' AND r.completed_at IS NOT NULL "
            f"AND r.expected_row_count > 0 "
            f"AND r.inserted_row_count = r.expected_row_count "
            f"AND (SELECT COUNT(*) FROM {WEEKLY_SCORES_TABLE} s "
            f"     WHERE s.run_id=r.id) = r.inserted_row_count",
            (int(season), SCORING_VERSION),
        ).fetchone()
    return int(row["w"]) if row and row.get("w") is not None else None


def has_any_weekly_snapshot(season: int) -> bool:
    """Whether history exists, including an incompatible old scoring version."""
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT 1 AS present FROM {WEEKLY_SCORES_TABLE} WHERE season = %s LIMIT 1",
            (int(season),),
        ).fetchone()
    return bool(row)


def load_previous_week_scores(season: int, before_week: int) -> Dict[str, Dict[str, Any]]:
    """Latest compatible score per player before a new snapshot (lifecycle input)."""
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    init_weekly_breakout_db()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT ON (player_id) player_id, as_of_week, breakout_score, "
            f"classification, evidence FROM {WEEKLY_SCORES_TABLE} s "
            f"WHERE season=%s AND as_of_week < %s AND scoring_version=%s "
            f"AND EXISTS (SELECT 1 FROM {WEEKLY_RUNS_TABLE} r WHERE r.id=s.run_id "
            f"AND r.status='completed' AND r.completed_at IS NOT NULL "
            f"AND r.expected_row_count=r.inserted_row_count) "
            f"ORDER BY player_id, as_of_week DESC",
            (int(season), int(before_week), SCORING_VERSION),
        ).fetchall()
    return {str(row["player_id"]): dict(row) for row in rows}


def get_latest_run(season: int) -> Optional[Dict[str, Any]]:
    """Most recent run row for a season (any status), for freshness reporting."""
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT * FROM {WEEKLY_RUNS_TABLE} WHERE season = %s "
            f"ORDER BY calculated_at DESC LIMIT 1",
            (int(season),),
        ).fetchone()
    return dict(row) if row else None


def get_completed_run(season: int, as_of_week: int, scoring_version: str) -> Optional[Dict[str, Any]]:
    """Completion metadata for the exact snapshot selected by the reader."""
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT * FROM {WEEKLY_RUNS_TABLE} WHERE season=%s AND as_of_week=%s "
            f"AND scoring_version=%s AND status='completed' AND completed_at IS NOT NULL "
            f"AND expected_row_count=inserted_row_count",
            (int(season), int(as_of_week), scoring_version),
        ).fetchone()
    return dict(row) if row else None


def load_weekly_candidates(
    season: int,
    as_of_week: Optional[int] = None,
    *,
    min_score: float = 0.0,
    classifications: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Load the weekly board for a season.

    When as_of_week is None, uses the latest snapshot that has rows (so a failed
    refresh still serves the last good week rather than an empty board). Returns
    a payload with candidates plus freshness metadata.
    """
    init_weekly_breakout_db()
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    week = as_of_week if as_of_week is not None else latest_scored_week(season)
    if week is None:
        return {
            "season": season, "as_of_week": None, "candidates": [], "count": 0,
            "data_available": False, "data_status": "unavailable",
        }

    params: List[Any] = [int(season), int(week), SCORING_VERSION, float(min_score)]
    clause = ""
    if classifications:
        clause = " AND classification = ANY(%s)"
        params.append(list(classifications))
    query = (
        f"SELECT s.* FROM {WEEKLY_SCORES_TABLE} s "
        f"JOIN {WEEKLY_RUNS_TABLE} r ON r.id=s.run_id "
        f"WHERE s.season = %s AND s.as_of_week = %s AND s.scoring_version = %s "
        f"AND r.status='completed' AND r.completed_at IS NOT NULL "
        f"AND r.expected_row_count=r.inserted_row_count "
        f"AND breakout_score >= %s{clause} "
        f"ORDER BY breakout_score DESC, confidence DESC"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]

    if limit and limit > 0:
        rows = rows[:limit]

    completed_run = get_completed_run(season, int(week), SCORING_VERSION)
    run = get_latest_run(season)
    latest_run_week = run.get("as_of_week") if run else week
    weeks_stale = max(0, int(latest_run_week or week) - int(week)) if latest_run_week else 0

    as_of_date = rows[0]["as_of_date"] if rows else (completed_run or {}).get("as_of_date")
    detail = (completed_run or {}).get("detail") or {}
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except ValueError:
            detail = {}
    return {
        "season": season,
        "as_of_week": int(week),
        "as_of_date": as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else as_of_date,
        "candidates": rows,
        "count": len(rows),
        "data_available": True,
        "data_status": "stale" if weeks_stale >= 1 else "ok",
        "weeks_stale": weeks_stale,
        "scoring_version": SCORING_VERSION,
        "last_run_status": (run or {}).get("status"),
        "snapshot_status": (completed_run or {}).get("status"),
        "completed_at": (completed_run or {}).get("completed_at"),
        "expected_row_count": (completed_run or {}).get("expected_row_count"),
        "inserted_row_count": (completed_run or {}).get("inserted_row_count"),
        "pipeline_telemetry": detail.get("pipeline_telemetry") or {},
    }


def get_weekly_candidate(
    player_id: str,
    season: int,
    as_of_week: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """One player's latest weekly row (or the given week's)."""
    init_weekly_breakout_db()
    week = as_of_week if as_of_week is not None else latest_scored_week(season)
    if week is None:
        return None
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT s.* FROM {WEEKLY_SCORES_TABLE} s "
            f"JOIN {WEEKLY_RUNS_TABLE} r ON r.id=s.run_id "
            f"WHERE player_id = %s AND season = %s AND as_of_week = %s "
            f"AND s.scoring_version = %s AND r.status='completed' "
            f"AND r.completed_at IS NOT NULL AND r.expected_row_count=r.inserted_row_count",
            (str(player_id), int(season), int(week), SCORING_VERSION),
        ).fetchone()
    return dict(row) if row else None
